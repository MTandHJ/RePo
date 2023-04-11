

import torch, os, math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp

from torch_geometric.data import HeteroData 
from torch_geometric.utils import to_scipy_sparse_matrix

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN
from freerec.utils import mkdirs, timemeter
from freeplot.utils import export_pickle, import_pickle


freerec.decalre(version='0.3.1')


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--alpha", type=int, default=0)
cfg.add_argument("--beta", type=float, default=2.)
cfg.add_argument("--req-vec", type=int, default=500)

cfg.set_defaults(
    description="SVD-GCN",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=100,
    batch_size=2048,
    optimizer='sgd',
    lr=15,
    weight_decay=1.e-4,
    seed=1
)
cfg.compile()


class SVDGCN(RecSysArch):

    def __init__(self, tokenizer: FieldModuleList) -> None:
        super().__init__()

        self.reg = cfg.weight_decay
        self.beta = cfg.beta
        self.User = tokenizer[USER, ID]
        self.Item = tokenizer[ITEM, ID]

        self.register_parameter(
            'FS', Parameter(
                torch.randn(cfg.req_vec, cfg.embedding_dim),
                requires_grad=True
            )
        )
        nn.init.uniform_(self.FS, -np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)), np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)))

    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    def save(self, data):
        path = os.path.join("filters", cfg.dataset, str(int(cfg.alpha)))
        mkdirs(path)
        file_ = os.path.join(path, "u_s_v.pickle")
        export_pickle(data, file_)

    @timemeter("SVDGCN/load")
    def load(self, graph: HeteroData):
        path = os.path.join("filters", cfg.dataset, str(int(cfg.alpha)))
        file_ = os.path.join(path, "u_s_v.pickle")
        try:
            data = import_pickle(file_)
        except ImportError:
            data = self.preprocess(graph)
            self.save(data)
        
        U, vals, V = data['U'], data['vals'], data['V']
        vals = self.weight_func(vals[:cfg.req_vec])
        U = U[:, :cfg.req_vec] * vals
        V = V[:, :cfg.req_vec] * vals
        self.register_buffer("U", U)
        self.register_buffer("V", V)

    def preprocess(self, graph: HeteroData):
        R = sp.lil_array(to_scipy_sparse_matrix(
            graph[graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        userDegs = R.sum(axis=1).squeeze() + cfg.alpha
        itemDegs = R.sum(axis=0).squeeze() + cfg.alpha
        userDegs = 1 / np.sqrt(userDegs)
        itemDegs = 1 / np.sqrt(itemDegs)
        userDegs[np.isinf(userDegs)] = 0.
        itemDegs[np.isinf(itemDegs)] = 0.
        R = (userDegs.reshape(-1, 1) * R * itemDegs).tocoo()
        rows = torch.from_numpy(R.row).long()
        cols = torch.from_numpy(R.col).long()
        vals = torch.from_numpy(R.data)
        indices = torch.stack((rows, cols), dim=0)
        R = torch.sparse_coo_tensor(
            indices, vals, size=R.shape
        )

        U, vals, V = torch.svd_lowrank(R, q=1000, niter=30)

        data = {'U': U.cpu(), 'vals': vals.cpu(), 'V': V.cpu()}
        return data

    def forward(
        self, users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        samp_user = users.flatten()
        pos_item = positives.flatten()
        neg_item = negatives.flatten()

        final_user, final_pos, final_nega = self.user_vector[samp_user].mm(self.FS), self.item_vector[pos_item].mm(self.FS), self.item_vector[neg_item].mm(self.FS)
        out = ((final_user*final_pos).sum(1) - (final_user*final_nega).sum(1)).sigmoid()
        regu_term = self.reg*(final_user**2 + final_pos**2+final_nega**2).sum()
        return (-torch.log(out).sum() + regu_term) / samp_user.size(0), samp_user.size(0)
    
    def recommend(self):
            return self.user_vector.mm(self.FS), self.item_vector.mm(self.FS)


class CoachForSVDGCN(Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            loss, bz = self.model(users, positives, negatives)
	
            loss.backward()
            with torch.no_grad():
                self.model.FS -= self.cfg.lr * self.model.FS.grad
                self.model.FS.grad.zero_()
            
            self.monitor(loss.item(), n=bz, mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        userFeats, itemFeats = self.model.recommend()
        for user, unseen, seen in self.dataloader:
            users = user.to(self.device).data
            seen = seen.to_csr().to(self.device).to_dense().bool()
            targets = unseen.to_csr().to(self.device).to_dense()
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[seen] = -1e10

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'RECALL']
            )


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1
    ).batch(cfg.batch_size).column_().tensor_()

    # validpipe
    validpipe = OrderedIDs(
        field=User
    ).sharding_filter().gen_valid_yielding_(
        dataset # return (user, unseen, seen)
    ).batch(1024).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    # testpipe
    testpipe = OrderedIDs(
        field=User
    ).sharding_filter().gen_test_yielding_(
        dataset
    ).batch(1024).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    tokenizer = FieldModuleList(dataset.fields)
    model = SVDGCN(tokenizer)

    coach = CoachForSVDGCN(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(
        cfg, 
        monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'],
        which4best='ndcg@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()

