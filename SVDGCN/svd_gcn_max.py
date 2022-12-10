

import torch, os
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import HeteroData 
from torch_geometric.utils import to_scipy_sparse_matrix

import freerec
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID

cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--alpha", type=int, default=0)
cfg.add_argument("--beta", type=float, default=2.)
cfg.add_argument("--req-vec", type=int, default=500)
cfg.add_argument("--coef_u", type=float, default=0.5)
cfg.add_argument("--coef_i", type=float, default=0.9)

cfg.set_defaults(
    description="SVD-GCN-Max",
    root="../../data",
    dataset='Gowalla_m1',
    eval_freq=1,
    epochs=15,
    batch_size=2048,
    optimizer='sgd',
    lr=10,
    weight_decay=0.01,
    seed=1
)
cfg.compile()


class SVDGCN(RecSysArch):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.reg = cfg.weight_decay
        self.beta = cfg.beta
        self.coef_u = cfg.coef_u
        self.coef_i = cfg.coef_i

        self.User = tokenizer[USER, ID]
        self.Item = tokenizer[ITEM, ID]

        path = os.path.join(cfg.dataset, str(cfg.alpha))

        svd_filter = self.weight_func(
            torch.Tensor(np.load(path+ r'/svd_value.npy')[:cfg.req_vec])
        )

        user_vector = (torch.Tensor(np.load(path+ r'/svd_u.npy')[:,:cfg.req_vec])) * svd_filter
        item_vector = (torch.Tensor(np.load(path+ r'/svd_v.npy')[:,:cfg.req_vec])) * svd_filter

        self.register_buffer('user_vector', user_vector)
        self.register_buffer('item_vector', item_vector)
        self.register_parameter(
            'FS', Parameter(
                torch.randn(cfg.req_vec, cfg.embedding_dim),
                requires_grad=True
            )
        )
        nn.init.uniform_(self.FS, -np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)), np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)))


    def weight_func(self, sig):
        return torch.exp(self.beta * sig)


    def forward(
        self, u = None, p = None, n = None, up = None, un = None, pp = None, pn = None
    ):
        if self.training:
            final_user,final_pos,final_nega=self.user_vector[u].mm(self.FS),self.item_vector[p].mm(self.FS),self.item_vector[n].mm(self.FS)
            final_user_p,final_user_n=self.user_vector[up].mm(self.FS),self.user_vector[un].mm(self.FS)
            final_pos_p,final_pos_n=self.item_vector[pp].mm(self.FS),self.item_vector[pn].mm(self.FS)

            out=((final_user*final_pos).sum(1)-(final_user*final_nega).sum(1)).sigmoid()
            self_loss_u=torch.log(((final_user*final_user_p).sum(1)-(final_user*final_user_n).sum(1)).sigmoid()).sum()
            self_loss_i=torch.log(((final_pos*final_pos_p).sum(1)-(final_pos*final_pos_n).sum(1)).sigmoid()).sum()
            regu_term=self.reg*(final_user**2+final_pos**2+final_nega**2+final_user_p**2+final_user_n**2+final_pos_p**2+final_pos_n**2).sum()
            return (-torch.log(out).sum()-self.coef_u*self_loss_u-self.coef_i*self_loss_i+regu_term) / len(u), len(u)
        else:
            return self.user_vector.mm(self.FS), self.item_vector.mm(self.FS)


class CoachForSVDGCN(Coach):

    def preprocess(self, graph):
        R = sp.lil_array(to_scipy_sparse_matrix(
            graph[graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        R = R.tocoo()
        rows = torch.from_numpy(R.row).long()
        cols = torch.from_numpy(R.col).long()
        vals = torch.from_numpy(R.data)
        indices = torch.stack((rows, cols), dim=0)
        R = torch.sparse_coo_tensor(
            indices, vals, size=R.shape
        )
        return R

    def prepare(self, bigraph: HeteroData):
        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]
        R = self.preprocess(bigraph)
        self.rate_matrix = R.to_dense()
        self.user_matrix = (torch.sparse.mm(R, R.t()).to_dense() != 0).float()
        self.item_matrix = (torch.sparse.mm(R.t(), R).to_dense() != 0).float()

    def get_batch(self):
        u=np.random.randint(0, self.User.count, self.cfg.batch_size)
        cur_rrr = self.rate_matrix[u].to(self.device)
        cur_uuu = self.user_matrix[u].to(self.device)
        cur_ppp = self.item_matrix[u].to(self.device)
        p=torch.multinomial(cur_rrr,1,True).squeeze(1)
        nega=torch.multinomial(1-cur_rrr,1,True).squeeze(1)
        up=torch.multinomial(cur_uuu,1,True).squeeze(1)
        un=torch.multinomial(1-cur_uuu,1,True).squeeze(1)
        pp=torch.multinomial(cur_ppp,1,True).squeeze(1)
        pn=torch.multinomial(1-cur_ppp,1,True).squeeze(1)
        return u, p, nega, up, un, pp, pn

    def train_per_epoch(self):
        for _ in range(self.cfg.datasize // self.cfg.batch_size):

            # u, p, nega, up, un, pp, pn = self.get_batch()
            loss, bz = self.model(*self.get_batch())
	
            loss.backward()
            with torch.no_grad():
                self.model.FS -= self.cfg.lr * self.model.FS.grad
                self.model.FS.grad.zero_()
            
            self.monitor(loss.item(), n=bz, mode="mean", prefix='train', pool=['LOSS'])

        
    def evaluate(self, prefix: str = 'valid'):
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        userFeats, itemFeats = self.model()
        for users, items in self.dataloader:
            users = users[User.name].to(self.device)
            targets = items[Item.name].to(self.device)
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[targets == -1] = -1e10
            targets[targets == -1] = 0

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE']
            )


def main():

    basepipe = getattr(freerec.data.datasets, cfg.dataset)(cfg.root)
    trainpipe = basepipe.uniform_sampling_(num_negatives=1).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=2048).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    cfg.datasize = basepipe.train().datasize

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    model = SVDGCN(tokenizer)


    coach = CoachForSVDGCN(
        model=model,
        dataset=dataset,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    User = basepipe.fields[USER, ID]
    Item = basepipe.fields[ITEM, ID]
    coach.prepare(basepipe.to_bigraph(User, Item))
    coach.fit()



if __name__ == "__main__":
    main()

