

from typing import Dict, Optional, Union

import torch, os
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import LGConv
from freeplot.utils import import_pickle, export_pickle

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN
from freerec.utils import infoLogger, mkdirs
from utils import calc_node_wise_norm, normalize_edge, \
                    jaccard_similarity, \
                    salton_cosine_similarity, \
                    leicht_holme_nerman_similarity, \
                    common_neighbors_similarity


freerec.decalre(version='0.3.5')


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--trend-type", type=str, choices=('jc', 'sc', 'lhn', 'cn'), default='jc')
cfg.add_argument("--trend-coeff", type=float, default=2)
cfg.add_argument("--fusion", type=str, choices=("TRUE", "FALSE"), default='FALSE')
cfg.set_defaults(
    description="CAGCN",
    root="../../data",
    dataset='Gowalla_m1',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()

cfg.fusion = True if cfg.fusion == 'TRUE' else False

class CAGCN(RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = fields
        self.conv = LGConv(False)
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = graph

        self.initialize()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: HeteroData):
        edge_type = '2'.join((self.User.name, self.Item.name))
        edge_index = graph[edge_type].edge_index
        R = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1)),
            size=(self.User.count, self.Item.count)
        )
        path = os.path.join("trends", cfg.dataset, cfg.trend_type)
        mkdirs(path)
        file_ = os.path.join(path, "data.pickle")
        try:
            data = import_pickle(file_)
            trend = data['trend']
            edge_index = data['edge_index']
            edge_weight = data['edge_weight']
            edge_norm = data['edge_norm']
            trend_norm = data['trend_norm']
        except ImportError:
            if cfg.trend_type == 'jc':
                edge_index, trend = jaccard_similarity(R)
            elif cfg.trend_type == 'sc':
                edge_index, trend = salton_cosine_similarity(R)
            elif cfg.trend_type == 'lhn':
                edge_index, trend = leicht_holme_nerman_similarity(R)
            elif cfg.trend_type == 'cn':
                edge_index, trend = common_neighbors_similarity(R)
            edge_weight, _ = normalize_edge(edge_index, self.User.count, self.Item.count)
            edge_norm = calc_node_wise_norm(edge_weight, edge_index[1], self.User.count, self.Item.count)
            trend_norm = calc_node_wise_norm(trend, edge_index[1], self.User.count, self.Item.count)

            data = {
                'trend': trend,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'edge_norm': edge_norm,
                'trend_norm': trend_norm
            }
            export_pickle(data, file_)

        if cfg.fusion:
            infoLogger("[CAGCN] >>> Use Trend and Edge Weight together ...")
            trend = cfg.trend_coeff * trend / trend_norm + edge_weight
        else:
            infoLogger("[CAGCN] >>> Use Trend only ...")
            trend = cfg.trend_coeff * trend * edge_norm / trend_norm 

        self.__graph = Data(edge_index=edge_index)
        self.__graph['edge_weight'] = trend
        T.ToSparseTensor()(self.__graph)

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(
        self, users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        users, items = users, torch.cat(
            [positives, negatives], dim=1
        )
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs

    def recommend(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats


class CoachForCAGCN(Coach):


    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum()
        loss = loss / userEmbds.size(0)
        return loss / 2

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            preds, users, items = self.model(users, positives, negatives)
            pos, neg = preds[:, 0], preds[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

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
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = CAGCN(
        tokenizer, dataset.train().to_bigraph((USER, ID), (ITEM, ID)), num_layers=cfg.layers
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
    criterion = BPRLoss()

    coach = CoachForCAGCN(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
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
