

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor, matmul, mul, fill_diag
from torch_sparse import sum as sparsesum
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN


freerec.declare(version='0.4.3')


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--mess-dropout", type=float, default=0.1)
cfg.set_defaults(
    description="NGCF",
    root="../data",
    dataset='Gowalla_m1',
    epochs=400,
    batch_size=1024,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1e-5,
    seed=1
)
cfg.compile()


class NGCFConv(MessagePassing):

    def __init__(
        self, in_features: int, out_features: int,
        dropout_rate: float
    ):
        super().__init__(aggr='add', flow='source_to_target')

        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, adj_t: SparseTensor):
        z = self.propagate(adj_t, x=x)
        return F.normalize(
            self.dropout(
                self.act(self.linear1(z.add(x))) + self.act(self.linear2(z.mul(x)))
            ),
            dim=-1
        )

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def message_and_aggregate(self, adj_t: SparseTensor, x: torch.Tensor) -> torch.Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
        

class NGCF(RecSysArch):

    def __init__(
        self, tokenizer: FieldModuleList, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.convs = nn.ModuleList([NGCFConv(cfg.embedding_dim, cfg.embedding_dim, cfg.mess_dropout) for _ in range(num_layers)])
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]
        self.graph = graph

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        adj_t = self.__graph.adj_t
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=torch.float32)
        adj_t = fill_diag(adj_t, 1.)
        deg = sparsesum(adj_t, dim=1) # column sum
        deg_inv = deg.pow_(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        self.__graph.adj_t = mul(adj_t, deg_inv.view(-1, 1))
        deg = sparsesum(self.__graph.adj_t, 1)

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
        allFeats = [features]
        for conv in self.convs:
            features = conv(features, self.graph.adj_t)
            allFeats.append(features)
        allFeats = torch.cat(allFeats, dim=-1)
        userFeats, itemFeats = torch.split(allFeats, (self.User.count, self.Item.count))

        users, items = users, torch.cat(
            [positives, negatives], dim=1
        )
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1), userFeats, itemFeats

    def recommend(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        allFeats = [features]
        for conv in self.convs:
            features = conv(features, self.graph.adj_t)
            allFeats.append(features)
        allFeats = torch.cat(allFeats, dim=-1)
        userFeats, itemFeats = torch.split(allFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats

class CoachForNGCF(Coach):


    def reg_loss(self, userFeats: torch.Tensor, itemFeats: torch.Tensor):
        loss = userFeats.norm() + itemFeats.norm()
        return loss / self.cfg.batch_size

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
    model = NGCF(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID)), num_layers=cfg.layers
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

    coach = CoachForNGCF(
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