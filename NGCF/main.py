

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
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--mess-dropout", type=float, default=0.1)
cfg.set_defaults(
    description="NGCF",
    root="../../data",
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
        self, tokenizer: Tokenizer, 
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
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
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

        if self.training: # Batch
            users, items = users[self.User.name], items[self.Item.name]
            userFeats = userFeats[users] # B x 1 x D
            itemFeats = itemFeats[items] # B x n x D
            userEmbs = self.User.look_up(users) # B x 1 x D
            itemEmbs = self.Item.look_up(items) # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1), userFeats, itemFeats
        else:
            return userFeats, itemFeats


class CoachForNGCF(Coach):


    def reg_loss(self, userFeats: torch.Tensor, itemFeats: torch.Tensor):
        loss = userFeats.norm() + itemFeats.norm()
        return loss / self.cfg.batch_size

    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            scores, users, items = self.model(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])

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
                pool=['NDCG', 'RECALL']
            )


def main():

    basepipe = getattr(freerec.data.datasets, cfg.dataset)(cfg.root)
    trainpipe = basepipe.shard_().uniform_sampling_(num_negatives=1).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = NGCF(
        tokenizer, basepipe.train().to_graph(User, Item), num_layers=cfg.layers
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
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()

