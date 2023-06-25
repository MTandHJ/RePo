

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv
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
cfg.set_defaults(
    description="LightGCN",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class LightGCN(RecSysArch):

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

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.__graph.adj_t = gcn_norm(
            self.__graph.adj_t, num_nodes=self.User.count + self.Item.count,
            add_self_loops=False
        )

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


class CoachForLightGCN(Coach):


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
        for users, _, items in self.dataloader:
            users = users.to(self.device) # (B, 1)
            items = items.to(self.device) # (B, 101)
            users = userFeats[users] # B x 1 x D
            items = itemFeats[items] # B x (101) x D
            scores = users.mul(items).sum(-1) # B x 101
            targets = torch.zeros_like(scores)
            targets[:, 0] = 1

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'NDCG', 'RECALL']
            )


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(cfg.root)
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
    ).sharding_filter().seq_valid_sampling_(
        dataset # yielding (user, items, (target + (100) negatives))
    ).lprune_(
        indices=[1], maxlen=1,
    ).batch(cfg.batch_size).column_().tensor_()

    # testpipe
    testpipe = OrderedIDs(
        field=User
    ).sharding_filter().seq_test_sampling_(
        dataset # yielding (user, items, (target + (100) negatives))
    ).lprune_(
        indices=[1], maxlen=1,
    ).batch(cfg.batch_size).column_().tensor_()

    tokenizer = FieldModuleList(dataset.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = LightGCN(
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

    coach = CoachForLightGCN(
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
        monitors=['loss', 'hitrate@1', 'hitrate@5', 'hitrate@10', 'ndcg@5', 'ndcg@10'],
        which4best='ndcg@10'
    )
    coach.fit()


if __name__ == "__main__":
    main()