

from typing import Dict, Optional, Union

import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=2)
cfg.add_argument("--alpha", type=float, default=0.2, help="margin")
cfg.add_argument("--lamb1", type=float, default=0.1, help="weight for ranking loss")
cfg.add_argument("--lamb2", type=float, default=0.5, help="weight for ranking loss")
cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--std", type=float, default=1, help="standard deviation for embedding initialization")
cfg.add_argument("--num-negs", type=int, default=5, help="one positive + [num_negs] negatives")
cfg.set_defaults(
    description="HS-GCN",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=100,
    batch_size=2048,
    optimizer='adam',
    lr=3e-4,
    weight_decay=1e-7,
    eval_freq=1,
    seed=1
)
cfg.compile()


class HashConv(MessagePassing):

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        m = self.propagate(edge_index, x=x)
        d = torch.mul(m, x)
        c = -F.relu(2 * -d) + 1
        return torch.mul(c, x)

    def update(self, aggr_out, x):
        return torch.clamp(aggr_out.float() + 2 * x, -1, 1)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class HSGCN(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        graph: Data,
        num_layers: int = 2
    ) -> None:
        super().__init__()

        self.fields = fields
        self.conv = HashConv()
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = graph
        self.sign = LBSign.apply

        self.dropout = nn.Dropout(p=cfg.dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=cfg.std)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        x = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        x = x.tanh()
        if self.training:
            h = x
            x = self.sign(x)
        else:
            h = torch.sign(x)
            x = torch.sign(x)
        for _ in range(self.num_layers):
            h = self.conv(h, self.graph.edge_index)
        h = self.dropout(h)
        return x, h

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        X, H = self.forward()
        userEmbds, itemEmbds = torch.split(X, (self.User.count, self.Item.count))
        userFeats, itemFeats = torch.split(H, (self.User.count, self.Item.count))
        userEmbds = userEmbds[users] # B x 1 x D
        itemEmbds = itemEmbds[items] # B x n x D
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        return userFeats, itemFeats, userEmbds, itemEmbds

    def recommend_from_full(self):
        _, H = self.forward()
        userFeats, itemFeats = torch.split(H, (self.User.count, self.Item.count))
        return userFeats, itemFeats


class CoachForHSGCN(freerec.launcher.GenCoach):

    def cross_loss(self, scores: torch.Tensor):
        r"""
        scores: torch.Tensor, (B, 2)
        """
        targets = torch.ones_like(scores)
        targets[:, 1].fill_(0.)
        return self.criterion(scores.flatten(), targets.flatten())

    def rank_loss(self, scores: torch.Tensor):
        r"""
        scores: torch.Tensor, (B, 2)
        """
        pos = scores[:, 0]
        neg = scores[:, 1]
        return (neg.sigmoid() - pos.sigmoid() + self.cfg.alpha).relu().mean()

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            userFeats, itemFeats, userEmbds, itemEmbds = self.model.predict(users, items)
            scores_feat = userFeats.mul(itemFeats).sum(-1) # (B, 2)
            scores_embd = userEmbds.mul(itemEmbds).sum(-1) # (B, 2)
            loss = self.cross_loss(scores_feat) + \
                self.rank_loss(scores_feat) * self.cfg.lamb1 + \
                self.rank_loss(scores_embd) * self.cfg.lamb2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    
    def evaluate(self, epoch: int, prefix: str = 'valid'):
        return super().evaluate(epoch, prefix)


def sample_neg(seenItems, allItems, user):
    neg = random.choice(allItems)
    while neg in seenItems[user]:
        neg = random.choice(allItems)
    return neg

def extend(dataset: freerec.data.datasets.RecDataSet):
    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]
    seenItems = [set() for _ in range(User.count)]

    for chunk in dataset.train():
        list(map(
            lambda user, item: seenItems[user].add(item),
            chunk[USER, ID], chunk[ITEM, ID]
        ))

    data = []
    for user, pos in dataset.train().to_pairs():
        for _ in range(cfg.num_negs):
            data.append((user, pos, sample_neg(seenItems, Item.enums, user)))
    return data


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=extend(dataset)
    ).sharding_filter().batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )

    tokenizer = FieldModuleList(dataset.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = HSGCN(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID)), num_layers=cfg.layers
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = freerec.criterions.BCELoss4Logits()

    coach = CoachForHSGCN(
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