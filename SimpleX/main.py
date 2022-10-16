

from typing import Dict, Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import bipartite_subgraph


from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BaseCriterion
from freerec.data.datasets import Gowalla_m1, Yelp18_m1, AmazonBooks_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID
from freerec.data.utils import errorLogger


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1000)
cfg.add_argument("--gamma", type=float, default=1.)
cfg.add_argument("--margin", type=float, default=.9)
cfg.add_argument("--weight-for-negative", type=float, default=150)
cfg.add_argument("--net-dropout", type=float, default=.1)
cfg.set_defaults(
    description="SimpleX",
    root="../../data",
    dataset="Yelp18_m1",
    epochs=100,
    batch_size=512,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1e-8,
    seed=2019
)
cfg.compile()


class AvgConv(MessagePassing):
    
    def __init__(self, embedding_dim: int, size: Tuple[int, int]):
        super().__init__(aggr="mean", flow='target_to_source')
        self.size = size
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, items: torch.Tensor, edge_index: torch.Tensor):
        """
        Parameters:
        ---

        x: item embeddings.
        edge_index: (2, N), torch.Tensor
            (users (i), items (j)): target2source
        """
        return self.propagate(edge_index=edge_index, items=items, size=self.size)

    def message(self, items_j: torch.Tensor) -> torch.Tensor:
        return items_j

    def update(self, items: torch.Tensor) -> torch.Tensor:
        return self.linear(items)


class SimpleX(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, graph: HeteroData, 
        gamma: float = cfg.gamma, dropout_rate: float = cfg.net_dropout
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User = self.tokenizer[USER, ID]
        self.Item = self.tokenizer[ITEM, ID]
        self.aggregator = AvgConv(
            embedding_dim=self.Item.dimension,
            size=(self.User.count, self.Item.count)
        )

        self.gamma = gamma
        self.graph = graph

        self.dropout = nn.Dropout(dropout_rate)

        self.initialize()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: HeteroData):
        self.__graph = graph

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def aggregate(self, users: torch.Tensor) -> torch.Tensor:
        users = users
        items = self.Item.embeddings.weight
        edge_index, _ = bipartite_subgraph(
            (users.flatten(), torch.arange(self.Item.count, device=self.device)),
            self.graph[self.graph.edge_types[0]].edge_index,
            relabel_nodes=False
        )
        userEmbds = self.User.look_up(users)
        return self.dropout(userEmbds * self.gamma + self.aggregator(items, edge_index)[users] * (1 - self.gamma))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                print(m)
                nn.init.normal_(m.weight, std=1e-4)

    def forward(
        self, users: Dict[str, torch.Tensor],
        items: Dict[str, torch.Tensor]
    ):
        users, items = users[self.User.name], items[self.Item.name]
        itemFeats = F.normalize(self.Item.look_up(items), dim=-1) # B x K x D
        userFeats = F.normalize(self.aggregate(users), dim=-1) # B x 1 x D
        scores = (userFeats * itemFeats).sum(-1) # cosine score, B x K
        return scores

    def recommend(self, users: Dict[str, torch.Tensor]):
        items = {self.Item.name: torch.arange(self.Item.count, device=self.device).view(1, -1)}
        return self(users, items)


class CosineContrastiveLoss(BaseCriterion):

    def __init__(
        self, margin: float = cfg.margin, 
        negative_weight: Optional[float] = cfg.weight_for_negative,
        reduction: str = 'mean'
    ):
        super(CosineContrastiveLoss, self).__init__(reduction)
        self.margin = margin
        self.negative_weight = negative_weight if negative_weight else 1.


    def regularize(self, params: Union[torch.Tensor, Iterable[torch.Tensor]], rtype: str = 'l2'):
        """Add regularization for given parameters.

        Parameters:
        ---

        params: List of parameters for regularization.
        rtype: Some kind of regularization including 'l1'|'l2'.
        """
        params = [params] if isinstance(params, torch.Tensor) else params
        if rtype == 'l1':
            return sum(param.abs().sum() for param in params)
        elif rtype == 'l2':
            return sum(param.pow(2).sum() for param in params) / 2
        else:
            errorLogger(f"{rtype} regularization is not supported ...", NotImplementedError)

    def forward(self, scores: torch.Tensor):
        logits_pos = scores[:, 0]
        loss_pos = (1 - logits_pos).relu()
        logits_neg = scores[:, 1:]
        loss_neg = (logits_neg - self.margin).relu()
        loss = loss_pos + loss_neg.mean(dim=-1) * self.negative_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CoachForSimpleX(Coach):


    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            scores = self.model(users, items)
            loss = self.criterion(scores)
            loss += self.criterion.regularize(
                self.model.tokenizer.parameters(),
                rtype='l2'
            ) * self.cfg.weight_decay

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])


    def evaluate(self, prefix: str = 'valid'):
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            targets = items[Item.name].to(self.device)
            scores = self.model.recommend(users)
            scores[targets == -1] = -1e10
            targets[targets == -1] = 0

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE']
            )


def main():

    if cfg.dataset == "Gowalla_m1":
        basepipe = Gowalla_m1(cfg.root)
    elif cfg.dataset == "Yelp18_m1":
        basepipe = Yelp18_m1(cfg.root)
    elif cfg.dataset == "AmazonBooks_m1":
        basepipe = AmazonBooks_m1(cfg.root)
    else:
        raise ValueError(f"DataSet should be Gowalla_m1 or Yelp18_m1 or AmazonBooks_m1 ...")
    trainpipe = basepipe.shard_().negatives_for_train_(num_negatives=cfg.num_negs).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = SimpleX(
        tokenizer, basepipe.train().to_bigraph(User, Item)
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            # weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            # weight_decay=cfg.weight_decay
        )
    criterion = CosineContrastiveLoss()

    coach = CoachForSimpleX(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()
