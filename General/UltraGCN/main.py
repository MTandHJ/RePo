

from typing import Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree, to_scipy_sparse_matrix

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.datasets import Gowalla_m1, Yelp18_m1, AmazonBooks_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1500)
cfg.add_argument("--num-neighbors", type=int, default=10)
cfg.add_argument("--neg-weight", type=float, default=300)
cfg.add_argument("--norm-weight", type=float, default=1e-4, help="for l2 normalization")
cfg.add_argument("--item-weight", type=float, default=5e-4, help="for item constraint")
cfg.add_argument("--w1", type=float, default=1e-6)
cfg.add_argument("--w2", type=float, default=1.)
cfg.add_argument("--w3", type=float, default=1e-6)
cfg.add_argument("--w4", type=float, default=1.)

cfg.set_defaults(
    description="UltraGCN",
    root="../../data",
    num_workers=4,
    dataset='Gowalla_m1',
    epochs=2000,
    batch_size=512,
    optimizer='adam',
    lr=1e-4,
    # weight_decay=1e-4, equal to norm_weight
    seed=2020
)
cfg.compile()



class UltraGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]
        self.graph = graph
        self.beta_for_user_item()
        self.beta_for_item_item()

        self.initialize()

    @property
    def graph(self) -> HeteroData:
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph

    def beta_for_user_item(self):
        row, col = self.graph[self.graph.edge_types[0]].edge_index
        userDeg = degree(row, self.User.count)
        itemDeg = degree(col, self.Item.count)
        userBeta = (userDeg + 1).sqrt() / userDeg
        itemBeta = (itemDeg + 1).pow(-0.5)
        userBeta[torch.isinf(userBeta)].fill_(0.)
        itemBeta[torch.isinf(itemBeta)].fill_(0.)

        self.register_buffer('userBeta', userBeta)
        self.register_buffer('itemBeta', itemBeta)

    def beta_for_item_item(self):
        A = sp.lil_array(to_scipy_sparse_matrix(
            self.graph[self.graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        G = A.T @ A # N x N
        # G.setdiag(0.)
        degs = G.sum(axis=1).squeeze()
        rowBeta = np.sqrt((degs + 1)) / degs
        colBeta = 1 / np.sqrt(degs + 1)
        rowBeta[np.isinf(rowBeta)] = 0.
        colBeta[np.isinf(colBeta)] = 0.
        G = rowBeta.reshape(-1, 1) * G * colBeta
        rows = torch.from_numpy(G.row).long()
        cols = torch.from_numpy(G.col).long()
        vals = torch.from_numpy(G.data)
        indices = torch.stack((rows, cols), dim=0)
        G = torch.sparse_coo_tensor(
            indices, vals, size=(self.Item.count, self.Item.count)
        )
        values, indices = torch.topk(G.to_dense(), cfg.num_neighbors, dim=-1)

        self.register_buffer('itemWeights', values.float())
        self.register_buffer('itemIndices', indices.long())


    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def constraint_for_user_item(
        self, users: torch.Tensor, items: torch.Tensor
    ):
        userEmbds = self.User.look_up(users) # B x 1 x D
        itemEmbds = self.Item.look_up(items) # B x (1 + K) x D
        scores = torch.mul(userEmbds, itemEmbds).sum(-1) # B x (1 + K)
        weights = self.userBeta[users] * self.itemBeta[items] # B x (1 + K)

        positives = scores[:, [0]]
        loss_pos = F.binary_cross_entropy_with_logits(
            positives, torch.ones_like(positives, dtype=torch.float32), 
            cfg.w1 + cfg.w2 * weights[:, [0]], 
            reduction='none'
        )

        negatives = scores[:, 1:]
        loss_neg = F.binary_cross_entropy_with_logits(
            negatives, torch.zeros_like(negatives, dtype=torch.float32), 
            cfg.w3 + cfg.w4 * weights[:, 1:], 
            reduction='none'
        ).mean(dim=-1)

        return (loss_pos + loss_neg * cfg.neg_weight).sum()

    def constraint_for_item_item(
        self, users: torch.Tensor, items: torch.Tensor
    ):
        userEmbds = self.User.look_up(users) # B x 1 x D
        posItems = items[:, 0]
        neighbors = self.Item.look_up(
            self.itemIndices[posItems]
        ) # return top-K neighbors, B x K x D
        weights = self.itemWeights[posItems.flatten()]

        scores = torch.mul(userEmbds, neighbors).sum(dim=-1)
        loss = - weights * scores.sigmoid().log()
        return loss.sum()

    def regularize(self):
        loss = 0.
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        users = users[self.User.name]
        items = items[self.Item.name]
        if self.training:
            loss = self.constraint_for_user_item(users, items) \
                    + self.constraint_for_item_item(users, items) * cfg.item_weight \
                        + self.regularize() * cfg.norm_weight
            return loss
        else:
            userEmbds = self.User.embeddings.weight # N x D
            itemEmbds = self.Item.embeddings.weight # M x D
            return userEmbds, itemEmbds


class CoachForUltraGCN(Coach):


    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum()
        loss = loss / userEmbds.size(0)
        return loss / 2

    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            loss = self.model(users, items)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=self.cfg.batch_size, mode="sum", prefix='train', pool=['LOSS'])

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

    if cfg.dataset == "Gowalla_m1":
        basepipe = Gowalla_m1(cfg.root)
    elif cfg.dataset == "Yelp18_m1":
        basepipe = Yelp18_m1(cfg.root)
    elif cfg.dataset == "AmazonBooks_m1":
        basepipe = AmazonBooks_m1(cfg.root)
    else:
        raise ValueError("Dataset should be Gowalla_m1, Yelp18_m1 or AmazonBooks_m1")
    trainpipe = basepipe.split_(cfg.batch_size).shard_().negatives_for_train_(num_negatives=cfg.num_negs, unseen_only=False).tensor_()
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = UltraGCN(
        tokenizer, basepipe.train().to_bigraph(User, Item)
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

    coach = CoachForUltraGCN(
        model=model,
        dataset=dataset,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()

