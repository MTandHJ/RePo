

from typing import Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import torch, random
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree, to_scipy_sparse_matrix

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1500)
cfg.add_argument("--num-neighbors", type=int, default=10)
cfg.add_argument("--neg-weight", type=float, default=300)
cfg.add_argument("--unseen-only", type=eval, default=False)
cfg.add_argument("--norm-weight", type=float, default=1e-4, help="for l2 normalization")
cfg.add_argument("--item-weight", type=float, default=5e-4, help="for item constraint")
cfg.add_argument("--init-weight", type=float, default=1e-4, help="std for init")
cfg.add_argument("--w1", type=float, default=1e-6)
cfg.add_argument("--w2", type=float, default=1.)
cfg.add_argument("--w3", type=float, default=1e-6)
cfg.add_argument("--w4", type=float, default=1.)

cfg.set_defaults(
    description="UltraGCN",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=300,
    batch_size=512,
    optimizer='adam',
    lr=1e-4,
    seed=1,
)
cfg.compile()

assert isinstance(cfg.unseen_only, bool)


class UltraGCN(freerec.models.RecSysArch):

    def __init__(
        self, tokenizer: FieldModuleList, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]
        self.graph = graph
        self.beta_for_user_item()
        if cfg.item_weight > 0.:
            self.beta_for_item_item()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=cfg.init_weight)

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

        self.register_buffer('userBeta', userBeta.flatten())
        self.register_buffer('itemBeta', itemBeta.flatten())

    def beta_for_item_item(self):
        A = sp.lil_array(to_scipy_sparse_matrix(
            self.graph[self.graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        G = A.T @ A # M x M
        # G.setdiag(0.)
        degs = G.sum(axis=1).squeeze()
        rowBeta = np.sqrt((degs + 1)) / degs
        colBeta = 1 / np.sqrt(degs + 1)
        rowBeta[np.isinf(rowBeta)] = 0.
        colBeta[np.isinf(colBeta)] = 0.
        G = rowBeta.reshape(-1, 1) * G * colBeta.reshape(1, -1)
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

        positives = scores[:, 0]
        loss_pos = F.binary_cross_entropy_with_logits(
            positives, torch.ones_like(positives, dtype=torch.float32), 
            cfg.w1 + cfg.w2 * weights[:, 0], 
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
        weights = self.itemWeights[posItems.flatten()] # B x K

        scores = torch.mul(userEmbds, neighbors).sum(dim=-1)
        loss = - weights * scores.sigmoid().log()
        return loss.sum()

    def regularize(self):
        loss = 0.
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self):
        userEmbds = self.User.embeddings.weight # N x D
        itemEmbds = self.Item.embeddings.weight # M x D
        return userEmbds, itemEmbds

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        loss = self.constraint_for_user_item(users, items) + \
                self.regularize() * cfg.norm_weight
        if cfg.item_weight > 0.:
            loss += self.constraint_for_item_item(users, items) * cfg.item_weight
        return loss
    
    def recommend_from_full(self):
        return self.forward()


class CoachForUltraGCN(freerec.launcher.GenCoach):

    def sample_negs_from_all(self, users, low, high):
        return torch.randint(low, high, size=(len(users), cfg.num_negs), device=self.device)

    def train_per_epoch(self, epoch: int):
        Item = self.fields[ITEM, ID]
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            if not self.cfg.unseen_only:
                negatives = self.sample_negs_from_all(users, 0, Item.count)
            items = torch.cat(
                [positives, negatives], dim=-1
            )
            loss = self.model.predict(users, items)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="sum", prefix='train', pool=['LOSS'])


@dp.functional_datapipe("gen_train_shuffle_uniform_sampling_")
class GenTrainShuffleSampler(freerec.data.postprocessing.sampler.GenTrainUniformSampler):

    def __iter__(self):
        for user, pos in self.source:
            if self._check(user):
                if cfg.unseen_only:
                    yield [user, pos, self._sample_neg(user)]
                else:
                    yield [user, pos, -1]

def take_all(dataset):
    data = []
    for chunk in dataset.train():
        data.extend(list(zip(chunk[USER, ID], chunk[ITEM, ID])))
    return data


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_pairs()
    ).sharding_filter().gen_train_shuffle_uniform_sampling_(
        dataset, num_negatives=cfg.num_negs
    ).batch(cfg.batch_size).column_().tensor_()

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
    model = UltraGCN(
        tokenizer, dataset.train().to_bigraph((USER, ID), (ITEM, ID))
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
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=None,
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