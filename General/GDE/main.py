

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData 
from torch_geometric.utils import to_scipy_sparse_matrix
from freeplot.utils import import_pickle, export_pickle

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID
from freerec.utils import mkdirs, timemeter

freerec.declare(version="0.4.3")

cfg = freerec.parpser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--filter-type", choices=('both', 'smooth'), default='smooth')
cfg.add_argument("--smooth-ratio", type=float, default=0.1)
cfg.add_argument("--rough-ratio", type=float, default=0.)
cfg.add_argument("--beta", type=float, default=5.)
cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--criterion", type=str, default='adaptive')

cfg.set_defaults(
    description="GDE",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=400,
    batch_size=256,
    optimizer='sgd',
    lr=0.03,
    weight_decay=0.01,
)
cfg.compile()


class GDE(freerec.models.RecSysArch):

    def __init__(
        self, tokenizer: FieldModuleList, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.graph = graph

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.dropout = torch.nn.Dropout(cfg.dropout_rate)
        self.load()

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
    def graph(self) -> HeteroData:
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph

    def weight_filter(self, vals):
        return torch.exp(cfg.beta * vals)

    def save(self, data: Dict):
        path = os.path.join("filters", cfg.dataset)
        mkdirs(path)
        file_ = os.path.join(path, "eig_vals_vecs.pickle")
        export_pickle(data, file_)

    @timemeter
    def load(self):
        path = os.path.join("filters", cfg.dataset)
        file_ = os.path.join(path, "eig_vals_vecs.pickle")
        try:
            data = import_pickle(file_)
        except ImportError:
            data = self.preprocess()
        
        if cfg.filter_type == 'smooth':
            userVals, userVecs = data['user']['smooth']
            itemVals, itemVecs = data['item']['smooth']
        elif cfg.filter_type == 'both':
            userVals = torch.cat(
                (data['user']['smooth'][0], data['user']['rough'][0]),
                dim=0
            )
            userVecs = torch.cat(
                (data['user']['smooth'][1], data['user']['rough'][1]),
                dim=1
            )
            itemVals = torch.cat(
                (data['item']['smooth'][0], data['item']['rough'][0]),
                dim=0
            )
            itemVecs = torch.cat(
                (data['item']['smooth'][1], data['item']['rough'][1]),
                dim=1
            )
        else:
            raise ValueError("Only 'smooth' or 'both' type supported !")

        # self.register_buffer("userVals", self.weight_filter(userVals))
        # self.register_buffer("userVecs", userVecs)
        # self.register_buffer("itemVals", self.weight_filter(itemVals))
        # self.register_buffer("itemVecs", itemVecs)

        self.register_buffer("A_u", (userVecs * self.weight_filter(userVals)).mm(userVecs.t()))
        self.register_buffer("A_i", (itemVecs * self.weight_filter(itemVals)).mm(itemVecs.t()))

    @timemeter
    def lobpcg(self, A, k: int, largest: bool = True, niter: int = 5):
        A = A.tocoo()
        rows = torch.from_numpy(A.row).long()
        cols = torch.from_numpy(A.col).long()
        vals = torch.from_numpy(A.data)
        indices = torch.stack((rows, cols), dim=0)
        A = torch.sparse_coo_tensor(
            indices, vals, size=A.shape
        )
        vals, vecs =  torch.lobpcg(A, k=k, largest=largest, niter=niter)
        return vals, vecs

    def preprocess(self):
        R = sp.lil_array(to_scipy_sparse_matrix(
            self.graph[self.graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        userDegs = R.sum(axis=1).squeeze()
        itemDegs = R.sum(axis=0).squeeze()
        userDegs = 1 / np.sqrt(userDegs)
        itemDegs = 1 / np.sqrt(itemDegs)
        userDegs[np.isinf(userDegs)] = 0.
        itemDegs[np.isinf(itemDegs)] = 0.
        R = userDegs.reshape(-1, 1) * R * itemDegs
        del userDegs, itemDegs

        data = dict(user=dict(), item=dict())
        A = R @ R.T # user x user
        if cfg.smooth_ratio != 0:
            k = int(self.User.count * cfg.smooth_ratio)
            eigVals, eigVecs = self.lobpcg(A, k, largest=True)
            data['user']['smooth'] = (eigVals, eigVecs)
        if cfg.rough_ratio != 0:
            k = int(self.User.count * cfg.rough_ratio)
            eigVals, eigVecs = self.lobpcg(A, k, largest=False)
            data['user']['rough'] = (eigVals, eigVecs)

        A = R.T @ R # item x item
        if cfg.smooth_ratio != 0:
            k = int(self.Item.count * cfg.smooth_ratio)
            eigVals, eigVecs = self.lobpcg(A, k, largest=True)
            data['item']['smooth'] = (eigVals, eigVecs)
        if cfg.rough_ratio != 0:
            k = int(self.Item.count * cfg.rough_ratio)
            eigVals, eigVecs = self.lobpcg(A, k, largest=False)
            data['item']['rough'] = (eigVals, eigVecs)

        self.save(data)
        return data

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
        return userEmbs, itemEmbs

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userEmbs, itemEmbs = self.forward()
        users, items = users[self.User.name], items[self.Item.name]
        if cfg.dropout_rate == 0:
            userFeats = self.A_u[users].matmul(userEmbs)
            itemFeats = self.A_i[items].matmul(itemEmbs)
        else:
            userFeats = self.dropout(self.A_u[users]).matmul(userEmbs) * (1 - cfg.dropout_rate)
            itemFeats = self.dropout(self.A_i[items]).matmul(itemEmbs) * (1 - cfg.dropout_rate)
        return (userFeats * itemFeats).sum(-1), userFeats, itemFeats

    def recommend_from_full(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        return self.A_u.mm(userEmbs), self.A_i.mm(itemEmbs)

class AdaptiveLoss(freerec.criterions.BaseCriterion):

    def forward(self, scores: torch.Tensor):
        positives = scores[:, 0]
        negatives = scores[:, 1]
        if cfg.criterion == 'adaptive':
            delta = (1 - (1 - negatives.sigmoid().clamp(max=0.99)).log10()).detach()
            out = (positives - negatives * delta).sigmoid()
        else:
            out = (positives - negatives).sigmoid()
        return -torch.log(out).mean()


class CoachForGDE(freerec.launcher.GenCoach):

    def reg_loss(self, userFeats, itemFeats):
        loss = userFeats.pow(2).sum() + itemFeats.pow(2).sum()
        loss = loss / userFeats.size(0)
        return loss

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            scores, users, items = self.model.predict(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1
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
    model = GDE(
        tokenizer,
        dataset.train().to_bigraph((USER, ID), (ITEM, ID))
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
    criterion = AdaptiveLoss()

    coach = CoachForGDE(
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