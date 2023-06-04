

from typing import Dict, Optional, Union

import torch, os
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData 
from torch_geometric.utils import to_scipy_sparse_matrix

import freerec
from freeplot.utils import export_pickle, import_pickle
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BaseCriterion
from freerec.data.datasets import Gowalla_m1, Yelp18_m1, AmazonBooks_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID
from freerec.utils import mkdirs, timemeter

freerec.declare(version="0.4.3")

cfg = Parser()
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
    dataset='Gowalla_m1',
    epochs=400,
    batch_size=256,
    optimizer='sgd',
    lr=0.03,
    weight_decay=0.01,
)
cfg.compile()



class GDE(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.graph = graph

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.dropout = torch.nn.Dropout(cfg.dropout_rate)
        self.load()

        self.initialize()

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

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight

        if self.training:
            users, items = users[self.User.name], items[self.Item.name]
            if cfg.dropout_rate == 0:
                userFeats = self.A_u[users].matmul(userEmbs)
                itemFeats = self.A_i[items].matmul(itemEmbs)
            else:
                userFeats = self.dropout(self.A_u[users]).matmul(userEmbs) * (1 - cfg.dropout_rate)
                itemFeats = self.dropout(self.A_i[items]).matmul(itemEmbs) * (1 - cfg.dropout_rate)
            return (userFeats * itemFeats).sum(-1), userFeats, itemFeats
        else:
            return self.A_u.mm(userEmbs), self.A_i.mm(itemEmbs)


class AdaptiveLoss(BaseCriterion):

    def forward(self, scores: torch.Tensor):
        positives = scores[:, 0]
        negatives = scores[:, 1]
        if cfg.criterion == 'adaptive':
            delta = (1 - (1 - negatives.sigmoid().clamp(max=0.99)).log10()).detach()
            out = (positives - negatives * delta).sigmoid()
        else:
            out = (positives - negatives).sigmoid()
        return -torch.log(out).mean()


class CoachForGDE(Coach):


    def reg_loss(self, userFeats, itemFeats):
        loss = userFeats.pow(2).sum() + itemFeats.pow(2).sum()
        loss = loss / userFeats.size(0)
        return loss

    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            scores, userFeats, itemFeats = self.model(users, items)
            reg_loss = self.reg_loss(userFeats, itemFeats)
            loss = self.criterion(scores) + reg_loss * self.cfg.weight_decay

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
    trainpipe = basepipe.shard_().uniform_sampling_(num_negatives=1).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = GDE(
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
    criterion = AdaptiveLoss()

    coach = CoachForGDE(
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



