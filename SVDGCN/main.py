

from typing import Dict, Optional, Union

import torch, os
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData 
from torch_geometric.utils import to_scipy_sparse_matrix


from freeplot.utils import export_pickle, import_pickle
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BaseCriterion
from freerec.data.datasets import Gowalla_m1, Yelp18_m1, AmazonBooks_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID
from freerec.utils import mkdirs, timemeter


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-features", type=int, default=90)
cfg.add_argument("--beta", type=float, default=2.)
cfg.add_argument("--alpha", type=float, default=3.)
cfg.add_argument("--user-weight", type=float, default=.5)
cfg.add_argument("--item-weight", type=float, default=.9)

cfg.set_defaults(
    description="SVDGAN",
    root="../../data",
    dataset='Gowalla_m1',
    epochs=50,
    batch_size=256,
    momentum=0.,
    optimizer='sgd',
    lr=9.,
    weight_decay=0.01,
)
cfg.compile()



class SVDGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.graph = graph

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]
        self.load()
        self.initialize()

        self.FS = torch.nn.Linear(cfg.num_features, cfg.embedding_dim, bias=False)
        torch.nn.init.uniform_(
            self.FS.weight,
            -np.sqrt(6. / (cfg.num_features + cfg.embedding_dim)),
            np.sqrt(6. / (cfg.num_features + cfg.embedding_dim))
        )

    @property
    def graph(self) -> HeteroData:
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph


    def save(self, data: Dict):
        path = os.path.join("filters", cfg.dataset)
        mkdirs(path)
        file_ = os.path.join(path, "eig_vals_vecs.pickle")
        export_pickle(data, file_)


    def weight_filter(self, vals):
        return torch.exp(cfg.beta * vals)

    @timemeter("GDE/load")
    def load(self):
        path = os.path.join("filters", cfg.dataset)
        file_ = os.path.join(path, "u_vals_v.pickle")
        try:
            data = import_pickle(file_)
        except ImportError:
            data = self.preprocess()
        
        U, vals, V = data['U'], data['vals'], data['V']
        vals = self.weight_filter(vals[:cfg.num_features])
        U = U[:, :cfg.num_features] * vals
        V = V[:, :cfg.num_features] * vals
        self.register_buffer("U", U)
        self.register_buffer("V", V)

    def preprocess(self):
        edge_type = self.graph.edge_types[0]
        edge_index = self.graph[edge_type].edge_index
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        R = torch.sparse_coo_tensor(
            edge_index, values, size=(self.User.count, self.Item.count)
        )
        userDegs = R.sum(axis=1).add(cfg.alpha).pow(-0.5)
        itemDegs = R.sum(axis=0).add(cfg.alpha).pow(-0.5)
        userDegs[torch.isinf(userDegs)] = 0.
        itemDegs[torch.isinf(itemDegs)] = 0.
        R = userDegs.view(-1, 1) * R * itemDegs
        del userDegs, itemDegs

        U, vals, V = torch.svd_lowrank(R, q=400, niter=30)

        data = {'U': U.cpu(), 'vals': vals.cpu(), 'V': V.cpu()}
        self.save(data)
        return data


    def criterion(self, x, yp, yn, weight):
        pos = (x * yp).sum(dim=1)
        neg = (x * yn).sum(dim=1)
        return F.softplus(neg - pos).mean() * weight

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        if self.training:
            users, items = users[self.User.name], items[self.Item.name]
            userEmbeds, itemEmbeds = self.U[users], self.V[items]
            userFeats, itemFeats = self.FS(userEmbeds), self.FS(itemEmbeds)

            u, up, un = torch.chunk(userFeats, 3, dim=1)
            p, n, pp, pn = torch.chunk(itemFeats, 4, dim=1)
            loss = 0.
            for params in [(u, p, n, 1), (u, up, un, cfg.user_weight), (p, pp, pn, cfg.item_weight)]:
                loss += self.criterion(*params)

            return loss, userFeats, itemFeats
        else:
            return self.FS(self.U), self.FS(self.V)


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


class CoachForSVDGAN(Coach):


    def set_sampler(self, graph: HeteroData):
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        edge_type = graph.edge_types[0]
        edge_index = graph[edge_type].edge_index
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        R = torch.sparse_coo_tensor(
            edge_index, values, size=(self.User.count, self.Item.count)
        )

        A_u = R @ R.t()
        self.A_u = (A_u != 0).float()
        A_i = R.t() @ R
        self.A_i = (A_i != 0).float()

    def sample(self, u, p, n):
        u, p = u.squeeze(), p.squeeze()
        up = torch.multinomial(self.A_u[u], 1, True)
        un = torch.multinomial(1 - self.A_u[u], 1, True)
        pp = torch.multinomial(self.A_i[p], 1, True)
        pn = torch.multinomial(1 - self.A_i[p], 1, True)
        u, p = u.unsqueeze(1), p.unsqueeze(1)
        users = torch.cat([u, up, un], dim=1) # B x 3
        items = torch.cat([p, n, pp, pn], dim=1) # B x 4
        return {self.User.name: users.to(self.device), self.Item.name: items.to(self.device)}

    def reg_loss(self, userFeats, itemFeats):
        loss = userFeats.pow(2).sum() + itemFeats.pow(2).sum()
        loss = loss / userFeats.size(0)
        return loss

    def train_per_epoch(self):
        for users, items in self.dataloader:
            u, items = users[self.User.name], items[self.Item.name]
            p, n = items[:, [0]], items[:, [1]]
            users, items = self.sample(u, p, n)

            loss, userFeats, itemFeats = self.model(users, items)
            reg_loss = self.reg_loss(userFeats, itemFeats)
            loss += reg_loss * self.cfg.weight_decay

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=u.size(0), mode="mean", prefix='train', pool=['LOSS'])

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



