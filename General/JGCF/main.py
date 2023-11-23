

from typing import Dict, Optional, Union, List
from torch_geometric.typing import Adj

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul
from functools import partial

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.5')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)

cfg.add_argument("--scaling-factor", type=float, default=3., help="hyper-parameter for rescaling")
cfg.add_argument("--alpha", type=float, default=1., help="hyper-parameter for Jacobi Polynomial")
cfg.add_argument("--beta", type=float, default=1., help="hyper-parameter for Jacobi Polynomial")
cfg.add_argument("--weight4mid", type=float, default=0.1, help="weight for scaling mid")

cfg.set_defaults(
    description="JGCF",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


def jacobi_conv(
    zs: List[torch.Tensor], A: Adj, l: int, 
    alpha: float = 1., beta: float = 1.
):
    r"""
    Polynomial convolution with [Jacobi bases](https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    (alpha, beta): float, two hyper-parameters for Jacobi Polynomial.
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    if l == 1:
        c = (alpha - beta) / 2
        return c * zs[-1] + (alpha + beta + 2) / 2 * matmul(A, zs[-1], reduce='sum')
    else:
        c0 = 2 * l \
                * (l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c1 = (2 * l + alpha + beta - 1) \
                * (alpha ** 2 - beta ** 2)
        c2 = (2 * l + alpha + beta - 1) \
                * (2 * l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c3 = 2 * (l + alpha - 1) \
                * (l + beta - 1) \
                * (2 * l + alpha + beta)
        
        part1 = c1 * zs[-1]
        part2 = c2 * matmul(A, zs[-1], reduce='sum')
        part3 = c3 * zs[-2]

        return (part1 + part2 - part3) / c0

class JacobiConv(nn.Module):

    def __init__(
        self, 
        scaling_factor: float = 3.,
        L: int = 3,
        alpha: float = 1.,
        beta: float = 1.,
    ):
        super().__init__()

        self.L = L
        self.scaling_factor = scaling_factor

        self.register_parameter(
            'gammas',
            nn.parameter.Parameter(
                torch.empty((L + 1, 1)).fill_(min(1 / scaling_factor, 1.)),
                requires_grad=False
            )
        )

        self.conv_fn = partial(jacobi_conv, alpha=alpha, beta=beta)

    def forward(self, x: torch.Tensor, A: Adj):
        zs = [self.conv_fn([x], A, 0)]
        for l in range(1, self.L + 1):
            z = self.conv_fn(zs, A, l)
            zs.append(z)
        coefs = (self.gammas.tanh() * self.scaling_factor).cumprod(dim=0) 
        zs = torch.stack(zs, dim=1) # (N, L + 1, D)
        return (zs * coefs).mean(1) # (N, D)


class JGCF(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.weight4mid = cfg.weight4mid

        self.fields = fields
        self.conv = JacobiConv(
            scaling_factor=cfg.scaling_factor, L=cfg.layers, 
            alpha=cfg.alpha, beta=cfg.beta
        )
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

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats_low = self.conv(features, self.graph.adj_t)
        avgFeats_mid = self.weight4mid * features - avgFeats_low
        avgFeats = torch.cat((avgFeats_low, avgFeats_mid), dim=1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats, itemFeats = self.forward()
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs

    def recommend_from_full(self):
        return self.forward()


class CoachForJGCF(freerec.launcher.GenCoach):

    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum()
        loss = loss / userEmbds.size(0)
        return loss / 2

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
    model = JGCF(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID))
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
    criterion = freerec.criterions.BPRLoss()

    coach = CoachForJGCF(
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