

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
cfg.add_argument("-K", "--K", type=int, default=4)
cfg.add_argument("--lr4time", type=float, default=1.e-6, help="the learning rate of timestamps")
cfg.add_argument("--timesplit", type=int, default=4)
cfg.add_argument("--solver", type=str, choices=('dopri5', 'euler', 'rk4', 'adaptive_heun', 'bosh3', 'explicit_adams', 'implicit_adams'), default='rk4')
cfg.add_argument("--adjoint", action='store_true', default=False, help="using torchdiffeq.odeint_adjoint")
cfg.add_argument("--frozentimes", action='store_true', default=False, help="using fixed timestamps")
cfg.add_argument('--rtol', type=float, default=1e-7, help="rtol for 'dopri5' solver")
cfg.add_argument('--atol', type=float, default=1e-9, help="atol for 'dopri5' solver")

cfg.set_defaults(
    description="LT-OCF",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()

if cfg.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ODEFunction(nn.Module):
    """[summary]
        LGC, non-time-dependent
    """

    def __init__(self, Graph: Data):
        super(ODEFunction, self).__init__()
        self.g = Graph
        self.conv = LGConv(normalize=False)

    def forward(self, t, x: torch.Tensor):
        """
        ODEFUNCTION(| --> only single layer --> |)
        """
        return self.conv(x, self.g.adj_t)

class ODEBlock(nn.Module):

    def __init__(self, odeFunction: ODEFunction, solver: str, **kwargs):
        super(ODEBlock, self).__init__()
        self.odefunc = odeFunction
        self.solver = solver
        for name, attr in kwargs.items():
            setattr(self, name, attr)

    def forward(self, x: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        t = torch.stack([start, end]).type_as(x)
        if self.solver == 'dopri5':
            out = odeint(func = self.odefunc, y0 = x, t = t, method=self.solver, rtol=self.rtol, atol=self.atol)
        else:
            out = odeint(func = self.odefunc, y0 = x, t = t, method=self.solver)
        return out[1]


class LTOCF(RecSysArch):

    def __init__(self, tokenizer: FieldModuleList, graph: Data) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]
        self.graph = graph
        self.init_ode()

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def init_ode(self):
        eta = cfg.K / cfg.timesplit
        if cfg.frozentimes:
            self.register_buffer(
                'odetimes',
                torch.arange(1, cfg.timesplit, dtype=torch.float32) * eta
            )
        else:
            self.register_parameter(
                'odetimes',
                nn.parameter.Parameter(
                    torch.arange(1, cfg.timesplit, dtype=torch.float32) * eta,
                    requires_grad=True
                )
            )
        self.register_buffer('start', torch.tensor([0.]))
        self.register_buffer('end', torch.tensor([cfg.K]))

        self.num_layers = cfg.timesplit
        self.odeBlocks = nn.ModuleList([ODEBlock(ODEFunction(self.graph), cfg.solver, rtol=cfg.rtol, atol=cfg.atol) for _ in range(self.num_layers)])

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

    @torch.no_grad()
    def reorder(self):
        """
        Make sure that 0 < t1 < t2 < ... < t_{K-1} < K.
        """
        if not cfg.frozentimes:
            eps = 1.e-6
            self.odetimes[0].clamp_(eps, min(self.odetimes[1]  - eps, cfg.K - (self.num_layers - 1) * eps))
            for l in range(1, self.num_layers - 2):
                self.odetimes[l].clamp_(
                    self.odetimes[l-1] + eps, 
                    min(self.odetimes[l+1]  - eps, cfg.K - (self.num_layers - l - 1) * eps)
                )
            self.odetimes[self.num_layers-2].clamp_(self.odetimes[self.num_layers-3] + eps, cfg.K - eps)

    def forward(
        self, users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        timeline1 = torch.cat([self.start, self.odetimes])
        timeline2 = torch.cat([self.odetimes, self.end])
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for l, (start, end) in enumerate(zip(timeline1, timeline2)):
            features = self.odeBlocks[l](features, start, end) - features # XXX: dual res = False
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
        timeline1 = torch.cat([self.start, self.odetimes])
        timeline2 = torch.cat([self.odetimes, self.end])
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for l, (start, end) in enumerate(zip(timeline1, timeline2)):
            features = self.odeBlocks[l](features, start, end) - features # XXX: dual res = False
            avgFeats += features / (self.num_layers + 1)

        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats

           

class CoachForLTOCF(Coach):

    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum()
        loss = loss / userEmbds.size(0)
        return loss / 2

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            scores, users, items = self.model(users, positives, negatives)
            pos, neg = scores[:, 0], scores[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.reorder()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        userFeats, itemFeats = self.model.recommend()
        for user, unseen, seen in self.dataloader:
            users = user.to(self.device).data
            seen = seen.to_csr().to(self.device).to_dense().bool()
            targets = unseen.to_csr().to(self.device).to_dense()
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[seen] = -1e10

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'RECALL']
            )


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
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
    ).sharding_filter().gen_valid_yielding_(
        dataset # return (user, unseen, seen)
    ).batch(1024).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    # testpipe
    testpipe = OrderedIDs(
        field=User
    ).sharding_filter().gen_test_yielding_(
        dataset
    ).batch(1024).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    tokenizer = FieldModuleList(dataset.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = LTOCF(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID))
    )

    params = [
        {'params': model.odeBlocks.parameters(), 'lr': cfg.lr},
        {'params': tokenizer.parameters(), 'lr': cfg.lr},
        {'params': model.odetimes, 'lr': cfg.lr4time}
    ]
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params,
            betas=(cfg.beta1, cfg.beta2),
        )
    criterion = BPRLoss()

    coach = CoachForLTOCF(
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

