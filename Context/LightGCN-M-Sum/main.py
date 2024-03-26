

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.7.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)

cfg.add_argument("--position", type=str, choices=('ego', 'post'), default="ego")

cfg.add_argument("--afile", type=str, default=None, help="the file of acoustic modality features")
cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="LightGCN",
    root="../../data",
    dataset='AmazonBaby_550_MMRec',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class LightGCN(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        data_path: str,
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = fields
        self.conv = LGConv(False)
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = graph

        self.load_feats(data_path)

        if cfg.vfile:
            self.vProjector = nn.Linear(self.vFeats.size(1), cfg.embedding_dim)

        if cfg.tfile:
            self.tProjector = nn.Linear(self.tFeats.size(1), cfg.embedding_dim)

        if cfg.afile:
            self.aProjector = nn.Linear(self.aFeats.size(1), cfg.embedding_dim)

        self.num_modality = len([file_ for file_ in (cfg.afile, cfg.vfile, cfg.tfile) if file_])
        assert self.num_modality > 0

        self.reset_parameters()

    def load_feats(self, path: str):
        from freeplot.utils import import_pickle
        if cfg.vfile:
            self.register_buffer(
                "vFeats", import_pickle(
                    os.path.join(path, cfg.vfile)
                )
            )
        if cfg.tfile:
            self.register_buffer(
                "tFeats", import_pickle(
                    os.path.join(path, cfg.tfile)
                )
            )
        if cfg.afile:
            self.register_buffer(
                "aFeats", import_pickle(
                    os.path.join(path, cfg.afile)
                )
            )

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

    def get_mEmbds(self):
        vEmbds = self.vProjector(self.vFeats) if cfg.vfile else 0.
        tEmbds = self.tProjector(self.tFeats) if cfg.tfile else 0.
        aEmbds = self.aProjector(self.aFeats) if cfg.afile else 0.
        return vEmbds, tEmbds, aEmbds

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight

        if cfg.position == 'ego':
            vEmbds, tEmbds, aEmbds = self.get_mEmbds()
            itemEmbs = (itemEmbs + vEmbds + tEmbds + aEmbds) / (self.num_modality + 1)

        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        if cfg.position == 'post':
            vEmbds, tEmbds, aEmbds = self.get_mEmbds()
            itemFeats = (itemFeats + vEmbds + tEmbds + aEmbds) / (self.num_modality + 1)
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


class CoachForLightGCN(freerec.launcher.GenCoach):

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

    dataset = getattr(freerec.data.datasets.context, cfg.dataset)(cfg.root)
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
    model = LightGCN(
        tokenizer,
        data_path=dataset.path,
        graph=dataset.train().to_graph((USER, ID), (ITEM, ID)), 
        num_layers=cfg.layers
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

    coach = CoachForLightGCN(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(
        cfg, 
        monitors=[
            'loss', 
            'recall@10', 'recall@20', 
            'precision@10', 'precision@20', 
            'ndcg@10', 'ndcg@20'
        ],
        which4best='ndcg@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()