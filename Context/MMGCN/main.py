

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID


freerec.declare(version="0.7.3")


cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--fusion-mode", type=str, choices=('cat', 'add'), default="cat")

cfg.add_argument("--afile", type=str, default=None, help="the file of acoustic modality features")
cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")


cfg.set_defaults(
    description="MMGCN",
    root="../../data",
    dataset='AmazonBaby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1e-5,
    seed=1
)
cfg.compile()


class GraphConvNet(nn.Module):

    def __init__(
        self, 
        num_users: int,
        feature_dim: int,
        embedding_dim: int,
        num_layers: int = 3,
        fusion_mode: str = "cat",
    ):
        super().__init__()

        self.register_parameter(
            'mUser',
            nn.parameter.Parameter(
                torch.empty((num_users, feature_dim)),
                requires_grad=True
            )
        )
        nn.init.xavier_normal_(self.mUser)

        self.L = num_layers
        self.act = nn.LeakyReLU()

        self.fusion_mode = fusion_mode

        self.aggr_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        ])
        self.m2id_layers = nn.ModuleList([
            nn.Linear(feature_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        ])

        if self.fusion_mode == "cat":
            self.fusion_layers = nn.ModuleList([
                nn.Linear(feature_dim + embedding_dim, embedding_dim),
                nn.Linear(embedding_dim + embedding_dim, embedding_dim),
                nn.Linear(embedding_dim + embedding_dim, embedding_dim),
            ])
        else:
            self.fusion_layers = nn.ModuleList([
                nn.Linear(feature_dim, embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
            ])

    def forward(self, mItem, idEmbds, A: torch.Tensor):
        x = torch.cat((self.mUser, mItem), dim=0) # (N, F_dim)
        x = F.normalize(x, dim=-1)

        for l in range(self.L):
            linear1 = self.aggr_layers[l]
            linear2 = self.m2id_layers[l]
            linear3 = self.fusion_layers[l]

            h = A @ linear1(x) # F/E_dim -> F/E_dim
            x_hat = self.act(linear2(x)) + idEmbds # F/E_dim -> E_dim
            if self.fusion_mode == "cat":
                x_hat = torch.cat((h, x_hat), dim=-1) # (F/E_dim + E_dim)
                x = self.act(linear3(x_hat)) # -> E_dim
            else:
                x = self.act(linear3(h) + x_hat)
        
        return x


class MMGCN(freerec.models.RecSysArch):

    def __init__(
        self, 
        dataset: freerec.data.datasets.base.RecDataSet,
    ) -> None:
        super().__init__()

        self.fields = FieldModuleList(dataset.fields)
        self.fields.embed(
            cfg.embedding_dim, ID
        )
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.load_graph(dataset.train().to_graph((USER, ID), (ITEM, ID)))
        self.load_feats(dataset.path)

        if cfg.vfile:
            self.vGCN = GraphConvNet(
                self.User.count,
                feature_dim=256, # 256 indicates the hidden size of visual features
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )
            self.vProjector = nn.Linear(self.vFeats.size(1), 256)

        if cfg.tfile:
            self.tGCN = GraphConvNet(
                self.User.count,
                feature_dim=self.tFeats.size(1),
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )

        if cfg.afile:
            self.aGCN = GraphConvNet(
                self.User.count,
                feature_dim=self.aFeats.size(1),
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )

        self.num_modality = len([file_ for file_ in (cfg.afile, cfg.vfile, cfg.tfile) if file_])
        assert self.num_modality > 0

        self.reset_parameters()

    def load_graph(self, graph: Data):
        edge_index = graph.edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, normalization='left'
        )
        Adj = freerec.graph.to_adjacency(
            edge_index, edge_weight, 
            num_nodes=self.User.count + self.Item.count
        ).to_sparse_csr()
        self.register_buffer(
            'Adj',
            Adj
        )

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

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        idEmbds = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D

        if cfg.vfile:
            vEmbds = self.vGCN(
                self.vProjector(self.vFeats), idEmbds, self.Adj
            )
        else:
            vEmbds = 0
        if cfg.tfile:
            tEmbds = self.tGCN(
                self.tFeats, idEmbds, self.Adj
            )
        else:
            tEmbds = 0
        if cfg.afile:
            aEmbds = self.aGCN(
                self.aFeats, idEmbds, self.Adj
            )
        else:
            aEmbds = 0
        avgFeats = (vEmbds + tEmbds + aEmbds) / self.num_modality

        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats, itemFeats = self.forward()
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs, self.vGCN.mUser.square().mean()

    def recommend_from_full(self):
        return self.forward()


class CoachForMMGCN(freerec.launcher.GenCoach):

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
            scores, users, items, v_reg_loss = self.model.predict(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) + v_reg_loss
            loss = self.criterion(pos, neg) + reg_loss * self.cfg.weight_decay

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

    model = MMGCN(
        dataset
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

    coach = CoachForMMGCN(
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
