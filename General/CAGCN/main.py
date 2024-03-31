

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
from freeplot.utils import import_pickle, export_pickle

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID
from freerec.utils import infoLogger, mkdirs
from utils import calc_node_wise_norm, normalize_edge, \
                    jaccard_similarity, \
                    salton_cosine_similarity, \
                    leicht_holme_nerman_similarity, \
                    common_neighbors_similarity


freerec.declare(version="0.7.5")


cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--trend-type", type=str, choices=('jc', 'sc', 'lhn', 'cn'), default='jc')
cfg.add_argument("--trend-coeff", type=float, default=2)
cfg.add_argument("--fusion", type=eval, choices=("True", "False"), default='True')
cfg.set_defaults(
    description="CAGCN",
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


assert cfg.fusion is True or cfg.fusion is False, "cfg.fusion should be `True' or `False' ..."


class CAGCN(freerec.models.RecSysArch):

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = FieldModuleList(dataset.fields)
        self.fields.embed(
            cfg.embedding_dim, ID
        )
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.num_layers = cfg.layers
        self.loadAdj(
            dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        )

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

    def loadAdj(self, edge_index: torch.Tensor):
        R = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1)),
            size=(self.User.count, self.Item.count)
        )
        path = os.path.join("trends", cfg.dataset, cfg.trend_type)
        mkdirs(path)
        file_ = os.path.join(path, "data.pickle")
        try:
            data = import_pickle(file_)
            trend = data['trend']
            edge_index = data['edge_index']
            edge_weight = data['edge_weight']
            edge_norm = data['edge_norm']
            trend_norm = data['trend_norm']
        except ImportError:
            if cfg.trend_type == 'jc':
                edge_index, trend = jaccard_similarity(R)
            elif cfg.trend_type == 'sc':
                edge_index, trend = salton_cosine_similarity(R)
            elif cfg.trend_type == 'lhn':
                edge_index, trend = leicht_holme_nerman_similarity(R)
            elif cfg.trend_type == 'cn':
                edge_index, trend = common_neighbors_similarity(R)
            edge_weight, _ = normalize_edge(edge_index, self.User.count, self.Item.count)
            edge_norm = calc_node_wise_norm(edge_weight, edge_index[1], self.User.count, self.Item.count)
            trend_norm = calc_node_wise_norm(trend, edge_index[1], self.User.count, self.Item.count)

            data = {
                'trend': trend,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'edge_norm': edge_norm,
                'trend_norm': trend_norm
            }
            export_pickle(data, file_)

        if cfg.fusion:
            infoLogger("[CAGCN] >>> Use Trend and Edge Weight together ...")
            trend = cfg.trend_coeff * trend / trend_norm + edge_weight
        else:
            infoLogger("[CAGCN] >>> Use Trend only ...")
            trend = cfg.trend_coeff * trend * edge_norm / trend_norm 

        self.register_buffer(
            'Adj',
            freerec.graph.to_adjacency(
                edge_index, trend
            )
        )

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features
            avgFeats += features / (self.num_layers + 1)
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


class CoachForCAGCN(freerec.launcher.GenCoach):

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

    model = CAGCN(dataset)

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

    coach = CoachForCAGCN(
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
        monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'],
        which4best='ndcg@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()
