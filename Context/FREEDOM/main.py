

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp
from torch_geometric.data import Data
from torch_geometric.utils import scatter

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.7.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num_ui_layers", type=int, default=2, help="the number of layers for U-I graph")
cfg.add_argument("--num_ii_layers", type=int, default=1, help="the number of layers for I-I graph")

cfg.add_argument("--knn-k", type=int, default=10, help="top-k knn graph")
cfg.add_argument("--weight4mAdj", type=float, default=0.1, help="weight for fusing vAdj and tAd")
cfg.add_argument("--weight4modality", type=float, default=1.e-5, help="weight for modality BPR loss")
cfg.add_argument("--sampling-ratio", type=float, default=0.2, help="sampling ratio for U-I graph")

cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="FREEDOM",
    root="../../data",
    dataset='AmazonBaby_550_MMRec',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=999
)
cfg.compile()


assert cfg.sampling_ratio >= 0., f"Invalid sampling ratio {cfg.sampling_ratio}"


class IISide(nn.Module):

    def __init__(
        self,
        num_items: int,
        num_layers: int,
        embedding_dim: int,
        dataset: freerec.data.datasets.base.RecDataSet
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.num_layers = num_layers
        self.load_feats(dataset.path)

        if cfg.vfile:
            self.vProjector = nn.Linear(self.vFeats.weight.size(1), embedding_dim)

        if cfg.tfile:
            self.tProjector = nn.Linear(self.tFeats.weight.size(1), embedding_dim)

    def load_feats(self, path: str):
        r"""
        Load v/t Features.

        Note: Following the offical implementation,
        they are stored as nn.Embedding and are trainable in default.
        I tried a frozen variant on Baby and found this operation makes no difference.
        """
        from freeplot.utils import import_pickle
        if cfg.vfile:
            vFeats = import_pickle(
                os.path.join(path, cfg.vfile)
            )
            self.vFeats = nn.Embedding.from_pretrained(vFeats, freeze=False)
            vAdj = self.get_knn_graph(vFeats)

        if cfg.tfile:
            tFeats = import_pickle(
                os.path.join(path, cfg.tfile)
            )
            self.tFeats = nn.Embedding.from_pretrained(tFeats, freeze=False)
            tAdj = self.get_knn_graph(tFeats)

        if cfg.vfile and cfg.tfile:
            mAdj = vAdj * cfg.weight4mAdj + tAdj * (1 - cfg.weight4mAdj)
        elif cfg.vfile:
            mAdj = vAdj
        elif cfg.tfile:
            mAdj = tAdj
        else:
            raise NotImplementedError("At least visual or texual modality should be given ...")
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr()
        )
        return

    def get_knn_graph(self, features: torch.Tensor):
        r"""
        Compute the kNN graph.

        Note: Following the offical implementation,
        this graph is not symmetric.
        """
        features = F.normalize(features, dim=-1) # (N, D)
        sim = features @ features.t() # (N, N)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, cfg.knn_k, symmetric=False
        )

        rows, cols = edge_index[0], edge_index[1]
        deg = 1.e-7 + scatter(torch.ones_like(rows), rows, dim=0, dim_size=self.num_items)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
        return torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.num_items, self.num_items)
        )

    def forward(self, itemEmbds: torch.Tensor):
        for _ in range(self.num_layers):
            itemEmbds = self.mAdj @ itemEmbds
        
        vFeats = self.vProjector(self.vFeats.weight) if cfg.vfile else None
        tFeats = self.tProjector(self.tFeats.weight) if cfg.tfile else None
        return itemEmbds, vFeats, tFeats


class FREEDOM(freerec.models.RecSysArch):

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__()

        self.fields = FieldModuleList(dataset.fields)
        self.fields.embed(
            cfg.embedding_dim, ID
        )
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.num_layers = cfg.num_ui_layers

        # I-I Branch
        self.iiSide = IISide( 
            self.Item.count,
            num_layers=cfg.num_ii_layers,
            embedding_dim=cfg.embedding_dim,
            dataset=dataset
        )

        # U-I Branch
        self.load_graph(dataset.train().to_graph((USER, ID), (ITEM, ID)))
        g = dataset.train().to_bigraph(
            (USER, ID), (ITEM, ID),
            edge_type='U2I'
        )
        self.interactions = g['U2I'].edge_index
        self.sampling_probs = self.normalize_graph(
            self.interactions
        )

        self.reset_parameters()

    def normalize_graph(self, edge_index: torch.Tensor):
        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones_like(row)
        row_sum = 1.e-7 + scatter(edge_weight, row, dim=0, dim_size=self.User.count)
        col_sum = 1.e-7 + scatter(edge_weight, col, dim=0, dim_size=self.Item.count)
        row_inv_sqrt = row_sum.pow(-0.5)
        col_inv_sqrt = col_sum.pow(-0.5)
        return row_inv_sqrt[row] * col_inv_sqrt[col]

    def sample_ui_graph(self):
        num_edges = self.sampling_probs.size(0)
        sampled_size = int(num_edges * cfg.sampling_ratio)
        sampled_indices = torch.multinomial(
            self.sampling_probs, 
            num_samples=sampled_size,
            replacement=False
        )
        edge_index = self.interactions[:, sampled_indices].clone()
        edge_index[1] += self.User.count

        num_nodes = self.User.count + self.Item.count
        edge_index = freerec.graph.to_undirected(
            edge_index,
            num_nodes=num_nodes
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, normalization='sym'
        )
        self.uiAdj = freerec.graph.to_adjacency(
            edge_index, edge_weight,
            num_nodes=self.User.count + self.Item.count
        ).to_sparse_csr().to(self.device)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.User.embeddings.weight)
        nn.init.xavier_normal_(self.Item.embeddings.weight)

    def load_graph(self, graph: Data):
        edge_index = graph.edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, normalization='sym'
        )
        Adj = freerec.graph.to_adjacency(
            edge_index, edge_weight,
            num_nodes=self.User.count + self.Item.count
        ).to_sparse_csr()
        self.register_buffer(
            'Adj', Adj
        )

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight

        A = self.uiAdj if self.training else self.Adj

        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = A @ features
            avgFeats += features / (self.num_layers + 1)
        
        iiEmbs, vFeats, aFeats = self.iiSide(itemEmbs)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        return userFeats, itemFeats + iiEmbs, vFeats, aFeats

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats, itemFeats, vFeats, tFeats = self.forward()
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        scores = torch.mul(userFeats, itemFeats).sum(-1)
        if cfg.vfile:
            vFeats = vFeats[items] # B x n x D
            vScores = torch.mul(userFeats, vFeats).sum(-1)
        else:
            vScores = None
        if cfg.tfile:
            tFeats = tFeats[items] # B x n x D
            tScores = torch.mul(userFeats, tFeats).sum(-1)
        else:
            tScores = None
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return scores, vScores, tScores, userEmbs, itemEmbs

    def recommend_from_full(self):
        return self.forward()[:2]


class CoachForFREEDOM(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        self.model.sample_ui_graph()
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            scores, vScores, tScores, users, items = self.model.predict(users, items)
            loss = self.criterion(scores[:, 0], scores[:, 1])
            mloss = 0
            if vScores is not None:
                mloss += self.criterion(vScores[:, 0], vScores[:, 1])
            if tScores is not None:
                mloss += self.criterion(tScores[:, 0], tScores[:, 1])
            loss = loss + mloss * self.cfg.weight4modality

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])


@dp.functional_datapipe("gen_train_pair_uniform_sampling_")
class GenTrainShuffleSampler(freerec.data.postprocessing.sampler.GenTrainUniformSampler):

    def __iter__(self):
        for user, pos in self.source:
            if self._check(user):
                yield [user, pos, self._sample_neg(user)]


def main():

    dataset = getattr(freerec.data.datasets.context, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_pairs()
    ).sharding_filter().gen_train_pair_uniform_sampling_(
        dataset, num_negatives=1
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )

    model = FREEDOM(dataset=dataset)

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

    coach = CoachForFREEDOM(
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