

from typing import Dict, Optional, Union

import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.set_defaults(
    description="GTE",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=0,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()

cfg.epochs = 0

class GTE(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = fields

        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]

        self.num_layers = num_layers
        self.graph = graph

        self.reset_parameters()

    def reset_parameters(self):
        self.User.embeddings.data.fill_(0.)
        self.Item.embeddings.data.copy_(
            torch.eye(self.Item.count)
        )

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.R = self.__graph[f"{self.User.name}2{self.Item.name}"].adj_t.to_torch_sparse_coo_tensor()

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(self):
        userFeats = self.User.embeddings.data
        itemFeats = self.Item.embeddings.data
        for _ in range(self.num_layers):
            userFeats_ = torch.sparse.mm(self.R, itemFeats) + userFeats
            itemFeats_ = torch.sparse.mm(self.R.transpose(0, 1), userFeats) + itemFeats
            userFeats = userFeats_
            itemFeats = itemFeats_
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


class CoachForGTE(freerec.launcher.GenCoach): ...


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
    model = GTE(
        tokenizer, dataset.train().to_bigraph((USER, ID), (ITEM, ID)), num_layers=cfg.layers
    )

    coach = CoachForGTE(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=None,
        optimizer=None,
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
