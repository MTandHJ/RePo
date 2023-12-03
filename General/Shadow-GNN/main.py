

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from torch_scatter import scatter

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, UNSEEN, SEEN

from sampler import ShadowHopSampler

freerec.declare(version='0.5.1')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)

cfg.add_argument("--khop", type=int, default=2, help="for k-hop sampling")
cfg.add_argument("--num-neighbors", type=int, default=5, help="maximum number of neighbors")
cfg.add_argument("--is_dynamic", action="store_false", default=True)

cfg.set_defaults(
    description="Shadow-GNN",
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


class ShadowGNN(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        sampler: ShadowHopSampler,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = fields
        self.conv = LGConv(normalize=False)
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.sampler = sampler

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

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        return super().to(device, dtype, non_blocking)

    @property
    def nodeEmbds(self):
        return torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)

    def readout(self, features, index, mask):
        # return scatter(features, index, dim=0, reduce='mean')
        return features[mask]

    def forward(self, users: torch.Tensor):
        batch, nodes, mask = [item.to(self.device) for item in self.sampler.sample(users)]
        features = self.nodeEmbds[nodes]
        allFeats = [features]
        for _ in range(self.num_layers):
            features = self.conv(features, batch.edge_index, batch.edge_weight)
            allFeats.append(features)
        allFeats = [self.readout(f, batch.batch, mask) for f in allFeats]
        avgFeats = torch.stack(allFeats, dim=1).mean(1)
        return avgFeats

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats = self.forward(users).unsqueeze(1) # B x 1 x D
        itemFeats = self.Item.look_up(items) # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1)

    def recommend_from_pool(self, users, pool):
        itemFeats = self.Item.look_up(pool) # (B, K, D)
        return self.forward(users).unsqueeze(1).mul(itemFeats).sum(-1) # (B, K)

    def recommend_from_full(self, users):
        itemFeats = self.Item.embeddings.weight
        return self.forward(users).matmul(itemFeats.T) # (B, N)


class CoachForShadow(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            scores = self.model.predict(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        r"""
        Evaluate recommender by sampled-based or full ranking
        according to the form of `data`:

        1. (users, pool):
            users: torch.Tensor, (B, 1)
            pool: torch.Tensor, (B, 101)
        2. (users, unseen, seen):
            users: BufferField
            unseen: BufferField
            seen: BufferField
        """
        for data in self.dataloader:
            if len(data) == 2:
                users, pool = [col.to(self.device) for col in data]
                scores = self.model.recommend(users=users, pool=pool)
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            elif len(data) == 3:
                users, unseen, seen = data
                users = users.to(self.device).data
                seen = seen.to_csr().to(self.device).to_dense().bool()
                targets = unseen.to_csr().to(self.device).to_dense()
                scores = self.model.recommend(users=users)
                scores[seen] = -1e23
            else:
                raise NotImplementedError(
                    f"GenCoach's `evaluate` expects the `data` to be the length of 2 or 3, but {len(data)} received ..."
                )

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )



def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1 # (user, positive, negatives)
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
    model = ShadowGNN(
        tokenizer,
        sampler=ShadowHopSampler(dataset, cfg.khop, cfg.num_neighbors, dynamic=cfg.is_dynamic),
        num_layers=cfg.layers
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
    criterion = freerec.criterions.BPRLoss()

    coach = CoachForShadow(
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