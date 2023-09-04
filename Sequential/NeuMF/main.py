

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=8)
cfg.add_argument("--num-negs", type=int, default=4)
cfg.add_argument("--hidden-sizes", type=str, default="64,32,16,8")
cfg.set_defaults(
    description="NeuMF",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=1024,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()

cfg.hidden_sizes = tuple(map(int, cfg.hidden_sizes.split(","))) 


class NeuMF(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]

        self.user4mlp = nn.Embedding(self.User.count, cfg.embedding_dim)
        self.item4mlp = nn.Embedding(self.Item.count, cfg.embedding_dim)

        self.user4mf = nn.Embedding(self.User.count, cfg.embedding_dim)
        self.item4mf = nn.Embedding(self.Item.count, cfg.embedding_dim)

        hidden_sizes = [cfg.embedding_dim * 2] + list(cfg.hidden_sizes)
        self.linears = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])]
        )
        self.act = nn.ReLU()

        self.fc = nn.Linear(
            hidden_sizes[-1] + cfg.embedding_dim,
            1
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

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        userEmbs4MLP = self.user4mlp(users) # (B, 1, D)
        itemEmbs4MLP = self.item4mlp(items) # (B, K, D)
        userEmbs4MLP, itemEmbs4MLP = self.broadcast(
            userEmbs4MLP, itemEmbs4MLP
        )
        features4MLP = torch.cat((userEmbs4MLP, itemEmbs4MLP), dim=-1)
        for linear in self.linears:
            features4MLP = self.act(linear(features4MLP)) # (B, K, D')

        userEmbs4MF = self.user4mf(users)
        itemEmbs4MF = self.item4mf(items)
        features4MF = userEmbs4MF.mul(itemEmbs4MF) # (B, K, D')

        features = torch.cat((features4MLP, features4MF), dim=-1) # (B, K, 2D')
        logits = self.fc(features).squeeze(-1) # (B, K)
        return logits

    def predict(self, users: torch.Tensor, items):
        return self.forward(users, items)

    def recommend_from_pool(self, users: torch.Tensor, pool: torch.Tensor):
        return self.forward(users, pool)

    def recommend_from_full(self, users: torch.Tensor):
        items = torch.tensor(range(self.Item.count), dtype=torch.long, device=self.device).unsqueeze(0)
        return self.forward(users, items)


class CoachForNeuMF(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=-1
            )
            logits = self.model(users, items)
            labels = torch.zeros_like(logits)
            labels[:, 0].fill_(1)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=logits.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
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

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=cfg.num_negs
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=128, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=128, ranking=cfg.ranking
    )

    tokenizer = FieldModuleList(dataset.fields)
    model = NeuMF(tokenizer)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = freerec.criterions.BCELoss4Logits()

    coach = CoachForNeuMF(
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
        monitors=[
            'loss', 
            'hitrate@1', 'hitrate@5', 'hitrate@10',
            'ndcg@5', 'ndcg@10'
        ],
        which4best='ndcg@10'
    )
    coach.fit()


if __name__ == "__main__":
    main()