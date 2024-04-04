

import torch
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version="0.7.5")

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)

cfg.add_argument("--num-negatives", type=int, default=500, help="number of negatives")
cfg.add_argument("--t1", type=float, default=0.15, help="temperature 1")
cfg.add_argument("--t2", type=float, default=1., help="temperature 2")
cfg.add_argument("--score-mode", type=str, choices=('cosine', 'inner'), default='cosine')

cfg.set_defaults(
    description="MF-BSL",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-8,
    seed=1
)
cfg.compile()


class BSLMF(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]

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

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        if cfg.score_mode == 'cosine':
            userEmbs = F.normalize(userEmbs, dim=-1)
            itemEmbs = F.normalize(itemEmbs, dim=-1)
        return torch.mul(userEmbs, itemEmbs).sum(-1)

    def recommend_from_full(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        if cfg.score_mode == 'cosine':
            userEmbs = F.normalize(userEmbs, dim=-1)
            itemEmbs = F.normalize(itemEmbs, dim=-1)
        return userEmbs, itemEmbs


class CoachForBSLMF(freerec.launcher.GenCoach):

    def sample_negatives(self, batch_size: int):
        Item = self.fields[ITEM, ID]
        return torch.randint(
            0, Item.count, size=(batch_size, self.cfg.num_negatives)
        ).to(self.device)

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives = [col.to(self.device) for col in data]
            negatives = self.sample_negatives(len(users))
            items = torch.cat(
                [positives, negatives], dim=1
            )
            scores = self.model.predict(users, items)
            pos, neg = scores[:, 0], scores[:, 1:]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])


class BSLoss(freerec.criterions.BaseCriterion):

    def __init__(
        self, 
        temperature1: float,
        temperature2: float,
        reduction: str = 'mean'
    ) -> None:
        super().__init__(reduction)

        self.temperature1 = temperature1
        self.temperature2 = temperature2

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor):
        positives = torch.exp(positives / self.temperature1)
        negatives = torch.exp(negatives / self.temperature1)
        negatives = negatives.sum(dim=-1)
        negatives = torch.pow(negatives, self.temperature2)
        if self.reduction == 'mean':
            return - torch.log(positives / negatives).mean()
        elif self.reduction == 'sum':
            return - torch.log(positives / negatives).sum()
        elif self.reduction == 'none':
            return - torch.log(positives / negatives)
        else:
            raise NotImplementedError

def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_yielding_(
        dataset
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
    model = BSLMF(tokenizer)

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
    criterion = BSLoss(
        temperature1=cfg.t1,
        temperature2=cfg.t2
    )

    coach = CoachForBSLMF(
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

