

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp

import freerec
from freerec.data.postprocessing import OrderedIDs, RandomShuffledSource, GenTrainUniformSampler
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN


freerec.declare(version='0.4.3')


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--loss-type", type=str, choices=('DNS', 'Softmax'), default='DNS')
cfg.add_argument("--num-negs", type=int, default=200)
cfg.add_argument("--weight", type=float, default=2, help="M for DNS; rho for Softmax")

cfg.add_argument("--stepsize", type=int, default=5, help="for StepLR")
cfg.add_argument("--gamma", type=int, default=0.95, help="for StepLR")

cfg.set_defaults(
    description="MF-OPAUC",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=200,
    batch_size=4096,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-2,
    seed=1
)
cfg.compile()


class MFOPAUC(RecSysArch):

    def __init__(self, tokenizer: FieldModuleList) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(
        self, users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        users, items = users, torch.cat(
            [positives, negatives], dim=1
        )
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userEmbs, itemEmbs).sum(-1)
    
    def recommend(self):
        return self.User.embeddings.weight, self.Item.embeddings.weight


class CoachForMFOPAUC(Coach):

    def uniform_sample_negs(self, users: torch.Tensor, num_items: int):
        bsz = users.size(0)
        return torch.randint(0, num_items, (bsz, self.cfg.num_negs), device=self.device)

    @torch.no_grad()
    def calc_importance(self, pos: torch.Tensor, negs: torch.Tensor):
        # pos: (B, 1)
        # negs: (B, num_negs)
        Z = F.logsigmoid(pos - negs).neg()
        tau = torch.sqrt(torch.var(Z, dim=1, keepdim=True).div(2 * self.cfg.weight))
        return F.softmax(Z.div(tau), dim=1) 

    def train_per_epoch(self, epoch: int):
        Item = self.fields[ITEM, ID]
        for data in self.dataloader:
            users, positives, _ = [col.to(self.device) for col in data]
            negatives = self.uniform_sample_negs(users, Item.count)
            scores = self.model(users, positives, negatives)
            pos, negs = scores[:, [0]], scores[:, 1:]
            if self.cfg.loss_type == 'DNS':
                loss = self.criterion(
                    pos, 
                    negs.topk(k=int(self.cfg.weight), dim=1)[0]
                )
                loss = loss.sum(-1)
            elif self.cfg.loss_type == 'Softmax':
                importance = self.calc_importance(pos, negs)
                loss = self.criterion(pos, negs) * importance
                loss = loss.sum(-1)
            loss = loss.sum(-1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="sum", prefix='train', pool=['LOSS'])
        
        self.lr_scheduler.step()

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


@dp.functional_datapipe("gen_train_uniform_yielding__")
class GenTrainShuffleSampler(GenTrainUniformSampler):

    def __iter__(self):
        for user, pos in self.source:
            if self._check(user):
                yield [user, pos, -1]

def take_all(dataset):
    data = []
    for chunk in dataset.train():
        data.extend(list(zip(chunk[USER, ID], chunk[ITEM, ID])))
    return data


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomShuffledSource(
        source=take_all(dataset)
    ).sharding_filter().gen_train_uniform_yielding__(
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
    model = MFOPAUC(tokenizer)

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
    criterion = BPRLoss(reduction='none')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.stepsize, cfg.gamma)

    coach = CoachForMFOPAUC(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
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

