

from typing import Dict, Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp

import freerec
from freerec.data.postprocessing import OrderedIDs, RandomShuffledSource
from freerec.data.postprocessing.sampler import GenTrainUniformSampler
from freerec.criterions import BaseCriterion
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN


freerec.declare(version='0.4.3')


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1000) # 100,500,1000
cfg.add_argument("--margin", type=float, default=.9) # 0:1:0.1
cfg.add_argument("--weight-for-negative", type=float, default=150)
cfg.add_argument("--dropout-rate", type=float, default=.1)
cfg.add_argument("--unseen-only", type=eval, choices=('True', 'False'), default=False)
cfg.set_defaults(
    description="MF-CCL",
    root="../../data",
    dataset="Yelp18_m1",
    epochs=100,
    batch_size=512,
    optimizer='adam',
    lr=1e-4, # 1e-3 5e-4 1e-4
    weight_decay=1e-9, # 1e-9:1e-2
    seed=1
)
cfg.compile()


assert isinstance(cfg.unseen_only, bool), f"`unseen_only' should be `bool' but type `{type(cfg.unseen_only)}' received ..."


class MFCCL(RecSysArch):

    def __init__(
        self, tokenizer: FieldModuleList,
        dropout_rate: float = cfg.dropout_rate
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User = self.tokenizer[USER, ID]
        self.Item = self.tokenizer[ITEM, ID]

        self.dropout = nn.Dropout(dropout_rate)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)

    def forward(
        self, 
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        users, items = users, torch.cat((positives, negatives), dim=-1)
        itemFeats = F.normalize(self.Item.look_up(items), dim=-1) # B x K x D
        userFeats = self.dropout(F.normalize(self.User.look_up(users), dim=-1)) # B x 1 x D
        scores = (userFeats * itemFeats).sum(-1) # cosine score, B x K
        return scores

    def recommend(self, users: torch.Tensor):
        itemFeats = F.normalize(self.Item.embeddings.weight) # N x D
        userFeats = self.dropout(F.normalize(self.User.look_up(users), dim=-1)).squeeze() # B x 1 x D
        return userFeats.matmul(itemFeats.t())


class CoachForMFCCL(Coach):

    def sample_negs_from_all(self, users, low, high):
        return torch.randint(low, high, size=(len(users), cfg.num_negs), device=self.device)

    def train_per_epoch(self, epoch: int):
        Item = self.fields[ITEM, ID]
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            if not self.cfg.unseen_only:
                negatives = self.sample_negs_from_all(users, 0, Item.count)
            scores = self.model(users, positives, negatives)
            loss = self.criterion(scores)
            loss += self.criterion.regularize(
                self.model.tokenizer.parameters(),
                rtype='l2'
            ) * self.cfg.weight_decay

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for user, unseen, seen in self.dataloader:
            users = user.to(self.device).data
            seen = seen.to_csr().to(self.device).to_dense().bool()
            targets = unseen.to_csr().to(self.device).to_dense()
            scores = self.model.recommend(users)
            scores[seen] = -1e10

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'RECALL']
            )


class CosineContrastiveLoss(BaseCriterion):

    def __init__(
        self, margin: float = cfg.margin, 
        negative_weight: Optional[float] = cfg.weight_for_negative,
        reduction: str = 'mean'
    ):
        super(CosineContrastiveLoss, self).__init__(reduction)
        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, scores: torch.Tensor):
        logits_pos = scores[:, 0]
        loss_pos = (1 - logits_pos).relu()
        logits_neg = scores[:, 1:]
        loss_neg = (logits_neg - self.margin).relu()
        loss = loss_pos + loss_neg.mean(dim=-1) * self.negative_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

@dp.functional_datapipe("gen_train_shuffle_uniform_sampling_")
class GenTrainShuffleSampler(GenTrainUniformSampler):

    def __iter__(self):
        for user, pos in self.source:
            if self._check(user):
                if cfg.unseen_only:
                    yield [user, pos, self._sample_neg(user)]
                else:
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
        take_all(dataset)
    ).sharding_filter().gen_train_shuffle_uniform_sampling_(
        dataset, num_negatives=cfg.num_negs
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
    model = MFCCL(
        tokenizer
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
    criterion = CosineContrastiveLoss()

    coach = CoachForMFCCL(
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
