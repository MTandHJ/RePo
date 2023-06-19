

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN


freerec.declare(version="0.4.3")


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.set_defaults(
    description="BPRMF",
    root="../../data",
    dataset='Gowalla_m1',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-8,
    seed=1
)
cfg.compile()

class BPRMF(RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

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


class CoachForBPRMF(Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            preds = self.model(users, positives, negatives)
            pos, neg = preds[:, 0], preds[:, 1]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

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


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
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
    model = BPRMF(tokenizer)

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
    criterion = BPRLoss()

    coach = CoachForBPRMF(
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

