

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp

import freerec
from freerec.data.postprocessing import OrderedIDs, SeqTrainUniformSampler, RandomShuffledSource
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BCELoss4Logits
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID

freerec.declare(version='0.4.3')

cfg = Parser()
cfg.add_argument("--maxlen", type=int, default=5)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-vert", type=int, default=4, help="number of vertical filters")
cfg.add_argument("--num-horiz", type=int, default=16, help="number of horizontal filters")
cfg.add_argument("--num-poss", type=int, default=3, help="number of positive samples")
cfg.add_argument("--num-negs", type=int, default=3, help="number of negative samples")


cfg.set_defaults(
    description="Caser",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=50,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-6,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class Caser(RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]

        self.vert = nn.Conv2d(
            in_channels=1, out_channels=cfg.num_vert,
            kernel_size=(cfg.maxlen, 1), stride=1
        )
        self.horizs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, out_channels=cfg.num_horiz,
                kernel_size=(k, cfg.hidden_size)
            )
            for k in range(1, cfg.maxlen + 1)
        ])
        self.pooling = nn.AdaptiveMaxPool1d((1,))

        self.fc_in_dims = cfg.num_vert * cfg.hidden_size + cfg.num_horiz * cfg.maxlen

        self.fc1 = nn.Linear(self.fc_in_dims, cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout_rate)

        self.W2 = nn.Embedding(self.Item.count + NUM_PADS, cfg.hidden_size * 2, padding_idx=0)
        self.b2 = nn.Embedding(self.Item.count + NUM_PADS, 1, padding_idx=0)

        self.initialize()


    def initialize(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, std=1. / cfg.hidden_size)
        self.b2.weight.data.zero_()


    def _forward(self, 
        seqs: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ):

        seqEmbs = self.Item.look_up(seqs).unsqueeze(1) # (B, 1, S, D)
        userEmbs = self.User.look_up(users).squeeze(1) # (B, D)

        vert_features = self.vert(seqEmbs).flatten(1)
        horiz_features = [
            self.pooling(F.relu(conv(seqEmbs).squeeze(3))).squeeze(2)
            for conv in self.horizs
        ]
        horiz_features = torch.cat(horiz_features, dim=1)

        features = self.dropout(torch.cat((vert_features, horiz_features), dim=1))
        features = F.relu(self.fc1(features))
        features = torch.cat([features, userEmbs], dim=1) # (B, 2D)


        itemEmbs = self.W2(items) # (B, K, 2D)
        itemBias = self.b2(items) # (B, K, 1)

        return torch.baddbmm(itemBias, itemEmbs, features.unsqueeze(2)).squeeze(-1)

    def forward(self, 
        seqs: torch.Tensor,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        items = torch.cat((positives, negatives), dim=1) 
        return self._forward(seqs, users, items)

    def recommend(
        self,
        seqs: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor
    ):
        return self._forward(seqs, users, items)


class CoachForCaser(Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            scores = self.model(seqs, users, positives, negatives)
            posLogits, negLogits = torch.split(scores, [self.cfg.num_poss, self.cfg.num_negs], dim=1)
            posLabels = torch.ones_like(posLogits)
            negLabels = torch.zeros_like(negLogits)
            loss = self.criterion(posLogits, posLabels) + self.criterion(negLogits, negLabels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for data in self.dataloader:
            users, seqs, items = [col.to(self.device) for col in data]
            scores = self.model.recommend(seqs, users, items)
            targets = torch.zeros_like(scores)
            targets[:, 0] = 1

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'NDCG', 'RECALL']
            )


@dp.functional_datapipe("caser_sampling_")
class SeqSampler(SeqTrainUniformSampler):

    def __init__(
        self, source_dp, dataset, num_negatives: int = 1
    ) -> None:
        super().__init__(source_dp, dataset)

        self.num_negatives = num_negatives

    def _sample_neg(self, user: int):
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        user: int 
            A user index.

        Returns:
        --------
        negatives: List[int] 
            A list of negative items that the user has not interacted with.
        """
        seen = self.posItems[user]
        return self.listmap(self._sample_from_pool, [seen] * self.num_negatives)

    def __iter__(self):
        for user, sequence in self.source:
            if len(sequence) < cfg.num_poss:
                continue
            positives = sequence[-cfg.num_poss:]
            seen = sequence[:-cfg.num_poss]
            yield [user, seen, positives, self._sample_neg(user)]


def take_all_train_seqs(dataset):
    User = dataset.fields[USER, ID]
    posItems = [[] for _ in range(User.count)]

    for chunk in dataset.train():
        list(map(
            lambda user, item: posItems[user].append(item),
            chunk[USER, ID], chunk[ITEM, ID]
        ))

    posItems = [tuple(items) for items in posItems]

    sequences = []
    SIZE = cfg.maxlen + cfg.num_poss
    for user, items in enumerate(posItems):
        if len(items) < SIZE:
            sequences.append(
                (user, items)
            )
        else:
            for end in range(len(items), SIZE-1, -1):
                sequences.append(
                    (user, items[end-SIZE:end])
                )
    return [tuple(seq) for seq in sequences]


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]
    sequences = take_all_train_seqs(dataset)

    # trainpipe
    trainpipe = RandomShuffledSource(
        sequences
    ).sharding_filter().caser_sampling_(
        dataset, num_negatives=cfg.num_negs # yielding (user, seqs, targets, negatives)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen, padding_value=0
    ).batch(cfg.batch_size).column_().tensor_()

    # validpipe
    validpipe = OrderedIDs(
        field=User
    ).sharding_filter().seq_valid_sampling_(
        dataset # yielding (user, items, (target + (100) negatives))
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen,
    ).rshift_(
        indices=[1, 2], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen, padding_value=0
    ).batch(cfg.batch_size).column_().tensor_()

    # testpipe
    testpipe = OrderedIDs(
        field=User
    ).sharding_filter().seq_test_sampling_(
        dataset # yielding (user, items, (target + (100) negatives))
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen,
    ).rshift_(
        indices=[1, 2], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen, padding_value=0
    ).batch(cfg.batch_size).column_().tensor_()

    Item.embed(
        cfg.hidden_size, padding_idx=0
    )
    User.embed(cfg.hidden_size)
    tokenizer = FieldModuleList(dataset.fields)
    model = Caser(
        tokenizer
    )

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
    criterion = BCELoss4Logits()

    coach = CoachForCaser(
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
        cfg, monitors=['loss', 'hitrate@1', 'hitrate@5', 'hitrate@10', 'ndcg@5', 'ndcg@10'],
        which4best='ndcg@10'
    )
    coach.fit()



if __name__ == "__main__":
    main()

