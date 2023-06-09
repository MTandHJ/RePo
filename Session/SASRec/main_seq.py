

import torch
import torch.nn as nn
import torchdata.datapipes as dp

import freerec
from freerec.data.datasets import RecDataSet
from freerec.data.postprocessing import RandomIDs, OrderedSource, SeqTrainUniformSampler
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BCELoss4Logits
from freerec.data.fields import FieldModuleList
from freerec.data.tags import SESSION, ITEM, ID, POSITIVE, UNSEEN, SEEN
from freerec.utils import timemeter

freerec.declare(version='0.4.3')

cfg = Parser()
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=100)
cfg.add_argument("--dropout-rate", type=float, default=0.2)

cfg.set_defaults(
    description="SASRec",
    root="../../data",
    dataset='Diginetica_250811_Chron',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    beta1=0.9,
    beta2=0.98,
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: int):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D)
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))) # -> (B, D, S)
        outputs = outputs.transpose(-1, -2) # -> (B, S, D)
        outputs += inputs
        return outputs


class SASRec(RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int,
        hidden_size: int = cfg.hidden_size,
        dropout_rate: float = cfg.dropout_rate,
        num_blocks: int = cfg.num_blocks,
        num_heads: int = cfg.num_heads,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.Position = nn.Embedding(maxlen, hidden_size)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.attnLNs = nn.ModuleList() # to be Q for self-attention
        self.attnLayers = nn.ModuleList()
        self.fwdLNs = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        self.lastLN = nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            self.attnLNs.append(nn.LayerNorm(
                hidden_size, eps=1e-8
            ))

            self.attnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True # !!!
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                hidden_size, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                hidden_size, dropout_rate
            ))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions)[:, -seqs.size(1):, :] # (1, S, D)
        return seqs + positions

    def get_attnMask(self, seqs: torch.Tensor):
        # False True  True ...
        # False False True ...
        # False False False ...
        # ....
        # True indices that the corresponding position is not allowed to attend !
        return torch.ones((seqs.size(1), seqs.size(1)), dtype=torch.bool, device=seqs.device).triu(diagonal=1)

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs, seqs, 
            attn_mask=self.get_attnMask(seqs),
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def forward(self, 
        seqs: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        padding_mask = (seqs == 0).unsqueeze(-1)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs *= self.Item.dimension ** 0.5
        seqs = self.embdDropout(self.position(seqs))
        seqs.masked_fill_(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)
        
        features = self.lastLN(seqs) # (B, S, D)

        posEmbds = self.Item.look_up(positives) # (B, S, D)
        negEmbds = self.Item.look_up(negatives) # (B, S, D)

        return features.mul(posEmbds).sum(-1), features.mul(negEmbds).sum(-1)

    def recommend(
        self,
        seqs: torch.Tensor,
    ):
        padding_mask = (seqs == 0).unsqueeze(-1)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs *= self.Item.dimension ** 0.5
        seqs = self.embdDropout(self.position(seqs))
        seqs.masked_fill_(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)
        
        features = self.lastLN(seqs)[:, -1, :].unsqueeze(-1) # (B, D, 1)
        others = self.Item.embeddings.weight[NUM_PADS:] # (#Items, D)

        return others.matmul(features).flatten(1) # (B, #Items)


class CoachForSASRec(Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            posLogits, negLogits = self.model(seqs, positives, negatives)
            posLabels = torch.ones_like(posLogits)
            negLabels = torch.zeros_like(negLogits)
            indices = positives != 0
            loss = self.criterion(posLogits[indices], posLabels[indices]) + self.criterion(negLogits[indices], negLabels[indices])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for sesses, seqs, unseen, seen in self.dataloader:
            sesses = sesses.data
            seqs = seqs.to(self.device).data
            scores = self.model.recommend(seqs)
            # Don't remove seens for session
            targets = unseen.to_csr().to(self.device).to_dense()

            self.monitor(
                scores, targets,
                n=len(sesses), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'MRR']
            )


@dp.functional_datapipe("sas_train_uniform_sampling_")
class SASRecTrainUniformSampler(SeqTrainUniformSampler):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.User = dataset.fields[SESSION, ID]
        self.posItems = [[] for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(dataset.datasize)

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[SESSION, ID], chunk[ITEM, ID]
            )

        self.posItems = [tuple(items) for items in self.posItems]


def main():

    dataset = getattr(freerec.data.datasets.session, cfg.dataset)(root=cfg.root)
    Session, Item = dataset.fields[SESSION, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomIDs(
        field=Session, datasize=Session.count
    ).sharding_filter().sas_train_uniform_sampling_(
        dataset # yielding (user, seqs, targets, negatives)
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).batch(cfg.batch_size).column_().lpad_col_(
        indices=[1, 2, 3], maxlen=None, padding_value=0
    ).tensor_()

    # validpipe
    validpipe = OrderedSource(
        dataset.valid().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_valid_yielding_(
        dataset # yielding (sesses, seqs, targets, seen)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(512).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_().field_(
        Session.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    # testpipe
    testpipe = OrderedSource(
        dataset.test().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_test_yielding_(
        dataset # yielding (sesses, seqs, targets, seen)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(512).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_().field_(
        Session.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    Item.embed(
        cfg.hidden_size, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)

    # maxlen
    trainlen = max(list(
        map(lambda seq: len(seq), dataset.train().to_seqs(keepid=False))
    ))
    validlen = max(list(
        map(lambda seq: len(seq), dataset.valid().to_seqs(keepid=False))
    ))
    testlen = max(list(
        map(lambda seq: len(seq), dataset.test().to_seqs(keepid=False))
    ))

    model = SASRec(
        tokenizer,
        maxlen=max(trainlen, validlen, testlen)
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
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = BCELoss4Logits()

    coach = CoachForSASRec(
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
        cfg, monitors=['loss', 'hitrate@10', 'hitrate@20', 'precision@10', 'precision@20', 'mrr@10', 'mrr@20'],
        which4best='mrr@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()

