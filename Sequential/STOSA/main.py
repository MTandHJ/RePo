

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdata.datapipes as dp

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN, POSITIVE

from modules import LayerNorm, DistSAEncoder, wasserstein_distance, kl_distance

freerec.declare(version='0.4.3')

cfg = Parser()
cfg.add_argument("--maxlen", type=int, default=100)
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-blocks", type=int, default=1)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.3)
cfg.add_argument("--attn-dropout-rate", type=float, default=0.)
cfg.add_argument("--distance-metric", type=str, choices=("wasserstein", "kl"), default="wasserstein")
cfg.add_argument("--pvn-weight", type=float, default=0.005, help="the weight for postives versus negatives")

cfg.set_defaults(
    description="STOSA",
    root="../../data",
    dataset='AmazonBeauty_550_Chron',
    epochs=500,
    batch_size=256,
    optimizer='adam',
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class STOSA(RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int = cfg.maxlen,
        num_blocks: int = cfg.num_blocks,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.fields = fields
        self.User, self.Item = self.fields[ID]

        self.item_mean_embds = nn.Embedding(self.Item.count + NUM_PADS, cfg.hidden_size, padding_idx=0)
        self.item_cov_embds = nn.Embedding(self.Item.count + NUM_PADS, cfg.hidden_size, padding_idx=0)

        self.pos_mean_embds = nn.Embedding(cfg.maxlen, cfg.hidden_size)
        self.pos_cov_embds = nn.Embedding(cfg.maxlen, cfg.hidden_size)

        self.embdLN = LayerNorm(cfg.hidden_size, eps=1e-12)
        self.embdDropout = nn.Dropout(p=cfg.hidden_dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, cfg.maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.encoder = DistSAEncoder(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_blocks,
            hidden_dropout_rate=cfg.hidden_dropout_rate,
            attn_dropout_rate=cfg.attn_dropout_rate,
            distance_metric=cfg.distance_metric
        )

        self.register_buffer(
            'attnMask',
            torch.ones((1, 1, maxlen, maxlen), dtype=torch.bool).tril() # (1, 1, maxlen, maxlen)
        )

        self.initialize()

    def initialize(self):
        """ Initialize the weights.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                module.weight.data.normal_(mean=0.01, std=0.02)
            elif isinstance(module, LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def mark_mean_pos(self, seqs: torch.Tensor):
        seqs = self.item_mean_embds(seqs)
        positions = self.pos_mean_embds(self.positions) # (1, maxlen, D)
        seqs = seqs + positions
        seqs = self.embdLN(seqs)
        seqs = self.embdDropout(seqs)
        return F.elu(seqs)

    def mark_cov_pos(self, seqs: torch.Tensor):
        seqs = self.item_cov_embds(seqs)
        positions = self.pos_cov_embds(self.positions) # (1, maxlen, D)
        seqs = seqs + positions
        seqs = self.embdLN(seqs)
        seqs = self.embdDropout(seqs)
        return F.elu(seqs) + 1 # positive semidefinite

    def forward(self, 
        seqs: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        attn_mask = (seqs > 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        attn_mask = attn_mask.logical_and(self.attnMask)
        attn_mask = (1. -  attn_mask.float()) * (- 2 ** 32 + 1)

        mean_seqs = self.mark_mean_pos(seqs)
        cov_seqs = self.mark_cov_pos(seqs)

        all_layer_items = self.encoder(
            mean_seqs, cov_seqs, attn_mask,
            output_all_encoded_layers=False
        )

        mean_outputs, cov_outputs, _ = all_layer_items[-1]

        mean_positives = self.item_mean_embds(positives)
        cov_positives = F.elu(self.item_cov_embds(positives)) + 1
        mean_negatives = self.item_mean_embds(negatives)
        cov_negatives = F.elu(self.item_cov_embds(negatives)) + 1

        if cfg.distance_metric == "wasserstein": 
            dist_func = wasserstein_distance
        elif cfg.distance_metric == "kl":
            dist_func = kl_distance

        posLogits = dist_func(
            mean_outputs, cov_outputs,
            mean_positives, cov_positives
        ).neg()

        negLogits = dist_func(
            mean_outputs, cov_outputs,
            mean_negatives, cov_negatives
        ).neg()

        pvnLogits = dist_func(
            mean_positives, cov_positives,
            mean_negatives, cov_negatives
        ).neg()

        return posLogits, negLogits, pvnLogits


    def recommend(
        self,
        seqs: torch.Tensor,
        items: torch.Tensor
    ):
        attn_mask = (seqs > 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        attn_mask = attn_mask.logical_and(self.attnMask)
        attn_mask = (1. -  attn_mask.float()) * (- 2 ** 32 + 1)

        mean_seqs = self.mark_mean_pos(seqs)
        cov_seqs = self.mark_cov_pos(seqs)

        all_layer_items = self.encoder(
            mean_seqs, cov_seqs, attn_mask,
            output_all_encoded_layers=False
        )

        mean_outputs, cov_outputs, _ = all_layer_items[-1]
        mean_outputs = mean_outputs[:, [-1], :]
        cov_outputs = cov_outputs[:, [-1], :]

        mean_items = self.item_mean_embds(items)
        cov_items = F.elu(self.item_cov_embds(items)) + 1

        if cfg.distance_metric == "wasserstein": 
            dist_func = wasserstein_distance
        elif cfg.distance_metric == "kl":
            dist_func = kl_distance

        return dist_func(
            mean_outputs, cov_outputs,
            mean_items, cov_items
        ).neg()


class CoachForSTOSA(Coach):

    def pvn_loss(self, posLogits: torch.Tensor, pvnLogits: torch.Tensor):
        return  (pvnLogits - posLogits).clamp(0.).mean()

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            posLogits, negLogits, pvnLogits = self.model(seqs, positives, negatives)
            indices = positives != 0
            loss = self.criterion(posLogits[indices], negLogits[indices])
            loss += self.pvn_loss(posLogits[indices], pvnLogits[indices]) * self.cfg.pvn_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for data in self.dataloader:
            users, seqs, items = [col.to(self.device) for col in data]
            scores = self.model.recommend(seqs, items)
            targets = torch.zeros_like(scores)
            targets[:, 0] = 1

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'NDCG', 'RECALL']
            )


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomIDs(
        field=User, datasize=User.count
    ).sharding_filter().seq_train_uniform_sampling_(
        dataset # yielding (user, seqs, targets, negatives)
    ).lprune_(
        indices=[1, 2, 3], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).lpad_(
        indices=[1, 2, 3], maxlen=cfg.maxlen, padding_value=0
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

    tokenizer = FieldModuleList(dataset.fields)
    model = STOSA(
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
    criterion = BPRLoss(reduction="mean")

    coach = CoachForSTOSA(
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

