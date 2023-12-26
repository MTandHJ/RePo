

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from modules import Encoder, LayerNorm

freerec.declare(version='0.5.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
cfg.add_argument("--num-blocks", default=2, type=int, help="number of filter-enhanced blocks")
cfg.add_argument("--num-heads", default=2, type=int)
cfg.add_argument("--hidden-act", default="gelu", type=str) # gelu relu
cfg.add_argument("--attention-probs-dropout-prob", default=0.5, type=float)
cfg.add_argument("--hidden-dropout-prob", default=0.5, type=float)
cfg.add_argument("--no-filters", action="store_true", help="if no filters, filter layers transform to self-attention")

cfg.set_defaults(
    description="FMLP-Rec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class FMLPRec(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int = cfg.maxlen,
        hidden_size: int = cfg.hidden_size,
        dropout_rate: float = cfg.hidden_dropout_prob,
        num_blocks: int = cfg.num_blocks,
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

        self.layerNorm = LayerNorm(cfg.hidden_size, eps=1.e-12)
        self.itemEncoder = Encoder(cfg)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0., std=0.02)
            elif isinstance(m, LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.)

    def position_encoding(self, seqs: torch.Tensor):
        S = seqs.size(1)
        positions = torch.arange(0, S, dtype=torch.long, device=self.device).unsqueeze(0)
        positions = self.Position(positions) # (1, maxlen, D)
        return self.embdDropout(self.layerNorm(seqs + positions))

    def create_mask(self, seqs: torch.Tensor):
        # seqs: (B, S)
        padding_mask = (seqs > 0).long() # (B, S)
        attnMask = padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        max_len = padding_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        causalMask = torch.triu(torch.ones(attn_shape), diagonal=1).to(self.device)
        causalMask = (causalMask == 0).unsqueeze(1).long() # (1, S, S)

        attnMask = attnMask * causalMask
        attnMask = (1.0 - attnMask) * -10000.0

        return attnMask # (B, 1, S, S)

    def encode(self, seqs: torch.Tensor):
        attnMask = self.create_mask(seqs)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs = self.position_encoding(seqs)

        features_layers = self.itemEncoder(
            seqs, attnMask,
            output_all_encoded_layers=True
        )

        return features_layers[-1]

    def predict(
        self, 
        seqs: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        features = self.encode(seqs)[:, [-1], :] # (B, 1, D)
        posEmbds = self.Item.look_up(positives) # (B, 1, D)
        negEmbds = self.Item.look_up(negatives) # (B, 1, D)
        return features.mul(posEmbds).sum(-1), features.mul(negEmbds).sum(-1)

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.encode(seqs)[:, [-1], :]  # (B, 1, D)
        others = self.Item.look_up(pool) # (B, K, D)
        return features.mul(others).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.encode(seqs)[:, -1, :]  # (B, D)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t()) # (B, N)


class CoachForFMLP(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            posLogits, negLogits = self.model.predict(seqs, positives, negatives.squeeze(-1))
            indices = positives != 0
            loss = self.criterion(posLogits[indices], negLogits[indices])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def to_roll_seqs(dataset: freerec.data.datasets.RecDataSet, minlen: int = 2):
    seqs = dataset.to_seqs(keepid=True)

    roll_seqs = []
    for id_, items in seqs:
        items = items[-cfg.maxlen:]
        for k in range(minlen, len(items) + 1):
            roll_seqs.append(
                (id_, items[:k])
            )

    return roll_seqs


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=to_roll_seqs(dataset.train())
    ).sharding_filter().seq_train_uniform_sampling_(
        dataset, leave_one_out=True # yielding (users, seqs, positives, negatives)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen,
    ).add_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_seq_lpad_validpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_seq_lpad_testpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )

    Item.embed(
        cfg.hidden_size, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = FMLPRec(
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
    criterion = freerec.criterions.BPRLoss(reduction='mean')

    coach = CoachForFMLP(
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