

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=200)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-size", type=int, default=64)

cfg.set_defaults(
    description="STAMP",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=30,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    eval_freq=1,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class STAMP(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        embedding_dim: int = cfg.embedding_dim,
        hidden_size: int = cfg.hidden_size
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.w1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w0 = nn.Linear(embedding_dim, 1, bias=False)
        self.register_parameter(
            'ba',
            nn.Parameter(torch.zeros(embedding_dim).view(1, 1, -1), requires_grad=True)
        )
        self.mlp_a = nn.Linear(embedding_dim, hidden_size, bias=True)
        self.mlp_b = nn.Linear(embedding_dim, hidden_size, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.002)

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        lens = masks.sum(dim=-1, keepdim=True) # (B, 1)
        seqs = self.Item.look_up(seqs) # (B, S, D)
        last = seqs[:, -1, :] # (B, D)
        ms = seqs.sum(dim=1).div(lens).unsqueeze(1) # (B, 1, D)

        alphas = self.w0(self.sigmoid(
            self.w1(seqs) + self.w2(last.unsqueeze(1)) + self.w3(ms) + self.ba
        )) # (B, S, 1)
        ma = alphas.mul(seqs).sum(1) + last

        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last))
        h = hs.mul(ht) # (B, D)

        return h

    def predict(
        self, 
        seqs: torch.Tensor, 
        positives: torch.Tensor, 
        negatives:torch.Tensor
    ):
        features = self.forward(seqs)
        posEmbds = self.Item.look_up(positives).squeeze(1) # (B, D)
        negEmbds = self.Item.look_up(negatives).squeeze(1) # (B, D)
        return features.mul(posEmbds).sum(-1), features.mul(negEmbds).sum(-1)

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs).unsqueeze(1) # (B, 1, D)
        items = self.Item.look_up(pool) # (B, K, D)
        return features.mul(items).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t())


class CoachForSTAMP(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            pos, neg = self.model.predict(seqs, positives, negatives)
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_roll_seqs(minlen=2)
    ).sharding_filter().seq_train_uniform_sampling_(
        dataset, leave_one_out=True # yielding (users, seqs, positives, negatives)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen,
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).batch(cfg.batch_size).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_()

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
        cfg.embedding_dim, padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = STAMP(
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
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )

    criterion = freerec.criterions.BPRLoss()

    coach = CoachForSTAMP(
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