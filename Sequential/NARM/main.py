

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
cfg.add_argument("--emb-dropout-rate", type=float, default=0.25)
cfg.add_argument("--ct-dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-gru-layers", type=int, default=1)

cfg.set_defaults(
    description="NARM",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=100,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-8,
    eval_freq=5,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class NARM(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.emb_dropout = nn.Dropout(cfg.emb_dropout_rate)
        self.gru = nn.GRU(
            cfg.embedding_dim,
            cfg.hidden_size,
            cfg.num_gru_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.a_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_t = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(cfg.ct_dropout_rate)
        self.b = nn.Linear(2 * cfg.hidden_size, cfg.embedding_dim, bias=False)

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

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0).unsqueeze(-1) # (B, S, 1)
        seqs = self.Item.look_up(seqs) # (B, S, D)
        seqs = self.emb_dropout(seqs)
        gru_out, _ = self.gru(seqs) # (B, S, H)

        c_global = ht = gru_out.gather(
            dim=1,
            index=masks.sum(1, keepdim=True).add(-1).expand((-1, 1, gru_out.size(-1)))
        ) # (B, 1, H)

        q1 = self.a_1(gru_out) # (B, S, H)
        q2 = self.a_2(ht) # (B, 1, H)

        alpha = self.v_t(masks * torch.sigmoid(q1 + q2)) # (B, S, 1)
        c_local = torch.sum(alpha * gru_out, 1) # (B, H)
        c_t = torch.cat([c_local, c_global.squeeze(1)], 1) # (B, 2H)
        c_t = self.ct_dropout(c_t)
        features = self.b(c_t) # (B, D)
        return features

    def predict(self, seqs: torch.Tensor):
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        features = self._forward(seqs)
        return features.matmul(items.t())

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


class CoachForNARM(freerec.launcher.SeqCoach):

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
    ).batch(cfg.batch_size).column_().rpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_()

    validpipe = freerec.data.dataloader.load_seq_rpad_validpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_seq_rpad_testpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )

    Item.embed(
        cfg.embedding_dim, padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = NARM(
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

    coach = CoachForNARM(
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