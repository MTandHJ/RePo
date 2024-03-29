

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, POSITIVE, UNSEEN, SEEN

from samplers import *

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=50)
cfg.add_argument("--dropout-rate", type=float, default=0.2)

cfg.add_argument('--l2_emb', default=5.e-5, type=float)
cfg.add_argument('--time_span', default=256, type=int)


cfg.set_defaults(
    description="TiSASRec",
    root="../../data",
    dataset='MovieLens1M',
    epochs=200,
    batch_size=128,
    optimizer='adam',
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


class TimeAwareMultiHeadAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(embed_dim, embed_dim)
        self.K_w = torch.nn.Linear(embed_dim, embed_dim)
        self.V_w = torch.nn.Linear(embed_dim, embed_dim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.dropout_rate = dropout_rate

    def forward(
        self, 
        queries: torch.Tensor, keys: torch.Tensor, 
        padding_mask: torch.Tensor, attn_mask: torch.Tensor, 
        abs_pos_K: torch.Tensor, abs_pos_V: torch.Tensor,
        time_matrix_K: torch.Tensor, time_matrix_V: torch.Tensor, 
    ):
        # Q, K, V: (B, S, D)
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # Q_, K_, V_: (B * num_heads, S, D / num_heads)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # abs_pos_K/V_: (B * num_heads, S, D / num_heads)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)
        # time_matrix_K/V_: (B * num_heads, S, S, D / num_heads)
        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2)) # (B * num_heads, S, S)
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        padding_mask = padding_mask.repeat(self.num_heads, 1, 1)
        padding_mask = padding_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.expand(attn_weights.shape[0], -1, -1)
        neg_infs = torch.empty_like(attn_weights).fill_(-2 ** 32 + 1)
        attn_weights = torch.where(padding_mask, neg_infs, attn_weights) # True: -inf
        attn_weights = torch.where(attn_mask, neg_infs, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights)
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (B * num_heads, S, D / num_head) -> (B, S, D)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class TiSASRec(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int = cfg.maxlen,
        hidden_size: int = cfg.hidden_size,
        dropout_rate: float = cfg.dropout_rate,
        num_blocks: int = cfg.num_blocks,
        num_heads: int = cfg.num_heads,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.fields = fields
        self.Item = self.fields[ITEM, ID]
        self.embdDropout = nn.Dropout(p=dropout_rate)

        self.Position4Key = nn.Embedding(maxlen, hidden_size)
        self.Position4Val = nn.Embedding(maxlen, hidden_size)
        self.embdDropout4Key = nn.Dropout(p=dropout_rate)
        self.embdDropout4Val = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.Time4Key = nn.Embedding(cfg.time_span + 1, cfg.hidden_size)
        self.Time4Val = nn.Embedding(cfg.time_span + 1, cfg.hidden_size)
        self.timeDropout4Key = nn.Dropout(p=dropout_rate)
        self.timeDropout4Val = nn.Dropout(p=dropout_rate)

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
                TimeAwareMultiHeadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                hidden_size, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                hidden_size, dropout_rate
            ))

        # False True  True ...
        # False False True ...
        # False False False ...
        # ....
        # True indices that the corresponding position is not allowed to attend !
        self.register_buffer(
            'attnMask',
            torch.ones((maxlen, maxlen), dtype=torch.bool).triu(diagonal=1)
        )

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

    def get_params(self):
        embeddings = []
        others = []
        for name, params in self.named_parameters():
            flags = [emb_name in name for emb_name in ('fields', 'Position4Key', 'Position4Val', 'Time4Key', 'Time4Val')]
            if any(flags):
                embeddings.append(params)
            else:
                others.append(params)
        assert len(embeddings) == 5, "Unknown errors cause some embeddings missed"
        return [{'params': embeddings, 'weight_decay': cfg.l2_emb}, {'params': others}]

    def after_one_block(
        self, seqs: torch.Tensor, 
        time_matrix_K: torch.Tensor, time_matrix_V: torch.Tensor,
        abs_pos_K: torch.Tensor, abs_pos_V: torch.Tensor,
        padding_mask: torch.Tensor, l: int
    ):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs,
            abs_pos_K=abs_pos_K, abs_pos_V=abs_pos_V,
            time_matrix_K=time_matrix_K, time_matrix_V=time_matrix_V,
            padding_mask=padding_mask, attn_mask=self.attnMask,
        ) + Q

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def forward(self, seqs: torch.Tensor, times: torch.Tensor):

        padding_mask = (seqs == 0).unsqueeze(-1) # (B, S, 1)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs *= self.Item.dimension ** 0.5
        seqs = self.embdDropout(seqs)

        abs_pos_K = self.embdDropout4Key(self.Position4Key(self.positions)) # (1, maxlen, D)
        abs_pos_V = self.embdDropout4Val(self.Position4Val(self.positions)) # (1, maxlen, D)

        time_matrix_K = self.timeDropout4Key(self.Time4Key(times)) # (B, maxlen, maxlen, D) ?
        time_matrix_V = self.timeDropout4Val(self.Time4Val(times)) # (B, maxlen, maxlen, D) ?

        seqs.masked_fill_(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(
                seqs, 
                time_matrix_K, time_matrix_V,
                abs_pos_K, abs_pos_V,
                padding_mask, l
            )
        
        features = self.lastLN(seqs) # (B, S, D)

        return features

    def predict(
        self,
        seqs: torch.Tensor, times: torch.Tensor,
        positives: torch.Tensor, negatives: torch.Tensor
    ):
        features = self.forward(seqs, times)

        posEmbds = self.Item.look_up(positives) # (B, S, D)
        negEmbds = self.Item.look_up(negatives) # (B, S, D)

        return features.mul(posEmbds).sum(-1), features.mul(negEmbds).sum(-1)

    def recommend_from_pool(
        self, seqs: torch.Tensor, times: torch.Tensor, pool: torch.Tensor
    ):
        features = self.forward(seqs, times)[:, -1, :].unsqueeze(-1) # (B, D, 1)
        others = self.Item.look_up(pool) # (B, 101, D)
        return others.matmul(features).flatten(1) # (B, 101)

    def recommend_from_full(self, seqs: torch.Tensor, times: torch.Tensor):
        features = self.forward(seqs, times)[:, -1, :].unsqueeze(-1) # (B, D, 1)
        others = self.Item.embeddings.weight[NUM_PADS:] # (#Items, D)
        return others.matmul(features).flatten(1) # (B, 101)


class CoachForTiSASRec(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, times, positives, negatives = [col.to(self.device) for col in data]
            posLogits, negLogits = self.model.predict(seqs, times, positives, negatives)
            posLabels = torch.ones_like(posLogits)
            negLabels = torch.zeros_like(negLogits)
            indices = positives != 0
            loss = self.criterion(posLogits[indices], posLabels[indices]) + self.criterion(negLogits[indices], negLabels[indices])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for data in self.dataloader:
            if len(data) == 4:
                users, seqs, times, pool = [col.to(self.device) for col in data]
                scores = self.model.recommend(seqs=seqs, times=times, pool=pool)
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            elif len(data) == 5:
                users, seqs, times, unseen, seen = data
                users = users.to(self.device).data
                times = times.to(self.device).data
                seqs = seqs.to(self.device).data
                scores = self.model.recommend(seqs=seqs, times=times)
                seen = seen.to_csr().to(self.device).to_dense().bool()
                scores[seen] = -1e23
                targets = unseen.to_csr().to(self.device).to_dense()
            else:
                raise NotImplementedError(
                    f"SeqCoach's `evaluate` expects the `data` to be the length of 3 or 4, but {len(data)} received ..."
                )

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )



def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]
    Time = dataset.fields[TIMESTAMP]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_seqs(keepid=True)
    ).sharding_filter().ti_seq_train_uniform_sampling_(
        dataset, leave_one_out=False # yielding (user, seqs, times, targets, negatives)
    ).lprune_(
        indices=[1, 2, 3, 4], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1, 3, 4], offset=NUM_PADS
    ).lpad_(
        indices=[1, 2, 3, 4], maxlen=cfg.maxlen, padding_value=0
    ).time2matrix_(
        indices=[2]
    ).batch(cfg.batch_size).column_().tensor_()


    # validpipe
    if cfg.ranking == 'full':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().ti_seq_valid_yielding_(
            dataset # yielding (user, seqs, times, unseens, seens)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).time2matrix_(
            indices=[2]
        ).batch(cfg.batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Time.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().ti_seq_valid_sampling_(
            dataset # yielding (user, seqs, times, (target + (100) negatives))
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).time2matrix_(
            indices=[2]
        ).batch(cfg.batch_size).column_().tensor_()

    # testpipe
    if cfg.ranking == 'full':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().ti_seq_test_yielding_(
            dataset # yielding (user, seqs, times, unseens, seens)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).time2matrix_(
            indices=[2]
        ).batch(cfg.batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Time.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().ti_seq_test_sampling_(
            dataset # yielding (user, seqs, times, (target + (100) negatives))
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).time2matrix_(
            indices=[2]
        ).batch(cfg.batch_size).column_().tensor_()

    Item.embed(
        cfg.hidden_size, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = TiSASRec(
        tokenizer
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.get_params(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.get_params(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = freerec.criterions.BCELoss4Logits()

    coach = CoachForTiSASRec(
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