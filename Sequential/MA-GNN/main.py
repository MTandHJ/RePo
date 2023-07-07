

from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torchdata.datapipes as dp

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID
from freerec.utils import timemeter

from modules import GNN, get_sinusoid_encoding_table

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)

cfg.add_argument("--maxlen", type=int, default=6 + 20, help="here, `maxlen` is equal to the length of left sequence plus `L`")
cfg.add_argument('--L', type=int, default=6, help="sequence size for short-term interest modeling")
cfg.add_argument('--T', type=int, default=2, help="num of targest")
cfg.add_argument('--K', type=int, default=3, help="K nearst neighbors")

# train arguments
cfg.add_argument('--layers', type=int, default=2, help='gnn propogation steps')
cfg.add_argument('--hidden-size', type=int, default=20, help='number of dimensions in attention')
cfg.add_argument('--memory-size', type=int, default=20, help='number of memory units')

cfg.set_defaults(
    description="MA-GNN",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=4096,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-3,
    seed=1,
)
cfg.compile()


NUM_PADS = 1



class MAGNN(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        L: int = cfg.L, T: int = cfg.T, maxlen: int = cfg.maxlen,
        K: int = cfg.K, layers: int = cfg.layers,
        embedding_dim: int = cfg.embedding_dim,
        hidden_size: int = cfg.hidden_size,
        memory_size: int = cfg.memory_size,
    ) -> None:
        super().__init__()

        self.fields = fields
        self.User= self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]

        self.L = L
        self.T = T

        self.gnn = GNN(embedding_dim, L, T, layers, K)
        self.register_buffer(
            'position_code_embedding',
            get_sinusoid_encoding_table(maxlen - L, embedding_dim)
        )

        self.register_buffer(
            "ones_production",
            torch.ones(1, maxlen - L)
        )

        self.feature_gate_item = nn.Linear(embedding_dim, embedding_dim)
        self.feature_gate_user = nn.Linear(embedding_dim, embedding_dim)

        self.register_parameter(
            'instance_gate_item',
            nn.parameter.Parameter(
                    torch.zeros(embedding_dim, 1, dtype=torch.float32), requires_grad=True
            )
        )

        self.register_parameter(
            'instance_gate_user',
            nn.parameter.Parameter(
                    torch.zeros(embedding_dim, L, dtype=torch.float32), requires_grad=True
            )
        )

        self.long_W1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.long_W2 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.long_W3 = nn.Parameter(torch.Tensor(embedding_dim, hidden_size))
        self.memory_K = nn.Parameter(torch.Tensor(embedding_dim, memory_size))
        self.memory_V = nn.Parameter(torch.Tensor(embedding_dim, memory_size))
        self.fusion_W1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.fusion_W2 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.fusion_W3 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.item_item_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))

        self.attention_item1 = nn.Linear(embedding_dim, embedding_dim)
        self.attention_item2 = nn.Linear(embedding_dim, hidden_size)

        self.W2 = nn.Embedding(self.Item.count + NUM_PADS, embedding_dim, padding_idx=0)
        self.b2 = nn.Embedding(self.Item.count + NUM_PADS, 1, padding_idx=0)

        self.reset_parameters(self.User.count, self.Item.count + NUM_PADS)

    def reset_parameters(self, num_users: int, num_items: int):
        nn.init.xavier_uniform_(self.instance_gate_item)
        nn.init.xavier_uniform_(self.instance_gate_user)
        nn.init.xavier_uniform_(self.item_item_W)
        nn.init.xavier_uniform_(self.long_W1)
        nn.init.xavier_uniform_(self.long_W2)
        nn.init.xavier_uniform_(self.long_W3)
        nn.init.xavier_uniform_(self.memory_K)
        nn.init.xavier_uniform_(self.memory_V)
        nn.init.xavier_uniform_(self.fusion_W1)
        nn.init.xavier_uniform_(self.fusion_W2)
        nn.init.xavier_uniform_(self.fusion_W3)

        self.User.embeddings.weight.data.normal_(
            0, 1.0 / num_users)
        self.Item.embeddings.weight.data.normal_(
            0, 1.0 / num_items)
        self.W2.weight.data.normal_(0, 1.0 / num_items)
        self.b2.weight.data.zero_()

    def forward(self, users: torch.Tensor, seqs: torch.Tensor):
        # left_seqs: (B, maxlen - L, D)
        left_seqs, seqs = seqs[:, :-self.L], seqs[:, -self.L:]

        item_embs = self.Item.look_up(seqs) # (B, L, D)
        left_item_embs = self.Item.look_up(left_seqs) # (B, maxlen - L, D)
        user_embs = self.User.look_up(users.squeeze(1)) # (B, D)

        # short
        short_embs = self.gnn(item_embs) # (B, L, D)
        short_arg_embs = short_embs.mean(1) # (B, D)

        # long
        long_hidden = left_item_embs + self.position_code_embedding # (B, maxlen - L, D)
        long_user_hidden = user_embs.matmul(self.long_W2).unsqueeze(2) * self.ones_production # (B, D, maxlen - L)

        long_hidden_head = (long_hidden.matmul(self.long_W1) + long_user_hidden.transpose(1, 2)).tanh() # (B, maxlen - L, D)
        long_hidden_head = long_hidden_head.matmul(self.long_W3).softmax(dim=2) # (B, maxlen - L, hidden_size)

        matrix_z = torch.bmm(long_hidden.permute(0, 2, 1), long_hidden_head) # (B, D, hidden_size)
        long_query = matrix_z.tanh().mean(dim=2) # (B, D)

        # memory units
        memory_hidden = long_query.matmul(self.memory_K).softmax(dim=1)
        memory_hidden = memory_hidden.matmul(self.memory_V.t())
        long_embs = long_query + memory_hidden  # [B, D]

        # fusion interest
        gate_unit = torch.sigmoid(
            short_arg_embs.matmul(self.fusion_W1)
            + long_embs.matmul(self.fusion_W2)
            + user_embs.matmul(self.fusion_W3)
        )
        fusion_embs = gate_unit * short_arg_embs + (1 - gate_unit) * long_embs  # (B, D)

        return user_embs, item_embs, fusion_embs

    def predict(
        self, 
        users: torch.Tensor,
        seqs: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        user_embs, item_embs, fusion_embs = self.forward(users, seqs)
        items = torch.cat((positives, negatives), dim=-1) # (B, 2T)
        w2 = self.W2(items) # (B, 2T, D)
        b2 = self.b2(items) # (B, 2T, 1)

        res = torch.baddbmm(b2, w2, user_embs.unsqueeze(2)).squeeze(-1) # (B, 2T)

        # union-level
        res += torch.bmm(fusion_embs.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        rel_score = item_embs.bmm(w2.permute(0, 2, 1))
        rel_score = torch.mean(rel_score, dim=1)
        res += rel_score
        return torch.split(res, [self.T, self.T], dim=1)

    def recommend_from_pool(self, users: torch.Tensor, seqs: torch.Tensor, pool: torch.Tensor):
        user_embs, item_embs, fusion_embs = self.forward(users, seqs)
        w2 = self.W2(pool)
        b2 = self.b2(pool)

        res = torch.baddbmm(b2, w2, user_embs.unsqueeze(2)).squeeze(-1)

        # union-level
        res += torch.bmm(fusion_embs.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        rel_score = item_embs.bmm(w2.permute(0, 2, 1))
        rel_score = torch.mean(rel_score, dim=1)
        res += rel_score
        return res

    def recommend_from_full(self, users: torch.Tensor, seqs: torch.Tensor):
        user_embs, item_embs, fusion_embs = self.forward(users, seqs)
        w2 = self.W2.weight[NUM_PADS:] # (N, D)
        b2 = self.b2.weight[NUM_PADS:] # (N, 1)

        res = user_embs.mm(w2.t()) + b2.squeeze(-1) # (B, N)

        # union-level
        res += fusion_embs.mm(w2.t())

        rel_score = torch.matmul(item_embs, w2.t().unsqueeze(0))
        rel_score = torch.mean(rel_score, dim=1)
        res += rel_score
        return res


class CoachForMAGNN(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            posScores, negScores = self.model.predict(users, seqs, positives, negatives)
            loss = self.criterion(posScores, negScores)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="sum", prefix='train', pool=['LOSS'])


def to_roll_seqs(dataset):
    seqs = dataset.train().to_seqs(keepid=True)

    roll_seqs = []
    for id_, items in seqs:
        if len(items) < (cfg.L + cfg.T):
                roll_seqs.append(
                    (id_, items)
                )
        else:
            for k in range(cfg.L + cfg.T, len(items) + 1):
                roll_seqs.append(
                    (id_, items[:k])
                )

    return roll_seqs


@dp.functional_datapipe("ma_seq_train_uniform_sampling_")
class MASeqTrainUniformSampler(freerec.data.postprocessing.sampler.SeqTrainUniformSampler):

    def __init__(
        self, 
        source_dp: dp.iter.IterableWrapper, 
        dataset: Optional[freerec.data.datasets.RecDataSet]
    ) -> None:
        super().__init__(source_dp, dataset, True)

        self.marker = cfg.T

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(dataset.train().datasize)
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(sorted(items)) for items in self.posItems]


    def _sample_neg(self, user: int, positives: Tuple) -> List[int]:
        r""" 
        Pre-verified with binary search.
        See reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html.

        `posItems` below should be ordered !!!
        """
        posItems = np.array(self.posItems[user])
        raw_samp = np.random.randint(0, self.Item.count - len(posItems), size=len(positives))
        pos_inds_adj = posItems - np.arange(len(posItems))
        ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
        neg_inds = raw_samp + ss
        return neg_inds.tolist()

    def __iter__(self):
        for user, seq in self.source:
            if self._check(seq):
                seen = seq[:-self.marker]
                positives = seq[-self.marker:]
                negatives = self._sample_neg(user, positives)
                yield [user, seen, positives, self._sample_neg(user, negatives)]


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=to_roll_seqs(dataset)
    ).sharding_filter().ma_seq_train_uniform_sampling_(
        dataset, # yielding (user, seqs, targets, negatives)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen, padding_value=0
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

    tokenizer = FieldModuleList(dataset.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID, padding_idx = 0
    )
    model = MAGNN(
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
    criterion = freerec.criterions.BPRLoss(reduction='sum')

    coach = CoachForMAGNN(
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