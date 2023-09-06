

from typing import Tuple, List

import torch, math
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID
from freerec.utils import infoLogger

from modules import TransformerEncoderLayer, TransformerEncoder, MultiheadAttention
from samplers import *

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=5)
cfg.add_argument("--num-heads", type=int, default=2)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=500)
cfg.add_argument("--dropout-rate", type=float, default=0.)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.5)
cfg.add_argument('--kernel_type', default='log-1', type=str, 
    help="kernels for timespan transformation: exp-3 equals to [exp, exp, exp]; exp-2-log-2 equals to [exp, exp, log, log]"
)


cfg.set_defaults(
    description="CTA",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=20,
    batch_size=100,
    optimizer='adagrad',
    momentum=0.1,
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SATT(freerec.models.RecSysArch):
    def __init__(
        self, fields: FieldModuleList,
        kernel_type='exp-1'
    ):
        super().__init__()
        
        self.fields = fields
        self.Item = fields[ITEM, ID]
        self.pe = PositionalEncoding(cfg.hidden_size, cfg.dropout_rate, max_len=cfg.maxlen)

        encoder_layer = TransformerEncoderLayer(cfg.hidden_size, cfg.num_heads, dim_feedforward=2048, dropout=cfg.hidden_dropout_rate)
        norm = nn.LayerNorm(cfg.hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, cfg.num_blocks, norm=norm).to(self.device)
        
        self.decoder = MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.hidden_dropout_rate)
        
        parts = kernel_type.split('-')
        kernel_types = []

        self.params = []
        for i in range( len(parts) ):
            pi = parts[i]
            if pi in {'exp', 'exp*', 'log', 'lin', 'exp^', 'exp*^', 'log^', 'lin^', 'ind', 'const', 'thres'}:
                if pi.endswith('^'):
                    var = (
                        nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) * 5 + 10), 
                        nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
                    )
                    kernel_types.append(pi[:-1])
                else:
                    var = (
                        nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) * 0.01), 
                        nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
                    )
                    kernel_types.append(pi)
                    
                self.register_parameter(pi + str(len(self.params)) + '_0',  var[0])
                self.register_parameter(pi + str(len(self.params)) + '_1',  var[1])
                
                self.params.append(var)
                
            elif pi.isdigit():
                val = int(pi)
                if val > 1:
                    pi = parts[i-1]
                    for j in range(val-1):
                        if pi.endswith('^'):
                            var = (
                                nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) * 5 + 10), 
                                nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
                            )
                            kernel_types.append(pi[:-1])
                        else:
                            var = (
                                nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) * 0.01), 
                                nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
                            )
                            kernel_types.append(pi)
                        
                        self.register_parameter(pi + str(len(self.params)) + '_0',  var[0])
                        self.register_parameter(pi + str(len(self.params)) + '_1',  var[1])
                        
                        self.params.append(var)

            else:
                raise KeyError(f"No matching kernel {pi} ...")
                
        self.kernel_num = len(kernel_types)
        infoLogger(kernel_types, self.params)
            
        def decay_constructor(t):
            kernels = []
            for i in range(self.kernel_num):
                pi = kernel_types[i]
                if pi == 'log':
                    kernels.append(torch.mul(self.params[i][0] , torch.log1p(t)) + self.params[i][1])
                elif pi == 'exp':
                    kernels.append(1000 * torch.exp(torch.mul(self.params[i][0], torch.neg(t))) + self.params[i][1])
                elif pi == 'exp*':
                    kernels.append(torch.mul(self.params[i][0], torch.exp(torch.neg(t))) + self.params[i][1])
                elif pi == 'lin':
                    kernels.append(self.params[i][0] * t  + self.params[i][1])
                elif pi == 'ind':
                    kernels.append(t)
                elif pi == 'const':
                    kernels.append(torch.ones(t.size(), device=self.device))
                elif pi == 'thres':
                    kernels.append(torch.reciprocal(1 + torch.exp(-self.params[i][0] * t + self.params[i][1])))
                    
            return torch.stack(kernels, dim=2)
                
        self.decay = decay_constructor   
            
        self.gru = nn.GRU(cfg.hidden_size, 10, num_layers=1, dropout=cfg.hidden_dropout_rate, batch_first=True, bidirectional=True)
        self.gru2context = nn.Linear(20, self.kernel_num)
        
        self.hidden_size = cfg.hidden_size

    def forward(
        self, seqs: torch.Tensor, t: torch.Tensor
    ):
        x_embed = self.Item.look_up(seqs) # (B, S, D)
        src_mask = (seqs == 0)
        src_mask_neg = (seqs != 0)

        # XXX: The official implementation replaces it with the next-item's timestamp,
        # but I think this operation leads to future information leakage.
        last_t = t[:, [-1]] # (B, 1)
        t = last_t - t 
        
        x = x_embed.transpose(0, 1) # (S, B, D)
      
        if self.pe != None:
            x = self.pe(x)

        # alpha stage
        x = self.encoder(x, src_key_padding_mask=src_mask)
        trg = self.Item.look_up(seqs[:, -1]).unsqueeze(0) # last input, (1, B, D)
        _, weight = self.decoder(trg, x, x, src_mask, softmax=True)
        alpha = (weight.squeeze(1) * src_mask_neg.float()).unsqueeze(2) # (B, S, 1)


        # beta stage
        t_decay = self.decay(t) # (B, S, K)
        beta = alpha * t_decay # (B, S, K)

        # gamma stage
        output, hidden = self.gru((x_embed))
        context = self.gru2context(output)
        context =  F.softmax(context, dim=-1)
        gamma = beta * context # (B, S, K)

        gamma = F.softmax(gamma.masked_fill(src_mask.unsqueeze(2), float('-inf')) , dim=1)
        gamma = torch.sum(gamma , dim=-1, keepdim=True)  # (B, S, 1)

        x_seq = torch.mul(gamma, x_embed) # (B, S, D)
        features = torch.sum(x_seq , dim=1) # (B, D)
        
        return features

    def predict(
        self, 
        seqs: torch.Tensor,
        times: torch.Tensor,
    ):
        features = self.forward(seqs, times) # (B, D)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t()) # (B, N)

    def recommend_from_pool(self, seqs: torch.Tensor, times: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs, times).unsqueeze(1)  # (B, 1, D)
        others = self.Item.look_up(pool) # (B, K, D)
        return features.mul(others).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor, times: torch.Tensor):
        features = self.forward(seqs, times)  # (B, D)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t()) # (B, N)


class CoachForCTA(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, times, positives, _ = [col.to(self.device) for col in data]
            logits = self.model.predict(seqs, times)
            logits = logits[:, positives.flatten()] # (B, B)
            loss = self.criterion(logits, positives)

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


class TOP1Loss(freerec.criterions.BaseCriterion):

    def forward(self, logit, target):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        # final loss
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


def to_roll_seqs(dataset, master: Tuple = (USER, ID)) -> List:
    r"""
    Rolling dataset in sequence.

    Parameters:
    -----------
    master: Tuple
        Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)
    minlen: int
        Shorest sequence
    
    Returns:
    --------
    List
    """
    seqs = dataset.train().to_seqs(master=master, keepid=True)

    roll_seqs = []
    for id_, items in seqs:
        if len(items) < cfg.maxlen:
            roll_seqs.append(
                (id_, items)
            )
            continue
        for k in range(cfg.maxlen, len(items) + 1):
            roll_seqs.append(
                (id_, items[:k])
            )
    return roll_seqs


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]
    Time = dataset.fields[TIMESTAMP]

    # trainpipe
    freerec.data.datasets.base
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=to_roll_seqs(dataset)
    ).sharding_filter().time_seq_train_uniform_sampling_(
        dataset, leave_one_out=False # yielding (user, seqs, times, targets, negatives)
    ).lprune_(
        indices=[1, 2, 3, 4], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).lpad_(
        indices=[1, 2, 3, 4], maxlen=cfg.maxlen, padding_value=0
    ).batch(cfg.batch_size).column_().tensor_()

    # validpipe
    if cfg.ranking == 'full':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().time_seq_valid_yielding_(
            dataset # yielding (user, seqs, times, unseens, seens)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Time.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().time_seq_valid_sampling_(
            dataset # yielding (user, seqs, times, (target + (100) negatives))
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()

    # testpipe
    if cfg.ranking == 'full':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().time_seq_test_yielding_(
            dataset # yielding (user, seqs, times, unseens, seens)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Time.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().time_seq_test_sampling_(
            dataset # yielding (user, seqs, times, (target + (100) negatives))
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen,
        ).rshift_(
            indices=[1, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()

    Item.embed(
        cfg.hidden_size, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = SATT(
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
    elif cfg.optimzier == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

    criterion = TOP1Loss()

    coach = CoachForCTA(
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