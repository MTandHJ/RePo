

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data
import torch_geometric.transforms as T

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=200)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=1)
cfg.add_argument('--lr-dc', type=float, default=0.1, help='learning rate decay rate')
cfg.add_argument('--lr-dc-step', type=int, default=3, help='the number of steps after which the learning rate decay')

cfg.set_defaults(
    description="SR-GNN",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=100,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-5,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class GNN(nn.Module):

    def __init__(self, hidden_size, num_layers: int = 1):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A: torch.Tensor, hidden: torch.Tensor):
        r"""
        Parameters:
        -----------
        A: torch.Tensor, (B, S, 2S)
            A = [A_in, A_out] includes two adjacency matrices, 
            which represents weighted connectioncs of incoming and outgoing edges in the session graph, respectively.
            `S' is the max length of the current batch of sessions.
        hidden: torch.Tensor, (B, S, D)
            The node features.
        """
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2) # (B, S, 2D)
        gi = F.linear(inputs, weight=self.w_ih, bias=self.b_ih) # (B, S, 3D)
        gh = F.linear(hidden, weight=self.w_hh, bias=self.b_hh) # (B, S, 3D)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy # (B, S, D)

    def forward(self, A: torch.Tensor, hidden: torch.Tensor):
        for _ in range(self.num_layers):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        hidden_size: int = cfg.embedding_dim
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]
        self.hidden_size = hidden_size

        self.gnn = GNN(hidden_size, num_layers=cfg.num_layers)
        self.linear_one = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_two = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_three = nn.Linear(hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(hidden_size * 2, hidden_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def getA(self, seqs: torch.Tensor, masks: torch.Tensor):
        def extract_from_seq(i: int, seq: np.ndarray, mask: np.ndarray):
            N = len(seq)
            seq_ = seq[mask]
            items = np.unique(seq_)
            mapper = {item:node for node, item in enumerate(items)}
            mapper[0] = N - 1
            seq_ = [mapper[item] for item in seq_]

            x = torch.empty((N, 0))
            
            graph = Data(
                x=x,
                edge_index=torch.LongTensor([
                    seq_[:-1],
                    seq_[1:]
                ])
            )

            T.ToSparseTensor()(graph)
            A  = graph.adj_t.t()

            A = A.to_dense()
            nodes = np.zeros_like(seq)
            nodes[:len(items)] = items
            alias_indices = np.array([mapper[item] for item in seq])
            return A, nodes, alias_indices
        A, nodes, alias_indices = zip(*map(extract_from_seq, range(len(seqs)), seqs.cpu().numpy(), masks.cpu().numpy()))

        A = torch.stack(A, dim=0).to(self.device) # (B, S, S)
        nodes = torch.from_numpy(np.array(nodes)).to(self.device)
        alias_indices = torch.from_numpy(np.array(alias_indices)).to(self.device)

        row = A.sum(dim=1, keepdim=True).clamp_min(1)
        col = A.sum(dim=2, keepdim=True).clamp_min(1)

        A_in = A / row
        A_out = A / col

        return torch.cat([A_in, A_out], dim=-1), nodes, alias_indices

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        A, unique_seqs, alias_indices  = self.getA(seqs, masks)
        unique_seqs = self.Item.look_up(unique_seqs) # (B, S', D)
        hidden: torch.Tensor = self.gnn(A, unique_seqs) # (B, S', D)
        hidden = hidden.gather(
            dim=1, 
            index=alias_indices.unsqueeze(-1).expand((-1, -1, hidden.size(-1)))
        ) # (B, S, D)

        last = hidden.gather(
            dim=1,
            index=masks.sum(-1).add(-1).view((-1, 1, 1)).expand((-1, 1, hidden.size(-1)))
        ).squeeze(1) # (B, D)
        q1 = self.linear_one(last).unsqueeze(1)  # (B, 1, D)
        q2 = self.linear_two(hidden)  # (B, S, D)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * masks.unsqueeze(-1).float(), 1)
        a = self.linear_transform(torch.cat([a, last], 1))
        return a

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


class CoachForSRGNN(freerec.launcher.SeqCoach):

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
        cfg.embedding_dim, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = SRGNN(
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_dc_step, gamma=cfg.lr_dc)

    coach = CoachForSRGNN(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
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