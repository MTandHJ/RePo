

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=100)
cfg.add_argument("--hidden-size", type=int, default=100)
cfg.add_argument("--num-layers", type=int, default=1)
cfg.add_argument('--lr-dc', type=float, default=0.1, help='learning rate decay rate')
cfg.add_argument('--lr-dc-step', type=int, default=3, help='the number of steps after which the learning rate decay')

cfg.set_defaults(
    description="SR-GNN",
    root="../../data",
    dataset='Diginetica_2507_Chron',
    epochs=30,
    batch_size=100,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-5,
    eval_freq=1,
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
        import math
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def getA(self, seqs: torch.Tensor):
        items, A, alias_indices = [], [], []
        N = seqs.size(1)
        for session in seqs.tolist():
            node = np.unique(session)
            items.append(node.tolist() + (N - len(node)) * [0])
            sess_A = np.zeros((N, N), dtype=np.float32)
            for i in np.arange(len(session) - 1):
                if session[i + 1] == 0:
                    break
                u = np.where(node == session[i])[0][0]
                v = np.where(node == session[i + 1])[0][0]
                sess_A[u][v] = 1
            u_sum_in = np.sum(sess_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            sess_A_in = np.divide(sess_A, u_sum_in)
            u_sum_out = np.sum(sess_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            sess_A_out = np.divide(sess_A.transpose(), u_sum_out)
            sess_A = np.concatenate([sess_A_in, sess_A_out]).transpose()
            A.append(sess_A)
            alias_indices.append([np.where(node == v)[0][0] for v in session])
        alias_indices = torch.LongTensor(alias_indices).to(self.device) # (B, S)
        A = torch.from_numpy(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)
        return alias_indices, A, items

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        alias_indices, A, unique_seqs = self.getA(seqs)
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

    def predict(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t())

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs).unsqueeze(1) # (B, 1, D)
        items = self.Item.look_up(pool) # (B, K, D)
        return features.mul(items).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t())


class CoachForSRGNN(freerec.launcher.SessCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            sesses, seqs, targets = [col.to(self.device) for col in data]
            scores = self.model.predict(seqs)
            loss = self.criterion(scores, targets.flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=sesses.size(0), mode="mean", prefix='train', pool=['LOSS'])
        self.lr_scheduler.step()


def main():

    dataset = getattr(freerec.data.datasets.session, cfg.dataset)(root=cfg.root)
    Session, Item = dataset.fields[SESSION, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_train_yielding_(
        dataset, leave_one_out=True # yielding (sess, seqs, target)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(cfg.batch_size).column_().rpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_()

    validpipe = freerec.data.dataloader.load_sess_rpad_validpipe(
        dataset, 
        NUM_PADS=NUM_PADS, padding_value=0, 
        batch_size=256, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_sess_rpad_testpipe(
        dataset, 
        NUM_PADS=NUM_PADS, padding_value=0, 
        batch_size=256, ranking=cfg.ranking
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
    criterion = freerec.criterions.CrossEntropy4Logits()
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
            'hitrate@10', 'hitrate@20', 
            'precision@10', 'precision@20', 
            'mrr@10', 'mrr@20'
        ],
        which4best='mrr@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()