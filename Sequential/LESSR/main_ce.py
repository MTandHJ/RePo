

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, softmax, coalesce

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

freerec.declare(version="0.4.3")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--feat-drop", type=float, default=0.2, help="the dropout rate for features")

cfg.set_defaults(
    description="LESSR",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=30,
    batch_size=512,
    optimizer='adamw',
    lr=1e-3,
    weight_decay=1.e-4,
    eval_freq=1,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class EOPA(MessagePassing):
    r"""Edge-Order Preserving Aggregation."""

    def __init__(
        self, input_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool = True, 
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        x = self.fc_self(x) + self.propagate(edge_index, x=x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def aggregate(
        self, x_j: torch.Tensor, edge_index_i: torch.Tensor, size_i: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        x_j: torch.Tensor
            source node features
        edge_index_i: torch.Tensor
            target node index
        size_i: torch.Tensor
            the number of target nodes
        
        Flows:
        ------
        1. `to_dense_batch` collects neighbors for each session
        2. `gru` aggregates neighbors for each session

        Notes:
        ------
        During the aggregation of `gru`, zero-padding is also involved.
        However, the official code seems ignore this issue, and thus I implement this in a similar way.
        """
        x = to_dense_batch(x_j, edge_index_i, batch_size=size_i)[0]
        _, hn = self.gru(x)
        return hn.squeeze(0)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_neigh(x)


class SGAT(MessagePassing):
    r"""
    Shortcut Graph Attention.

    SGAT removes repeated edges.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool = True, 
    ):
        super().__init__(aggr='add')

        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_k(x)

        alpha = self.edge_updater(edge_index, q=q, k=k)

        x = self.propagate(edge_index, x=v, alpha=alpha)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def edge_update(
        self, q_j: torch.Tensor, k_j: torch.Tensor,
        edge_index_i: torch.Tensor, size_i: int
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        edge_index_i: torch.Tensor
            target node index, i.e., edge_index[1]
        size_i: int 
            the number of target nodes
        """
        alpha =  self.fc_e((q_j + k_j).sigmoid())
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        return alpha

    def aggregate(
        self, x_j: torch.Tensor, alpha: torch.Tensor,
        edge_index_i: torch.Tensor, size_i: int
    ) -> torch.Tensor:
        return super().aggregate(x_j.mul(alpha), edge_index_i, dim_size=size_i)


class AttnReadout(MessagePassing):

    def __init__(
        self,
        input_dim: int, hidden_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool =True,
    ):
        super().__init__(aggr='add')
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(
        self, x: torch.Tensor, lasts: torch.Tensor, 
        edge_index: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        # edge_index: (BS, D)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        x_u = self.fc_u(x) # (*, D)
        x_v = self.fc_v(lasts) # (*, D)

        alpha = self.edge_updater(edge_index, q=x_u, k=x_v, groups=groups) # (BS, D)

        x = self.propagate(edge_index, x=x, alpha=alpha, groups=groups)
        if self.fc_out is not None:
            x  = self.fc_out(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def edge_update(
        self, q: torch.Tensor, k: torch.Tensor, 
        edge_index_i: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        groups: torch.Tensor
            ptr for grouping
        """
        alpha =  self.fc_e((q + k).sigmoid())
        alpha = softmax(alpha, ptr=groups)
        return alpha

    def message(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def aggregate(
        self, x: torch.Tensor, edge_index_i: torch.Tensor,
        alpha: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        return super().aggregate(x.mul(alpha), index=None, ptr=groups, dim_size=len(groups) - 1)


class LESSR(freerec.models.RecSysArch):

    def __init__(
        self, 
        fields: FieldModuleList,
        embedding_dim: int = cfg.embedding_dim,
        num_layers: int = cfg.num_layers,
        dropout_rate: float = cfg.feat_drop,
        batch_norm: bool = True
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        input_dim = embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    activation=nn.PReLU(embedding_dim),
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def get_multi_graphs(self, seqs: torch.Tensor, masks: torch.Tensor):
        def extract_from_seq(i: int, seq: np.ndarray, mask: np.ndarray):
            last = seq[-1]
            seq = seq[mask]
            items = np.unique(seq)
            mapper = {item:node for node, item in enumerate(items)}
            seq = [mapper[item] for item in seq]
            last = mapper[last]

            nums = len(items)
            x = torch.empty((nums, 0))
            
            # EOP
            graph_eop = Data(
                x=x,
                edge_index=torch.LongTensor(
                    [seq[:-1], seq[1:]]
                )
            )
            graph_eop.nodes = torch.from_numpy(items)

            # Shortcut
            graph_cut = Data(
                x=x,
                edge_index= coalesce(graph_eop.edge_index)
            )

            # Session
            graph_sess = Data(
                x=x,
                edge_index=torch.LongTensor([
                    [last] * nums, # for last items
                    [i] * nums,
                ])
            )

            return graph_eop, graph_cut, graph_sess
        graph_eop, graph_cut, graph_sess = zip(*map(extract_from_seq, range(len(seqs)), seqs.cpu().numpy(), masks.cpu().numpy()))
        # Batch graphs into a disconnected graph,
        # i.e., the edge_index will be re-ordered.
        graph_eop = Batch.from_data_list(graph_eop)
        graph_cut = Batch.from_data_list(graph_cut)
        graph_sess = Batch.from_data_list(graph_sess)
        return graph_eop.to(self.device), graph_cut.to(self.device), graph_sess.to(self.device)

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        graph_eop, graph_cut, graph_sess = self.get_multi_graphs(
            seqs, masks
        )

        features = self.Item.look_up(graph_eop.nodes) # (*, D)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(features, graph_eop.edge_index)
            else:
                out = layer(features, graph_cut.edge_index)
            features = torch.cat([out, features], dim=1)

        last_features = features[graph_sess.edge_index[0]]
        sr_g = self.readout(features, last_features, graph_sess.edge_index, graph_sess.ptr)
        sr_l = features[graph_sess.edge_index[0].unique(sorted=False)]
        sr = torch.cat([sr_l, sr_g], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))
        return sr

    def predict(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:]
        return features.matmul(items.t())

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs).unsqueeze(1) # (B, 1, D)
        items = self.Item.look_up(pool) # (B, K, D)
        return features.mul(items).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t())


class CoachForLESSR(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, targets = [col.to(self.device) for col in data]
            scores = self.model.predict(seqs)
            loss = self.criterion(scores, targets.flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_roll_seqs(minlen=2)
    ).sharding_filter().seq_train_yielding_(
        dataset, leave_one_out=True # yielding (users, seqs, target)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen,
    ).rshift_(
        indices=[1], offset=NUM_PADS
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
        cfg.embedding_dim, padding_idx=0, max_norm=1
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = LESSR(
        tokenizer
    )

    if cfg.weight_decay > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )

    criterion = freerec.criterions.CrossEntropy4Logits()

    coach = CoachForLESSR(
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