

from typing import Optional, Union

import torch
import torch.nn as nn

from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv

import freerec
from freerec.data.postprocessing import RandomShuffledSource, OrderedSource
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.fields import FieldModuleList
from freerec.data.tags import SESSION, ITEM, ID, POSITIVE, UNSEEN, SEEN
from utils import get_session_item_graph, get_session_graph

freerec.declare(version='0.4.3')

cfg = Parser()
cfg.add_argument("--embedding-dim", type=int, default=100)
cfg.add_argument('--num_layers', type=float, default=3, help='the number of layer used')
cfg.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')

cfg.set_defaults(
    description="DHCN",
    root="../../data",
    dataset='Diginetica_250811_Chron',
    epochs=30,
    batch_size=100,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1.e-8,
    eval_freq=1,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class HyperConv(nn.Module):

    def __init__(self, adj, num_layers: int = 2) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.adj = adj
        self.conv = LGConv(normalize=False)

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.adj = self.adj.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(self, features: torch.Tensor):
        avgFeats = features.div(self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.adj)
            avgFeats += features.div(self.num_layers + 1)
        return avgFeats


class LinConv(nn.Module):

    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.conv = LGConv(normalize=False)

    def forward(self, adj: torch.Tensor, seqEmbs: torch.Tensor, seqLens: torch.Tensor):
        # seqEmbs: (B, S, D)
        # detaching LinConv according to the official code
        features = torch.sum(seqEmbs, 1).div(seqLens).detach()
        avgFeats = features.div(self.num_layers + 1)
        for _ in range(self.num_layers):
            features = adj.matmul(features)
            avgFeats += features.div(self.num_layers + 1)
        return avgFeats


class DHCN(RecSysArch):

    def __init__(
        self, 
        graph: Data,
        fields: FieldModuleList,
        embedding_dim: int = cfg.embedding_dim,
        maxlen: int = 200,
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]
        self.pos_embedding = nn.Embedding(maxlen, embedding_dim)
        self.HyperGraph = HyperConv(graph, cfg.num_layers)
        self.LineGraph = LinConv(cfg.num_layers)

        self.embedding_dim = embedding_dim

        self.w_1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.w_2 = nn.Parameter(torch.Tensor(embedding_dim, 1))
        self.glu1 = nn.Linear(embedding_dim, embedding_dim)
        self.glu2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        self.HyperGraph.to(device)
        return super().to(device, dtype, non_blocking)

    def reset_parameters(self):
        """Initializes the module parameters."""
        import math
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def criterion(self, scores: torch.Tensor, targets: torch.Tensor):
        return self.loss_function(scores, targets.flatten())

    def calc_sess_emb(
        self, 
        itemEmbsI: torch.Tensor,
        seqs: torch.Tensor,
        seqLens: torch.Tensor,
        masks: torch.Tensor
    ):
        seqh = itemEmbsI[seqs] # (B, S, D)
        positions = self.pos_embedding.weight[:seqs.size(-1)].unsqueeze(0).expand_as(seqh) # (B, S, D)

        hs = seqh.sum(1).div(seqLens).unsqueeze(1) # (B, 1, D)
        nh = self.w_1(torch.cat([positions, seqh], -1)).tanh()
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        alpha = torch.matmul(nh, self.w_2) # (B, S, 1)
        alpha = alpha * masks.unsqueeze(-1)
        sessEmbsI = torch.sum(alpha * seqh, 1) # (B, D)
        return sessEmbsI

    def topk_func_random(
        self, 
        scores: torch.Tensor,
        itemEmbs: torch.Tensor
    ):
        _, indices = scores.topk(self.num, dim=1, largest=True, sorted=True)

        positives = itemEmbs[indices[:, :self.K]] # (B, K, D)
        random_slices = torch.randint(self.K, self.num, (1,self.K), device=self.device).expand((indices.size(0), -1)) # (B, K)
        random_slices = indices.gather(1, random_slices)
        negatives = itemEmbs[random_slices] # (B, K, D)
        return positives, negatives

    def _shuffle(self, features: torch.Tensor):
        B, D = features.shape
        shuffled = features[torch.randperm(B)]
        shuffled = features[:, torch.randperm(D)]
        return shuffled

    def SSL(self, sessEmbsH: torch.Tensor, sessEmbsL: torch.Tensor):
        posScores = sessEmbsL.mul(sessEmbsH).sum(-1)
        negScores = sessEmbsL.mul(
            self._shuffle(sessEmbsH)
        ).sum(-1)
        return torch.sum(
            posScores.sigmoid().add(1e-8).log().neg() + 
            (1 - negScores.sigmoid()).add(1e-8).log().neg()
        )

    def forward(self, seqs: torch.Tensor, targets: torch.Tensor):
        r"""
        Parameters:
        -----------
        seqs: torch.Tensor, (B, S)
            Each row is [0, 0, ..., s1, s2, ..., sm]
        targets: torch.Tensor, (B, 1)
        """
        masks = seqs.not_equal(0)
        seqLens = masks.sum(-1, keepdim=True)

        # HyperGraph
        itemEmbsH = self.HyperGraph(
            self.Item.embeddings.weight[NUM_PADS:]
        )
        sessEmbsH = self.calc_sess_emb(
            torch.cat([
                torch.zeros(NUM_PADS, itemEmbsH.size(-1), device=self.device),
                itemEmbsH
            ], dim=0),
            seqs, seqLens, masks
        )

        # LineGraph
        A_s_hat = get_session_graph(seqs, n=self.Item.count)
        D_s_hat = A_s_hat.sum(-1, keepdim=True)
        sessEmbsS = self.LineGraph(
            D_s_hat.mul(A_s_hat),
            self.Item.look_up(seqs) * masks.unsqueeze(-1), # (B, S, D)
            seqLens,
        )

        # Contrastive Learning
        loss_ssl = self.SSL(sessEmbsH, sessEmbsS)

        # Main loss
        scores = sessEmbsH.matmul(itemEmbsH.t())
        loss_item = self.criterion(scores + 1e-8, targets)
        return loss_item + loss_ssl * cfg.beta

    def recommend(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        seqLens = masks.sum(-1, keepdim=True)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        itemEmbsH = self.HyperGraph(items)
        sessEmbsH = self.calc_sess_emb(
            torch.cat([
                torch.zeros(NUM_PADS, itemEmbsH.size(-1), device=self.device),
                itemEmbsH
            ], dim=0),
            seqs, seqLens, masks
        )
        scores = torch.mm(sessEmbsH, itemEmbsH.t())
        return scores


class CoachForDHCN(Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            sesses, seqs, targets = [col.to(self.device) for col in data]
            loss = self.model(seqs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=sesses.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        for sesses, seqs, unseen, seen in self.dataloader:
            sesses = sesses.data
            seqs = seqs.to(self.device).data
            scores = self.model.recommend(seqs)
            # Don't remove seens for session
            targets = unseen.to_csr().to(self.device).to_dense()

            self.monitor(
                scores, targets,
                n=len(sesses), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'MRR']
            )


def main():

    dataset = getattr(freerec.data.datasets.session, cfg.dataset)(root=cfg.root)
    Session, Item = dataset.fields[SESSION, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomShuffledSource(
        dataset.train().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_train_yielding_(
        None # yielding (sesses, seqs, targets)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(cfg.batch_size).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_()

    # validpipe
    # Shuffling for the following reason:
    # https://github.com/xiaxin1998/DHCN/issues/2
    validpipe = RandomShuffledSource(
        dataset.valid().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_valid_yielding_(
        dataset # yielding (sesses, seqs, targets, seen)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(512).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_().field_(
        Session.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    # testpipe
    testpipe = RandomShuffledSource(
        dataset.test().to_roll_seqs(minlen=2)
    ).sharding_filter().sess_test_yielding_(
        dataset # yielding (sesses, seqs, targets, seen)
    ).rshift_(
        indices=[1], offset=NUM_PADS
    ).batch(512).column_().lpad_col_(
        indices=[1], maxlen=None, padding_value=0
    ).tensor_().field_(
        Session.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    Item.embed(
        cfg.embedding_dim, padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = DHCN(
        get_session_item_graph(dataset),
        tokenizer,
        maxlen=max(list(
            map(lambda seq: len(seq), dataset.to_seqs(keepid=False))
        ))
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

    coach = CoachForDHCN(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(
        cfg, monitors=['loss', 'hitrate@10', 'hitrate@20', 'precision@10', 'precision@20', 'mrr@10', 'mrr@20'],
        which4best='mrr@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()
