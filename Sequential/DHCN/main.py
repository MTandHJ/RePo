

from typing import Optional, Union

import torch
import torch.nn as nn

from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, POSITIVE, UNSEEN, SEEN

from utils import get_session_item_graph, get_session_graph

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument('--num_layers', type=int, default=3, help='the number of layer used')
cfg.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')

cfg.set_defaults(
    description="DHCN",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=100,
    batch_size=100,
    optimizer='adam',
    lr=1e-4,
    weight_decay=1.e-8,
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
        avgFeats = features.div(self.num_layers + 1)#.detach()
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
        # detaching LinConv according to the official code ?
        features = torch.sum(seqEmbs, 1).div(seqLens)#.detach()
        avgFeats = features.div(self.num_layers + 1)
        for _ in range(self.num_layers):
            features = adj.matmul(features)
            avgFeats += features.div(self.num_layers + 1)
        return avgFeats


class DHCN(freerec.models.RecSysArch):

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

        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = freerec.criterions.BPRLoss()

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

    def criterion(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ):
        return self.loss_function(positives, negatives)

    def calc_sess_emb(
        self, 
        itemEmbsI: torch.Tensor,
        seqs: torch.Tensor,
        seqLens: torch.Tensor,
        masks: torch.Tensor
    ):
        seqh = itemEmbsI[seqs] # (B, S, D)
        positions = self.pos_embedding.weight[-seqs.size(-1):].unsqueeze(0).expand_as(seqh) # (B, S, D)

        hs = seqh.sum(1).div(seqLens).unsqueeze(1) # (B, 1, D)
        nh = self.w_1(torch.cat([positions, seqh], -1)).tanh()
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        alpha = torch.matmul(nh, self.w_2) # (B, S, 1)
        alpha = alpha * masks.unsqueeze(-1)
        sessEmbsI = torch.sum(alpha * seqh, 1) # (B, D)
        return sessEmbsI

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

    def predict(
        self, seqs: torch.Tensor, 
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ):
        r"""
        Parameters:
        -----------
        seqs: torch.Tensor, (B, S)
            Each row is [0, 0, ..., s1, s2, ..., sm]
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
        ) # (B, D)

        # Contrastive Learning
        loss_ssl = self.SSL(sessEmbsH, sessEmbsS)

        # Main loss
        positives = itemEmbsH[positives].squeeze(1) # (B, D)
        negatives = itemEmbsH[negatives].squeeze(1) # (B, D)
        scores_pos = sessEmbsS.mul(positives).sum(-1)
        scores_neg = sessEmbsS.mul(negatives).sum(-1)
        loss_item = self.criterion(scores_pos, scores_neg)
        return loss_item + loss_ssl * cfg.beta

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
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
        sessEmbsH = sessEmbsH.unsqueeze(1) # (B, 1, D)
        items = itemEmbsH[pool - NUM_PADS] # (B, K, D)
        return sessEmbsH.mul(items).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
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


class CoachForDHCN(freerec.launcher.SessCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            sesses, seqs, positives, negatives = [col.to(self.device) for col in data]
            loss = self.model.predict(seqs, positives, negatives)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=sesses.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    Session, Item = dataset.fields[SESSION, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_roll_seqs(minlen=2)
    ).sharding_filter().seq_train_uniform_sampling_(
        dataset, leave_one_out=True # yielding (user, seqs, positives, negatives)
    ).lprune_(
        indices=[1], maxlen=cfg.maxlen
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
        cfg.embedding_dim, padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = DHCN(
        get_session_item_graph(dataset),
        tokenizer,
        maxlen=max(
            dataset.train().maxlen,
            dataset.valid().maxlen,
            dataset.test().maxlen
        )
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