

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_sparse import mul, fill_diag
from torch_sparse import sum as sparsesum
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv

import freerec
from freerec.data.postprocessing import RandomShuffledSource, OrderedSource
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import CrossEntropy4Logits
from freerec.data.fields import FieldModuleList
from freerec.data.tags import SESSION, ITEM, ID, POSITIVE, UNSEEN, SEEN
from utils import get_item_graph, get_session_graph

freerec.decalre(version="0.4.3")

cfg = Parser()
cfg.add_argument("--embedding-dim", type=int, default=100)
cfg.add_argument('--num_layers', type=float, default=2, help='the number of layer used')
cfg.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
cfg.add_argument('--lam', type=float, default=0.005, help='diff task maginitude')
cfg.add_argument('--eps', type=float, default=0.2, help='eps')
cfg.add_argument('--K', type=int, default=10, help='[co-training] top-K for positives')
cfg.add_argument('--neg-pool-size', type=int, default=5000, help='[co-training] pooling size for negatives')
cfg.add_argument('--scale4sessEmbsI', type=float, default=10., help='scale factor for sessEmbsI')

cfg.set_defaults(
    description="COTREC",
    root="../../../data",
    dataset='Diginetica_250811_Chron',
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


class ItemConv(nn.Module):

    def __init__(
        self,
        graph,
        num_layers: int = 2,
        embedding_dim: int = 100
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.graph = graph
        self.conv = LGConv(normalize=False)
        self.weights = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(self.num_layers)
        ])

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor(attr='edge_weight')(self.__graph)
        adj_t = self.__graph.adj_t
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=torch.float32)
        adj_t = fill_diag(adj_t, 1.)
        deg = sparsesum(adj_t, dim=1) # column sum
        deg_inv = deg.pow(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        self.__graph.adj_t = mul(adj_t, deg_inv.view(-1, 1)) # TODO: ???

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(self, features: torch.Tensor):
        avgFeats = features.div(self.num_layers + 1)
        for i in range(self.num_layers):
            features = self.weights[i](features)
            features = self.conv(features, self.graph.adj_t)
            avgFeats += F.normalize(
                features
            ).div(self.num_layers + 1)
        return avgFeats


class SessConv(nn.Module):

    def __init__(
        self,
        num_layers: int = 2,
        embedding_dim: int = 100
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.conv = LGConv(normalize=False)
        self.weights = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, adj: torch.Tensor, seqEmbs: torch.Tensor, seqLens: torch.Tensor):
        # seqEmbs: (B, S, D)
        features = torch.sum(seqEmbs, 1).div(seqLens)
        avgFeats = features.div(self.num_layers + 1)
        for i in range(self.num_layers):
            features = self.weights[i](features)
            features = adj.matmul(features)
            avgFeats += F.normalize(
                features
            ).div(self.num_layers + 1)
        return avgFeats


class COTREC(RecSysArch):

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
        self.ItemGraph = ItemConv(graph, cfg.num_layers, embedding_dim)
        self.SessGraph = SessConv(cfg.num_layers, embedding_dim)

        self.K = cfg.K
        self.w_k = cfg.scale4sessEmbsI
        self.num = cfg.neg_pool_size
        self.eps = cfg.eps
        self.embedding_dim = embedding_dim

        self.w_1 = nn.Parameter(torch.Tensor(2 * embedding_dim, embedding_dim))
        self.w_2 = nn.Parameter(torch.Tensor(embedding_dim, 1))
        self.w_i = nn.Linear(embedding_dim, embedding_dim)
        self.w_s = nn.Linear(embedding_dim, embedding_dim)
        self.glu1 = nn.Linear(embedding_dim, embedding_dim)
        self.glu2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.register_buffer(
            "adv_item",
            torch.zeros(self.Item.count, embedding_dim)
        )

        self.register_buffer(
            "adv_sess",
            torch.zeros(self.Item.count, embedding_dim)
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.initialize()

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        self.ItemGraph.to(device)
        return super().to(device, dtype, non_blocking)

    def initialize(self):
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
        nh = torch.matmul(torch.cat([positions, seqh], -1), self.w_1).tanh()
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

    def SSL_topk(
        self, 
        anchor: torch.Tensor, sessEmbs: torch.Tensor, 
        positives: torch.Tensor, negatives: torch.Tensor
    ):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        anchor = F.normalize(anchor + sessEmbs, p=2, dim=-1).unsqueeze(1) # (B, D)
        positives = positives + sessEmbs.unsqueeze(1) # (B, K, D)
        negatives = negatives + sessEmbs.unsqueeze(1) # (B, K, D)
        pos_score = score(anchor.unsqueeze(1), F.normalize(positives, p=2, dim=-1))
        neg_score = score(anchor.unsqueeze(1), F.normalize(negatives, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / 0.2), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss

    def craft_perturbations(
        self, 
        itemEmbs: torch.Tensor, sessEmbs: torch.Tensor, 
        targets: torch.Tensor, perturbations: torch.Tensor
    ):
        perturbations.requires_grad_(True)
        adv_item_emb = itemEmbs + perturbations
        score = torch.mm(sessEmbs, adv_item_emb.t())
        loss = self.criterion(score, targets)
        grad = torch.autograd.grad(loss, perturbations, retain_graph=True)[0]
        adv = grad.detach()
        perturbations.requires_grad_(False)
        return F.normalize(adv, p=2, dim=1) * self.eps

    def diff(self, score_item, score_sess, score_adv2, score_adv1, diff_mask):
        # compute KL(score_item, score_adv2), KL(score_sess, score_adv1)
        score_item = F.softmax(score_item, dim=1)
        score_sess = F.softmax(score_sess, dim=1)
        score_adv2 = F.softmax(score_adv2, dim=1)
        score_adv1 = F.softmax(score_adv1, dim=1)
        score_item = torch.mul(score_item, diff_mask)
        score_sess = torch.mul(score_sess, diff_mask)
        score_adv1 = torch.mul(score_adv1, diff_mask)
        score_adv2 = torch.mul(score_adv2, diff_mask)

        h1 = torch.sum(torch.mul(score_item, torch.log(1e-8 + ((score_item + 1e-8)/(score_adv2 + 1e-8)))))
        h2 = torch.sum(torch.mul(score_sess, torch.log(1e-8 + ((score_sess + 1e-8)/(score_adv1 + 1e-8)))))

        return h1 + h2

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

        # Item-view
        itemEmbsI = self.ItemGraph(
            self.Item.embeddings.weight[NUM_PADS:]
        )
        sessEmbsI = self.calc_sess_emb(
            torch.cat([
                torch.zeros(NUM_PADS, itemEmbsI.size(-1), device=self.device),
                itemEmbsI
            ], dim=0),
            seqs, seqLens, masks
        )

        sessEmbsI = self.w_k * F.normalize(sessEmbsI, dim=-1, p=2) # (B, D)
        itemEmbsI = F.normalize(itemEmbsI, dim=-1, p=2) # (N, D)
        scoresI = torch.mm(sessEmbsI, itemEmbsI.t()) # (B, N)
        loss_item = self.criterion(scoresI, targets)

        # Session-view
        A_s_hat = get_session_graph(seqs, n=self.Item.count)
        D_s_hat = A_s_hat.sum(-1, keepdim=True)
        sessEmbsS = self.SessGraph(
            D_s_hat.mul(A_s_hat),
            self.Item.look_up(seqs) * masks.unsqueeze(-1), # (B, S, D)
            seqLens,
        )
        scoresS = torch.mm(sessEmbsS, itemEmbsI.t())

        # Co-training
        scoresI4S = sessEmbsI.matmul(itemEmbsI.t()).softmax(-1) # (B, N)
        scoresS4I = sessEmbsS.matmul(self.Item.embeddings.weight[NUM_PADS:].t()).softmax(-1) # (B, N)
        positivesI, negativesI = self.topk_func_random(scoresS4I, itemEmbsI)
        positivesS, negativesS = self.topk_func_random(scoresI4S, self.Item.embeddings.weight[NUM_PADS:])

        # Contrastive learning
        loss_ssl = self.SSL_topk(
            itemEmbsI[seqs[:, -1] - NUM_PADS],
            sessEmbsI, positivesI, negativesI
        )
        loss_ssl += self.SSL_topk(
            self.Item.look_up(seqs[:, -1]),
            sessEmbsS, positivesS, negativesS
        )

        # FGSM for adversarial training
        self.adv_item.data.copy_(self.craft_perturbations(itemEmbsI, sessEmbsI, targets, self.adv_item))
        self.adv_sess.data.copy_(self.craft_perturbations(itemEmbsI, sessEmbsI, targets, self.adv_sess))

        adv_emb_item = itemEmbsI + self.adv_item.detach()
        adv_emb_sess = itemEmbsI + self.adv_sess.detach()

        score_adv1 = torch.mm(sessEmbsS, adv_emb_item.t())
        score_adv2 = torch.mm(sessEmbsI, adv_emb_sess.t())
        # add difference constraint
        diff_mask = torch.ones((seqs.size(0), self.Item.count), device=self.device) / self.Item.count
        diff_mask.scatter_(
            1, targets, 1.
        )
        loss_diff = self.diff(scoresI, scoresS, score_adv2, score_adv1, diff_mask)

        return loss_item + loss_ssl * cfg.beta + loss_diff * cfg.lam

    def recommend(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0)
        seqLens = masks.sum(-1, keepdim=True)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        itemEmbsI = self.ItemGraph(items)
        sessEmbsI = self.calc_sess_emb(
            torch.cat([
                torch.zeros(NUM_PADS, itemEmbsI.size(-1), device=self.device),
                itemEmbsI
            ], dim=0),
            seqs, seqLens, masks
        )
        sessEmbsI = F.normalize(sessEmbsI, dim=-1, p=2)
        itemEmbsI = F.normalize(itemEmbsI, dim=-1, p=2)
        scores = torch.mm(sessEmbsI, itemEmbsI.t())
        return scores


class CoachForCOTREC(Coach):

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
    validpipe = OrderedSource(
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
    testpipe = OrderedSource(
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
    model = COTREC(
        get_item_graph(dataset),
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

    criterion = CrossEntropy4Logits()

    coach = CoachForCOTREC(
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
        cfg, monitors=['loss', 'hitrate@10', 'hitrate@20', 'precision@10', 'precision@20', 'mrr@10', 'mrr@20'],
        which4best='mrr@20'
    )
    coach.fit()


if __name__ == "__main__":
    main()

