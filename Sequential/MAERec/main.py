

from typing import Union, Optional

import torch, os
import numpy as np
import scipy.sparse as sp

import freerec
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID
from freerec.utils import mkdirs

from modules import Encoder, Decoder, SASRec, RandomMaskSubgraphs, LocalGraph

freerec.declare(version='0.4.3')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--hidden-size", type=int, default=32, help="embedding size")
cfg.add_argument('--batch-size-con', default=2048, type=int, help='batch size for reconstruction task')
cfg.add_argument('--num-reco-neg', default=40, type=int, help='number of negative items for reconstruction task')
cfg.add_argument('--mask-depth', default=3, type=int, help='k steps for generating transitional path')
cfg.add_argument('--path-prob', default=0.5, type=float, help='random walk sample probability')
cfg.add_argument("--num-heads", type=int, default=4, help='number of heads in attention')
cfg.add_argument('--num-gcn-layers', default=2, type=int, help='number of gcn layers')
cfg.add_argument('--num-trm-layers', default=2, type=int, help='number of gcn layers')
cfg.add_argument('--num-mask-cand', default=50, type=int, help='number of seeds in patch masking')
cfg.add_argument('--mask-steps', default=10, type=int, help='steps to train on the same sampled graph')
cfg.add_argument('--eps', default=0.2, type=float, help='scaled weight for task-adaptive function')
cfg.add_argument('--weight4reco', default=0.2, type=float, help='weight for reconstruction loss')
cfg.add_argument("--attention-probs-dropout-prob", type=float, default=0.3, help="attention dropout p")
cfg.add_argument("--hidden-dropout-prob", type=float, default=0.3, help="hidden dropout p")
cfg.add_argument("--ii-dist", type=int, default=3, help="distance for i-i graph construction")

cfg.set_defaults(
    description="MAERec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=20,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-6,
    eval_freq=1,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class MAERec(freerec.models.RecSysArch):

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__()

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.recommender = SASRec(cfg)
        self.masker = RandomMaskSubgraphs(cfg)
        self.sampler = LocalGraph(cfg)

        self.trn =  self.construct_ii_graph(dataset) # Sparse Item-Item graph in coo format
        self.ii_dok = self.trn.todok() # Sparse Item-Item graph in dok format
        self.ii_adj: torch.Tensor = self.make_torch_adj(self.trn) # normalized adjacency matrix 
        self.ii_adj_all_one: torch.Tensor = self.make_all_one_adj(self.ii_adj) # unnormalized adjacency matrix 

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device is not None:
            self.ii_adj = self.ii_adj.to(device)
            self.ii_adj_all_one = self.ii_adj_all_one.to(device)
        return super().to(device, dtype, non_blocking)

    def construct_ii_graph(self, dataset: freerec.data.datasets.RecDataSet):
        from freeplot.utils import import_pickle, export_pickle
        Item = dataset.fields[ITEM, ID]
        path = os.path.join(cfg.dataset, str(cfg.ii_dist))
        try:
            indices = import_pickle(
                os.path.join(path, "row_col_indices.pickle")
            )
            row_indices, col_indices = indices['row'], indices['col']
        except ImportError:
            row_indices = []
            col_indices = []
            for seq in dataset.train().to_seqs():
                for h in range(1, cfg.ii_dist + 1):
                    row_indices.extend(seq[+h:])
                    row_indices.extend(seq[:-h])
                    col_indices.extend(seq[:-h])
                    col_indices.extend(seq[+h:])

            row_indices = [idx + NUM_PADS for idx in row_indices]
            col_indices = [idx + NUM_PADS for idx in col_indices]

            mkdirs(path)
            export_pickle(
                {'row': row_indices, 'col': col_indices},
                os.path.join(path, "row_col_indices.pickle")
            )
        values = np.ones_like(row_indices)
        trn = sp.csr_matrix((values, (row_indices, col_indices)), shape=(Item.count + NUM_PADS, Item.count + NUM_PADS)).astype(np.float32)
        return sp.coo_matrix(trn)

    def normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def make_torch_adj(self, mat):
        mat = (mat + sp.eye(mat.shape[0]))
        mat = (mat != 0) * 1.0
        mat = self.normalize(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)

    def make_all_one_adj(self, adj):
        idxs = adj._indices()
        vals = torch.ones_like(adj._values())
        shape = adj.shape
        return torch.sparse.FloatTensor(idxs, vals, shape)

    def sample(self):
        sample_scr, candidates = self.sampler(self.ii_adj_all_one, self.encoder.get_ego_embeds())
        masked_adj, masked_edg = self.masker(self.ii_adj, candidates)
        return sample_scr, masked_adj, masked_edg

    def sample_pos_edges(self, masked_edges: torch.Tensor):
        return masked_edges[torch.randperm(masked_edges.shape[0])[:cfg.batch_size_con]]

    def sample_neg_edges(self, pos: torch.Tensor):
        neg = []
        for u, v in pos:
            cu_neg = []
            num_samp = cfg.num_reco_neg // 2
            for i in range(num_samp):
                while True:
                    v_neg = np.random.randint(1, cfg.num_items)
                    if (u, v_neg) not in self.ii_dok:
                        break
                cu_neg.append([u, v_neg])
            for i in range(num_samp):
                while True:
                    u_neg = np.random.randint(1, cfg.num_items)
                    if (u_neg, v) not in self.ii_dok:
                        break
                cu_neg.append([u_neg, v])
            neg.append(cu_neg)
        return torch.Tensor(neg).long()

    def calc_reg_loss(self):
        reg_loss_encoder = 0
        for param in self.encoder.parameters():
            reg_loss_encoder += param.norm(2).square()
        
        reg_loss_decoder = 0
        for param in self.decoder.parameters():
            reg_loss_decoder += param.norm(2).square()

        reg_loss_recommender = 0
        for param in self.recommender.parameters():
            reg_loss_recommender += param.norm(2).square()
        return reg_loss_encoder + reg_loss_decoder + reg_loss_recommender

    def calc_cross_entropy(self, seq_out, pos_emb, neg_emb, tar_msk):
        seq_emb = seq_out.view(-1, cfg.hidden_size)
        pos_emb = pos_emb.view(-1, cfg.hidden_size)
        neg_emb = neg_emb.view(-1, cfg.hidden_size)
        pos_scr = torch.sum(pos_emb * seq_emb, -1)
        neg_scr = torch.sum(neg_emb * seq_emb, -1)
        tar_msk = tar_msk.view(-1).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_scr) + 1e-24) * tar_msk -
            torch.log(1 - torch.sigmoid(neg_scr) + 1e-24) * tar_msk
        ) / torch.sum(tar_msk)
        return loss
        
    def predict(
        self,
        seqs: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor,
        masked_adj: torch.Tensor, masked_edg: torch.Tensor
    ):
        r"""
        Parameters:
        -----------
        seqs: torch.Tensor
            a batch of sequence of items
        positives: torch.Tensor
            positive items for SASRec
        negtives: torch.Tensor
            negative items for SASRec
        masked_adj: torch.Tensor
            masked adjacency matrix for reconstruction tasks
        masked_edg: torch.Tensor
            masked edges for reconstruction tasks.

        Flows:
        -----
        1. Encoder returns the item embeddinsg.
        2. Seq embeddings will be given by SASRec and then the (binary) cross entropy loss will be calculated.
        3. Positive and negative edges will be sampled.
        4. Decoder predicts the masked edges and the correspoding loss will be calculated.

        Returns
        -------
        loss_main: the main loss for SASRec
        loss_reco: the reconstruction loss
        loss_regu: the regularization loss
        """
        item_emb, item_emb_his = self.encoder(masked_adj)
        seq_emb = self.recommender(seqs, item_emb)
        tar_msk = positives > 0
        loss_main = self.calc_cross_entropy(seq_emb, item_emb[positives], item_emb[negatives], tar_msk)

        pos = self.sample_pos_edges(masked_edg)
        neg = self.sample_neg_edges(pos)
        loss_reco = self.decoder(item_emb_his, pos, neg)       

        loss_regu = self.calc_reg_loss()
        return loss_main, loss_reco * cfg.weight4reco, loss_regu * cfg.weight_decay

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        item_emb, item_emb_his = self.encoder(self.ii_adj)
        seq_emb = self.recommender(seqs, item_emb)
        seq_emb = seq_emb[:, -1, :].unsqueeze(-1)  # (B, D, 1)
        item_emb = item_emb[pool] # (B, K, D)
        return item_emb.matmul(seq_emb).flatten(1) # (B, K)

    def recommend_from_full(self, seqs: torch.Tensor):
        item_emb, item_emb_his = self.encoder(self.ii_adj)
        seq_emb = self.recommender(seqs, item_emb)
        seq_emb = seq_emb[:, -1, :]  # (B, D)
        item_emb = item_emb[NUM_PADS:] # (N, D)
        return seq_emb.matmul(item_emb.t()) # (B, N)


class CoachForMAERec(freerec.launcher.SeqCoach):

    def calc_reward(self, lastLosses, eps):
        if len(lastLosses) < 3:
            return 1.0
        curDecrease = lastLosses[-2] - lastLosses[-1]
        avgDecrease = 0
        for i in range(len(lastLosses) - 2):
            avgDecrease += lastLosses[i] - lastLosses[i + 1]
        avgDecrease /= len(lastLosses) - 2
        return 1 if curDecrease > avgDecrease else eps

    def prepare(self):
        self.loss_his = []

    def train_per_epoch(self, epoch: int):
        for i, data in enumerate(self.dataloader):
            if i % self.cfg.mask_steps == 0:
                sample_scr, masked_adj, masked_edg = self.model.sample()

            users, seqs, targets, negatives = [col.to(self.device) for col in data]
            loss_main, loss_reco, loss_regu = self.model.predict(seqs, targets, negatives, masked_adj, masked_edg)

            loss = loss_main + loss_reco + loss_regu

            self.loss_his.append(loss_main) # detach ?
            if i % self.cfg.mask_steps == 0:
                reward = self.calc_reward(self.loss_his, self.cfg.eps)
                loss_mask = -sample_scr.mean() * reward
                self.loss_his = self.loss_his[-1:]
                loss += loss_mask

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    cfg.num_items = Item.count + NUM_PADS

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_seqs(keepid=True)
    ).sharding_filter().seq_train_uniform_sampling_(
        dataset, leave_one_out=False # yielding (user, seqs, targets, negatives)
    ).lprune_(
        indices=[1, 2, 3], maxlen=cfg.maxlen
    ).rshift_(
        indices=[1, 2, 3], offset=NUM_PADS
    ).lpad_(
        indices=[1, 2, 3], maxlen=cfg.maxlen, padding_value=0
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

    model = MAERec(dataset)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=0.
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=0.
        )

    coach = CoachForMAERec(
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
    coach.prepare()
    coach.fit()


if __name__ == "__main__":
    main()