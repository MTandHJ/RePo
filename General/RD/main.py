

from typing import Optional

import torch, os
import torch.nn.functional as F
from torch_geometric.data.data import Data

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from models import BPRMF, LightGCN

freerec.declare(version="0.4.3")

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=200, help="embedding dim for Teacher model")
cfg.add_argument("--ratio", type=int, default=10, help="embedding dim / ratio for student model")
cfg.add_argument("--path", type=str, default=None, help="the path of Teacher model")
cfg.add_argument("--filename", type=str, default='best.pt', help="the filename of Teacher model")
cfg.add_argument("--model", type=str, choices=('MF', 'LightGCN'), default='MF')
cfg.add_argument("--num-layers", type=int, default=3, help="Valid for LightGCN")

cfg.add_argument("--num-negs", type=int, default=1)
cfg.add_argument("--num-dynamic-samples", type=int, default=100)
cfg.add_argument("--dynamic-start-epoch", type=int, default=10)
cfg.add_argument("--weight-renormalize", type=str, default='False')


cfg.add_argument("--K", type=int, default=10, help="top-K positions for ranking distillation")
cfg.add_argument("--alpha", type=float, default=1., help="weight for balancing ranking loss and distillation loss")
cfg.add_argument("--lamda", type=float, default=1., help="hyperparameter for tuning the sharpness of position importance weight")
cfg.add_argument("--mu", type=float, default=0.1, help="hyperparameter for tuning the sharpness of ranking discrepancy weight")


cfg.set_defaults(
    description="RD",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-6,
    seed=1
)
cfg.compile()

if cfg.weight_renormalize == 'True':
    cfg.weight_renormalize = True
elif cfg.weight_renormalize == 'False':
    cfg.weight_renormalize = False
else:
    raise ValueError("weight_renormalize should be 'True' or 'False' ...")

if cfg.model == 'MF':
    BACKBONE = BPRMF
elif cfg.model == 'LightGCN':
    BACKBONE = LightGCN
else:
    raise ValueError(f"Only 'MF' or 'LightGCN' is supported ...")


class RD(freerec.models.RecSysArch):

    def __init__(
        self, 
        fields: FieldModuleList,
        graph: Optional[Data] = None,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.teacher = BACKBONE(fields, cfg.embedding_dim, graph=graph, num_layers=num_layers).requires_grad_(False)
        self.student = BACKBONE(fields, cfg.embedding_dim // cfg.ratio, graph=graph, num_layers=num_layers)

        self.teacher.load_state_dict(torch.load(os.path.join(cfg.path, cfg.filename), map_location='cpu'))
        self.teacher.eval()

    @torch.no_grad()
    def get_teacher_topk(self):
        from freeplot.utils import import_pickle, export_pickle
        k = cfg.K
        path = f"teacher_topks/{cfg.dataset}/{str(k)}"
        file_ = os.path.join(path, "topk.pickle")
        try:
            topks = import_pickle(file_).to(self.device)
        except ImportError:
            userFeats, itemFeats = self.teacher.recommend_from_full()
            num_users, num_items = len(userFeats), len(itemFeats)
            assert k <= num_items, "k > number of Item is invalid ..."

            topks = torch.empty((0, k), dtype=torch.long).to(self.device)
            for users in torch.tensor(range(num_users)).to(self.device).split(256):
                users = userFeats[users].flatten(1) # B x D
                items = itemFeats.flatten(1) # N x D
                scores = users.matmul(items.T) # B x N
                values, indices = scores.topk(k, dim=1, largest=True, sorted=True)
                topks = torch.cat([topks, indices])
            topks = topks
            freerec.utils.mkdirs(path)
            export_pickle(topks.cpu(), file_)
        finally:
            self.register_buffer("teacher_topks", topks)

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        r"""
        Parameters:
        -----------
        users: a mini-batch of users, (B, 1)
        items: a mini-batch of items, (B, 1 + num_negs + num_dynamic_samples)
        """
        batch_candidates = self.teacher_topks[users.flatten()] # (B, K)
        items = torch.cat((items, batch_candidates), dim=1)
        logits = self.student.predict(users, items)
        return logits.split_with_sizes((1, cfg.num_negs, cfg.num_dynamic_samples, cfg.K), 1)

    def recommend_from_full(self):
        return self.student.recommend_from_full()


class CoachForRD(freerec.launcher.GenCoach):

    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        Item = self.fields[ITEM, ID]
        weight_static = (-torch.tensor(range(1, cfg.K + 1)).float() / self.cfg.lamda).exp()
        self.weight_static = (weight_static / weight_static.sum()).to(self.device)
        seqs = dataset.to_seqs(keepid=False)
        self.num_pools = torch.tensor(
            [Item.count - len(seq) for seq in seqs], 
            dtype=torch.long, device=self.device
        )

    def get_position_weight(
        self, epoch: int,
        users: torch.Tensor,
        logits_topk: torch.Tensor,
        logits_dynamic: torch.Tensor,
    ):
        num_pools = self.num_pools[users] - 1 # (B, 1) for estimating Students' ranking
        num_pools = num_pools.float()
        if epoch > self.cfg.dynamic_start_epoch:
            logits_topk = logits_topk.unsqueeze(-1) # (B, K, 1)
            logits_dynamic = logits_dynamic.unsqueeze(1) # (B, 1, D)
            relative_rank = (logits_topk < logits_dynamic).sum(dim=-1).float().div(self.cfg.num_dynamic_samples)
            predicted_rank = torch.floor(num_pools * relative_rank)
            weight = torch.tanh(self.cfg.mu * (predicted_rank - logits_topk.squeeze(-1))).clamp_min(0.)
            if self.cfg.weight_renormalize:
                weight = F.normalize(weight, p=1, dim=1)
        else:
            weight = self.weight_static
        return weight

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            logits_pos, logits_neg, logits_dynamic, logits_topk = self.model.predict(users, items)
            loss = self.criterion(
                logits_pos, logits_neg, logits_topk,
                self.get_position_weight(epoch, users, logits_topk, logits_dynamic)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class RDLoss(freerec.criterions.BaseCriterion):

    def __init__(self, alpha: float = None) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        logits_pos: torch.Tensor, logits_neg: torch.Tensor, 
        logits_topk: torch.Tensor,
        weight: torch.Tensor
    ):
        loss1 = -torch.log(torch.sigmoid(logits_pos))
        loss0 = -torch.log(1 - torch.sigmoid(logits_neg))

        loss_cand = -torch.log(torch.sigmoid(logits_topk))

        if weight is not None:
            loss_cand = loss_cand * weight.expand_as(loss_cand)

        if self.alpha is not None:
            loss_cand = loss_cand * self.alpha

        loss = torch.sum(torch.cat((loss1, loss0, loss_cand), 1), dim=1)
        # reg_loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

        return loss.mean() # , reg_loss.mean()


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=cfg.num_negs + cfg.num_dynamic_samples
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )

    tokenizer = FieldModuleList(dataset.fields)

    model = RD(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID)), 
        num_layers=cfg.num_layers
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.student.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.student.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = RDLoss(cfg.alpha)

    coach = CoachForRD(
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
        monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'],
        which4best='ndcg@20'
    )
    model.get_teacher_topk()
    coach.prepare(dataset)
    coach.fit()


if __name__ == "__main__":
    main()