

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

cfg.add_argument("--weight4dist", type=float, default=1., help="weight for Distance-wise loss")
cfg.add_argument("--weight4angle", type=float, default=1., help="weight for Angle-wise loss")

cfg.set_defaults(
    description="RKD",
    root="../../data",
    dataset='Gowalla_10100811_Chron',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-8,
    seed=1
)
cfg.compile()


if cfg.model == 'MF':
    BACKBONE = BPRMF
elif cfg.model == 'LightGCN':
    BACKBONE = LightGCN
else:
    raise ValueError(f"Only 'MF' or 'LightGCN' is supported ...")


class RKD(freerec.models.RecSysArch):

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

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats_s, itemFeats_s = self.student.recommend_from_full()
        with torch.no_grad():
            userFeats_t, itemFeats_t = self.teacher.recommend_from_full()
        return userFeats_s[users], itemFeats_s[items], userFeats_t[users], itemFeats_t[items]

    def recommend_from_full(self):
        return self.student.recommend_from_full()


class CoachForRKD(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            userFeats_s, itemFeats_s, userFeats_t, itemFeats_t = self.model.predict(users, items)
            loss = self.criterion(
                userFeats_s, itemFeats_s,
                userFeats_t, itemFeats_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class DistLoss(freerec.criterions.BaseCriterion):

    def forward(
        self, 
        userFeats_s: torch.Tensor, itemFeats_s: torch.Tensor,
        userFeats_t: torch.Tensor, itemFeats_t: torch.Tensor,
    ):
        with torch.no_grad():
            t_d = (userFeats_t - itemFeats_t).norm(p=2, dim=-1)
            mean_td = t_d.mean()
            t_d = t_d / mean_td

        d = (userFeats_s - itemFeats_s).norm(p=2, dim=-1)
        mean_d = d.mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction=self.reduction)
        return loss

class AngleLoss(freerec.criterions.BaseCriterion):

    def forward(
        self, 
        userFeats_s: torch.Tensor, itemFeats_s: torch.Tensor,
        userFeats_t: torch.Tensor, itemFeats_t: torch.Tensor,
    ):
        with torch.no_grad():
            td = userFeats_t - itemFeats_t
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = userFeats_s - itemFeats_s
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction=self.reduction)
        return loss       


class RKDLoss(freerec.criterions.BaseCriterion):

    def __init__(self, weight4dist: float, weight4angle: float) -> None:
        super().__init__()

        self.main_loss = freerec.criterions.BPRLoss('mean')
        self.dist_loss = DistLoss()
        self.angle_loss = AngleLoss()

        self.weight4dist = weight4dist
        self.weight4angle = weight4angle

    def forward(
        self, 
        userFeats_s: torch.Tensor, itemFeats_s: torch.Tensor,
        userFeats_t: torch.Tensor, itemFeats_t: torch.Tensor,
    ):
        logits_s = torch.mul(userFeats_s, itemFeats_s).sum(-1)
        loss = self.main_loss(logits_s[:, 0], logits_s[:, 1])
        loss += self.weight4dist * self.dist_loss(
            userFeats_s, itemFeats_s,
            userFeats_t, itemFeats_t
        )
        loss += self.weight4angle * self.angle_loss(
            userFeats_s, itemFeats_s,
            userFeats_t, itemFeats_t
        )

        return loss


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1
    ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_gen_validpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_gen_testpipe(
        dataset, batch_size=512, ranking=cfg.ranking
    )

    tokenizer = FieldModuleList(dataset.fields)

    model = RKD(
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
    criterion = RKDLoss(cfg.weight4dist, cfg.weight4angle)

    coach = CoachForRKD(
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
    coach.fit()


if __name__ == "__main__":
    main()