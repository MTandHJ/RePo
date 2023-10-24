

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
cfg.add_argument("--temperature", type=float, default=1.)
cfg.add_argument("--weight4hard", type=float, default=1.)
cfg.set_defaults(
    description="KD",
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


class KD(freerec.models.RecSysArch):

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

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        logits_s = self.student.predict(users, items)
        with torch.no_grad():
            logits_t = self.teacher.predict(users, items)
        return logits_s, logits_t

    def recommend_from_full(self):
        return self.student.recommend_from_full()


class CoachForKD(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            logits_s, logits_t = self.model.predict(users, items)
            loss = self.criterion(logits_s, logits_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class KLDivLoss4Logits(freerec.criterions.BaseCriterion):
    """KLDivLoss with logits"""

    def __init__(self, reduction: str = 'batchmean') -> None:
        super().__init__('mean')
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        assert logits.size() == targets.size()
        inputs = F.log_softmax(logits, dim=-1)
        targets = F.softmax(targets, dim=-1)
        return F.kl_div(inputs, targets, reduction=self.reduction)

class KDLoss(freerec.criterions.BaseCriterion):

    def __init__(self, temperature: float, weight4hard: float) -> None:
        super().__init__()

        self.main_loss = freerec.criterions.BPRLoss('mean')
        self.kdiv_loss = KLDivLoss4Logits('batchmean')
        self.temperature = temperature
        self.weight4hard = weight4hard

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor):
        hard_loss = self.main_loss(logits_s[:, 0], logits_s[:, 1])
        soft_loss = self.kdiv_loss(logits_s / self.temperature, logits_t / self.temperature) * (self.temperature ** 2)
        return soft_loss * (1 - self.weight4hard) + hard_loss * self.weight4hard


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

    model = KD(
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
    criterion = KDLoss(cfg.temperature, cfg.weight4hard)

    coach = CoachForKD(
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