

import torch, os
import torch.nn as nn
import torch.nn.functional as F


import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from models import BPRMF, GRU4Rec, SASRec
from coaches import CoachForBPRMF, CoachForGRU4Rec, CoachForSASRec
from utils import load_datapipes

freerec.declare(version='0.5.1')

cfg = freerec.parser.Parser()

cfg.add_argument("-eb", "--embedding-dim", type=int, default=200, help="embedding dim for Teacher model")
cfg.add_argument("--ratio", type=int, default=10, help="embedding dim / ratio for student model")
cfg.add_argument("--path", type=str, default=None, help="the path of Teacher model")
cfg.add_argument("--filename", type=str, default='best.pt', help="the filename of Teacher model")
cfg.add_argument("--model", type=str, choices=('MF', 'GRU4Rec', 'SASRec'), default='SASRec')

# hyper-parameters for model
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1, help="for SASRec")
cfg.add_argument("--num-blocks", type=int, default=2, help="for SASRec")
cfg.add_argument("--num-gru-layers", type=int, default=1, help="for GRU4Rec")
cfg.add_argument("--dropout-rate", type=float, default=0.2)

# hyper-parameters for KD
cfg.add_argument("--temperature", type=float, default=1.)
cfg.add_argument("--alpha", type=float, default=1.)
cfg.add_argument("--beta", type=float, default=8.)

cfg.set_defaults(
    description="KD",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=128,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


if cfg.model == 'MF':
    BACKBONE = BPRMF
    COACH = CoachForBPRMF
    cfg.NUM_PADS = 0
elif cfg.model == 'GRU4Rec':
    BACKBONE = GRU4Rec
    COACH = CoachForGRU4Rec
    cfg.NUM_PADS = 1
elif cfg.model == 'SASRec':
    BACKBONE = SASRec
    COACH = CoachForSASRec
    cfg.NUM_PADS = 1
else:
    raise ValueError(f"Only 'MF' or 'GRU4Rec' or 'SASRec' is supported ...")


class DKD(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.teacher = BACKBONE(fields, cfg.embedding_dim, cfg).requires_grad_(False)
        self.student = BACKBONE(fields, cfg.embedding_dim // cfg.ratio, cfg)

        self.teacher.load_state_dict(torch.load(os.path.join(cfg.path, cfg.filename), map_location='cpu'))
        self.teacher.eval()

    def marked_params(self):
        return self.student.parameters()

    def predict(
        self, 
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        r"""
        users: (B, *)
            - (B, 1): users for MF
            - (B, S): seqs for GRU4Rec or SASRec
        positives: (B, *)
            - (B, 1): users for MF
            - (B, S): seqs for GRU4Rec or SASRec
        negatives: (B, *)
            - (B, 1): users for MF
            - (B, S): seqs for GRU4Rec or SASRec
        """
        items = torch.stack((positives, negatives), dim=-1) # (B, *, 2)
        userFeats_s = self.student(users).unsqueeze(-2) # (B, *, 1, D)
        itemFeats_s = self.student.Item.look_up(items) # (B, *, 2, D)
        logits_s = userFeats_s.mul(itemFeats_s).sum(-1) # (B, *, 2)
        logits_full_s = userFeats_s @ self.student.Item.embeddings.weight[cfg.NUM_PADS:].t() # (B, *, 1, N)
        with torch.no_grad():
            userFeats_t = self.teacher(users).unsqueeze(-2)
            logits_full_t = userFeats_t @ self.teacher.Item.embeddings.weight[cfg.NUM_PADS:].t() # (B, *, 1, N)
        
        # (B x *, 2/N)
        return logits_s.flatten(end_dim=-2), logits_full_s.flatten(end_dim=-2), logits_full_t.flatten(end_dim=-2)

    def recommend(self, **kwargs):
        return self.student.recommend(**kwargs)


def _get_gt_mask(logits: torch.Tensor, target: torch.Tensor):
    mask = F.one_hot(target, num_classes=logits.size(1))
    return mask.bool()

def _get_other_mask(logits: torch.Tensor, target: torch.Tensor):
    target = target.view(-1, 1)
    mask = torch.ones_like(logits).scatter(1, target, 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKDLoss(freerec.criterions.BaseCriterion):

    def __init__(
        self, 
        alpha: float = 1.,
        beta: float = 8.,
        temperature: float = 1.,
        reduction: str = 'batchmean'
    ) -> None:
        super().__init__(reduction)

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
        self, 
        logits_s: torch.Tensor, logits_t: torch.Tensor, 
        target: torch.Tensor
    ):
        gt_mask = _get_gt_mask(logits_s, target)
        other_mask = _get_other_mask(logits_s, target)
        pred_student = F.softmax(logits_s / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_t / self.temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_t / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_s / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss


class BPR_DKD_Loss(freerec.criterions.BaseCriterion):

    def __init__(
        self, 
        alpha: float = 1.,
        beta: float = 8.,
        temperature: float = 1.,
    ) -> None:
        super().__init__()

        self.main_loss = freerec.criterions.BPRLoss('mean')
        self.dkd_loss = DKDLoss(alpha, beta, temperature)

    def forward(
        self, logits_s: torch.Tensor, positives: torch.Tensor,
        logits_full_s: torch.Tensor, logits_full_t: torch.Tensor
    ):
        hard_loss = self.main_loss(logits_s[:, 0], logits_s[:, 1])
        soft_loss = self.dkd_loss(logits_full_s, logits_full_t, positives)
        return hard_loss + soft_loss


def main():

    dataset, trainpipe, validpipe, testpipe = load_datapipes(cfg)
    tokenizer = FieldModuleList(dataset.fields)
    model = DKD(tokenizer)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.marked_params(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.marked_params(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = BPR_DKD_Loss(cfg.alpha, cfg.beta, cfg.temperature)

    coach = COACH(
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