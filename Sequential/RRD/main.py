

import torch, os
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from models import BPRMF, GRU4Rec, SASRec
from coaches import CoachForBPRMF, CoachForGRU4Rec, CoachForSASRec
from utils import load_datapipes

freerec.declare(version='0.4.5')

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
cfg.add_argument("--K", type=int, default=40, help="#interesting items")
cfg.add_argument("--L", type=int, default=40, help="#uninteresting items")
cfg.add_argument("--T", type=int, default=10, help="#temperature")
cfg.add_argument("--num-topks", type=int, default=500)
cfg.add_argument("--weight4soft", type=float, default=0.001)


cfg.set_defaults(
    description="RRD",
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


assert cfg.num_topks > cfg.K, "#interesting items > NUM_TOPKS"
NUM_TOPKS = cfg.num_topks


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


class RRD(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.num_users = fields[USER, ID].count
        self.num_items = fields[ITEM, ID].count

        self.teacher = BACKBONE(fields, cfg.embedding_dim, cfg).requires_grad_(False)
        self.student = BACKBONE(fields, cfg.embedding_dim // cfg.ratio, cfg)

        self.teacher.load_state_dict(torch.load(os.path.join(cfg.path, cfg.filename), map_location='cpu'))
        self.teacher.eval()

        self.register_buffer(
            "position_probs",
            torch.arange(1, NUM_TOPKS + 1).float().neg().div(cfg.T).softmax(0).unsqueeze(0) # (1, NUM_TOPKS)
        )

    def marked_params(self):
        return self.student.parameters()

    @torch.no_grad()
    def sampling_interesting(self, logits_full_t: torch.Tensor):
        size = (len(logits_full_t), 1)
        indices = torch.multinomial(self.position_probs.repeat(size), cfg.K, replacement=False)
        _, topks = logits_full_t.topk(NUM_TOPKS, dim=-1, largest=True, sorted=True)
        return topks.gather(1, indices) + cfg.NUM_PADS

    @torch.no_grad()
    def sampling_uninteresting(self, logits_full_t: torch.Tensor):
        size = (len(logits_full_t), cfg.L)
        return torch.randint(cfg.NUM_PADS, self.num_items + cfg.NUM_PADS, size=size, device=self.device)

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
        userFeats_s = self.student(users).unsqueeze(-2).flatten(end_dim=-3) # (B x *, 1, D)
        itemFeats_s = self.student.Item.look_up(items).flatten(end_dim=-3) # (B x *, 2, D)
        logits_s = userFeats_s.mul(itemFeats_s).sum(-1) # (B x *, 2)
        with torch.no_grad():
            userFeats_t = self.teacher(users).unsqueeze(-2)
            logits_full_t = userFeats_t @ self.teacher.Item.embeddings.weight[cfg.NUM_PADS:].t() # (B, *, 1, N)
            logits_full_t = logits_full_t.flatten(end_dim=-2) # (B x *, N)
        interesting_items = self.sampling_interesting(logits_full_t) # (B x *, K)
        uninteresting_items = self.sampling_uninteresting(logits_full_t) # (B x *, L)
        logits_i = userFeats_s.mul(self.student.Item.look_up(interesting_items)).sum(-1)
        logits_u = userFeats_s.mul(self.student.Item.look_up(uninteresting_items)).sum(-1)
        # (B x *, 2/N)
        return logits_s, logits_i, logits_u 

    def recommend(self, **kwargs):
        return self.student.recommend(**kwargs)


def relaxed_ranking_loss(S1, S2):

	above = S1.sum(1, keepdims=True)

	below1 = S1.flip(-1).exp().cumsum(1)		
	below2 = S2.exp().sum(1, keepdims=True)		

	below = (below1 + below2).log().sum(1, keepdims=True)
	
	return -(above - below).sum()


class RRDLoss(freerec.criterions.BaseCriterion):

    def __init__(self, weight4soft: float) -> None:
        super().__init__()

        self.main_loss = freerec.criterions.BPRLoss('sum')
        self.soft_loss = relaxed_ranking_loss
        self.weight4soft = weight4soft

    def forward(
        self, logits_s: torch.Tensor, 
        logits_i: torch.Tensor, logits_u: torch.Tensor
    ):
        hard_loss = self.main_loss(logits_s[:, 0], logits_s[:, 1])
        soft_loss = self.soft_loss(logits_i, logits_u)
        return hard_loss + self.weight4soft * soft_loss


def main():

    dataset, trainpipe, validpipe, testpipe = load_datapipes(cfg)
    tokenizer = FieldModuleList(dataset.fields)
    model = RRD(tokenizer)

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
    criterion = RRDLoss(cfg.weight4soft)

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