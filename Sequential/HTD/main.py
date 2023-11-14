

import torch, os
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from models import BPRMF, GRU4Rec, SASRec
from coaches import CoachForBPRMF, CoachForGRU4Rec, CoachForSASRec
from utils import load_datapipes

freerec.declare(version='0.4.3')

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

cfg.add_argument("--similarity", type=str, choices=('cosine', 'inner'), default='cosine')
cfg.add_argument("--topology", type=str, choices=('group2group', 'group2entity'), default='group2group')
cfg.add_argument("--weight4htd", type=float, default=0.001, help="weight for FTD loss")
cfg.add_argument("--gamma", type=float, default=0.5, help="(0, 1), balacing topology loss and group assignment loss")
cfg.add_argument("--K", type=int, default=30, help="#Experts")


cfg.set_defaults(
    description="HTD",
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


T_START = 1.
T_END = 1.e-10


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


class Projector(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]))

    def forward(self, x):
        return self.net(x)


class HTD(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.teacher = BACKBONE(fields, cfg.embedding_dim, cfg).requires_grad_(False)
        self.student = BACKBONE(fields, cfg.embedding_dim // cfg.ratio, cfg)

        self.teacher.load_state_dict(torch.load(os.path.join(cfg.path, cfg.filename), map_location='cpu'))
        self.teacher.eval()

        dims = [
            cfg.embedding_dim // cfg.ratio, 
            (cfg.embedding_dim + cfg.embedding_dim // cfg.ratio) // 2, 
            cfg.embedding_dim
        ]
        self.userProjectors = nn.ModuleList([Projector(dims) for i in range(self.K)])
        self.itemProjectors = nn.ModuleList([Projector(dims) for i in range(self.K)])

        self.userClassifier = nn.Sequential(
            nn.Linear(cfg.embedding_dim, self.K),
            nn.Softmax(dim=1)
        )
        self.itemClassifier = nn.Sequential(
            nn.Linear(cfg.embedding_dim, self.K),
            nn.Softmax(dim=1)
        )

    def marked_params(self):
        from itertools import chain
        return chain(
            self.student.parameters(),
            self.userProjectors.parameters(),
            self.itemProjectors.parameters(),
            self.userClassifier.parameters(),
            self.itemClassifier.parameters()
        )

    def anneal_T(self, epoch: int):
        self.T = T_START * ((T_END / T_START) ** (epoch / cfg.epochs))

    def calculate_similarity(self, x: torch.Tensor, y: torch.Tensor):
        r"""
        Pairwise similarity.

        Parameters:
        -----------
        x: torch.Tensor, (m, d)
        y: torch.Tensor, (n, d)

        Returns:
        --------
        Similarity matrix: torch.Tensor, (m, n)
        """
        return x @ y.t()

    def group(
        self,
        feats_t: torch.Tensor, 
        classifier: nn.Module
    ):
        return classifier(feats_t).argmax(-1)

    def group_assignment_loss(
        self, 
        feats_s: torch.Tensor, feats_t: torch.Tensor, 
        projectors: nn.Module, classifier: nn.Module
    ):
        alpha = classifier(feats_t) + 1.e-10 # (n, K)
        g = torch.distributions.Gumbel(0, 1).sample(alpha.size()).to(self.device)
        z = F.softmax(
            (alpha.log() + g) / self.T, dim=1
        ).unsqueeze(1).repeat((1, cfg.embedding_dim, 1)) # (n, D, K)

        feat_s2t = torch.cat([p(feats_s).unsqueeze(-1) for p in projectors], dim=-1) # (n, D, K)
        feat_s2t = (feat_s2t * z).sum(-1) # (n, D)

        return F.mse_loss(feat_s2t, feats_t, reduction='sum')

    def topology_distillation_loss(
        self,
        feats_s: torch.Tensor, feats_t: torch.Tensor,
        groups: torch.Tensor
    ):
        Z = F.one_hot(groups).float()	
        G_set = groups.unique()

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(feats_s)[G_set]
            P_t = tmp.mm(feats_t)[G_set]

        mask = Z.mm(Z.t()) > 0

        if cfg.similarity == 'cosine':
            feats_t = F.normalize(feats_t, dim=1)
            feats_s = F.normalize(feats_s, dim=1)
            P_s = F.normalize(P_s)
            P_t = F.normalize(P_t)

        sim_t = self.calculate_similarity(feats_t, feats_t).masked_select(mask)
        sim_s = self.calculate_similarity(feats_s, feats_s).masked_select(mask)

        if cfg.topology == 'group2group':
            sim_g_t = self.calculate_similarity(P_t, P_t)
            sim_g_s = self.calculate_similarity(P_s, P_s)
        else:
            sim_g_t = self.calculate_similarity(P_t, feats_t)
            sim_g_s = self.calculate_similarity(P_s, feats_s)

        return F.mse_loss(sim_s, sim_t, reduction='sum') + \
            F.mse_loss(sim_g_s, sim_g_t, reduction='sum')


    def htd_loss(
        self, 
        userFeats_s: torch.Tensor, 
        userFeats_t: torch.Tensor,
        items: torch.Tensor
    ):
        items = items.flatten().unique()

        with torch.no_grad():
            itemFeats_t = self.teacher.Item.embeddings.weight
        itemFeats_s = self.student.Item.embeddings.weight

        itemFeats_t = itemFeats_t[items]
        itemFeats_s = itemFeats_s[items]

        group_loss = self.group_assignment_loss(
            userFeats_s, userFeats_t, 
            self.userProjectors, self.userClassifier
        )
        group_loss += self.group_assignment_loss(
            itemFeats_s, itemFeats_t, 
            self.itemProjectors, self.itemClassifier
        )

        feats_t = torch.cat((userFeats_t, itemFeats_t), dim=0) # (B, D)
        feats_s = torch.cat((userFeats_s, itemFeats_s), dim=0) # (B, d)
        groups = torch.cat((
            self.group(userFeats_t, self.userClassifier),
            self.group(itemFeats_t, self.itemClassifier) + cfg.K
        ), dim=0)


        topology_loss = self.topology_distillation_loss(
            feats_s, feats_t, groups
        )

        return topology_loss * cfg.gamma + group_loss * (1 - cfg.gamma)

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
        userFeats_s = self.student(users).unsqueeze(-2) # (B, *, D)
        itemFeats_s = self.student.Item.look_up(items) # (B, *, 2, D)
        with torch.no_grad():
            userFeats_t = self.teacher(users).unsqueeze(-2)
        # (B x *, 1/2, D)
        userFeats_s = userFeats_s.flatten(end_dim=1)
        itemFeats_s = itemFeats_s.flatten(end_dim=1)
        userFeats_t = userFeats_t.flatten(end_dim=1)

        logits_s = userFeats_s.mul(itemFeats_s).sum(-1)
        return logits_s, self.htd_loss(userFeats_s, userFeats_t, items)

    def recommend(self, **kwargs):
        return self.student.recommend(**kwargs)


def main():

    dataset, trainpipe, validpipe, testpipe = load_datapipes(cfg)
    tokenizer = FieldModuleList(dataset.fields)
    model = HTD(tokenizer)

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
    criterion = freerec.criterions.BPRLoss(reduction='sum')

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