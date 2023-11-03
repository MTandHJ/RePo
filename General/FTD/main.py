

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

cfg.add_argument("--similarity ", type=str, choices=('cosine', 'inner'), default='cosine')
cfg.add_argument("--weight4ftd", type=float, default=1., help="weight for FTD loss")

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


class FTD(freerec.models.RecSysArch):

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
    
    def ftd_loss(self, users: torch.Tensor, items: torch.Tensor):
        users = users.flatten().unique()
        items = items.flatten().unique()

        with torch.no_grad():
            userFeats_t, itemFeats_t = self.teacher.recommend_from_full()
        userFeats_s, itemFeats_s = self.student.recommend_from_full()

        feats_t = torch.cat((userFeats_t[users], itemFeats_t[items]), dim=0)
        feats_s = torch.cat((userFeats_s[users], itemFeats_s[items]), dim=0)

        if cfg.similarity == 'cosine':
            feats_t = F.normalize(feats_t, dim=1)
            feats_s = F.normalize(feats_s, dim=1)

        sim_t = self.calculate_similarity(feats_t, feats_t)
        sim_s = self.calculate_similarity(feats_s, feats_s)

        return (sim_t - sim_s).square().sum()

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        logits_s = self.student.predict(users, items)
        return logits_s, self.fid_loss(users, items)

    def recommend_from_full(self):
        return self.student.recommend_from_full()


class CoachForFTD(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            items = torch.cat(
                [positives, negatives], dim=1
            )
            logits_s, logits_t, ftd_loss = self.model.predict(users, items)
            loss = self.criterion(logits_s, logits_t) + self.cfg.weight4ftd * ftd_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


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

    model = FTD(
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
    criterion = freerec.criterions.BPRLoss(reduction='sum')

    coach = CoachForFTD(
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