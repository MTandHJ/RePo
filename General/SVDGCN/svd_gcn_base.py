

import torch, os, math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

import freerec
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID

cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--alpha", type=int, default=0)
cfg.add_argument("--beta", type=float, default=2.)
cfg.add_argument("--req-vec", type=int, default=500)

cfg.set_defaults(
    description="SVD-GCN",
    root="../../data",
    dataset='Gowalla_m1',
    epochs=100,
    batch_size=2048,
    optimizer='sgd',
    lr=15,
    weight_decay=1.e-4,
    seed=1
)
cfg.compile()

class SVDGCN(RecSysArch):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.reg = cfg.weight_decay
        self.beta = cfg.beta
        self.User = tokenizer[USER, ID]
        self.Item = tokenizer[ITEM, ID]

        path = os.path.join(cfg.dataset, str(cfg.alpha))

        svd_filter = self.weight_func(
            torch.Tensor(np.load(path+ r'/svd_value.npy')[:cfg.req_vec])
        )

        user_vector = (torch.Tensor(np.load(path+ r'/svd_u.npy')[:,:cfg.req_vec])) * svd_filter
        item_vector = (torch.Tensor(np.load(path+ r'/svd_v.npy')[:,:cfg.req_vec])) * svd_filter

        self.register_buffer('user_vector', user_vector)
        self.register_buffer('item_vector', item_vector)
        self.register_parameter(
            'FS', Parameter(
                torch.randn(cfg.req_vec, cfg.embedding_dim),
                requires_grad=True
            )
        )
        nn.init.uniform_(self.FS, -np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)), np.sqrt(6. / (cfg.req_vec + cfg.embedding_dim)))


    def weight_func(self, sig):
        return torch.exp(self.beta * sig)


    def forward(self, users = None, items = None):
        if self.training:
            samp_user = users[self.User.name].flatten()
            samp_item = items[self.Item.name]
            pos_item, neg_item = samp_item[:, 0], samp_item[:, 1]

            final_user, final_pos, final_nega = self.user_vector[samp_user].mm(self.FS), self.item_vector[pos_item].mm(self.FS), self.item_vector[neg_item].mm(self.FS)
            out = ((final_user*final_pos).sum(1) - (final_user*final_nega).sum(1)).sigmoid()
            regu_term = self.reg*(final_user**2 + final_pos**2+final_nega**2).sum()
            return (-torch.log(out).sum() + regu_term) / samp_user.size(0), samp_user.size(0)
        else:
            return self.user_vector.mm(self.FS), self.item_vector.mm(self.FS)


class CoachForSVDGCN(Coach):


    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            loss, bz = self.model(users, items)
	
            loss.backward()
            with torch.no_grad():
                self.model.FS-= self.cfg.lr * self.model.FS.grad
                self.model.FS.grad.zero_()
            
            self.monitor(loss.item(), n=bz, mode="mean", prefix='train', pool=['LOSS'])

        
    def evaluate(self, prefix: str = 'valid'):
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        userFeats, itemFeats = self.model()
        for users, items in self.dataloader:
            users = users[User.name].to(self.device)
            targets = items[Item.name].to(self.device)
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[targets == -1] = -1e10
            targets[targets == -1] = 0

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE']
            )


def main():

    basepipe = getattr(freerec.data.datasets, cfg.dataset)(cfg.root)
    trainpipe = basepipe.uniform_sampling_(num_negatives=1).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=2048).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    model = SVDGCN(tokenizer)


    coach = CoachForSVDGCN(
        model=model,
        dataset=dataset,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()

