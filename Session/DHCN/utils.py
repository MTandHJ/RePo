

import numpy as np
import torch

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_sparse import matmul
from torch_sparse import sum as sparsesum

from freerec.data.datasets.base import RecDataSet
from freerec.data.tags import SESSION, ITEM, ID


def get_session_item_graph(dataset: RecDataSet):
    Item = dataset.fields[ITEM, ID]
    seqs = dataset.to_roll_seqs(keepid=False)
    src = []
    trg = []
    for s, seq in seqs:
        seq = np.unique(seq).tolist()
        src.extend([s] * len(seq))
        trg.extend(seq)
    edge_index = torch.LongTensor(
        [src, trg]
    )
    graph = HeteroData()
    graph[SESSION.name].x = torch.empty((len(seqs), 0))
    graph[ITEM.name].x = torch.empty((Item.count, 0))
    graph[SESSION.name, 'to', ITEM.name].edge_index = edge_index
    T.ToSparseTensor()(graph)
    H = graph[SESSION.name, 'to', ITEM.name].adj_t
    B = sparsesum(H, dim=0)
    D = sparsesum(H, dim=1)

    B_inv = B.pow(-1)
    B_inv.masked_fill_(B_inv == float('inf'), 0.)
    D_inv = D.pow(-1)
    D_inv.masked_fill_(D_inv == float('inf'), 0.)

    adj = matmul(
        H.mul(D_inv.view(-1, 1)), 
        H.t().mul(B_inv.view(-1, 1))
    )
    return adj


def get_session_graph(seqs: torch.Tensor, n: int):
    R = torch.zeros(seqs.size(0), n + 1, device=seqs.device)
    R.scatter_(
        1, seqs, 1
    )
    R = R[:, 1:]

    s_cap_s = R.matmul(R.t())
    s_cup_s = (R.sum(-1) + R.sum(-1, keepdim=True)) - s_cap_s

    return s_cap_s / s_cup_s