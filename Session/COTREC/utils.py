

import torch

from torch_geometric.data.data import Data
from collections import defaultdict

from freerec.data.datasets.base import RecDataSet
from freerec.data.tags import ITEM, ID


def get_item_graph(dataset: RecDataSet):
    Item = dataset.fields[ITEM, ID]
    seqs = dataset.to_seqs(keepid=False)
    edge = defaultdict(int)
    for seq in seqs:
        for i in range(len(seq) - 1):
            edge[(seq[i], seq[i + 1])] += 1
    edge_index, edge_weight = zip(*edge.items())
    edge_index = torch.LongTensor(
        edge_index
    ).t()
    graph = Data()
    graph.x = torch.empty((Item.count, 0))
    graph.edge_index = edge_index
    graph.edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return graph


def get_session_graph(seqs: torch.Tensor, n: int):
    R = torch.zeros(seqs.size(0), n + 1, device=seqs.device)
    R.scatter_(
        1, seqs, 1
    )
    R = R[:, 1:]

    s_cap_s = R.matmul(R.t())
    s_cup_s = (R.sum(-1) + R.sum(-1, keepdim=True)) - s_cap_s

    return s_cap_s / s_cup_s