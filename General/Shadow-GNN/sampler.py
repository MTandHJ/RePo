

import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import freerec
from freerec.data.tags import USER, ITEM, ID


class ShadowHopSampler:
    r"""
    K-hop sampling with depth/width limitation.

    Parameters:
    -----------
    dataset: freerec.data.datasets.BaseSet
    depth: int, depth-hop
    num_neighbors: int
        The maximum number of neighbors sampled for each node
    replace: bool, default `False'
    """

    def __init__(
        self, dataset: freerec.data.datasets.BaseSet,
        depth: int, num_neighbors: int, 
        replace: bool = False, dynamic: bool = True
    ) -> None:

        self.graph = dataset.train().to_graph(
            (USER, ID), (ITEM, ID)
        )
        self.User = dataset.fields[USER, ID]
        self.Item = dataset.fields[ITEM, ID]
        N = self.User.count + self.Item.count

        row, col = self.graph.edge_index
        A = SparseTensor(
            row=row, col=col, sparse_sizes=(N, N)
        )
        self.row_ptr, self.col_indices, _ = A.csr()

        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.dynamic = dynamic
        self._cached_batch = []
        self._cached_nodes = []
        self._cached_mask = []

        if not self.dynamic:
            for user in range(self.User.count):
                batch, nodes, mask = self._sample(user)
                self._cached_batch.append(batch)
                self._cached_nodes.append(nodes)
                self._cached_mask.append(mask)
            self._cached_batch = tuple(self._cached_batch)
            self._cached_nodes = tuple(self._cached_nodes)
            self._cached_mask = tuple(self._cached_mask)

    def _sample(self, root: int):
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            self.row_ptr, self.col_indices, torch.tensor(root), self.depth, self.num_neighbors, self.replace)
        rowptr, col, nodes_ori, edges_ori, ptr, _ = out

        A = SparseTensor(
                rowptr=rowptr, col=col,
                sparse_sizes=(nodes_ori.numel(), nodes_ori.numel()),
                is_sorted=True, trust_data=True
            )

        row, col, _ = A.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=nodes_ori.numel(), add_self_loops=True
        )
        return Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=nodes_ori.numel()), nodes_ori, nodes_ori == root

    def sample(self, roots: torch.Tensor):
        roots = roots.flatten().cpu().tolist()
        if self.dynamic:
            batchs, nodes, mask = zip(*map(
                self._sample, roots
            ))
        else:
            batchs = [self._cached_batch[root] for root in roots]
            nodes = [self._cached_nodes[root] for root in roots]
            mask = [self._cached_mask[root] for root in roots]

        batchdata = Batch.from_data_list(batchs)
        nodes = torch.cat(nodes)
        mask = torch.cat(mask)
        return batchdata, nodes, mask