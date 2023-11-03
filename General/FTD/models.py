

from typing import Optional, Union

import torch
import torch.nn as nn
from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID


class BPRMF(nn.Module):

    def __init__(self, fields: FieldModuleList, embedding_dim: int, **kwargs) -> None:
        super().__init__()

        self.fields = deepcopy(fields)
        self.fields.embed(
            embedding_dim, ID
        )
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userEmbs = self.User.look_up(users) # B x 1 x D
        itemEmbs = self.Item.look_up(items) # B x n x D
        return torch.mul(userEmbs, itemEmbs).sum(-1)

    def recommend_from_full(self):
        return self.User.embeddings.weight, self.Item.embeddings.weight


class LightGCN(nn.Module):

    def __init__(
        self, fields: FieldModuleList, embedding_dim: int,
        graph: Data, num_layers: int = 3,
        **kwargs
    ) -> None:
        super().__init__()

        self.fields = deepcopy(fields)
        self.fields.embed(
            embedding_dim, ID
        )
        self.conv = LGConv(False)
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = deepcopy(graph)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.__graph.adj_t = gcn_norm(
            self.__graph.adj_t, num_nodes=self.User.count + self.Item.count,
            add_self_loops=False
        )

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(self):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats

    def predict(self, users: torch.Tensor, items: torch.Tensor):
        userFeats, itemFeats = self.forward()
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1)

    def recommend_from_full(self):
        return self.forward()