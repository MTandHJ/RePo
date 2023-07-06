

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


def pos_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

    Returns:
    Tensor with shape [length, hidden_size]
    """
    position = torch.FloatTensor(torch.range(0, length - 1))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (torch.FloatTensor(num_timescales) - 1))
    inv_timescales = min_timescale * \
        torch.exp(torch.FloatTensor(torch.range(0, num_timescales - 1))
                  * -log_timescale_increment)
    scaled_time = torch.unsqueeze(
        position, 1) * torch.unsqueeze(inv_timescales, 0)
    signal = torch.cat(
        (torch.sin(scaled_time), torch.cos(scaled_time)), axis=1)
    return signal


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class GNN(nn.Module):

    def __init__(
        self, 
        emebdding_dim: int, 
        L: int, T: int, layers: int, K: int = 3
    ):
        super(GNN, self).__init__()
        self.L = L
        self.T = T
        self.layers = layers
        self.hidden_size = emebdding_dim
        self.input_size = emebdding_dim * 2

        self.W1 = Parameter(torch.Tensor(self.input_size, self.hidden_size))
        nn.init.xavier_uniform_(self.W1)

        A = torch.ones((L, L))
        A = A.triu(1) * (1 - A.triu(K))
        A = A + A.t()
        A = A.mul(A.sum(-1, keepdim=True))
        self.register_buffer(
            'A', A
        )

    def GNNCell(self, hidden):
        input_in1 = torch.matmul(self.A, hidden)
        input_in_item1 = torch.cat((input_in1, hidden), dim=2)

        # no b have item
        item_hidden1 = torch.matmul(input_in_item1, self.W1)
        item_embs1 = item_hidden1

        item_embs = torch.tanh(item_embs1) 
      
        return item_embs

    def forward(self, hidden):
        for _ in range(self.layers):
            hidden = self.GNNCell(hidden)
        return hidden