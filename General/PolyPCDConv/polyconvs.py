

from typing import List

import torch
import torch.nn as nn
from torch_sparse import matmul
from torch_geometric.typing import Adj


class PolyPCDConv(nn.Module):
    r"""
    Polynomial convolution in which the coefficients are learnt through Polynomial Coefficient Decomposition (PCD).
    See [[here](https://arxiv.org/pdf/2205.11172.pdf)] for more details.


    Parameters:
    ----------
    embedding_dim: int
    scaling_factor: float, default to 1.
    L: int, default to 3,
        The number of layers (depth).
    poly_base: str
        - `monomial`: Monomial bases;
        - `legendre`: Legendre bases;
        - `chebyshev`: Chebyshev bases;
        - `jacobi`: Jacobi bases
    (alpha, beta): hyper-parameters for Jacobi bases.
    fixed: bool, default to `False`
        - `True`: unlearnable coefficients
        - `False`: learnable coefficients

    """

    def __init__(
        self, 
        embedding_dim: int,
        scaling_factor: float = 1.,
        L: int = 3,
        poly_base: str = 'jacobi',
        alpha: float = 1.,
        beta: float = 1.,
        fixed: bool = False
    ):
        self.L = L
        self.scaling_factor = scaling_factor

        self.register_parameter(
            'gammas',
            nn.parameter.Parameter(
                torch.empty((L + 1, embedding_dim)).fill_(min(1 / scaling_factor, 1.)),
                requires_grad=not fixed
            )
        )

        poly_base = poly_base.lower()
        if poly_base == 'monomial':
            self.conv_fn = power_conv
        elif poly_base == 'legendre':
            self.conv_fn = legendre_conv
        elif poly_base == 'chebyshev':
            self.conv_fn = chebyshev_conv
        elif poly_base == 'jacobi':
            from functools import partial
            self.conv_fn = partial(jacobi_conv, alpha=alpha, beta=beta)
        else:
            raise NotImplementedError(
                f"{poly_base} as a polunomial base is not supported ..."
            )

    def forward(self, x: torch.Tensor, A: Adj):
        zs = [self.conv_fn([x], A, 0)]
        for l in range(1, self.L + 1):
            z = self.conv_fn(zs, A, l)
            zs.append(z)
        coefs = (self.gammas.tanh() * self.scaling_factor).cumprod(dim=0) 
        zs = torch.stack(zs, dim=1) # (N, L + 1, D)
        return (zs * coefs).sum(1) # (N, D)


def power_conv(
    zs: List[torch.Tensor], A: Adj, l: int
):
    r"""
    Polynomial convolution with [Monomial bases](https://en.wikipedia.org/wiki/Monomial).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    return matmul(A, zs[-1], reduce='sum')

def legendre_conv(
    zs: List[torch.Tensor], A: Adj, l: int
):
    r"""
    Polynomial convolution with [Legendre bases](https://en.wikipedia.org/wiki/Legendre_polynomials#Recurrence_relations).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    if l == 1:    
        return matmul(A, zs[-1], reduce='sum')
    else:
        part1 = (2 - 1 / l) * matmul(A, zs[-1], reduce='sum')
        part2 = (1 - 1 / l) * zs[-2]
        return part1 - part2

def chebyshev_conv(
    zs: List[torch.Tensor], A: Adj, l: int
):
    r"""
    Polynomial convolution with [Chebyshev bases](https://en.wikipedia.org/wiki/Chebyshev_polynomials).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    if l == 1:
        return matmul(A, zs[-1], reduce='sum')
    else:
        part1 = 2 * matmul(A, zs[-1], reduce='sum')
        part2 = zs[-2]
        return part1 - part2

def jacobi_conv(
    zs: List[torch.Tensor], A: Adj, l: int, 
    alpha: float = 1., beta: float = 1.
):
    r"""
    Polynomial convolution with [Jacobi bases](https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    (alpha, beta): float, two hyper-parameters for Jacobi Polynomial.
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    if l == 1:
        c = (alpha - beta) / 2
        return c + (alpha + beta + 2) / 2 * zs[-1]
    else:
        c0 = 2 * l \
                * (l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c1 = (2 * l + alpha + beta - 1) \
                * (alpha ** 2 - beta ** 2)
        c2 = (2 * l + alpha + beta - 1) \
                * (2 * l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c3 = 2 * (l + alpha - 1) \
                * (l + beta - 1) \
                * (2 * l + alpha + beta)
        
        part1 = c1 * zs[-1]
        part2 = c2 * matmul(A, zs[-1], reduce='sum')
        part3 = c3 * zs[-2]

        return (part1 + part2 - part3) / c0