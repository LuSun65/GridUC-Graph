"""V3 interface skeleton; computation is not implemented."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class STConvV3Config:
    f_in: int
    f_edge: int
    f_hidden: int = 64
    k_t: int = 3
    n_heads: int = 4


class STConvBlockV3(nn.Module):
    """Temporal convolution, per-time ESA, then temporal convolution; [B, T, N, H]."""

    def __init__(self, config: STConvV3Config):
        super().__init__()
        self.config = config

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        raise NotImplementedError("STConvBlockV3 computation is not implemented")
