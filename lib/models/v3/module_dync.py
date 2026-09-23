"""V3 interface skeleton; computation is not implemented."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DyncModuleV3Config:
    n_period: int
    f_node: int
    f_edge: int
    f_hidden: int = 64
    k_t: int = 3
    n_heads: int = 4


class DyncModuleV3(nn.Module):
    """Two temporal-ESA blocks and full-period collapse: [B, T, N, f_node] to [B, N, H]."""

    def __init__(self, config: DyncModuleV3Config):
        super().__init__()
        self.config = config

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        raise NotImplementedError("DyncModuleV3 computation is not implemented")
