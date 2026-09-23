"""V3 interface skeleton; computation is not implemented."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StaticModuleV3Config:
    f_node: int
    f_edge: int
    f_hidden: int = 64
    n_heads: int = 4


class StaticModuleV3(nn.Module):
    """Encode bus-mapped static features [B, N, f_node] into [B, N, H]."""

    def __init__(self, config: StaticModuleV3Config):
        super().__init__()
        self.config = config

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        raise NotImplementedError("StaticModuleV3 computation is not implemented")
