"""Static bus encoder for V3."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.esa.layer_esa_spatial import ESASpatial, ESASpatialConfig


@dataclass
class StaticModuleV3Config:
    f_node: int
    f_edge: int
    f_hidden: int = 64
    n_heads: int = 4


class StaticModuleV3(nn.Module):
    """Encode bus-mapped generator features to [B, N, H]."""

    def __init__(self, config: StaticModuleV3Config):
        super().__init__()
        self.spatial = ESASpatial(ESASpatialConfig(
            config.f_node, config.f_edge, config.f_hidden, config.n_heads
        ))

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        return self.spatial(x, edge_index, edge_attr, edge_mask)
