"""Dynamic bus encoder for V3."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.esa.layer_st_conv import STConvBlockV3, STConvV3Config


@dataclass
class DyncModuleV3Config:
    n_period: int
    f_node: int
    f_edge: int
    f_hidden: int = 64
    k_t: int = 3
    n_heads: int = 4


class DyncModuleV3(nn.Module):
    """Two temporal-ESA blocks and full-period collapse to [B, N, H]."""

    def __init__(self, config: DyncModuleV3Config):
        super().__init__()
        self.block1 = STConvBlockV3(STConvV3Config(
            config.f_node, config.f_edge, config.f_hidden, config.k_t, config.n_heads
        ))
        self.block2 = STConvBlockV3(STConvV3Config(
            config.f_hidden, config.f_edge, config.f_hidden, config.k_t, config.n_heads
        ))
        self.collapse = nn.Conv2d(config.f_hidden, 2 * config.f_hidden, (config.n_period, 1))
        self.proj = MLPLayer(MLPConfig(
            f_in=config.f_hidden, f_out=config.f_hidden,
            f_hidden=config.f_hidden, n_layers=2,
        ))

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        x = self.block1(x, edge_index, edge_attr, edge_mask)
        x = self.block2(x, edge_index, edge_attr, edge_mask)
        p, q = self.collapse(x.permute(0, 3, 1, 2)).chunk(2, dim=1)
        return self.proj((p * torch.sigmoid(q)).squeeze(2).permute(0, 2, 1))
