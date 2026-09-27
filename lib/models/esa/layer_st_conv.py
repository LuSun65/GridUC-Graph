"""Temporal-spatial-temporal block for ESA."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_temp_conv import TemporalConvConfig, TemporalConvLayer
from lib.models.esa.layer_esa_spatial import ESASpatial, ESASpatialConfig


@dataclass
class STConvESAConfig:
    f_in: int
    f_edge: int
    f_hidden: int = 64
    k_t: int = 3
    n_heads: int = 4


class STConvBlockESA(nn.Module):
    """Causal temporal convolution, per-time ESA, causal convolution."""

    def __init__(self, config: STConvESAConfig):
        super().__init__()
        self.tconv1 = TemporalConvLayer(TemporalConvConfig(config.f_in, config.f_hidden, config.k_t))
        self.spatial = ESASpatial(ESASpatialConfig(
            config.f_hidden, config.f_edge, config.f_hidden, config.n_heads
        ))
        self.tconv2 = TemporalConvLayer(TemporalConvConfig(config.f_hidden, config.f_hidden, config.k_t))
        self.norm = nn.LayerNorm(config.f_hidden)

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        x = self.tconv1(x)
        batch, period, n_node, width = x.shape
        spatial = self.spatial(
            x.reshape(batch * period, n_node, width),
            edge_index,
            edge_attr[:, None].expand(-1, period, -1, -1).reshape(batch * period, *edge_attr.shape[1:]),
            edge_mask[:, None].expand(-1, period, -1).reshape(batch * period, -1),
        )
        x = self.tconv2(torch.relu(spatial.reshape(batch, period, n_node, width)))
        return self.norm(x)
