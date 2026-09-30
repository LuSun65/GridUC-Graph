from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.common.layer_cheb_conv import ChebConvConfig, ChebConvLayer
from lib.models.common.layer_temp_conv import TemporalConvConfig, TemporalConvLayer


@dataclass
class STConv2Config:
    n_node: int
    k_t: int = 3
    k_s: int = 3
    f_in: int = 1
    f_hidden: int = 64
    f_out: int = 64


class ResidualChebConvBlock(nn.Module):
    """ChebConv update with a projected skip connection and LayerNorm."""

    def __init__(self, config: ChebConvConfig):
        super().__init__()
        self.graph_conv = ChebConvLayer(config)
        self.skip = (
            nn.Identity()
            if config.f_in == config.f_out
            else nn.Linear(config.f_in, config.f_out, bias=False)
        )
        self.alpha = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(config.f_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)
        update = self.graph_conv(x, edge_index)
        return self.norm(residual + self.alpha * update)


class STConvBlock2(nn.Module):
    """stgcn_attn_2 ST block with two Chebyshev graph convolutions."""

    N_CHEB_CONV = 2

    def __init__(self, config: STConv2Config):
        super().__init__()
        self.tconv1 = TemporalConvLayer(TemporalConvConfig(
            f_in=config.f_in,
            f_out=config.f_out,
            k_t=config.k_t,
        ))
        self.cheb_convs = nn.ModuleList([
            ResidualChebConvBlock(ChebConvConfig(
                n_node=config.n_node,
                f_in=config.f_out if index == 0 else config.f_hidden,
                f_out=config.f_hidden,
                k_s=config.k_s,
            ))
            for index in range(self.N_CHEB_CONV)
        ])
        self.tconv2 = TemporalConvLayer(TemporalConvConfig(
            f_in=config.f_hidden,
            f_out=config.f_out,
            k_t=config.k_t,
        ))
        self.ln = nn.LayerNorm(config.f_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tconv1(x)

        batch_size, n_period, n_node, channels = x.shape
        x = x.reshape(batch_size * n_period, n_node, channels)
        for block in self.cheb_convs:
            x = block(x, edge_index)
        x = x.reshape(batch_size, n_period, n_node, -1)

        return self.ln(self.tconv2(x))
