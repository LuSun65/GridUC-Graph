import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.layers.layer_pre_norm import PreNormLayer
from lib.layers.layer_mlp import MLPLayer, MLPConfig
from lib.layers.layer_st_conv import STConvBlock, STConvConfig


@dataclass
class DyncGraph:
    """
    Data format for dynamic graph input.

    x          [B, n_period, N, f_node]  — per sample node features
    edge_index [2, E]                    — shared across all samples
    edge_mask  [B, E]                    — 1=active, 0=broken line (per sample)
    """
    x:          torch.Tensor   # [B, n_period, N, f_node]
    edge_index: torch.Tensor   # [2, E]
    edge_mask:  torch.Tensor   # [B, E]

    @property
    def f_node(self) -> int:
        return self.x.shape[-1]


@dataclass
class DyncModuleConfig:
    n_period: int          # number of time steps
    n_node:   int          # number of buses
    f_node:   int          # input node feature dim
    f_hidden: int = 64     # hidden channels throughout
    f_out:    int = 64     # output dim per node
    k_t:      int = 3      # temporal kernel size


class DyncModule(nn.Module):
    """
    input:  DyncGraph  —  x [B, n_period, N, f_node],  edge_index [2, E],  edge_mask [E]
    output: [B, N, f_out]
    """

    def __init__(self, config: DyncModuleConfig):
        super().__init__()

        self.norm = PreNormLayer(config.f_node)

        self.st_block1 = STConvBlock(STConvConfig(
            n_node   = config.n_node,
            k_t      = config.k_t,
            f_in     = config.f_node,
            f_hidden = config.f_hidden,
            f_out    = config.f_hidden,
        ))
        self.st_block2 = STConvBlock(STConvConfig(
            n_node   = config.n_node,
            k_t      = config.k_t,
            f_in     = config.f_hidden,
            f_hidden = config.f_hidden,
            f_out    = config.f_hidden,
        ))

        # No causal padding — collapses all time steps into one
        self.collapse = nn.Conv2d(config.f_hidden, 2 * config.f_hidden, (config.n_period, 1))

        self.proj = MLPLayer(MLPConfig(
            f_in     = config.f_hidden,
            f_out    = config.f_out,
            f_hidden = config.f_hidden,
            n_layers = 2,
        ))

    def prenorm_update_on(self):
        self.norm.update_on()

    def prenorm_update_off(self):
        self.norm.update_off()

    def forward(self, data: DyncGraph) -> torch.Tensor:
        x = self.norm(data.x)                         # [B, n_period, N, f_node]

        x = self.st_block1(x, data.edge_index, data.edge_mask)   # [B, n_period, N, f_hidden]
        x = self.st_block2(x, data.edge_index, data.edge_mask)   # [B, n_period, N, f_hidden]

        x = x.permute(0, 3, 1, 2)                         # [B, f_hidden, n_period, N]
        p, q = torch.chunk(self.collapse(x), 2, dim=1)  # [B, f_hidden, 1, N] each
        x = (p * torch.sigmoid(q)).squeeze(2).permute(0, 2, 1)  # [B, N, f_hidden]
        return self.proj(x)                            # [B, N, f_out]
