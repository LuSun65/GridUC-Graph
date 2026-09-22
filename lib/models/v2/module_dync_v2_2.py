from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.common.layer_pre_norm import PreNormLayer
from lib.models.v2.layer_st_conv_v2_2 import STConvBlockV2_2, STConvV22Config
from lib.models.model_input import DyncGraph


@dataclass
class DyncModuleV22Config:
    n_period: int
    n_node: int
    f_node: int
    f_hidden: int = 64
    f_out: int = 64
    k_t: int = 3
    k_s: int = 3


class DyncModuleV2_2(nn.Module):
    """V2.2 dynamic encoder with two ST blocks of two ChebConv layers."""

    def __init__(self, config: DyncModuleV22Config):
        super().__init__()
        self.norm = PreNormLayer(config.f_node)
        self.st_blocks = nn.ModuleList([
            STConvBlockV2_2(STConvV22Config(
                n_node=config.n_node,
                k_t=config.k_t,
                k_s=config.k_s,
                f_in=config.f_node if index == 0 else config.f_hidden,
                f_hidden=config.f_hidden,
                f_out=config.f_hidden,
            ))
            for index in range(2)
        ])
        self.collapse = nn.Conv2d(
            config.f_hidden,
            2 * config.f_hidden,
            (config.n_period, 1),
        )
        self.proj = MLPLayer(MLPConfig(
            f_in=config.f_hidden,
            f_out=config.f_out,
            f_hidden=config.f_hidden,
            n_layers=2,
        ))

    def prenorm_update_on(self):
        self.norm.update_on()

    def prenorm_update_off(self):
        self.norm.update_off()

    def forward(self, data: DyncGraph) -> torch.Tensor:
        x = self.norm(data.x)
        for st_block in self.st_blocks:
            x = st_block(x, data.edge_index, data.edge_mask)

        x = x.permute(0, 3, 1, 2)
        p, q = torch.chunk(self.collapse(x), 2, dim=1)
        x = (p * torch.sigmoid(q)).squeeze(2).permute(0, 2, 1)
        return self.proj(x)
