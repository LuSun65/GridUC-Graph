from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.v2.layer_graph_attention import (
    GraphAttentionConfig,
    GraphAttentionLayer,
)
from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.model_input import FusionGraph


@dataclass
class FusionModuleV22Config:
    n_node: int
    n_gen: int
    n_period: int
    f_in: int
    f_hidden: int = 64
    n_layers: int = 2
    n_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class ResidualGraphAttentionBlock(nn.Module):
    """Graph-attention update with a projected skip and LayerNorm."""

    def __init__(self, config: GraphAttentionConfig):
        super().__init__()
        self.graph_conv = GraphAttentionLayer(config)
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
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)
        update = self.graph_conv(x, edge_index, edge_mask)
        return self.norm(residual + self.alpha * update)


class FusionModuleV2_2(nn.Module):
    """V2.2 fusion encoder with two graph-attention layers."""

    N_ATTENTION = 2

    def __init__(self, config: FusionModuleV22Config):
        super().__init__()
        self.n_gen = config.n_gen
        self.n_period = config.n_period
        self.fusion_attentions = nn.ModuleList([
            ResidualGraphAttentionBlock(GraphAttentionConfig(
                n_node=config.n_node,
                f_in=2 * config.f_in if index == 0 else config.f_hidden,
                f_out=config.f_hidden,
                n_heads=config.n_heads,
                dropout=config.attention_dropout,
                negative_slope=config.attention_negative_slope,
            ))
            for index in range(self.N_ATTENTION)
        ])
        self.map_layer = MLPLayer(MLPConfig(
            f_in=config.f_hidden,
            f_out=config.n_period * config.f_hidden,
            f_hidden=config.f_hidden,
            n_layers=config.n_layers,
        ))
        self.out_layer = MLPLayer(MLPConfig(
            f_in=config.f_hidden,
            f_out=1,
            f_hidden=config.f_hidden,
            n_layers=config.n_layers,
        ))

    def forward(self, data: FusionGraph) -> torch.Tensor:
        batch_size = data.x_static.shape[0]
        x = torch.cat((data.x_static, data.x_dynamic), dim=-1)
        for block in self.fusion_attentions:
            x = block(x, data.edge_index, data.edge_mask)

        x = x[:, data.gen_bus, :]
        x = self.map_layer(x)
        x = x.reshape(batch_size, self.n_gen, self.n_period, -1)
        return self.out_layer(x).squeeze(-1)
