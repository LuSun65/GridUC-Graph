from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.stgcn_attn.layer_graph_attention import (
    GraphAttentionConfig,
    GraphAttentionLayer,
)
from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.model_input import FusionGraph


@dataclass
class AttentionFusionModuleConfig:
    n_node: int
    n_gen: int
    n_period: int
    f_in: int
    f_hidden: int = 64
    n_layers: int = 2
    n_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class AttentionFusionModule(nn.Module):
    """Fuse static and dynamic embeddings using graph attention.

    This is the attention counterpart of ``FusionModule`` and intentionally
    accepts the same ``FusionGraph`` input and returns the same logits shape.

    input:  FusionGraph
    output: [B, G, n_period]
    """

    def __init__(self, config: AttentionFusionModuleConfig):
        super().__init__()
        self.n_gen = config.n_gen
        self.n_period = config.n_period

        self.fusion_attention = GraphAttentionLayer(GraphAttentionConfig(
            n_node=config.n_node,
            f_in=2 * config.f_in,
            f_out=config.f_hidden,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
            negative_slope=config.attention_negative_slope,
        ))

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
        B, N, _ = data.x_static.shape
        x = torch.cat((data.x_static, data.x_dynamic), dim=-1)
        x = torch.relu(self.fusion_attention(
            x, data.edge_index, data.edge_mask
        ))

        x_gen = x[:, data.gen_bus, :]
        x_map = self.map_layer(x_gen)
        x_map = x_map.reshape(B, self.n_gen, self.n_period, -1)
        return self.out_layer(x_map).squeeze(-1)
