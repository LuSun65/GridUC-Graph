from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.v2.layer_graph_attention import (
    GraphAttentionConfig,
    GraphAttentionLayer,
)
from lib.models.common.layer_temp_conv import TemporalConvConfig, TemporalConvLayer


@dataclass
class AttentionSTConvConfig:
    n_node: int
    k_t: int = 3
    f_in: int = 1
    f_hidden: int = 64
    f_out: int = 64
    n_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class AttentionSTConvBlock(nn.Module):
    """Spatio-temporal block with graph attention for spatial messages.

    The input and output contract matches ``STConvBlock``. Graph attention is
    applied independently at every time step after folding the batch and time
    dimensions together.

    input:  x [B, T, N, f_in], edge_index [2, E], edge_mask [B, E]
    output: [B, T, N, f_out]
    """

    def __init__(self, config: AttentionSTConvConfig):
        super().__init__()
        self.tconv1 = TemporalConvLayer(TemporalConvConfig(
            f_in=config.f_in,
            f_out=config.f_out,
            k_t=config.k_t,
        ))
        self.graph_attention = GraphAttentionLayer(GraphAttentionConfig(
            n_node=config.n_node,
            f_in=config.f_out,
            f_out=config.f_hidden,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
            negative_slope=config.attention_negative_slope,
        ))
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
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tconv1(x)

        B, T, N, C = x.shape
        x_bt = x.reshape(B * T, N, C)
        mask_bt = edge_mask.unsqueeze(1).expand(B, T, -1)
        mask_bt = mask_bt.reshape(B * T, -1)
        x_bt = torch.relu(self.graph_attention(
            x_bt, edge_index, mask_bt
        ))
        x = x_bt.reshape(B, T, N, -1)

        x = self.tconv2(x)
        return self.ln(x)
