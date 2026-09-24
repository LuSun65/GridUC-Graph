from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.common.layer_pre_norm import PreNormLayer
from lib.models.stgcn_attn.layer_st_conv_attention import (
    AttentionSTConvBlock,
    AttentionSTConvConfig,
)
from lib.models.model_input import DyncGraph


@dataclass
class AttentionDyncModuleConfig:
    n_period: int
    n_node: int
    f_node: int
    f_hidden: int = 64
    f_out: int = 64
    k_t: int = 3
    n_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class AttentionDyncModule(nn.Module):
    """Dynamic graph encoder using attention-based ST-Conv blocks.

    This is the attention counterpart of ``DyncModule`` and intentionally
    accepts the same ``DyncGraph`` input and returns the same node embeddings.

    input:  DyncGraph
    output: [B, N, f_out]
    """

    def __init__(self, config: AttentionDyncModuleConfig):
        super().__init__()
        self.norm = PreNormLayer(config.f_node)

        common = {
            "n_node": config.n_node,
            "k_t": config.k_t,
            "f_hidden": config.f_hidden,
            "f_out": config.f_hidden,
            "n_heads": config.n_heads,
            "attention_dropout": config.attention_dropout,
            "attention_negative_slope": config.attention_negative_slope,
        }
        self.st_block1 = AttentionSTConvBlock(AttentionSTConvConfig(
            f_in=config.f_node,
            **common,
        ))
        self.st_block2 = AttentionSTConvBlock(AttentionSTConvConfig(
            f_in=config.f_hidden,
            **common,
        ))

        # No causal padding: collapse all time steps into one embedding.
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
        x = self.st_block1(x, data.edge_index, data.edge_mask)
        x = self.st_block2(x, data.edge_index, data.edge_mask)

        x = x.permute(0, 3, 1, 2)
        p, q = torch.chunk(self.collapse(x), 2, dim=1)
        x = (p * torch.sigmoid(q)).squeeze(2).permute(0, 2, 1)
        return self.proj(x)
