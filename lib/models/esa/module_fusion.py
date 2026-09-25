"""UC prediction head for V3."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_mlp import MLPConfig, MLPLayer
from lib.models.esa.layer_esa_spatial import ESASpatial, ESASpatialConfig


@dataclass
class FusionModuleV3Config:
    n_period: int
    f_edge: int
    f_hidden: int = 64
    n_heads: int = 4


class FusionModuleV3(nn.Module):
    """Fuse bus embeddings and produce generator logits [B, G, T]."""

    def __init__(self, config: FusionModuleV3Config):
        super().__init__()
        self.n_period = config.n_period
        self.f_hidden = config.f_hidden
        self.spatial = ESASpatial(ESASpatialConfig(
            2 * config.f_hidden, config.f_edge, config.f_hidden, config.n_heads
        ))
        self.time_map = MLPLayer(MLPConfig(
            f_in=config.f_hidden, f_out=config.n_period * config.f_hidden,
            f_hidden=config.f_hidden, n_layers=2,
        ))
        self.output = MLPLayer(MLPConfig(
            f_in=config.f_hidden, f_out=1,
            f_hidden=config.f_hidden, n_layers=2,
        ))

    def forward(self, x_static, x_dynamic, edge_index, edge_attr, edge_mask, gen_bus) -> torch.Tensor:
        x = torch.cat((x_static, x_dynamic), dim=-1)
        x = torch.relu(self.spatial(x, edge_index, edge_attr, edge_mask))
        batch = x.shape[0]
        x = self.time_map(x[:, gen_bus])
        return self.output(x.reshape(batch, gen_bus.numel(), self.n_period, self.f_hidden)).squeeze(-1)
