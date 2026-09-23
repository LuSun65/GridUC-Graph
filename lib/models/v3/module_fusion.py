"""V3 interface skeleton; computation is not implemented."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FusionModuleV3Config:
    n_period: int
    f_edge: int
    f_hidden: int = 64
    n_heads: int = 4


class FusionModuleV3(nn.Module):
    """Fuse bus embeddings with ESA and produce generator UC logits [B, G, T]."""

    def __init__(self, config: FusionModuleV3Config):
        super().__init__()
        self.config = config

    def forward(self, x_static, x_dynamic, edge_index, edge_attr, edge_mask, gen_bus) -> torch.Tensor:
        raise NotImplementedError("FusionModuleV3 computation is not implemented")
