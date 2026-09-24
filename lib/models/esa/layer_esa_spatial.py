"""V3 interface skeleton; computation is not implemented."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ESASpatialConfig:
    f_in: int
    f_edge: int
    f_out: int
    n_heads: int = 4


class ESASpatial(nn.Module):
    """Bus features [B, N, f_in] to [B, N, f_out] via edge attention."""

    def __init__(self, config: ESASpatialConfig):
        super().__init__()
        self.config = config

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        raise NotImplementedError("ESASpatial computation is not implemented")
