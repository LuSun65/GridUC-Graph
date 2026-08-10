import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TemporalConvConfig:
    f_in:  int
    f_out: int
    k_t:   int = 3    # temporal kernel size


class TemporalConvLayer(nn.Module):
    """
    Causal gated temporal convolution.

    input:  [B, T, N, f_in]
    output: [B, T, N, f_out]   (T preserved via causal padding)
    """

    def __init__(self, config: TemporalConvConfig):
        super().__init__()
        self.k_t  = config.k_t
        self.conv = nn.Conv2d(config.f_in, 2 * config.f_out, (config.k_t, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)                     # [B, f_in, T, N]
        x = F.pad(x, (0, 0, self.k_t - 1, 0))        # causal pad on T dim
        p, q = torch.chunk(self.conv(x), 2, dim=1)
        return (p * torch.sigmoid(q)).permute(0, 2, 3, 1)  # [B, T, N, f_out]
