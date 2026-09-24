from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_mlp import MLPConfig, MLPLayer


@dataclass
class STGCNOutput:
    uc_logits: torch.Tensor  # [B, G, T]
    lmp_pred: torch.Tensor   # [B, N, T], normalized price


class UCHead(nn.Module):
    """Use V1's time expansion and output layers on generator buses."""

    def __init__(self, map_layer, out_layer, n_period):
        super().__init__()
        self.map_layer = map_layer
        self.out_layer = out_layer
        self.n_period = n_period

    def forward(self, x, gen_bus):
        x = self.map_layer(x[:, gen_bus, :])
        x = x.reshape(x.shape[0], gen_bus.numel(), self.n_period, -1)
        return self.out_layer(x).squeeze(-1)


class LMPHead(MLPLayer):
    """Predict all bus prices with an unrestricted linear output."""

    def __init__(self, f_hidden, n_period, n_layers):
        super().__init__(MLPConfig(
            f_in=f_hidden, f_out=n_period,
            f_hidden=f_hidden, n_layers=n_layers,
        ))
