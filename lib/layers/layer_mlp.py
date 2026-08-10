import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MLPConfig:
    f_in:     int
    f_out:    int
    f_hidden: int = 64
    n_layers: int = 2        # total layers including output layer
    dropout:  float = 0.1


class MLPLayer(nn.Module):
    """
    input:  [..., f_in]
    output: [..., f_out]
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        layers = []
        in_dim = config.f_in
        for _ in range(config.n_layers - 1):
            layers += [nn.Linear(in_dim, config.f_hidden), nn.ReLU()]
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.f_hidden
        layers.append(nn.Linear(in_dim, config.f_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
