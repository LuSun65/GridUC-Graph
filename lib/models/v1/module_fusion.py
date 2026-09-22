import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.models.model_input import FusionGraph

from lib.models.common.layer_cheb_conv import ChebConvLayer, ChebConvConfig
from lib.models.common.layer_mlp import MLPLayer, MLPConfig


@dataclass
class FusionModuleConfig:
    n_node:   int          # number of buses
    n_gen:    int          # number of generators
    n_period: int          # number of time steps
    f_in:     int          # feature dim from each branch (must match StaticModule/DyncModule f_out)
    f_hidden: int = 64     # hidden channels throughout
    k_s:      int = 3      # Chebyshev order for fusion conv
    n_layers: int = 2      # MLP depth for time_mlp and out_layer


class FusionModule(nn.Module):
    """
    Fuses static and dynamic branch outputs, then predicts per-generator per-timestep logits.

    input:  FusionGraph
    output: [B, G, n_period]  — logits
    """

    def __init__(self, config: FusionModuleConfig):
        super().__init__()

        self.n_gen    = config.n_gen
        self.n_period = config.n_period

        self.fusion_conv = ChebConvLayer(ChebConvConfig(
            n_node = config.n_node,
            f_in   = 2 * config.f_in,
            f_out  = config.f_hidden,
            k_s    = config.k_s,
        ))

        # Expand each generator embedding to n_period time steps
        self.map_layer = MLPLayer(MLPConfig(
            f_in     = config.f_hidden,
            f_out    = config.n_period * config.f_hidden,
            f_hidden = config.f_hidden,
            n_layers = config.n_layers,
        ))

        # Per time-step binary output
        self.out_layer = MLPLayer(MLPConfig(
            f_in     = config.f_hidden,
            f_out    = 1,
            f_hidden = config.f_hidden,
            n_layers = config.n_layers,
        ))

    def forward(self, data: FusionGraph) -> torch.Tensor:   # [B, G, n_period]
        B, N, _ = data.x_static.shape

        x = torch.cat([data.x_static, data.x_dynamic], dim=-1)  # [B, N, 2*f_in]

        # ChebConv expects [B*1, N, F] — run per sample
        x_bt = x.reshape(B, N, -1)
        x_bt = torch.relu(self.fusion_conv(x_bt, data.edge_index, data.edge_mask))  # [B, N, f_hidden]

        # Extract generator nodes
        x_gen = x_bt[:, data.gen_bus, :]                            # [B, G, f_hidden]

        # Expand to time steps
        x_map = self.map_layer(x_gen)                               # [B, G, n_period*f_hidden]
        x_map = x_map.reshape(B, self.n_gen, self.n_period, -1)    # [B, G, n_period, f_hidden]

        return self.out_layer(x_map).squeeze(-1)                    # [B, G, n_period]  logits
