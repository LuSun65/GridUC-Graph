import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.models.common.layer_temp_conv import TemporalConvLayer, TemporalConvConfig
from lib.models.common.layer_cheb_conv import ChebConvLayer, ChebConvConfig


# ---------------------------------------------------------------------------
# ST-Conv block:  TemporalConv -> ChebConv -> TemporalConv -> LayerNorm
# ---------------------------------------------------------------------------

@dataclass
class STConvConfig:
    n_node: int
    k_t:    int = 3    # temporal kernel size
    k_s:    int = 3    # Chebyshev order
    f_in:   int = 1    # input channels
    f_hidden:  int = 64   # graph conv output channels
    f_out:  int = 64   # temporal conv output channels


class STConvBlock(nn.Module):
    """
    input:  x [B, T, N, f_in],  edge_index [2, E],  edge_mask [E]
    output:   [B, T, N, f_out]
    """

    def __init__(self, config: STConvConfig):
        super().__init__()
        self.tconv1     = TemporalConvLayer(TemporalConvConfig(
            f_in=config.f_in, f_out=config.f_out, k_t=config.k_t,
        ))
        self.cheb_conv  = ChebConvLayer(ChebConvConfig(
            n_node = config.n_node,
            f_in   = config.f_out,
            f_out  = config.f_hidden,
            k_s    = config.k_s,
        ))
        self.tconv2     = TemporalConvLayer(TemporalConvConfig(
            f_in=config.f_hidden, f_out=config.f_out, k_t=config.k_t,
        ))
        self.ln         = nn.LayerNorm(config.f_out)

    def forward(
        self,
        x:          torch.Tensor,   # [B, T, N, f_in]
        edge_index: torch.Tensor,   # [2, E]
        edge_mask:  torch.Tensor,   # [B, E]  1=active, 0=broken
    ) -> torch.Tensor:              # [B, T, N, f_out]
        x = self.tconv1(x)                                                    # [B, T, N, f_out]

        B, T, N, C = x.shape
        x_bt = x.reshape(B * T, N, C)                                        # [B*T, N, C]
        mask_bt = edge_mask.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1) # [B*T, E]
        x_bt = torch.relu(self.cheb_conv(x_bt, edge_index, mask_bt))         # [B*T, N, f_hidden]
        x    = x_bt.reshape(B, T, N, -1)                                     # [B, T, N, f_hidden]

        x = self.tconv2(x)                                                    # [B, T, N, f_out]
        x = self.ln(x)                                                        # [B, T, N, f_out]
        return x
