import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ChebConvConfig:
    n_node: int
    f_in: int
    f_out:  int
    k_s:    int = 3     # Chebyshev order (matches PyG ChebConv K parameter)


class ChebConvLayer(nn.Module):
    """
    Chebyshev spectral graph convolution.

    input:  x [B, n_node, f_in],  edge_index [2, E],  edge_mask [B, E] (1=active, 0=broken)
    output: [B, n_node, f_out]
    """

    def __init__(self, config: ChebConvConfig):
        super().__init__()
        self.n_node = config.n_node
        self.k_s    = config.k_s
        self.lins   = nn.ModuleList(
            [nn.Linear(config.f_in, config.f_out, bias=False) for _ in range(config.k_s)]
        )
        self.bias   = nn.Parameter(torch.zeros(config.f_out))

    def _edge_weights(self, edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """Compute L_hat edge weights: -1 / sqrt(deg_src * deg_dst), masked by edge_mask.
        edge_mask [B, E] -> returns [B, E]
        """
        src, dst = edge_index[0], edge_index[1]
        B, E     = edge_mask.shape
        deg      = torch.zeros(B, self.n_node, device=edge_index.device, dtype=torch.float)
        deg.scatter_add_(1, dst.unsqueeze(0).expand(B, E), edge_mask.float())
        deg_inv  = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0.0
        return -deg_inv[:, src] * deg_inv[:, dst] * edge_mask.float()   # [B, E]

    def _graph_mul(
        self,
        x:           torch.Tensor,   # [B, n_node, F]
        src:         torch.Tensor,   # [E]
        dst:         torch.Tensor,   # [E]
        edge_weight: torch.Tensor,   # [B, E]  already zeroed for broken edges
    ) -> torch.Tensor:               # [B, n_node, F]
        B, N, F = x.shape
        msg = x[:, src, :] * edge_weight.unsqueeze(-1)   # [B, E, F]
        out = torch.zeros(B, N, F, device=x.device, dtype=x.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, F)
        out.scatter_add_(1, idx, msg)
        return out

    def forward(
        self,
        x:          torch.Tensor,   # [B, n_node, f_in]
        edge_index: torch.Tensor,   # [2, E]
        edge_mask:  torch.Tensor,   # [B, E]  1=active, 0=broken
    ) -> torch.Tensor:              # [B, n_node, f_out]
        src, dst    = edge_index[0], edge_index[1]
        edge_weight = self._edge_weights(edge_index, edge_mask)   # [B, E]

        z_prev, z_cur = x, self._graph_mul(x, src, dst, edge_weight)
        out = self.lins[0](z_prev)
        if self.k_s > 1:
            out = out + self.lins[1](z_cur)
        for lin in self.lins[2:]:
            z_next = 2.0 * self._graph_mul(z_cur, src, dst, edge_weight) - z_prev
            out    = out + lin(z_next)
            z_prev, z_cur = z_cur, z_next

        return out + self.bias
