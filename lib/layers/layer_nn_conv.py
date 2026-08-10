import torch
import torch.nn as nn
from dataclasses import dataclass



@dataclass
class StaticGraph:
    """
    Data format for static graph input.

    edge_index  [2, E]          — shared across all samples
    edge_attr   [B, E, F_edge]  — per sample edge features
    node_feat   [B, N, F_node]  — per sample node features (already bus-indexed)
    edge_mask   [E]             — 1=active, 0=broken line (shared across all samples)
    """
    edge_index:  torch.Tensor   # [2, E]
    edge_attr:   torch.Tensor   # [B, E, F_edge]
    node_feat:   torch.Tensor   # [B, N, F_node]
    edge_mask:   torch.Tensor   # [B, E]  1=active, 0=broken line (per sample)

    @property
    def f_in(self) -> int:
        return self.node_feat.shape[-1]

    @property
    def f_edge(self) -> int:
        return self.edge_attr.shape[-1]


@dataclass
class NNConvConfig:
    n_node:   int         # number of buses
    f_in:   int         # input node feature dim
    f_edge: int         # input edge feature dim
    n_layers: int = 2     # number of layers in edge_mlp
    f_hidden: int = 64    # neurons per layer (also the node output dim)


class NNConvLayer(nn.Module):
    """
    Single NNConv layer. Forward takes a StaticGraph.

    Node output dim = f_hidden.

    edge_attr [B, E, f_edge]  ->  edge_mlp  ->  W [B, E, f_in, f_hidden]
    x_src     [B, E, f_in]  @   W          ->  msg [B, E, f_hidden]
    msg       * edge_mask      ->  masked msg [B, E, f_hidden]  (broken edges zeroed)
    msg                        ->  scatter_add(dst)  ->  out [B, n_node, f_hidden]
    """

    def __init__(self, config: NNConvConfig):
        super().__init__()
        self.n_node   = config.n_node
        self.f_in     = config.f_in
        self.f_hidden = config.f_hidden

        # edge_mlp: f_edge -> [f_hidden] * n_layers -> f_in * f_hidden
        layers = []
        in_dim = config.f_edge
        for _ in range(config.n_layers):
            layers += [nn.Linear(in_dim, config.f_hidden), nn.ReLU()]
            in_dim = config.f_hidden
        layers.append(nn.Linear(in_dim, config.f_in * config.f_hidden))
        self.edge_mlp = nn.Sequential(*layers)

    def _gather(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        # x [B, n_node, f_in], src [E] -> [B, E, f_in]
        return x[:, src, :]

    def _scatter_add(self, msg: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # msg [B, E, f_hidden], dst [E] -> [B, n_node, f_hidden]
        B, E, f_hidden = msg.shape
        out = torch.zeros(B, self.n_node, f_hidden, device=msg.device, dtype=msg.dtype)
        idx = dst.view(1, E, 1).expand(B, E, f_hidden)
        out.scatter_add_(1, idx, msg)
        return out

    def forward(self, data: StaticGraph) -> torch.Tensor:  # [B, n_node, f_hidden]
        src, dst = data.edge_index[0], data.edge_index[1]
        E = data.edge_attr.shape[1]

        x     = data.node_feat                                                          # [B, n_node, f_in]
        W     = self.edge_mlp(data.edge_attr).view(-1, E, self.f_in, self.f_hidden)   # [B, E, f_in, f_hidden]
        x_src = self._gather(x, src)                                                    # [B, E, f_in]
        msg   = torch.einsum('bef, befo -> beo', x_src, W)                             # [B, E, f_hidden]
        msg   = msg * data.edge_mask.unsqueeze(-1).float()                              # [B, E, f_hidden]  broken edges → 0
        return self._scatter_add(msg, dst)                                              # [B, n_node, f_hidden]
