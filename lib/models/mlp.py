import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.models.model_input import STGCNInput


@dataclass
class MLPConfig:
    n_gen:      int        # number of generators
    n_period:   int        # number of time steps
    n_node:     int        # number of buses
    f_node_s:   int        # static generator feature dim
    f_node_d:   int        # dynamic node feature dim
    f_edge:     int        # edge feature dim
    n_edge:     int        # number of edges (2 * num_line)
    hidden_dim: int = 512  # hidden layer width
    n_hidden:   int = 3    # number of hidden layers


class MLP(nn.Module):
    """
    Simple MLP baseline for UC prediction.

    Flattens all features from STGCNInput into a single vector,
    passes through hidden layers, and outputs [B, G, T] logits.

    input:  STGCNInput  (same as STGCN)
    output: [B, G, n_period]  — logits
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        # Input dim: flatten all features
        in_dim = (
            config.n_gen * config.f_node_s          # node_feat_s [G, F_node_s]
            + config.n_period * config.n_node * config.f_node_d  # node_feat_d [T, N, F_node_d]
            + config.n_edge * config.f_edge          # edge_attr [E, F_edge]
            + config.n_edge                           # edge_mask [E]
        )
        out_dim = config.n_gen * config.n_period

        layers = []
        prev = in_dim
        for _ in range(config.n_hidden):
            layers.append(nn.Linear(prev, config.hidden_dim))
            layers.append(nn.ReLU())
            prev = config.hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def default_config(cls, data: STGCNInput) -> "MLPConfig":
        """Infer MLPConfig dimensions from a batched STGCNInput."""
        B, G, f_node_s = data.node_feat_s.shape
        B, T, N, f_node_d = data.node_feat_d.shape
        f_edge = data.edge_attr.shape[-1]
        n_edge = data.edge_attr.shape[1]
        return MLPConfig(
            n_gen    = G,
            n_period = T,
            n_node   = N,
            f_node_s = f_node_s,
            f_node_d = f_node_d,
            f_edge   = f_edge,
            n_edge   = n_edge,
        )

    def forward(self, data: STGCNInput) -> torch.Tensor:  # [B, G, n_period]
        B = data.node_feat_s.shape[0]
        parts = [
            data.node_feat_s.reshape(B, -1),   # [B, G*F_node_s]
            data.node_feat_d.reshape(B, -1),   # [B, T*N*F_node_d]
            data.edge_attr.reshape(B, -1),     # [B, E*F_edge]
            data.edge_mask.reshape(B, -1),     # [B, E]
        ]
        x = torch.cat(parts, dim=-1)          # [B, in_dim]
        logits = self.net(x)                   # [B, G*T]
        return logits.reshape(B, self.config.n_gen, self.config.n_period)
