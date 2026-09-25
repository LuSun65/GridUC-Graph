"""Edge-token attention followed by a bus-level readout."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ESASpatialConfig:
    f_in: int
    f_edge: int
    f_out: int
    n_heads: int = 4
    dropout: float = 0.0


class ESASpatial(nn.Module):
    """Map [B, N, f_in] to [B, N, f_out] using directed edge tokens.

    Each batch element is an independent graph, including a time slice when
    called from a spatiotemporal block.
    """

    def __init__(self, config: ESASpatialConfig):
        super().__init__()
        if config.f_out % config.n_heads:
            raise ValueError("f_out must be divisible by n_heads")
        self.config = config
        self.edge_projection = nn.Linear(2 * config.f_in + config.f_edge, config.f_out)
        self.attention = nn.MultiheadAttention(
            config.f_out, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.edge_norm1 = nn.LayerNorm(config.f_out)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.f_out, 2 * config.f_out),
            nn.ReLU(),
            nn.Linear(2 * config.f_out, config.f_out),
        )
        self.edge_norm2 = nn.LayerNorm(config.f_out)
        self.node_projection = nn.Linear(config.f_in, config.f_out)
        self.node_norm = nn.LayerNorm(config.f_out)

    def forward(self, x, edge_index, edge_attr, edge_mask) -> torch.Tensor:
        batch, n_node, _ = x.shape
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        n_edge = edge_index.shape[1]
        if edge_attr.shape != (batch, n_edge, self.config.f_edge):
            raise ValueError("edge_attr must have shape [B, E, f_edge]")
        if edge_mask.shape != (batch, n_edge):
            raise ValueError("edge_mask must have shape [B, E]")

        residual = self.node_projection(x)
        if n_edge == 0:
            return self.node_norm(residual)

        source, target = edge_index.unbind(0)
        valid_index = (source >= 0) & (source < n_node) & (target >= 0) & (target < n_node)
        active = edge_mask.to(torch.bool)
        if (active & ~valid_index.unsqueeze(0)).any():
            raise ValueError("active edges must have valid endpoint indices")
        # Padding may use -1 endpoints. Its mask is false, and clamped indices
        # only permit a safe gather/scatter before the padded token is removed.
        source = source.clamp(0, n_node - 1)
        target = target.clamp(0, n_node - 1)
        tokens = self.edge_projection(torch.cat(
            (x[:, source], x[:, target], edge_attr), dim=-1
        ))
        # MultiheadAttention needs at least one unmasked key per sample.
        # The temporary key cannot contribute to an empty graph's readout.
        safe_active = active.clone()
        safe_active[:, 0] |= ~active.any(dim=1)
        attended, _ = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        tokens = self.edge_norm1(tokens + attended)
        tokens = self.edge_norm2(tokens + self.feed_forward(tokens))
        tokens = tokens * active.unsqueeze(-1)

        incoming = torch.zeros_like(residual)
        incoming.index_add_(1, target, tokens)
        degree = torch.zeros(batch, n_node, 1, dtype=x.dtype, device=x.device)
        degree.index_add_(1, target, active.unsqueeze(-1).to(x.dtype))
        return self.node_norm(residual + incoming / degree.clamp_min(1))
