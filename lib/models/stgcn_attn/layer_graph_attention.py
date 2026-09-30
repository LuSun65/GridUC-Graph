from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphAttentionConfig:
    n_node: int
    f_in: int
    f_out: int
    n_heads: int = 4
    dropout: float = 0.0
    negative_slope: float = 0.2


class GraphAttentionLayer(nn.Module):
    """Multi-head attention over the shared graph, with one self-loop per node.

    Input: x [B, N, F], edge_index [2, E]. Output: [B, N, f_out].
    Attention is normalized over each destination's incoming edges.
    """

    def __init__(self, config: GraphAttentionConfig):
        super().__init__()
        if config.n_heads < 1:
            raise ValueError("n_heads must be at least 1")
        if config.f_out % config.n_heads != 0:
            raise ValueError(
                f"f_out ({config.f_out}) must be divisible by "
                f"n_heads ({config.n_heads})"
            )
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_node = config.n_node
        self.n_heads = config.n_heads
        self.head_dim = config.f_out // config.n_heads
        self.dropout = config.dropout
        self.negative_slope = config.negative_slope

        self.proj = nn.Linear(
            config.f_in, config.n_heads * self.head_dim, bias=False
        )
        self.attn_src = nn.Parameter(
            torch.empty(config.n_heads, self.head_dim)
        )
        self.attn_dst = nn.Parameter(
            torch.empty(config.n_heads, self.head_dim)
        )
        self.bias = nn.Parameter(torch.zeros(config.f_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def _edges_with_self_loops(
        self, edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source and destination with exactly one loop per node."""
        src, dst = edge_index
        non_self = src != dst
        loops = torch.arange(self.n_node, device=edge_index.device)
        return torch.cat((src[non_self], loops)), torch.cat((dst[non_self], loops))

    def _incoming_softmax(
        self,
        logits: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax over incoming edges for each batch, node, and head."""
        B, E, H = logits.shape
        index = dst.view(1, E, 1).expand(B, E, H)

        max_per_node = torch.full(
            (B, self.n_node, H),
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        max_per_node.scatter_reduce_(
            1, index, logits, reduce="amax", include_self=True
        )
        stabilized = logits - max_per_node.gather(1, index)
        exp_logits = torch.exp(stabilized)

        denominator = torch.zeros(
            B, self.n_node, H, device=logits.device, dtype=logits.dtype
        )
        denominator.scatter_add_(1, index, exp_logits)
        return exp_logits / denominator.gather(1, index).clamp_min(
            torch.finfo(logits.dtype).tiny
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        if N != self.n_node:
            raise ValueError(f"expected {self.n_node} nodes, got {N}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        src, dst = self._edges_with_self_loops(edge_index)

        node_value = self.proj(x).view(B, N, self.n_heads, self.head_dim)
        src_value = node_value[:, src, :, :]
        dst_value = node_value[:, dst, :, :]

        logits = (src_value * self.attn_src).sum(dim=-1)
        logits = logits + (dst_value * self.attn_dst).sum(dim=-1)
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        alpha = self._incoming_softmax(logits, dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = src_value * alpha.unsqueeze(-1)

        E = src.shape[0]
        index = dst.view(1, E, 1, 1).expand(
            B, E, self.n_heads, self.head_dim
        )
        output = torch.zeros(
            B,
            self.n_node,
            self.n_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        output.scatter_add_(1, index, messages)
        return output.reshape(B, self.n_node, -1) + self.bias
