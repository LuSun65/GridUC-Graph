"""Encoder-only adapter of upstream SABComplete/ESA from edge-set-attention."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import SAB
from .mlp import SmallMLP

UPSTREAM_COMMIT = '861a0b12d445ae80cc69b3948af623b4331321d4'


def edge_adjacency(edges):
    """SDPA boolean mask: True permits attention, including the diagonal."""
    source, target = edges
    return ((source[:, None] == source[None, :])
            | (target[:, None] == target[None, :])
            | (source[:, None] == target[None, :])
            | (target[:, None] == source[None, :]))


class EncoderBlock(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        width = config.f_hidden
        self.norm = nn.LayerNorm(width, eps=1e-8)
        self.norm_mlp = nn.LayerNorm(width, eps=1e-8)
        self.sab = SAB(width, width, config.n_heads,
                       config.attention_dropout, 'torch')
        self.mlp = SmallMLP(width, config.mlp_hidden, width,
                            dropout_p=config.mlp_dropout,
                            num_layers=config.mlp_layers)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x, mask):
        # Preserve upstream's Pre-LN residual, which uses normalized X.
        x = self.norm(x)
        x = x + self.sab(x, mask if self.kind == 'M' else None)
        x = x + self.mlp(self.norm_mlp(x))
        return self.dropout(x)


class ESAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(EncoderBlock(config, k) for k in config.layer_types)
        self.output = nn.Linear(config.f_hidden, config.f_hidden)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(self, x, mask):
        if x.shape[1] == 0:
            return x
        initial = x
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        # Upstream encoder adds its input and applies a final projection + Mish.
        return torch.nn.functional.mish(self.output(x + initial))
