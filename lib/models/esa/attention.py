"""Adapted from upstream esa/mha.py at the revision pinned in encoder.py."""
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0, xformers_or_torch_attn="torch"):
        super(MAB, self).__init__()

        if xformers_or_torch_attn != "torch":
            raise ValueError("This adapter supports only torch SDPA")
        self.xformers_or_torch_attn = xformers_or_torch_attn

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.fc_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)

        # NOTE: xavier_uniform_ might work better for a few datasets
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)


    def forward(self, Q, K, adj_mask=None):
        batch_size = Q.size(0)
        E_total = self.dim_V
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads

        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)

        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)

        if self.xformers_or_torch_attn in ["torch"]:
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)


        # SDPA broadcasts a shared mask over batch and heads.

        # PyTorch 2.3 CPU fused SDPA yields NaNs for large sparse masks even
        # with a valid diagonal. Math is portable; CUDA keeps automatic dispatch.
        backend = sdpa_kernel(SDPBackend.MATH) if Q.device.type == 'cpu' else nullcontext()
        with backend:
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=adj_mask,
                dropout_p=self.dropout_p if self.training else 0, is_causal=False,
            )
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)


        out = out + F.mish(self.fc_o(out))

        return out


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)
