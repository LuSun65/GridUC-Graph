"""Shared model inputs, independent of architecture versions."""

from dataclasses import dataclass

import torch


@dataclass
class STGCNInput:
    """
    Batched input for STGCN.

    node_feat_s  [B, G, F_node_s]  — static generator node features (pre-scatter)
    node_feat_d  [B, T, N, F_node_d]  — dynamic node features (e.g. demand)
    edge_index   [2, E]            — shared topology
    edge_attr    [B, E, F_edge]    — per-sample edge features
    gen_bus      [G]               — bus index of each generator
    edge_mask    [B, E]            — 1=active, 0=broken line (per sample)
    uc_target    [B, G, T]         — ground truth (optional, for training)
    lmp_target   [B, N, T]         — prices in original units (optional)
    """
    node_feat_s: torch.Tensor   # [B, G, F_node_s]
    node_feat_d: torch.Tensor   # [B, T, N, F_node_d]
    edge_index:  torch.Tensor   # [2, E]
    edge_attr:   torch.Tensor   # [B, E, F_edge]
    gen_bus:     torch.Tensor   # [G]
    edge_mask:   torch.Tensor   # [B, E]
    uc_target:   torch.Tensor = None  # [B, G, T]
    lmp_target:  torch.Tensor = None  # [B, N, T], original price units

    def to(self, device) -> "STGCNInput":
        return STGCNInput(
            node_feat_s = self.node_feat_s.to(device),
            node_feat_d = self.node_feat_d.to(device),
            edge_index  = self.edge_index.to(device),
            edge_attr   = self.edge_attr.to(device),
            gen_bus     = self.gen_bus.to(device),
            edge_mask   = self.edge_mask.to(device),
            uc_target   = self.uc_target.to(device) if self.uc_target is not None else None,
            lmp_target  = self.lmp_target.to(device) if self.lmp_target is not None else None,
        )


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
class DyncGraph:
    """
    Data format for dynamic graph input.

    x          [B, n_period, N, f_node]  — per sample node features
    edge_index [2, E]                    — shared across all samples
    edge_mask  [B, E]                    — 1=active, 0=broken line (per sample)
    """
    x:          torch.Tensor   # [B, n_period, N, f_node]
    edge_index: torch.Tensor   # [2, E]
    edge_mask:  torch.Tensor   # [B, E]

    @property
    def f_node(self) -> int:
        return self.x.shape[-1]


@dataclass
class FusionGraph:
    """
    Data format for fusion module input.

    x_static:  [B, N, f_in]  — output of StaticModule
    x_dynamic: [B, N, f_in]  — output of DyncModule
    edge_index [2, E]         — shared across all samples
    edge_mask  [B, E]         — 1=active, 0=broken line (per sample)
    gen_bus:   [G]            — bus index of each generator
    """
    x_static:   torch.Tensor  # [B, N, f_in]
    x_dynamic:  torch.Tensor  # [B, N, f_in]
    edge_index: torch.Tensor  # [2, E]
    edge_mask:  torch.Tensor  # [B, E]
    gen_bus:    torch.Tensor  # [G]
