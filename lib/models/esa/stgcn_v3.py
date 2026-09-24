"""V3 model interface; ESA branches are not implemented yet."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.model_input import STGCNInput


@dataclass
class STGCNV3Config:
    n_node: int
    n_gen: int
    n_period: int
    f_node_s: int
    f_node_d: int
    f_edge: int
    f_hidden: int = 64
    k_t: int = 3
    n_heads: int = 4


class STGCN_V3(nn.Module):
    """ESA model skeleton with the shared input and UC logits [B, G, T] interface."""

    def __init__(self, config: STGCNV3Config):
        super().__init__()
        self.config = config
        # Fail before training can treat this parameter-free skeleton as a model.
        raise NotImplementedError("STGCN V3 is registered but its ESA implementation is not yet available")

    @classmethod
    def default_config(cls, data: STGCNInput) -> STGCNV3Config:
        _, n_gen, f_node_s = data.node_feat_s.shape
        _, n_period, n_node, f_node_d = data.node_feat_d.shape
        return STGCNV3Config(
            n_node=n_node,
            n_gen=n_gen,
            n_period=n_period,
            f_node_s=f_node_s,
            f_node_d=f_node_d,
            f_edge=data.edge_attr.shape[-1],
        )

    def forward(self, data: STGCNInput) -> torch.Tensor:
        raise NotImplementedError("STGCN V3 forward is not implemented")
