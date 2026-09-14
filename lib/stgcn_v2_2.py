from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.layers.layer_nn_conv import StaticGraph
from lib.layers.module_dync import DyncGraph
from lib.layers.module_dync_v2_2 import DyncModuleV2_2, DyncModuleV22Config
from lib.layers.module_fusion import FusionGraph
from lib.layers.module_fusion_v2_2 import (
    FusionModuleV2_2,
    FusionModuleV22Config,
)
from lib.layers.module_static_v2_2 import (
    StaticModuleV2_2,
    StaticModuleV22Config,
)
from lib.stgcn import STGCNInput


@dataclass
class STGCNV22Config:
    n_node: int
    n_gen: int
    n_period: int
    f_node_s: int
    f_node_d: int
    f_edge: int
    f_hidden: int = 64
    n_layers: int = 2
    k_t: int = 3
    k_s: int = 3
    attention_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class STGCN_V2_2(nn.Module):
    """Depth-expanded STGCN V2.2 for UC prediction."""

    def __init__(self, config: STGCNV22Config):
        super().__init__()
        if not isinstance(config, STGCNV22Config):
            config = STGCNV22Config(
                n_node=config.n_node,
                n_gen=config.n_gen,
                n_period=config.n_period,
                f_node_s=config.f_node_s,
                f_node_d=config.f_node_d,
                f_edge=config.f_edge,
                f_hidden=config.f_hidden,
                n_layers=config.n_layers,
                k_t=config.k_t,
                k_s=getattr(config, "k_s", 3),
                attention_heads=getattr(config, "attention_heads", 4),
                attention_dropout=getattr(config, "attention_dropout", 0.0),
                attention_negative_slope=getattr(
                    config, "attention_negative_slope", 0.2
                ),
            )
        self.config = config

        self.static_branch = StaticModuleV2_2(StaticModuleV22Config(
            n_node=config.n_node,
            f_node=config.f_node_s,
            f_edge=config.f_edge,
            n_layers=config.n_layers,
            f_hidden=config.f_hidden,
            f_out=config.f_hidden,
        ))
        self.dynamic_branch = DyncModuleV2_2(DyncModuleV22Config(
            n_period=config.n_period,
            n_node=config.n_node,
            f_node=config.f_node_d,
            f_hidden=config.f_hidden,
            f_out=config.f_hidden,
            k_t=config.k_t,
            k_s=config.k_s,
        ))
        self.fusion = FusionModuleV2_2(FusionModuleV22Config(
            n_node=config.n_node,
            n_gen=config.n_gen,
            n_period=config.n_period,
            f_in=config.f_hidden,
            f_hidden=config.f_hidden,
            n_layers=config.n_layers,
            n_heads=config.attention_heads,
            attention_dropout=config.attention_dropout,
            attention_negative_slope=config.attention_negative_slope,
        ))

    @classmethod
    def default_config(cls, data: STGCNInput) -> STGCNV22Config:
        _, n_gen, f_node_s = data.node_feat_s.shape
        _, n_period, n_node, f_node_d = data.node_feat_d.shape
        return STGCNV22Config(
            n_node=n_node,
            n_gen=n_gen,
            n_period=n_period,
            f_node_s=f_node_s,
            f_node_d=f_node_d,
            f_edge=data.edge_attr.shape[-1],
        )

    def prenorm_update_on(self):
        self.static_branch.prenorm_update_on()
        self.dynamic_branch.prenorm_update_on()

    def prenorm_update_off(self):
        self.static_branch.prenorm_update_off()
        self.dynamic_branch.prenorm_update_off()

    def _gen_to_nodal(
        self,
        gen_feat: torch.Tensor,
        gen_bus: torch.Tensor,
        n_node: int,
    ) -> torch.Tensor:
        batch_size, _, feature_size = gen_feat.shape
        x = torch.zeros(
            batch_size,
            n_node,
            feature_size,
            device=gen_feat.device,
            dtype=gen_feat.dtype,
        )
        x[:, gen_bus, :] = gen_feat
        return x

    def forward(self, data: STGCNInput) -> torch.Tensor:
        node_feat_s = self._gen_to_nodal(
            data.node_feat_s, data.gen_bus, self.config.n_node
        )
        x_static = self.static_branch(StaticGraph(
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            node_feat=node_feat_s,
            edge_mask=data.edge_mask,
        ))
        x_dynamic = self.dynamic_branch(DyncGraph(
            x=data.node_feat_d,
            edge_index=data.edge_index,
            edge_mask=data.edge_mask,
        ))
        return self.fusion(FusionGraph(
            x_static=x_static,
            x_dynamic=x_dynamic,
            edge_index=data.edge_index,
            edge_mask=data.edge_mask,
            gen_bus=data.gen_bus,
        ))
