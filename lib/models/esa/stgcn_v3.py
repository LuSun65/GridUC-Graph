"""V3 edge set attention model for UC prediction."""

from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_pre_norm import PreNormLayer
from lib.models.esa.module_dync import DyncModuleV3, DyncModuleV3Config
from lib.models.esa.module_fusion import FusionModuleV3, FusionModuleV3Config
from lib.models.esa.module_static import StaticModuleV3, StaticModuleV3Config
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
    """ESA model with the shared STGCN input and [B, G, T] output."""

    def __init__(self, config: STGCNV3Config):
        super().__init__()
        self.config = config
        self.static_norm = PreNormLayer(config.f_node_s)
        self.dynamic_norm = PreNormLayer(config.f_node_d)
        self.edge_norm = PreNormLayer(config.f_edge)
        self.static_branch = StaticModuleV3(StaticModuleV3Config(
            config.f_node_s, config.f_edge, config.f_hidden, config.n_heads
        ))
        self.dynamic_branch = DyncModuleV3(DyncModuleV3Config(
            config.n_period, config.f_node_d, config.f_edge,
            config.f_hidden, config.k_t, config.n_heads,
        ))
        self.fusion = FusionModuleV3(FusionModuleV3Config(
            config.n_period, config.f_edge, config.f_hidden, config.n_heads
        ))

    @classmethod
    def default_config(cls, data: STGCNInput) -> STGCNV3Config:
        _, n_gen, f_node_s = data.node_feat_s.shape
        _, n_period, n_node, f_node_d = data.node_feat_d.shape
        return STGCNV3Config(
            n_node=n_node, n_gen=n_gen, n_period=n_period,
            f_node_s=f_node_s, f_node_d=f_node_d,
            f_edge=data.edge_attr.shape[-1],
        )

    def prenorm_update_on(self):
        self.static_norm.update_on()
        self.dynamic_norm.update_on()
        self.edge_norm.update_on()

    def prenorm_update_off(self):
        self.static_norm.update_off()
        self.dynamic_norm.update_off()
        self.edge_norm.update_off()

    def forward(self, data: STGCNInput) -> torch.Tensor:
        batch = data.node_feat_s.shape[0]
        # Generators on the same bus are summed; V1's indexed assignment keeps
        # only one of them, nondeterministically on CUDA.
        bus_features = data.node_feat_s.new_zeros(batch, self.config.n_node, self.config.f_node_s)
        bus_features.index_add_(1, data.gen_bus, self.static_norm(data.node_feat_s))
        edge_attr = self.edge_norm(data.edge_attr)
        x_static = self.static_branch(bus_features, data.edge_index, edge_attr, data.edge_mask)
        x_dynamic = self.dynamic_branch(
            self.dynamic_norm(data.node_feat_d), data.edge_index, edge_attr, data.edge_mask
        )
        return self.fusion(
            x_static, x_dynamic, data.edge_index, edge_attr, data.edge_mask, data.gen_bus
        )
