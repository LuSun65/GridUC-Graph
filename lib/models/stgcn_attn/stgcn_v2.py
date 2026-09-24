import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.models.model_input import STGCNInput
from lib.models.model_input import StaticGraph
from lib.models.stgcn.module_static import StaticModule, StaticModuleConfig
from lib.models.stgcn.module_dync import DyncModule, DyncModuleConfig
from lib.models.model_input import DyncGraph
from lib.models.model_input import FusionGraph
from lib.models.stgcn_attn.module_fusion_attention import (
    AttentionFusionModule,
    AttentionFusionModuleConfig,
)


@dataclass
class STGCNV2Config:
    n_node: int
    n_gen: int
    n_period: int
    f_node_s: int
    f_node_d: int
    f_edge: int
    f_hidden: int = 64
    n_layers: int = 2
    k_t: int = 3
    attention_heads: int = 4
    attention_dropout: float = 0.0
    attention_negative_slope: float = 0.2


class STGCN_V2(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for UC prediction.

    Static branch (NNConv) + Dynamic branch (ST-Conv)
    -> Fusion graph attention -> [B, G, T]

    input:  STGCNInput
    output: [B, G, n_period]  — logits (training) or sigmoid probs (eval)
    """

    def __init__(self, config: STGCNV2Config):
        super().__init__()
        # Keep the in-progress trainer compatible with a V1 STGCNConfig while
        # ensuring V2 checkpoints always store the complete V2 configuration.
        if not isinstance(config, STGCNV2Config):
            config = STGCNV2Config(
                n_node=config.n_node,
                n_gen=config.n_gen,
                n_period=config.n_period,
                f_node_s=config.f_node_s,
                f_node_d=config.f_node_d,
                f_edge=config.f_edge,
                f_hidden=config.f_hidden,
                n_layers=config.n_layers,
                k_t=config.k_t,
            )
        self.config = config

        self.static_branch = StaticModule(StaticModuleConfig(
            n_node   = config.n_node,
            f_node   = config.f_node_s,
            f_edge   = config.f_edge,
            n_layers = config.n_layers,
            f_hidden = config.f_hidden,
            f_out    = config.f_hidden,
        ))

        self.dynamic_branch = DyncModule(DyncModuleConfig(
            n_period = config.n_period,
            n_node   = config.n_node,
            f_node   = config.f_node_d,
            f_hidden = config.f_hidden,
            f_out    = config.f_hidden,
            k_t      = config.k_t,
        ))

        fusion_cfg = AttentionFusionModuleConfig(
            n_node   = config.n_node,
            n_gen    = config.n_gen,
            n_period = config.n_period,
            f_in     = config.f_hidden,
            f_hidden = config.f_hidden,
            n_layers = config.n_layers,
            n_heads  = config.attention_heads,
            attention_dropout = config.attention_dropout,
            attention_negative_slope = config.attention_negative_slope,
        )
        self.fusion = AttentionFusionModule(fusion_cfg)

    @classmethod
    def default_config(cls, data: STGCNInput) -> STGCNV2Config:
        """Infer dimensions from input data and retain V2 hyperparameter defaults."""
        B, G, f_node_s = data.node_feat_s.shape
        B, T, N, f_node_d = data.node_feat_d.shape
        f_edge = data.edge_attr.shape[-1]
        return STGCNV2Config(
            n_node      = N,
            n_gen       = G,
            n_period    = T,
            f_node_s    = f_node_s,
            f_node_d    = f_node_d,
            f_edge      = f_edge,
        )

    def prenorm_update_on(self):
        self.static_branch.prenorm_update_on()
        self.dynamic_branch.prenorm_update_on()

    def prenorm_update_off(self):
        self.static_branch.prenorm_update_off()
        self.dynamic_branch.prenorm_update_off()

    def _gen_to_nodal(
        self,
        gen_feat: torch.Tensor,   # [B, G, F_node_s]
        gen_bus:  torch.Tensor,   # [G]
        n_node:   int,
    ) -> torch.Tensor:            # [B, n_node, F_node_s]
        B, G, F = gen_feat.shape
        x = torch.zeros(B, n_node, F, device=gen_feat.device, dtype=gen_feat.dtype)
        x[:, gen_bus, :] = gen_feat
        return x

    def forward(self, data: STGCNInput) -> torch.Tensor:   # [B, G, n_period]
        # Static branch: scatter gen features to nodes, then NNConv
        node_feat_s = self._gen_to_nodal(data.node_feat_s, data.gen_bus, self.config.n_node)  # [B, N, f_node_s]
        x_static = self.static_branch(StaticGraph(
            edge_index = data.edge_index,
            edge_attr  = data.edge_attr,
            node_feat  = node_feat_s,
            edge_mask  = data.edge_mask,
        ))  # [B, N, f_hidden]

        # Dynamic branch: [B, T, N, f_node_d]
        x_dynamic = self.dynamic_branch(DyncGraph(
            x          = data.node_feat_d,
            edge_index = data.edge_index,
            edge_mask  = data.edge_mask,
        ))  # [B, N, f_hidden]

        # Fusion -> per-generator per-timestep prediction
        logits = self.fusion(FusionGraph(
            x_static   = x_static,
            x_dynamic  = x_dynamic,
            edge_index = data.edge_index,
            edge_mask  = data.edge_mask,
            gen_bus    = data.gen_bus,
        ))  # [B, G, n_period]
        return logits
