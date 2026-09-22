import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.models.model_input import STGCNInput

from lib.models.model_input import StaticGraph
from lib.models.v1.module_static import StaticModule, StaticModuleConfig
from lib.models.v1.module_dync import DyncModule, DyncModuleConfig
from lib.models.model_input import DyncGraph
from lib.models.v1.module_fusion import FusionModule, FusionModuleConfig
from lib.models.model_input import FusionGraph


@dataclass
class STGCNConfig:
    n_node:   int        # number of buses
    n_gen:    int        # number of generators
    n_period: int        # number of time steps
    f_node_s: int        # static branch node feature dim (generator features)
    f_node_d: int        # dynamic branch node feature dim
    f_edge:   int        # edge feature dim
    f_hidden: int = 64      # hidden channels throughout
    n_layers: int = 2       # MLP depth in static edge_mlp and fusion MLPs
    k_t:      int = 3       # temporal kernel size
    k_s:      int = 3       # Chebyshev order


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for UC prediction.

    Static branch (NNConv)  +  Dynamic branch (ST-Conv)  ->  Fusion  ->  [B, G, T]

    input:  STGCNInput
    output: [B, G, n_period]  — logits (training) or sigmoid probs (eval)
    """

    def __init__(self, config: STGCNConfig):
        super().__init__()
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

        fusion_cfg = FusionModuleConfig(
            n_node   = config.n_node,
            n_gen    = config.n_gen,
            n_period = config.n_period,
            f_in     = config.f_hidden,
            f_hidden = config.f_hidden,
            k_s      = config.k_s,
            n_layers = config.n_layers,
        )
        self.fusion = FusionModule(fusion_cfg)

    @classmethod
    def default_config(cls, data: STGCNInput) -> STGCNConfig:
        """Infer STGCNConfig dimensions from a batched STGCNInput, keeping all hyperparams at defaults."""
        B, G, f_node_s = data.node_feat_s.shape
        B, T, N, f_node_d = data.node_feat_d.shape
        f_edge = data.edge_attr.shape[-1]
        return STGCNConfig(
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
