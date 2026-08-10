import torch
import torch.nn as nn
from dataclasses import dataclass

from lib.layers.layer_nn_conv import StaticGraph, NNConvConfig, NNConvLayer
from lib.layers.layer_pre_norm import PreNormLayer


@dataclass
class StaticModuleConfig:
    n_node:   int        # number of buses
    f_node:   int        # input node feature dim
    f_edge:   int        # edge feature dim
    n_layers: int = 2    # number of layers in each NNConv's edge_mlp
    f_hidden: int = 64   # hidden channels throughout
    f_out:    int = 64   # final output dim per bus


class StaticModule(nn.Module):
    """
    input:  StaticGraph  —  node_feat [B, N, f_node],  edge_attr [B, E, f_edge],  edge_index [2, E]
    output: [B, N, f_out]
    """

    def __init__(self, config: StaticModuleConfig):
        super().__init__()

        self.node_norm = PreNormLayer(config.f_node)
        self.edge_norm = PreNormLayer(config.f_edge)

        self.nn_conv1 = NNConvLayer(NNConvConfig(
            n_node   = config.n_node,
            f_in     = config.f_node,
            f_edge   = config.f_edge,
            n_layers = config.n_layers,
            f_hidden = config.f_hidden,
        ))
        self.nn_conv2 = NNConvLayer(NNConvConfig(
            n_node   = config.n_node,
            f_in     = config.f_hidden,
            f_edge   = config.f_edge,
            n_layers = config.n_layers,
            f_hidden = config.f_hidden,
        ))

        self.proj = nn.Linear(config.f_hidden, config.f_out)

    def prenorm_update_on(self):
        self.node_norm.update_on()
        self.edge_norm.update_on()

    def prenorm_update_off(self):
        self.node_norm.update_off()
        self.edge_norm.update_off()

    def forward(self, data: StaticGraph) -> torch.Tensor:   # [B, N, f_out]
        data = StaticGraph(
            edge_index = data.edge_index,
            edge_attr  = self.edge_norm(data.edge_attr),
            node_feat  = self.node_norm(data.node_feat),
            edge_mask  = data.edge_mask,
        )

        x = torch.relu(self.nn_conv1(data))              # [B, N, f_hidden]
        mid = StaticGraph(
            edge_index = data.edge_index,
            edge_attr  = data.edge_attr,
            node_feat  = x,
            edge_mask  = data.edge_mask,
        )
        x = torch.relu(self.nn_conv2(mid))               # [B, N, f_hidden]
        return self.proj(x)                              # [B, N, f_out]
