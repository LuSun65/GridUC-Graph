from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.layers.layer_nn_conv import NNConvConfig, NNConvLayer, StaticGraph
from lib.layers.layer_pre_norm import PreNormLayer


@dataclass
class StaticModuleV22Config:
    n_node: int
    f_node: int
    f_edge: int
    n_layers: int = 2
    f_hidden: int = 64
    f_out: int = 64


class DegreeNormalizedNNConvLayer(NNConvLayer):
    """NNConv whose incoming messages are averaged by active node degree."""

    def forward(self, data: StaticGraph) -> torch.Tensor:
        src, dst = data.edge_index[0], data.edge_index[1]
        edge_count = data.edge_attr.shape[1]

        weights = self.edge_mlp(data.edge_attr).view(
            -1, edge_count, self.f_in, self.f_hidden
        )
        source_features = self._gather(data.node_feat, src)
        messages = torch.einsum(
            'bef, befo -> beo', source_features, weights
        )
        active_edges = data.edge_mask.unsqueeze(-1).to(messages.dtype)
        messages = messages * active_edges
        summed_messages = self._scatter_add(messages, dst)

        batch_size = messages.shape[0]
        degree = torch.zeros(
            batch_size,
            self.n_node,
            device=messages.device,
            dtype=messages.dtype,
        )
        degree.scatter_add_(
            1,
            dst.unsqueeze(0).expand(batch_size, edge_count),
            data.edge_mask.to(messages.dtype),
        )
        return summed_messages / degree.clamp_min(1.0).unsqueeze(-1)


class ResidualNNConvBlock(nn.Module):
    """NNConv update with a projected skip connection and LayerNorm."""

    def __init__(self, config: NNConvConfig):
        super().__init__()
        self.graph_conv = DegreeNormalizedNNConvLayer(config)
        self.skip = (
            nn.Identity()
            if config.f_in == config.f_hidden
            else nn.Linear(config.f_in, config.f_hidden, bias=False)
        )
        self.alpha = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(config.f_hidden)

    def forward(self, data: StaticGraph) -> torch.Tensor:
        residual = self.skip(data.node_feat)
        update = self.graph_conv(data)
        return self.norm(residual + self.alpha * update)


class StaticModuleV2_2(nn.Module):
    """V2.2 static encoder with six NNConv layers."""

    N_NN_CONV = 6

    def __init__(self, config: StaticModuleV22Config):
        super().__init__()
        self.node_norm = PreNormLayer(config.f_node)
        self.edge_norm = PreNormLayer(config.f_edge)

        self.nn_convs = nn.ModuleList([
            ResidualNNConvBlock(NNConvConfig(
                n_node=config.n_node,
                f_in=config.f_node if index == 0 else config.f_hidden,
                f_edge=config.f_edge,
                n_layers=config.n_layers,
                f_hidden=config.f_hidden,
            ))
            for index in range(self.N_NN_CONV)
        ])
        self.proj = nn.Linear(config.f_hidden, config.f_out)

    def prenorm_update_on(self):
        self.node_norm.update_on()
        self.edge_norm.update_on()

    def prenorm_update_off(self):
        self.node_norm.update_off()
        self.edge_norm.update_off()

    def forward(self, data: StaticGraph) -> torch.Tensor:
        edge_attr = self.edge_norm(data.edge_attr)
        x = self.node_norm(data.node_feat)

        for block in self.nn_convs:
            x = block(StaticGraph(
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                node_feat=x,
                edge_mask=data.edge_mask,
            ))
        return self.proj(x)
