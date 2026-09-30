"""Full-horizon edge set attention with endpoint-mean UC readout."""
from dataclasses import dataclass

import torch
from torch import nn

from lib.models.common.layer_pre_norm import PreNormLayer
from lib.models.model_input import STGCNInput
from .encoder import ESAEncoder, UPSTREAM_COMMIT, edge_adjacency
from .mlp import SmallMLP


@dataclass
class ESAConfig:
    n_node: int
    n_gen: int
    n_period: int
    f_node_s: int
    f_node_d: int
    f_edge: int
    f_hidden: int = 64
    n_heads: int = 4
    layer_types: str = 'MSMS'
    mlp_hidden: int = 128
    mlp_layers: int = 2
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    mlp_dropout: float = 0.0
    gradient_checkpointing: bool = False
    upstream_commit: str = UPSTREAM_COMMIT
    normalization: str = 'pre_ln'
    attention_backend: str = 'torch_sdpa'
    input_encoding: str = 'full_horizon_flatten'
    edge_encoding: str = 'paired_halves_first_direction'
    readout: str = 'endpoint_mlp_mean'
    generator_order: str = 'input_gen_bus'


class EdgeToNode(nn.Module):
    def __init__(self, config, node_width):
        super().__init__()
        self.width = config.f_hidden
        args = dict(inter_dim=config.mlp_hidden, out_dim=self.width,
                    num_layers=config.mlp_layers, dropout_p=config.mlp_dropout)
        self.message = SmallMLP(in_dim=self.width + node_width, **args)
        self.node = SmallMLP(in_dim=node_width + self.width, **args)

    def aggregate(self, edge_features, node_features, edges):
        batch, nodes, _ = node_features.shape
        # Both endpoints receive messages even though each line has one token.
        endpoints = edges.reshape(-1)
        tokens = edge_features.repeat(1, 2, 1)
        messages = self.message(torch.cat((tokens, node_features[:, endpoints]), dim=-1))
        summed = messages.new_zeros(batch, nodes, self.width)
        summed.index_add_(1, endpoints, messages)
        degree = torch.bincount(endpoints, minlength=nodes).to(summed.dtype)
        return summed / degree.clamp_min(1)[None, :, None]

    def forward(self, edge_features, node_features, edges):
        mean = self.aggregate(edge_features, node_features, edges)
        return self.node(torch.cat((node_features, mean), dim=-1))


class ESA(nn.Module):
    """STGCNInput -> UC logits [B,G,T]; calibration skips the attention stack."""
    def __init__(self, config: ESAConfig):
        super().__init__()
        for key in ('n_node', 'n_gen', 'n_period', 'f_node_s', 'f_node_d',
                    'f_edge', 'f_hidden', 'n_heads', 'mlp_hidden'):
            if getattr(config, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if config.f_hidden % config.n_heads:
            raise ValueError('f_hidden must be divisible by n_heads')
        if not config.layer_types or set(config.layer_types) - set('MS'):
            raise ValueError('layer_types must be a nonempty sequence of M/S')
        if config.mlp_layers < 2:
            raise ValueError('mlp_layers must be >= 2 to retain linear output logits')
        for key in ('attention_dropout', 'residual_dropout', 'mlp_dropout'):
            if not 0 <= getattr(config, key) < 1:
                raise ValueError(f'{key} must be in [0, 1)')
        expected = dict(upstream_commit=UPSTREAM_COMMIT, normalization='pre_ln',
                        attention_backend='torch_sdpa', input_encoding='full_horizon_flatten',
                        edge_encoding='paired_halves_first_direction',
                        readout='endpoint_mlp_mean', generator_order='input_gen_bus')
        for key, value in expected.items():
            if getattr(config, key) != value:
                raise ValueError(f'Unsupported {key}: {getattr(config, key)!r}')
        self.config = config
        width = config.f_node_s + config.n_period * config.f_node_d
        self.node_norm = PreNormLayer(width)
        self.edge_norm = PreNormLayer(config.f_edge)
        self.edge_embedding = SmallMLP(2 * width + config.f_edge, config.mlp_hidden,
                                       config.f_hidden, config.mlp_dropout, config.mlp_layers)
        self.encoder = ESAEncoder(config)
        self.readout = EdgeToNode(config, width)
        self.uc_head = SmallMLP(config.f_hidden, config.mlp_hidden, config.n_period,
                               config.mlp_dropout, config.mlp_layers)
        # Shared topology only; no B/T-expanded masks and no checkpoint pollution.
        self.register_buffer('_mask_edges', torch.empty(2, 0, dtype=torch.long), persistent=False)
        self.register_buffer('_mask', torch.empty(0, 0, dtype=torch.bool), persistent=False)

    @classmethod
    def default_config(cls, data: STGCNInput):
        return ESAConfig(data.node_feat_d.shape[2], data.node_feat_s.shape[1],
                         data.node_feat_d.shape[1], data.node_feat_s.shape[2],
                         data.node_feat_d.shape[3], data.edge_attr.shape[2])

    def prenorm_update_on(self):
        self.node_norm.update_on()
        self.edge_norm.update_on()

    def prenorm_update_off(self):
        self.node_norm.update_off()
        # An edgeless graph has no edge statistics; identity is appropriate.
        if self.edge_norm._count.item():
            self.edge_norm.update_off()
        else:
            self.edge_norm.is_updating = False

    def prepare_inputs(self, data):
        """Validate paired physical lines and map raw features before PreNorm."""
        c = self.config
        batch = data.node_feat_s.shape[0]
        if data.node_feat_s.shape != (batch, c.n_gen, c.f_node_s):
            raise ValueError('node_feat_s shape does not match ESA config')
        if data.node_feat_d.shape != (batch, c.n_period, c.n_node, c.f_node_d):
            raise ValueError('node_feat_d shape does not match ESA config')
        if data.gen_bus.dtype != torch.long or data.gen_bus.shape != (c.n_gen,):
            raise ValueError('gen_bus must be a long tensor of shape [G]')
        if (data.gen_bus < 0).any() or (data.gen_bus >= c.n_node).any():
            raise ValueError('gen_bus out of range')
        if data.gen_bus.unique().numel() != c.n_gen:
            raise ValueError('ESA requires at most one generator per bus')
        edges = data.edge_index
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] % 2:
            raise ValueError('edge_index must contain paired halves with shape [2,2L]')
        lines = edges.shape[1] // 2
        if (edges < 0).any() or (edges >= c.n_node).any():
            raise ValueError('edge endpoint out of range')
        if not torch.equal(edges[:, lines:], edges[:, :lines].flip(0)):
            raise ValueError('reverse edges must match the first half in physical-line order')
        if data.edge_attr.shape != (batch, 2 * lines, c.f_edge):
            raise ValueError('edge_attr shape does not match ESA config')
        if not torch.equal(data.edge_attr[:, :lines], data.edge_attr[:, lines:]):
            raise ValueError('reverse line attributes/maintenance flags must match')
        if hasattr(data, 'edge_mask'):
            raise ValueError('retired edge_mask input; rerun process_data')
        static = data.node_feat_s.new_zeros(batch, c.n_node, c.f_node_s)
        static[:, data.gen_bus] = data.node_feat_s
        dynamic = data.node_feat_d.permute(0, 2, 1, 3).reshape(batch, c.n_node, -1)
        return torch.cat((static, dynamic), dim=-1), edges[:, :lines], data.edge_attr[:, :lines]

    def topology_mask(self, edges):
        if self._mask.shape[0] != edges.shape[1] or not torch.equal(self._mask_edges, edges):
            self._mask_edges = edges.detach().clone()
            self._mask = edge_adjacency(edges)
        return self._mask[None, None]

    def forward(self, data: STGCNInput):
        nodes, edges, attrs = self.prepare_inputs(data)
        nodes = self.node_norm(nodes)
        attrs = self.edge_norm(attrs)
        if self.node_norm.is_updating:
            # Trainer ignores these values during training-set-only calibration.
            return nodes.new_zeros(nodes.shape[0], self.config.n_gen, self.config.n_period)
        src, dst = edges
        tokens = self.edge_embedding(torch.cat((nodes[:, src], nodes[:, dst], attrs), dim=-1))
        mask = self.topology_mask(edges) if 'M' in self.config.layer_types else None
        encoded = self.encoder(tokens, mask)
        node_repr = self.readout(encoded, nodes, edges)
        return self.uc_head(node_repr[:, data.gen_bus])
