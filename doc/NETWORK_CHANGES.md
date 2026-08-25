# Network Changes

## 1. Previous Architecture

The current STGCN implementation is defined mainly in `lib/stgcn.py`, with reusable layers and submodules under `lib/layers/`.

### 1.1 Input Representation

The model consumes one `STGCNInput` dataclass and produces UC logits:

- `node_feat_s [B, G, F_node_s]`: static generator features before bus scatter.
- `node_feat_d [B, T, N, F_node_d]`: dynamic bus-level time-series features.
- `edge_index [2, E]`: shared directed graph topology. Each physical line is represented in both directions.
- `edge_attr [B, E, F_edge]`: per-sample edge features.
- `gen_bus [G]`: bus index for each generator.
- `edge_mask [B, E]`: per-sample active-line mask, where `1` means active and `0` means removed or under maintenance.
- `uc_target [B, G, T]`: binary UC schedule target, used during training.
- Output `logits [B, G, T]`: one binary on/off logit per generator and time period.

`lib/data_loader.py` builds this input from `UCData`:

- Static generator features include `pmax`, `pmin`, ramp up/down, min up/down, production cost, and startup cost.
- Dynamic node features include bus demand by time period plus bid-price segment features aggregated from generators to their buses.
- Edge features include line reactance and line flow limit. For SCUC data, an additional contingency-monitor indicator is appended.
- Maintenance lines are encoded through `edge_mask`, zeroing both directed copies of the affected line.

### 1.2 Top-Level STGCN Flow

`STGCN.forward()` has three branches/stages:

```text
node_feat_s [B, G, F_node_s]
-> StaticModule
-> x_static [B, N, H]

node_feat_d [B, T, N, F_node_d]
-> DyncModule
-> x_dynamic [B, N, H]

x_static [B, N, H] + x_dynamic [B, N, H]
-> FusionModule
-> logits [B, G, T]
```

Default model hyperparameters are inferred from the processed data by `STGCN.default_config()`:

- `f_hidden = 64`
- `n_layers = 2`
- `k_t = 3`
- `k_s = 3`

The input dimensions and top-level configuration parameters correspond as follows: `n_node = N`, `n_gen = G`, `n_period = T`, `f_hidden = H`, `f_node_s = F_node_s`, `f_node_d = F_node_d`, and `f_edge = F_edge`. The three submodules receive these configured dimensions as follows:

- `StaticModule`: `f_node = f_node_s`, `f_out = f_hidden`.
- `DyncModule`: `f_node = f_node_d`, `f_out = f_hidden`.
- `FusionModule`: `f_in = f_hidden`.

### 1.3 Static Branch

The static branch is implemented by `StaticModule` in `lib/layers/module_static.py`. It converts generator-level static features into bus-level static graph embeddings.

```text
node_feat_s [B, G, F_node_s]
-> scatter generator features to bus nodes with gen_bus [B, N, F_node_s]
-> node PreNormLayer [B, N, F_node_s]

edge_attr [B, E, F_edge]
-> edge PreNormLayer [B, E, F_edge]


first NNConvLayer (f_in = f_node = F_node_s):
edge_attr [B, E, f_edge]
-> edge_mlp hidden Linear + ReLU, repeated n_layers times [B, E, H]
-> edge_mlp output Linear [B, E, f_in * H]
-> reshape W [B, E, f_in, H]

node features [B, N, f_in]
-> gather source-node features x_src [B, E, f_in]

x_src [B, E, f_in] and W [B, E, f_in, H]
-> einsum message [B, E, H]
-> apply edge_mask [B, E, H]
-> scatter_add to destination nodes [B, N, H]
-> ReLU [B, N, H]


second NNConvLayer (f_in = H):
edge_attr [B, E, f_edge]
-> edge_mlp hidden Linear + ReLU, repeated n_layers times [B, E, H]
-> edge_mlp output Linear [B, E, f_in * H]
-> reshape W [B, E, f_in, H] = [B, E, H, H]

node features [B, N, f_in] = [B, N, H]
-> gather source-node features x_src [B, E, f_in]

x_src [B, E, f_in] and W [B, E, f_in, H]
-> einsum message [B, E, H]
-> apply edge_mask [B, E, H]
-> scatter_add to destination nodes [B, N, H]
-> ReLU [B, N, H]
-> Linear projection [B, N, f_out]
-> x_static [B, N, f_out] = [B, N, H]
```

The two `NNConvLayer` blocks are edge-conditioned graph convolutions. Each block uses `edge_attr` to generate and reshape one weight matrix `W [B, E, f_in, H]` per directed edge, applies `edge_mask` to remove inactive lines, and aggregates source-node messages to destination buses. The final projection has shape `[B, N, f_out]`; it is also `[B, N, H]` in `STGCN` because `StaticModuleConfig.f_out = STGCNConfig.f_hidden`.

### 1.4 Dynamic Branch

The dynamic branch is implemented by `DyncModule` in `lib/layers/module_dync.py`. It converts bus-level time-series features into one dynamic embedding per bus.

```text
node_feat_d [B, T, N, F_node_d]
-> dynamic PreNormLayer [B, T, N, F_node_d]
-> STConvBlock [B, T, N, H]
-> STConvBlock [B, T, N, H]

-> permute for temporal collapse [B, H, T, N]
-> full-horizon Conv2d [B, 2 * H, 1, N]
-> chunk p, q [B, H, 1, N] each
-> p * sigmoid(q) [B, H, 1, N]
-> squeeze and permute [B, N, H]

projection MLP:
-> hidden Linear [B, N, H]
-> ReLU + Dropout [B, N, H]
-> output Linear [B, N, f_out]
-> x_dynamic [B, N, f_out] = [B, N, H]
```

Each `STConvBlock` follows:

```text
input [B, T, N, f_in]
-> permute and causal pad [B, f_in, T + k_t - 1, N]
-> first TemporalConv Conv2d [B, 2 * f_out, T, N]
-> chunk p, q [B, f_out, T, N] each
-> p * sigmoid(q), then permute [B, T, N, f_out]
-> reshape for graph convolution [B * T, N, f_out]
-> ChebConvLayer [B * T, N, H]
-> ReLU [B * T, N, H]
-> reshape [B, T, N, H]
-> permute and causal pad [B, H, T + k_t - 1, N]
-> second TemporalConv Conv2d [B, 2 * f_out, T, N]
-> chunk p, q [B, f_out, T, N] each
-> p * sigmoid(q), then permute [B, T, N, f_out]
-> LayerNorm [B, T, N, f_out]
```

The temporal convolutions preserve the time dimension with causal padding. The `ChebConvLayer` performs masked spectral graph convolution at each time step, using `edge_mask [B * T, E]` after expanding and reshaping `edge_mask [B, E]`. Both `STConvBlock` instances set `f_out = f_hidden`; the first sets `f_in = f_node = F_node_d`, while the second sets `f_in = f_hidden`. The projection output is `[B, N, f_out]`, which becomes `x_dynamic [B, N, H]` because `DyncModuleConfig.f_out = STGCNConfig.f_hidden`.

### 1.5 Fusion Branch

The fusion branch is implemented by `FusionModule` in `lib/layers/module_fusion.py`. It combines static and dynamic bus embeddings, then converts bus-level features back to generator-level time predictions.

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * H]
-> ChebConvLayer [B, N, H]
-> ReLU [B, N, H]
-> select generator buses with gen_bus [B, G, H]

time expansion MLP:
-> hidden Linear [B, G, H]
-> ReLU + Dropout [B, G, H]
-> output Linear [B, G, T * H]
-> reshape [B, G, T, H]

output MLP:
-> hidden Linear [B, G, T, H]
-> ReLU + Dropout [B, G, T, H]
-> output Linear [B, G, T, 1]
-> squeeze [B, G, T]
-> logits [B, G, T]
```

The Fusion `ChebConvLayer` propagates information across the masked bus graph after static and dynamic embeddings have been combined. Its input is `[B, N, 2 * f_in]`, which is also `[B, N, 2 * H]` because `FusionModuleConfig.f_in = STGCNConfig.f_hidden`. The generator-bus selection maps bus embeddings back to generator-level representations before producing one commitment logit per generator and time period. Here `n_period = T`.

## 2. Completed Changes

### 2.1 STGCN V2.0 (`stgcn_v2`)

V2.0 keeps the V1 input/output contract and the existing Static and Dynamic
branches unchanged. The only structural change is in the Fusion branch, where
`ChebConvLayer` is replaced by a masked multi-head `GraphAttentionLayer`:

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * H]
-> masked multi-head GraphAttentionLayer [B, N, H]
-> generator-bus selection and output MLPs
-> logits [B, G, T]
```

Attention is normalized over each destination bus's active incoming edges using
`edge_mask`. Each bus also has an always-active self-loop so that its own
representation is retained and isolated buses remain valid.

### 2.2 STGCN V2.1 (`stgcn_v2.1`)

V2.1 keeps the V1 input/output contract and Static branch unchanged, and retains
the V2.0 attention-based Fusion branch. The Dynamic branch now also replaces the
spatial `ChebConvLayer` in both `STConvBlock`s with masked multi-head
`GraphAttentionLayer`s applied independently at each time step:

```text
node_feat_d [B, T, N, F_node_d]
-> dynamic PreNormLayer
-> two blocks of TemporalConv -> GraphAttentionLayer -> TemporalConv
-> full-horizon temporal collapse and projection MLP
-> x_dynamic [B, N, H]

x_static [B, N, H] + x_dynamic [B, N, H]
-> attention-based Fusion branch
-> logits [B, G, T]
```

Both Dynamic and Fusion attention use `edge_mask` and always-active self-loops.
The default configuration uses four attention heads.

## 3. Planned Changes

### 3.1 Graph Attention

Graph attention can be introduced in the current STGCN architecture in two main places.

The first target is the Fusion branch. The current Fusion branch combines static and dynamic node embeddings, then applies one `ChebConvLayer`:

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * f_in]
-> ChebConvLayer [B, N, H]
-> generator-bus selection [B, G, H]
-> time expansion MLP [B, G, T * H]
-> reshape [B, G, T, H]
-> output MLP [B, G, T, 1]
-> squeeze [B, G, T]
```

This `ChebConvLayer` is the best initial replacement point:

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * f_in]
-> GraphAttentionLayer [B, N, H]
-> generator-bus selection [B, G, H]
-> time expansion MLP [B, G, T * H]
-> reshape [B, G, T, H]
-> output MLP [B, G, T, 1]
-> squeeze [B, G, T]
```

Reasons:

- Fusion already has both static and dynamic information, so attention can learn which neighboring buses matter most for the final UC decision.
- The change is localized to `FusionModule`, with limited impact on the rest of the model.
- It is cheaper and lower-risk than applying attention inside every dynamic time step.

The second target is the Dynamic branch. Each `STConvBlock` currently follows:

```text
TemporalConv
-> ChebConvLayer
-> TemporalConv
-> LayerNorm
```

The middle `ChebConvLayer` could be replaced with a graph attention layer:

```text
TemporalConv
-> GraphAttentionLayer
-> TemporalConv
-> LayerNorm
```

This would let the model learn attention-based spatial message passing at each time step. However, the graph layer receives reshaped input `[B * T, N, f_out]` and produces `[B * T, N, H]`, so this option is more expensive, especially for large cases such as `case2383` and `case6515`.

The Static branch is not the best first replacement target. It currently uses `NNConvLayer`, where `edge_attr` is passed through an edge MLP to generate per-edge weight matrices. This directly uses line reactance, flow limits, and SCUC contingency indicators. Replacing it with a standard graph attention layer could lose edge-feature conditioning unless the new attention layer is explicitly edge-aware.

Suggested implementation order:

1. Replace or add attention in the Fusion branch first.
2. If Fusion attention improves performance, test attention inside the Dynamic branch `STConvBlock`.
3. Modify the Static branch only if using an edge-aware graph attention design.

### 3.2 Node Features

TBD

### 3.3 Multi-Task Learning

- Locational marginal price
- Power output

### 3.4 Topo & Scuc Together

TBD

## 4. Architecture Decisions

TBD
