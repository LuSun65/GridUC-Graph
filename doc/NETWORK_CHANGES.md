# Network Changes

Model family mapping: V1 → `lib/models/stgcn/`; stgcn_attn_0/stgcn_attn_1/stgcn_attn_2 →
`lib/models/stgcn_attn/`; ESA → `lib/models/esa/`; multitask → `lib/models/multitask/`.
See [the registry mapping table](README.md#model-names-and-directories) for
canonical names and aliases. Section 2.3 describes the implemented ESA encoder;
multitask is the implemented UC/LMP multitask model.

## 1. Previous Architecture

The current STGCN implementation is defined mainly in `lib/models/stgcn/stgcn.py`, with version-specific modules under `lib/models/stgcn/` and shared layers under `lib/models/common/`.

### 1.1 Input Representation

The model consumes one `STGCNInput` dataclass and produces UC logits:

- `node_feat_s [B, G, F_node_s]`: static generator features before bus scatter.
- `node_feat_d [B, T, N, F_node_d]`: dynamic bus-level time-series features.
- `edge_index [2, E]`: shared directed graph topology. Each physical line is represented in both directions.
- `edge_attr [B, E, F_edge]`: per-sample edge features.
- `gen_bus [G]`: bus index for each generator.
- `uc_target [B, G, T]`: binary UC schedule target, used during training.
- Output `logits [B, G, T]`: one binary on/off logit per generator and time period.

`lib/data_loader.py` builds this input from `UCData`:

- Static generator features include `pmax`, `pmin`, ramp up/down, min up/down, production cost, and startup cost.
- Dynamic node features include bus demand by time period plus bid-price segment features aggregated from generators to their buses.
- Edge features include line reactance and line flow limit. For SCUC data, an additional contingency-monitor indicator is appended.
- Maintenance uses soft outage: all connections are retained, both directed copies have doubled reactance, and their capacity limits remain unchanged, matching the TOPO solver.

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

The static branch is implemented by `StaticModule` in `lib/models/stgcn/module_static.py`. It converts generator-level static features into bus-level static graph embeddings.

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
-> scatter_add to destination nodes [B, N, H]
-> ReLU [B, N, H]
-> Linear projection [B, N, f_out]
-> x_static [B, N, f_out] = [B, N, H]
```

The two `NNConvLayer` blocks are edge-conditioned graph convolutions. Each block uses `edge_attr` to generate and reshape one weight matrix `W [B, E, f_in, H]` per directed edge, and aggregates source-node messages to destination buses. The final projection has shape `[B, N, f_out]`; it is also `[B, N, H]` in `STGCN` because `StaticModuleConfig.f_out = STGCNConfig.f_hidden`.

### 1.4 Dynamic Branch

The dynamic branch is implemented by `DyncModule` in `lib/models/stgcn/module_dync.py`. It converts bus-level time-series features into one dynamic embedding per bus.

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

The temporal convolutions preserve the time dimension with causal padding. The `ChebConvLayer` performs spectral graph convolution at each time step using the complete shared topology. Both `STConvBlock` instances set `f_out = f_hidden`; the first sets `f_in = f_node = F_node_d`, while the second sets `f_in = f_hidden`. The projection output is `[B, N, f_out]`, which becomes `x_dynamic [B, N, H]` because `DyncModuleConfig.f_out = STGCNConfig.f_hidden`.

### 1.5 Fusion Branch

The fusion branch is implemented by `FusionModule` in `lib/models/stgcn/module_fusion.py`. It combines static and dynamic bus embeddings, then converts bus-level features back to generator-level time predictions.

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

The Fusion `ChebConvLayer` propagates information across the bus graph after static and dynamic embeddings have been combined. Its input is `[B, N, 2 * f_in]`, which is also `[B, N, 2 * H]` because `FusionModuleConfig.f_in = STGCNConfig.f_hidden`. The generator-bus selection maps bus embeddings back to generator-level representations before producing one commitment logit per generator and time period. Here `n_period = T`.

## 2. Completed Changes

### 2.1 Graph Attention Layer

This section covers the introduction of graph attention layers in the Fusion and Dynamic branches, increased network depth, and normalization and residual connection improvements for training stability.

#### 2.1.1 Fusion Attention (stgcn_attn_0)

stgcn_attn_0 keeps the V1 input/output contract and the existing Static and Dynamic branches unchanged. The only structural change is in the Fusion branch, where `ChebConvLayer` is replaced by a multi-head `GraphAttentionLayer`:

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * H]
-> multi-head GraphAttentionLayer [B, N, H]
-> generator-bus selection and output MLPs
-> logits [B, G, T]
```

Attention is normalized over each destination bus's incoming edges in the complete topology. Each bus also has an always-active self-loop so that its own representation is retained and isolated buses remain valid.

#### 2.1.2 Dynamic Attention (stgcn_attn_1)

stgcn_attn_1 keeps the V1 input/output contract and Static branch unchanged, and retains the stgcn_attn_0 attention-based Fusion branch. The Dynamic branch now also replaces the spatial `ChebConvLayer` in both `STConvBlock`s with multi-head `GraphAttentionLayer`s applied independently at each time step:

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

Both Dynamic and Fusion attention retain all graph edges and add one self-loop per bus. The default configuration uses four attention heads.

#### 2.1.3 Network Depth and Stability (stgcn_attn_2)

stgcn_attn_2 changes network depth. Current receptive-field baseline（for case 2383):

| Model        | Parameters (case2383 SCUC) |    Static encoder |                             Dynamic encoder |                Fusion layer | Max static-to-output spatial RF | Max dynamic-to-output spatial RF | Temporal RF before collapse |          Final temporal RF |
| ------------ | -------------------------: | ----------------: | ------------------------------------------: | --------------------------: | ------------------------------: | -------------------------------: | --------------------------: | -------------------------: |
| V1           |                    751,169 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops | Cheb (`k_s = 3`) = 2 hops |                          4 hops |                           6 hops |                     9 steps | Full input horizon (`T`) |
| stgcn_attn_0 |                    734,913 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops |               1 GAT = 1 hop |                          3 hops |                           5 hops |                     9 steps | Full input horizon (`T`) |
| stgcn_attn_1 |                    718,785 | 2 NNConv = 2 hops |                    2 GAT ST blocks = 2 hops |               1 GAT = 1 hop |                          3 hops |                           3 hops |                     9 steps | Full input horizon (`T`) |
| stgcn_attn_2 |                  1,846,529 | 6 NNConv = 6 hops | 2 ST blocks, each with 2 ChebConv = 8 hops |             2 GAT = 2 hops |                          8 hops |                          10 hops |                     9 steps | `Full input horizon (T)` |

Assumptions: Cheb `k_s = 3` = max 2 hops; NNConv/GAT = 1 hop per layer. Temporal RF: `1 + 2 * dynamic_st_blocks * (k_t - 1)`; current value = 9.

stgcn_attn_2 also includes the following stability changes:

1. PRENORM is an independent calibration process with no gradients or parameter updates. When calibration uses the full training set, `PreNormLayer` accumulates the mean and variance instead of overwriting them after each batch.
2. NNConv, ChebConv, and GAT use normalized residual blocks, such as `x = LayerNorm(x + alpha * GraphConv(x))`. The residual branch uses a linear projection when the input and output dimensions differ.
3. NNConv uses a degree-normalized sum or mean for neighbor aggregation to limit layer-by-layer growth in activation magnitude.

### 2.2 Multi-Task Learning (multitask)

#### 2.2.1 Generating Local Marginal Price Labels

`lib/lmp_sover.py` validates the saved commitment schedule, derives startup and shutdown states assuming all units start off, and builds a continuous fixed-commitment LP directly from the sample. It checks minimum up/down transitions, includes no balance slack, and solves with Gurobi feasibility and optimality tolerances of `1e-6`. Network constraints are added for violated base-case and, for SCUC, contingency flows until all scenarios satisfy the feasibility tolerance. Only `OPTIMAL` LP solutions are accepted. The final solve produces dispatch and short-term LMP from the balance and active network-constraint duals:

`LMP = balance dual + PTDFᵀ × sum of line upper/lower-bound duals + contingency term`

Non-SCUC cases omit the contingency term. The pricing solve rebuilds its network scenarios from `UCData`, so it does not depend on whether the sample's UC solution used dense or lazy network constraints. Finite-difference checks of the LMP convention are present in the pricing tests.

#### 2.2.2 Adding Labels to Existing Samples

`label_pricing.py` calls `lib/label_pricing.py` to add the fields in `TARGET_FIELDS`. `PricingResult` uses the same field names, so every field is copied from the solver result to the sample unchanged. A sample counts as fully labeled (`REQUIRED_FIELDS`) when all of these fields and the existing `uc_sol` are present:

| Field                     | Content                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `p_target`              | `[G,T]` unrounded dispatch, MW                                                                        |
| `lmp_target`            | `[N,T]`, currency/MWh; equals energy + congestion                                                     |
| `lmp_energy_target`     | `[T]` power-balance dual                                                                                |
| `lmp_congestion_target` | `[N,T]` PTDF-mapped line duals summed over all network scenarios                                      |
| `shadow_price_target`   | Sparse `{"shape": (S,L,T,2), "indices", "values"}` of line-capacity duals, last axis upper/lower       |
| `line_flow_target`      | `[S,L,T]` float64 signed flow, MW, positive `line_from → line_to`                                     |
| `line_binding_target`   | `[S,L,T,2]` bool, upper/lower bound active within `binding_tol_mw`                                    |
| `pricing_obj`           | LP objective, including the fixed startup costs                                                         |
| `pricing_solve_time`    | Sum of Gurobi solve times across constraint-generation rounds, in seconds                               |
| `pricing_metadata`      | LP status, solver version/parameters, period length, reference bus, binding tolerance, index/flow/bound conventions, per-scenario outage and capacity factors, constraint-generation statistics |

See section 3.1.1 for the full label definitions.

Run in the `lu_uc` environment; the case and UC type are inferred from the path:

```bash
python label_pricing.py --sample-dir ../GridUC-Graph/data/case5/samples/tcuc
```

By default, all numbered sample files are processed; `--sample-id 1` selects one file, `--sample-path` selects a file directly, and `--overwrite` recomputes labels that pass the current shape and finiteness checks. The CLI uses spawned worker processes, up to the detected CPU count or the number of samples. Successful results are written atomically to each sample file. Failures leave the original file intact, are summarized in the console and logs under `log/<case>/lmp_labels/`, and cause a nonzero exit code. The current labels have no pricing-rule version or commitment hash, and the skip check does not verify provenance. Processing and multitask training can consume labels added to existing samples; automatic pricing during new-sample generation remains pending.

#### 2.2.3 Model Interface

multitask returns `STGCNOutput` with UC logits `[B, G, T]` and normalized LMP predictions `[B, N, T]`. Both heads share V1's encoders and bus fusion; the UC head selects generator buses, while the LMP head covers every bus.

`lmp_target [B, N, T]` stays in original price units through processing and batching. Multitask training requires labels in both splits. Processing computes `lmp_normalization` from training labels only; the model checkpoint stores its mean and standard deviation. Recover prices with `lmp_pred * lmp_std + lmp_mean`.

#### 2.2.4 Training Process

The current `multitask` training loop updates the shared encoders and both task heads jointly from the start. It uses binary cross-entropy with logits for UC and Huber loss against normalized LMP labels:

```text
L = L_uc_BCE + lambda_lmp * L_lmp_Huber
```

`lambda_lmp` and the Huber `delta` are configurable and both default to `1.0`. Each epoch logs UC loss, normalized LMP loss, and their weighted total for the training and held-out test splits. The best checkpoint is selected by test-split UC loss. There is no separate validation split, pretraining stage, automatic `lambda_lmp` tuning, shared-encoder gradient-scale logging, or LMP MAE/RMSE reporting in the current pipeline.

### 2.3 Edge Set Attention

The model uses the encoder of the [official ESA implementation](https://github.com/davidbuterez/edge-set-attention) as its backbone: edge tokens are passed through a stack of ESA attention layers to obtain edge representations, which are converted to node representations and fed to an MLP that predicts UC logits. The previous design's static, dynamic, and fusion branches, `ESASpatial`, spatio-temporal convolution blocks, full-horizon convolutional collapse, and temporal expansion are all removed from this design.

**Implementation status**: The new design is implemented in `lib/models/esa/` and registered as `esa`, with checkpoint architecture version `3.1`; old three-branch `3.0` checkpoints are incompatible. The official encoder is pinned to `861a0b12d445ae80cc69b3948af623b4331321d4` (upstream is MIT-licensed). Test results, parameter counts, and memory data from the old model do not count as validation of this design.

#### 2.3.1 Overall Structure

One sample corresponds to the grid graph for one complete scheduling horizon. The full horizon is flattened in a fixed time order and encoded into the input features. Each physical line keeps one direction and forms one edge token; ESA processes the whole horizon's information over the full set of lines. Spatial attention is therefore no longer run separately for each time step.

```text
Generator static features [B,G,F_node_s]
-> assigned directly to buses by gen_bus [B,N,F_node_s] (same as STGCN)
-> concatenated with each bus's full-horizon load/bid features
-> bus input [B,N,F_node_s + T*F_node_d]

Endpoint bus features of each line + line attributes (including line capacity limit)
-> edge token embedding [B, L, H]
-> ESA encoder: stack of topology-masked attention (M) and global self-attention (S)
-> edge representations [B, L, H]
-> edge representation concatenated with the receiving bus input, passed through a shared message MLP to get endpoint messages
-> mean over each node's incident line messages [B, N, H]
-> concatenated with the bus input, passed through a node MLP to get node representations [B, N, H]
-> generator-bus representations selected by gen_bus [B, G, H]
-> shared MLP (H -> T)
-> logits [B, G, T]
```

`L` is the number of physical lines (one edge token per line), and `H` is the hidden dimension. The ESA encoder's M/S attention, feed-forward layers, residuals, and normalization reuse the official implementation where possible, pinned to a fixed upstream version; edge-to-node aggregation and the UC MLP output head are adapted in this project. The layer sequence, depth, number of heads, and dropout are exposed as configuration; the first experiment configuration should be recorded clearly and does not reuse the old STGCN `k_t`, `k_s`, or three-branch settings.

#### 2.3.2 UC Input and Output Adaptation

Existing data fields are kept, the training output remains `logits [B,G,T]`, and the loss is UC BCE-with-logits. The data interface name `STGCNInput` may be reused for now.

- **Time information**: `node_feat_d [B,T,N,F_node_d]` is reordered in fixed time order to `[B,N,T*F_node_d]` as part of the bus input. Time positions are distinguished by feature slots; the first version assumes a fixed scheduling length T. The design uses the known load and bids for the whole forecast horizon at once and drops temporal convolution and per-step causal prediction assumptions.
- **Generator information**: The current experiments guarantee at most one generator per bus. The mapping is the same as STGCN's `_gen_to_nodal()`: create an all-zero bus tensor `x [B,N,F_node_s]`, then set `x[:, gen_bus, :] = node_feat_s`; buses without generators stay zero. After mapping, it is concatenated with the flattened dynamic features. For case2383, static features are 8-dimensional and dynamic features are `24*4=96`-dimensional, so the bus input is `[B,2383,104]`.
- **Line information**: Line reactance, capacity limit, the contingency-monitor indicator in SCUC data, and the maintenance indicator in TOPO data are kept. The model uses these features for prediction.
- **Topology validity**: The current experiments keep all line tokens, and the data and model interfaces no longer contain `edge_mask`. TOPO maintenance lines use soft outages: reactance is doubled in both directions, capacity is unchanged, and the maintenance indicator is set to 1. No base topology has isolated buses, and soft outages create none; a fixed case has a constant number of lines, so no padding is needed. Attention stays isolated within each sample in the batch, and M layers still use a topology mask based on shared line endpoints.

The readout uses an edge-to-node scheme: `ESA encoder → edge representations [B,L,H] → node representations [B,N,H] → generator-bus representations selected by gen_bus [B,G,H] → shared MLP → UC logits [B,G,T]`. The encoder keeps a one-to-one correspondence with edges, and edge representations are aggregated into node representations through the line-endpoint mapping: each line generates a message to each of its two endpoints through a shared message MLP, each node takes the mean of its incident line messages, and a node MLP produces the node representation. PMA graph-level pooling is not used. The input side uses the mapped bus features; the output side predicts UC logits for each generator in the existing label order. The shared MLP maps each generator's H-dimensional representation to logits for T periods; the output head requires a fixed T, and its parameter dimensions do not depend on G.

The edge-to-node conversion is defined below, where `x_v` is the normalized concatenation of bus static features and flattened full-horizon features, with dimension `F = F_node_s + T*F_node_d`, and `h_e` is the H-dimensional edge representation output by the ESA encoder:

```text
m_(e->v) = MLP_message(concat(h_e, x_v))          # [H], computed separately for both endpoints of the line
mean_v   = sum(m_(e->v) for e incident to v) / degree(v)
h_v      = MLP_node(concat(x_v, mean_v))         # [H]
logits_g = MLP_uc(h_(gen_bus[g]))                # [T], no sigmoid on output
```

`MLP_message` shares parameters across all lines and both endpoints; `MLP_node` shares parameters across all nodes; `MLP_uc` shares parameters across all generators. The three are independent networks and do not share parameters with each other. Both the message MLP and the node MLP map `H+F`-dimensional inputs to H dimensions; including the receiving bus features in the message lets one line produce different messages for its two endpoints. The mean is normalized by the number of physical lines incident to each node, parallel lines are counted separately, and aggregation stays within each sample. If a node has no incident lines, `mean_v` is set to zero and the node MLP still keeps the node's own input. This conversion follows the ESA attention encoder.

#### 2.3.3 Topology, Normalization, and Configuration

M layers build local connectivity from shared line endpoints; S layers allow all edges within the same sample to interact. In the PyTorch SDPA mask, True means attention is allowed and the diagonal is kept; this corrects the pinned upstream torch branch's behavior of inverting the adjacency. The model caches a shared `[1,1,L,L]` topology mask without expanding the B/T dimensions; the current fixed cases need no padding or validity mask.

Line tokens use a single-direction representation: each physical line keeps one arbitrary direction, the endpoint order is fixed after preprocessing, and it stays the same for training and inference. The direction only affects storage; edge-to-node aggregation passes information to both endpoints. Shared data still keeps bidirectional edges, and only the ESA input adaptation extracts one copy per line. Reverse pairs must be checked for matching topology, attributes, and maintenance flags; deduplication pairs edges by physical line and keeps parallel lines. An arbitrarily chosen direction does not make the model naturally invariant to swapping endpoints; if the edge embedding concatenates ordered endpoint features, the chosen order must always be kept.

Normalization, residuals, and dropout inside the attention layers follow the selected ESA configuration. Input feature normalization is handled separately and uses training-set statistics only. `PreNormLayer` is used; warmup only collects statistics for the concatenated bus input and single-direction line attributes and does not run the ESA backbone, so the statistics match the final input adaptation.

The new configuration must save the upstream version, input encoding, edge representation, ESA layer sequence, dimensions and number of heads, normalization and dropout, edge-to-node mean aggregation and direction handling, message MLP/node MLP/UC MLP settings, and the G/T and generator-order conventions. The new architecture requires retraining, and direct compatibility with old three-branch checkpoints is not promised.

#### 2.3.4 Large-Graph Memory and Validation

The new design's attention sequence is the L edges of each sample, and the full horizon is already in the feature dimension. If the score matrix is materialized, its size is `[B,A,L,L]`, without the old dynamic branch's T-fold spatial attention batch. The old design's `spatial_time_chunk_size`, time-step chunking, and corresponding equivalence checks are therefore removed.

Actual memory still depends on the number of edges, number of layers, dtype, mask, and attention backend. Officially supported efficient attention implementations available on the target device are preferred; AMP and ESA block-level gradient checkpointing are evaluated when needed. A local mask does not by itself mean sparse computation and does not guarantee lower quadratic memory; the old model's peak memory is not used to conclude that the new model will fit.

Implementation and acceptance order:

1. Following the full-horizon flattened input and one token per physical line, use the shared message MLP, mean aggregation, and node MLP, pin the official ESA version, and complete the UC adaptation.
2. On small graphs, verify the generator-to-bus mapping (matching STGCN's direct assignment, zero-filled buses without generators), the two-endpoint mapping and mean normalization of edge-to-node aggregation, permutation invariance over incident lines, zero-degree node handling, output shape, generator order, soft-outage bidirectional reactance and capacity, parallel lines, and sample isolation; check for a finite loss, backward gradients, and one optimizer update.
3. Check that feature normalization uses training-set statistics only; verify prediction consistency after saving and loading a new checkpoint.
4. On the case2383 target GPU, start with `batch_size=1` and measure peak memory and time separately for preprocessing/warmup, training forward, backward, optimizer update, and validation; record the device, runtime versions, dtype, and actual attention backend.
5. Run training and validation epochs, then tune batch size, ESA depth, and hidden dimension, and compare UC prediction quality with existing baselines.

Record of this implementation and validation:

- Defaults are `f_hidden=64`, `n_heads=4`, `layer_types="MSMS"` (the sequence length is the depth), Pre-LN, standard Mish MLPs with `mlp_hidden=128` and `mlp_layers=2`; attention, residual, and MLP dropout are all 0. These dimension, layer-sequence, head, MLP, and dropout parameters can be set through `config_overrides`, and `gradient_checkpointing=True` enables per-block recomputation.
- Upstream's normalized-input Pre-LN residual, outer encoder input residual, and final Linear+Mish are kept. Only PyTorch SDPA is used, so ESA, xformers, and flash-attn need not be installed; CPU forces the math backend to avoid NaNs from the current PyTorch 2.3.1 fused CPU attention on large sparse masks, while CUDA selects an available backend automatically.
- `OMP_NUM_THREADS=1 python -m unittest discover -s tests/models -p 'test_esa_contract.py' -v`: 9 tests pass, covering input mapping, reverse-pair checks, parallel lines, local/global attention, sample isolation, two-endpoint mean, isolated nodes, permutation properties, warmup statistics, forward/backward/optimizer update, checkpoint round trip and old-version rejection, deterministic inference with dropout, gradient recomputation, and large sparse masks.
- case5 and case2383 TOPO inputs pass CPU training/validation pipeline checks. case2383 uses 2,896 edge tokens, bus input `[1,2383,104]`, and output `[1,327,24]`; the default model has 244,632 parameters. With FP32, batch=1, and gradient recomputation enabled, single-sample warmup takes about 0.005 s, a training update about 3.0 s, and validation about 0.81 s (current environment PyTorch 2.3.1+cu121, one CPU thread). The checks use zero placeholder UC labels and validate numerics and the pipeline, not prediction quality or full-dataset training results.
- The NVIDIA driver is currently unavailable, so peak GPU memory on the target GPU, the actual CUDA attention backend, and training/evaluation with real labels are not yet verified.

#### 2.3.5 Confirmed Design Choices

1. The full horizon is flattened in a fixed order and used as bus input features.
2. Each physical line keeps one direction and forms one edge token; the endpoint order is fixed and parallel lines are kept.
3. The ESA encoder outputs edge representations, which are converted to node representations through a shared message MLP, two-endpoint mean aggregation, and a node MLP; generator-bus representations are then selected and passed through the UC MLP to predict logits.

## 3. Planned Changes

### 3.1 Multi-task Labels Generating

本节方案已实现（2026-10-13），用于对现有 `.pkl` 样本补齐 dispatch、unit commitment、LMP congestion、LMP energy component、shadow price、total objective value、线路潮流和线路紧约束状态。入口仍为根目录 `label_pricing.py`，由 `lib/label_pricing.py` 管理标签读写，`lib/lmp_sover.py` 负责求解和提取。无需重新抽样；所有连续标签及派生的紧约束状态应来自同一次固定 UC 的最终最优 LP。

#### 3.1.1 标签口径与字段

现有样本已经保存 `uc_sol`；当前定价流程已有 `p_target`、`lmp_target`、`pricing_obj`，但没有保存 LMP 分量和线路对偶值。建议采用以下契约，其中 G/N/L/T 分别为机组、母线、物理线路、时段数，S 为网络场景数：

| 标签 | 建议字段及单样本形状 | 生成方式与含义 |
| --- | --- | --- |
| Unit commitment | 复用 `uc_sol [G,T]` | 校验现有 0/1 开停机状态；训练端继续映射为 `uc_target`，不另存重复副本 |
| Dispatch | 复用 `p_target [G,T]` | 固定 `uc_sol` 后的 LP 最优出力，MW，保存未四舍五入的值 |
| LMP energy component | 新增 `lmp_energy_target [T]` | 最终 LP 的 `balance.Pi`；需要母线维度时广播为 `[N,T]` |
| LMP congestion | 新增 `lmp_congestion_target [N,T]` | 基态及所有 SCUC 场景线路对偶值经各自 PTDF 映射后的总和 |
| Total LMP | 复用 `lmp_target [N,T]` | energy 与 congestion 相加；保留现有训练接口 |
| Shadow price | 新增 `shadow_price_target` 稀疏结构 | 各网络场景、线路、时段、上下界的非负容量边际价值，逻辑形状 `[S,L,T,2]`，最后一维依次为 upper/lower |
| 线路潮流 | 新增 `line_flow_target [S,L,T]` | 最终 dispatch 对应的各场景有符号有功潮流，MW，正方向为 `line_from → line_to`，保存 float64 未取整值 |
| 线路紧约束状态 | 新增 `line_binding_target [S,L,T,2]` | 布尔标签，最后一维依次为 upper/lower，表示潮流是否在容差内达到该侧容量限额 |
| Total objective value | 复用 `pricing_obj` 标量 | 当前固定 UC 下，全时域的分段发电成本（含 no-load）与启动成本之和，货币单位 |

这里先将 shadow price 明确定义为**线路容量约束**的影子价格；母线功率平衡的影子价格已由 energy 标签表示。机组出力上下界、爬坡等约束的影子价格不纳入第一版，如果后续需要，应另按约束族扩展字段，不能混入线路标签。

`pricing_obj` 是这套 dispatch/UC 标签对应的总成本，不代表重新求得了混合整数 UC 的全局最优值。已有 `uc_obj` 应保留：它来自原始 UC 求解，当前采样器仅在 `attach_result=True` 时附带，并且求解结果中已做两位小数取整；缺失时不能以 `pricing_obj` 冒充原始 MIP 结果。若需要经最优性证明的 UC objective，须另行重求 MIP 并保存 status、bound、gap，这不属于本次补标签默认流程。

#### 3.1.2 同一次 LP 提取价格、分量和总成本

继续调用 `solve_pricing(data, data.uc_sol, uc_type)`：校验 UC、按当前“初始全部停机”的规则推导 startup/shutdown 并检查最小开停机时间，建立连续 LP，以约束生成方式加入违反的网络约束。仅当最终 LP 为 `OPTIMAL` 且所有网络场景通过容差检查时，提取整套标签；不从 MIP 直接读取价格，也不对不同标签分别求解。

沿用现有约束方向，令 `H_s` 为场景 s 的 PTDF，`d` 为母线负荷，线路两侧约束写为：

```text
H_s[:, gen_bus] @ p <=  F_s + H_s @ d      # upper, dual pi_upper <= 0
H_s[:, gen_bus] @ p >= -F_s + H_s @ d      # lower, dual pi_lower >= 0

energy[t]         = balance.Pi[t]
mu_upper[s,l,t]   = -pi_upper[s,l,t]
mu_lower[s,l,t]   =  pi_lower[s,l,t]
congestion[:,t]   = sum_s H_s.T @ (mu_lower[s,:,t] - mu_upper[s,:,t])
lmp[:,t]          = energy[t] + congestion[:,t]
```

`mu_upper/mu_lower` 表示放宽相应侧容量 1 MW 所减少的最优成本；同时放宽同一场景某线路两侧容量时，对应价值为二者之和。该定义针对场景约束的容量 RHS；修改原始 `line_fmax` 会同时影响多个场景，敏感度还必须乘各场景的容量系数并求和。允许容差内的对偶符号误差，超出容差则拒绝写入；不要取绝对值掩盖符号错误。

当前模型是无损 DC 模型，因此不增加 loss component。所有场景统一沿用当前 `ref_bus=0` 和线路方向；energy/congestion 的拆分依赖参考母线，必须在 metadata 中记录。当前目标函数未显式乘时段长度，第一版沿用一小时口径，价格记为货币/MWh；若未来加入非一小时时段，须同步调整目标函数和价格单位并升级规则版本。

目标值直接读取同一个最终 LP 的 `ObjVal`，并用以下公式核对，不能使用 `sum(LMP * demand)` 代替系统成本：

```text
startup[:,0] = uc_sol[:,0]
startup[:,t] = max(uc_sol[:,t] - uc_sol[:,t-1], 0)
pricing_obj = sum(g,t) [bid_no_load[g] * uc_sol[g,t]
                       + sum(k) bid_price[g,k,t] * dp[g,k,t]
                       + gen_startup_cost[g] * startup[g,t]]
```

#### 3.1.3 网络场景、线路标签与 shadow price 存储

场景 0 为基态，TCUC/TOPO 的 S=1，SCUC 的 S=`1 + len(cc_monitor)`。SCUC 后续场景严格按 `cc_monitor` 顺序编号，并保存每个场景对应的线路索引和限额规则。沿用现有模型：TOPO 检修线路电抗加倍、容量不变；SCUC 各事故场景从原始网络构建，事故线路电抗加倍，容量为原容量的 `0.5 * 1.05`，其他线路容量为 `1.05` 倍。不能将场景维与物理线路维混为一体，也不能使用图神经网络双向边数代替 L。

线路潮流与紧约束状态必须覆盖所有场景的全部物理线路，包括从未加入 LP 的网络约束，不能只遍历 `added` 或非零 shadow price。在最终全场景可行性检查通过后，复用 `networks` 中的机组 PTDF 列和负荷乘积计算，无需另一次优化：

```text
flow[s]             = H_s[:, gen_bus] @ p_target - H_s @ demand
slack_upper[s]      = F_s[:,None] - flow[s]
slack_lower[s]      = F_s[:,None] + flow[s]
binding[s,:,:,0]    = abs(slack_upper[s]) <= binding_tol_mw
binding[s,:,:,1]    = abs(slack_lower[s]) <= binding_tol_mw
```

第一版 `binding_tol_mw=1e-6`，与当前网络可行性容差一致，并写入 `pricing_metadata`；之后若需要按改变后的规则或容差重新生成已有完整标签，由用户指定 `--overwrite`。先检查两侧 slack 均不低于负的可行性容差，再生成状态，超容差越限必须报错，不可作为正常紧约束样本保存。使用未取整 float64 潮流判定，不以利用率取整或对偶值判定。两个布尔值通常分别对应非紧约束 `(False,False)`、上界紧约束 `(True,False)`、下界紧约束 `(False,True)`；保留双布尔表示以支持零容量等情况下两侧同时紧约束。

紧约束只表示达到容量限额，不等于非零 shadow price；退化时允许紧约束且对偶为零。`line_flow_target` 和 `line_binding_target` 使用稠密数组，并始终保留 S 维（TCUC/TOPO 为 1），与 shadow price 共用场景、线路和上下界顺序。存储和 worker 内存预算需计入每样本约 `8*S*L*T` 字节的潮流以及 `2*S*L*T` 字节的布尔状态；SCUC 批量回填前先测量实际文件增量及峰值内存。

建议 `shadow_price_target` 使用可直接 pickle 的 COO 风格字典：`shape=(S,L,T,2)`、整数 `indices [K,4]`、浮点 `values [K]`。从最终 `added` 字典的约束 `.Pi` 提取并转换符号；未加入约束的对偶值按零补全，在所有场景已通过可行性检查的前提下，这构成完整 LP 的一个有效对偶解。保存所有非零对偶，不人为设置稀疏截断阈值；K=0 也是完整有效标签。零对偶并不表示线路一定未达到容量，退化情况下紧约束也可能是零对偶。

复用当前逐场景重建 PTDF 的方式，同时累积 congestion 和输出稀疏 shadow price，避免在大算例中常驻 `[S,L,N]` 的 PTDF 或稠密 shadow 张量。元数据保存场景顺序、母线/机组/线路索引约定、参考母线及容量规则。LP 退化可能导致不同的最优对偶和 dispatch；记录求解器版本与配置，不承诺跨算法逐元素一致。

#### 3.1.4 文件修改与旧样本迁移

现有样本保存的是 LP 的部分结果，而非可恢复的完整 LP 模型及对偶解。需要生成标签时，统一使用样本输入和原有 `uc_sol` 重新建立并求解固定 UC 的连续 LP；不复用旧 LMP、dispatch 或 objective 来拼接新结果，不重新求解 UC 的混合整数问题。一次定价流程仍可能包含多轮网络约束生成，但所有标签均从最终同一个最优解提取。

| 样本情况 | 未指定 `--overwrite` | 指定 `--overwrite` |
| --- | --- | --- |
| 没有 LMP 标签 | 重建 LP，生成整套标签 | 重建 LP，生成整套标签 |
| 已有 LMP，但缺少任一必需标签 | 重建 LP，整体替换定价标签 | 重建 LP，整体替换定价标签 |
| 已有完整标签 | 跳过，不求解 LP | 重建 LP，整体替换定价标签 |

不设计标签过期判断，不通过版本、输入 hash 或 UC hash 变化触发重算。跳过条件只检查整套必需标签是否存在且非 `None`（合法的空稀疏 shadow price 也算存在）；`--overwrite` 是对已有完整标签进行重算的唯一开关。生成结果的数值和物理校验独立于跳过判断：发现已有完整标签需要修复时，也由用户指定 `--overwrite`，不自动重算。求解或新结果校验失败时原文件保持不变，记录为失败，不能把保留的旧标签计作本次生成成功。

1. **`lib/lmp_sover.py`**：扩展 `PricingResult`，在当前 `solve_pricing()` 的最终解提取处一次性返回 energy、congestion、shadow price、线路潮流和线路紧约束状态，值均从本次最终 LP 重新提取。`PricingResult` 的字段名与样本字段一致（包括 `pricing_obj`、`pricing_solve_time`、`pricing_metadata`），写回时按 `TARGET_FIELDS` 逐项同名复制，不需要再单独映射。当前生产路径是直接固定 UC 的约束生成 LP，不能只修改旧的 dense `_calculate_lmp()` 辅助函数。
2. **`lib/case_loader.py`**：为新字段增加默认 `None` 的可选声明；旧 pickle 的访问使用 `getattr(..., None)`，确保缺少新字段时仍可读取。
3. **`lib/label_pricing.py`**：将“已有标签是否齐全”和“新结果是否有效”分开：新增 `has_pricing_labels()` 检查必需字段（`REQUIRED_FIELDS`，即 `TARGET_FIELDS` 加上已有的 `uc_sol`）存在且非 `None`，仅用于 `if not overwrite and has_pricing_labels(data): return "skipped"`；`valid_pricing_labels()` 用于写入前检查新结果的字段完整性、形状、有限性、UC 二值性、稀疏索引范围/唯一性、对偶符号、分量求和及 metadata。对线路标签检查 float64/bool 类型、场景数、潮流容量可行性及按记录容差重新计算的紧约束状态一致性。新结果全部校验通过后才原子写入，整套替换定价标签；保留原 `uc_sol` 和 `uc_obj`，失败保持原文件不变，不留下部分新标签。
4. **求解记录**：`pricing_metadata` 保存 LP status、求解器版本/参数、时段长度、场景信息、参考母线、紧约束判定容差和约束生成统计，供解释与复现实验使用。不增加 `uc_hash`、`input_hash` 或标签版本匹配机制，元数据变化不作为自动重算依据。
5. **根目录 `label_pricing.py`**：保留 `--sample-dir`、`--sample-path`、`--sample-id`、`--overwrite` 及失败非零退出码；沿用原有并行方式：worker 数为 CPU 数与样本数的较小值，Gurobi 使用默认线程设置，不新增并发控制参数。日志记录首次生成、缺少标签重算、完整标签跳过、强制覆盖及失败原因与耗时，并检查传入 UC 类型和样本场景数据一致。`--overwrite` 表示对所有选中的样本重新求解，不受已有标签情况影响。
6. **训练接入另一步完成**：本次先补齐原始样本。若后续模型要消费这些新任务，需扩展 `lib/data_loader.py`、模型输入/输出和损失，重新生成处理数据；energy 保持 `[T]`，线路标签保留场景映射，变长 SCUC 场景需使用稀疏批处理或 padding/mask。各连续任务归一化只使用训练集统计量。

#### 3.1.5 验证与执行顺序

先扩展 `tests/test_pricing_generation.py`，再在少量现有样本副本上试跑，最后按 case/UC 类型批量回填：

- 检查原 `uc_sol` 不变，dispatch 满足平衡、出力、爬坡及所有网络场景约束；对照最终 LP 验证总成本。
- 从最终 dispatch 和 PTDF 独立重建所有场景线路潮流，核对方向、物理线路顺序及容量规则；覆盖 S=1、多个 SCUC 场景、未加入 LP 的线路和平行线路。紧约束测试覆盖正向满载、反向满载、非紧约束、容差边界、超容差越限拒绝、零容量双侧紧约束，以及紧约束但零对偶的退化情形。检查线路对偶与相应 slack 的互补性，不能要求每个紧约束都有正 shadow price。
- 检查 `lmp_target == lmp_energy_target[None,:] + lmp_congestion_target`，并用保存的 shadow price 和各场景 PTDF 独立重建 congestion。覆盖无拥塞、上界拥塞、下界拥塞、TOPO 与 SCUC。
- 小算例固定同一 UC，轻微扰动母线负荷，用 objective 差分验证总 LMP；单独放宽指定场景的某侧容量，用差分验证 shadow price 的符号和幅值。避免跨越最优基变化点；退化点使用单侧敏感度或区间检查，不强制得到唯一导数。
- 与小算例完整 dense LP 比较 objective、可行性和价格重建关系；有退化时不要求 dispatch 或每个约束对偶逐元素相同。
- 验证无 LMP 和只有旧 LMP 的样本均重新求解并写入完整标签；完整标签在未指定 `--overwrite` 时跳过，指定后强制重算。确认输入、UC 或求解配置变化不会触发自动重算；确认空稀疏 shadow price 算作存在。覆盖不可行、非最优、新结果校验失败及写入异常，保证原文件保持不变且统计为失败。

实施后沿用现有调用方式，例如先用 `python label_pricing.py --sample-dir <case>/samples/tcuc --sample-id 1` 验证单样本，再去掉 `--sample-id` 批量处理。回填完成应汇总各标签的有效样本数、失败 ID、shape、数值范围和最大重建误差。代码已完成，尚未对原始样本批量回填。

实施与验证记录（2026-10-13）：

- `label_pricing.py` 保留原有并行方式和 Gurobi 默认线程设置。示例：`python label_pricing.py --sample-dir data/case2383/samples/tcuc`；加 `--overwrite` 强制重算选中的所有样本。
- `tests/test_pricing_generation.py` 共 15 项测试通过，涵盖 TCUC/TOPO/SCUC 与 dense LP 对照、负荷/容量差分、价格和潮流重建、平行线路、零容量双侧紧约束及零对偶、补齐/跳过/覆盖和失败原子保护。该测试文件位于仓库现有 `.gitignore` 忽略的本地 `tests/` 目录。
- 使用 `lu_uc` 的 Python/Gurobi 13.0.1；该环境未安装 pytest，测试运行时从 base 环境追加加载 pytest。沙箱内 Gurobi HostID 不可用，实际求解测试在获准的沙箱外运行。
- 对 case2383 TCUC 样本 619 的 `/tmp` 副本完成 CLI 验证：生成约 2.2 秒，线路潮流形状 `[1,2896,24]`，非零 shadow price 109 个，LMP 分量重建最大误差约 `1.42e-14`，文件从 1,631,449 增至 2,789,452 字节；再次运行正确跳过。原始样本未修改，大规模 SCUC 文件增长及峰值内存仍需在批量回填前测量。

### 3.2 模型训练与样本生成同步进行

TBD

### 3.3 Topo & Scuc Together

TBD

### 3.4 Minor Changes

已将 TOPO 模型输入与求解器统一为 soft outage（检修线路双向电抗加倍、容量不变、保留连接），TOPO 线路特征新增双向 maintenance 0/1 指示列，并从数据接口、NNConv/ChebConv/GAT 及 MLP 中移除 `edge_mask`；旧处理数据须从已有原始样本重新转换（无需重新求解），MLP 输入维度缩小，TOPO 的 `f_edge` 由 2 增至 3；`architecture_version` 检查保留，旧 TOPO 模型与旧 MLP 因输入维度变化无法加载，需按新输入重新训练。

## 4. Architecture Decisions

TBD
