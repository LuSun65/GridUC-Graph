# Network Changes

Model family mapping: V1 → `lib/models/stgcn/`; stgcn_attn_0/stgcn_attn_1/stgcn_attn_2 →
`lib/models/stgcn_attn/`; ESA → `lib/models/esa/`; multitask → `lib/models/multitask/`.
See [the registry mapping table](README.md#model-names-and-directories) for
canonical names and aliases. Section 3.1 describes the planned direct ESA
architecture; the previous three-branch ESA implementation and registry entry
have been removed. multitask is the implemented UC/LMP multitask model.

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

`label_pricing.py` calls `lib/label_pricing.py` to add:

| Field                  | Content                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `lmp_target`         | `[N,T]`, currency/MWh                                                                                 |
| `p_target`           | `[G,T]` unrounded dispatch, MW                                                                        |
| `pricing_obj`        | LP objective, including the fixed startup costs                                                         |
| `pricing_solve_time` | Sum of Gurobi solve times across constraint-generation rounds, in seconds                               |
| `pricing_metadata`   | `uc_type`, `constraint_generation_rounds`, `network_constraints_added`, `max_network_violation` |

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

## 3. Planned Changes

### 3.1 Edge Set Attention

以 [ESA 官方实现](https://github.com/davidbuterez/edge-set-attention) 为完整模型主干，使用边 token 输入、ESA 注意力堆叠与读出，再适配 UC 预测任务。原方案中的静态、动态、融合三分支、`ESASpatial`、时空卷积块、全周期卷积压缩与时间展开均从本方案移除。

**实现状态**：旧三分支实现及 ESA 注册项已移除，`lib/models/esa/esa.py` 暂为空文件；新方案尚未实现。旧模型的测试结果、参数量及显存数据不能作为本方案的验证结果。

#### 3.1.1 整体结构

一个样本对应一个完整调度周期的电网图。建议把完整时域按固定时间顺序编码进输入特征，每条线路形成一个边 token；ESA 在整个线路集合上处理该周期的信息。这样不再对每个时刻分别运行空间注意力。

```text
机组静态特征 [B,G,F_node_s]
-> 按 gen_bus 直接赋值到母线 [B,N,F_node_s]（与 STGCN 一致）
-> 拼接母线完整时域的负荷/报价特征
-> 母线输入 [B,N,F_node_s + T*F_node_d]

线路两端的母线特征 + 线路属性（含线路容量约束）
-> 边 token 嵌入 [B, L, H]
-> ESA：拓扑 masked attention（M）与全局 self-attention（S）堆叠
-> 读出（待确认）
-> UC 输出头
-> logits [B, G, T]
```

`L` 为适配后的边 token 数，`H` 为隐藏维度。ESA 的 M/S 注意力、前馈层、残差、归一化与读出优先复用官方实现，接入时固定上游版本。具体层序列、深度、头数和 dropout 作为配置暴露；首轮实验配置应记录清楚，不沿用旧 STGCN 的 `k_t`、`k_s` 或三分支配置。

#### 3.1.2 UC 输入与输出适配

继续使用现有数据字段，并保持训练输出为 `logits [B,G,T]`，采用 UC BCE-with-logits 损失。数据接口名 `STGCNInput` 可以暂时复用。

- **时间信息**：建议将 `node_feat_d [B,T,N,F_node_d]` 按固定时序重排为 `[B,N,T*F_node_d]`，作为母线输入的一部分。时间位置由特征槽位区分；首版对应固定调度长度 T。该方案一次使用完整预测周期的已知负荷与报价，不保留时间卷积或逐时因果预测假设。
- **机组信息**：当前实验保证每条母线至多有一台机组。采用与 STGCN 的 `_gen_to_nodal()` 相同的映射：先创建全零母线张量 `x [B,N,F_node_s]`，再执行 `x[:, gen_bus, :] = node_feat_s`，无机组母线保持零值。映射后与展平的动态特征拼接。以 case2383 为例，静态特征为 8 维，动态特征为 `24*4=96` 维，拼接后的母线输入为 `[B,2383,104]`。
- **线路信息**：保留线路电抗、容量上限，以及 SCUC 数据中的 contingency-monitor 指示特征和 TOPO 数据中的 maintenance 指示特征。模型使用这些特征学习预测。
- **拓扑有效性**：当前实验保留全部线路 token，数据与模型接口不再包含 `edge_mask`；TOPO 检修线路采用 soft outage，两个方向的电抗加倍、容量不变，并将 maintenance 指示特征置 1。各算例基础拓扑没有孤立母线，soft outage 不新增孤立母线；固定算例的线路数相同，无需 padding。注意力保持 batch 内样本隔离，M 层仍按线路共享端点使用拓扑 mask。

读出建议采用原版图级方案：`ESA → PMA 图表示 → UC MLP → [B,G*T] → [B,G,T]`，具体采用与否仍待确认。输入端使用映射后的母线特征；输出端继续按现有标签的固定机组顺序输出 UC logits，不改为母线级标签。该输出头要求固定 G、T。

#### 3.1.3 拓扑、归一化与配置

M 层按线路共享端点构造局部连接，S 层允许同一样本内所有有效边交互；具体 mask 的布尔含义与对角线行为应按接入版本核对。拓扑连接与有效性 mask 分别管理，拓扑相同的样本可复用结构信息。

线路 token 的粒度待确认：保留现有双向边最便于接入；每条物理线路一个 token 可以减少序列长度，但需要明确端点顺序和方向编码。若从现有双向数据提取物理线路，必须校验反向配对的拓扑、属性与停运状态，并保留平行线路。

注意力层内部的归一化、残差与 dropout 沿用选定的 ESA 配置。输入特征的归一化另行处理，只使用训练集统计量。若继续使用 `PreNormLayer`，预热仅收集原始输入统计，不必执行 ESA 主干；其统计口径应与最终输入适配一致。

新配置需保存上游版本、输入编码方式、边表示方式、ESA 层序列、维度与头数、归一化与 dropout、读出类型，以及 G/T 和机组顺序约定。新架构需要重新训练，旧三分支 checkpoint 不作直接兼容承诺。

#### 3.1.4 大图显存与验证

新方案的注意力序列是每个样本的 L 条边，完整时域已进入特征维度。若物化分数矩阵，其规模为 `[B,A,L,L]`，不再具有旧动态分支的 T 倍空间注意力批次；因此删除旧方案的 `spatial_time_chunk_size`、时段分块及对应等价性验证。

实际显存仍取决于边数、层数、dtype、mask 与注意力后端。优先使用官方支持且目标设备可用的高效注意力实现，必要时评估 AMP 与 ESA block 级梯度检查点。局部 mask 本身不代表采用稀疏计算，也不保证降低二次内存；不以旧模型峰值估算新模型一定可运行。

实施与验收顺序：

1. 确定输入编码、读出与边 token 方案，固定官方 ESA 版本并完成 UC 适配。
2. 小图验证机组到母线的映射（与 STGCN 直接赋值结果一致、无机组母线填零）、输出形状、机组顺序、soft outage 的双向电抗与容量、平行线路以及样本隔离；完成有限损失、反向梯度和一次优化器更新检查。
3. 检查特征归一化只使用训练集统计；验证新 checkpoint 保存/加载后的预测一致性。
4. 在 case2383 目标 GPU 上以 `batch_size=1` 起步，分别测量预处理/预热、训练前向、反向、优化器更新及验证阶段的峰值显存与耗时；记录设备、运行版本、dtype 和实际注意力后端。
5. 跑通训练及验证 epoch，再调整 batch size、ESA 深度与隐藏维度，比较 UC 预测效果与现有基线。

#### 3.1.5 待确认的设计选择

1. 是否采用完整时域展平输入，并使用原版 ESA + PMA + 固定 `[G*T]` UC 输出头？
2. 边 token 保留双向，还是每条物理线路一个 token？

### 3.2 学习线路拥堵任务

TBD

### 3.3 模型训练与样本生成同步进行

TBD

### 3.4 Topo & Scuc Together

TBD

### 3.5 Minor Changes

已将 TOPO 模型输入与求解器统一为 soft outage（检修线路双向电抗加倍、容量不变、保留连接），TOPO 线路特征新增双向 maintenance 0/1 指示列，并从数据接口、NNConv/ChebConv/GAT 及 MLP 中移除 `edge_mask`；旧处理数据须从已有原始样本重新转换（无需重新求解），MLP 输入维度缩小，TOPO 的 `f_edge` 由 2 增至 3；`architecture_version` 检查保留，旧 TOPO 模型与旧 MLP 因输入维度变化无法加载，需按新输入重新训练。

## 4. Architecture Decisions

TBD
