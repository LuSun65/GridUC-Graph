# Network Changes

Model family mapping: V1 → `lib/models/stgcn/`; V2/V2.1/V2.2 →
`lib/models/stgcn_attn/`; V3 → `lib/models/esa/`; V4 → `lib/models/multitask/`.
See [the registry mapping table](README.md#model-directories-and-versions) for
canonical names and aliases. V3 implements the ESA-based UC model described
in Section 3.1; V4 is the implemented UC/LMP multitask model.

## 1. Previous Architecture

The current STGCN implementation is defined mainly in `lib/models/stgcn/stgcn.py`, with version-specific modules under `lib/models/stgcn/` and shared layers under `lib/models/common/`.

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

The temporal convolutions preserve the time dimension with causal padding. The `ChebConvLayer` performs masked spectral graph convolution at each time step, using `edge_mask [B * T, E]` after expanding and reshaping `edge_mask [B, E]`. Both `STConvBlock` instances set `f_out = f_hidden`; the first sets `f_in = f_node = F_node_d`, while the second sets `f_in = f_hidden`. The projection output is `[B, N, f_out]`, which becomes `x_dynamic [B, N, H]` because `DyncModuleConfig.f_out = STGCNConfig.f_hidden`.

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

The Fusion `ChebConvLayer` propagates information across the masked bus graph after static and dynamic embeddings have been combined. Its input is `[B, N, 2 * f_in]`, which is also `[B, N, 2 * H]` because `FusionModuleConfig.f_in = STGCNConfig.f_hidden`. The generator-bus selection maps bus embeddings back to generator-level representations before producing one commitment logit per generator and time period. Here `n_period = T`.

## 2. Completed Changes

### 2.1 Graph Attention Layer

This section covers the introduction of masked graph attention layers in the Fusion and Dynamic branches, increased network depth, and normalization and residual connection improvements for training stability.

#### 2.1.1 Fusion Attention (V2.0)

V2.0 keeps the V1 input/output contract and the existing Static and Dynamic branches unchanged. The only structural change is in the Fusion branch, where `ChebConvLayer` is replaced by a masked multi-head `GraphAttentionLayer`:

```text
x_static [B, N, H] + x_dynamic [B, N, H]
-> concat [B, N, 2 * H]
-> masked multi-head GraphAttentionLayer [B, N, H]
-> generator-bus selection and output MLPs
-> logits [B, G, T]
```

Attention is normalized over each destination bus's active incoming edges using `edge_mask`. Each bus also has an always-active self-loop so that its own representation is retained and isolated buses remain valid.

#### 2.1.2 Dynamic Attention (V2.1)

V2.1 keeps the V1 input/output contract and Static branch unchanged, and retains the V2.0 attention-based Fusion branch. The Dynamic branch now also replaces the spatial `ChebConvLayer` in both `STConvBlock`s with masked multi-head `GraphAttentionLayer`s applied independently at each time step:

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

Both Dynamic and Fusion attention use `edge_mask` and always-active self-loops. The default configuration uses four attention heads.

#### 2.1.3 Network Depth and Stability (V2.2)

V2.2 changes network depth. Current receptive-field baseline（for case 2383):

| Model | Parameters (case2383 SCUC) |    Static encoder |                             Dynamic encoder |                Fusion layer | Max static-to-output spatial RF | Max dynamic-to-output spatial RF | Temporal RF before collapse |          Final temporal RF |
| ----- | -------------------------: | ----------------: | ------------------------------------------: | --------------------------: | ------------------------------: | -------------------------------: | --------------------------: | -------------------------: |
| V1    |                    751,169 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops | Cheb (`k_s = 3`) = 2 hops |                          4 hops |                           6 hops |                     9 steps | Full input horizon (`T`) |
| V2.0  |                    734,913 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops |               1 GAT = 1 hop |                          3 hops |                           5 hops |                     9 steps | Full input horizon (`T`) |
| V2.1  |                    718,785 | 2 NNConv = 2 hops |                    2 GAT ST blocks = 2 hops |               1 GAT = 1 hop |                          3 hops |                           3 hops |                     9 steps | Full input horizon (`T`) |
| V2.2  |                  1,846,529 | 6 NNConv = 6 hops | 2 ST blocks, each with 2 ChebConv = 8 hops |             2 GAT = 2 hops |                          8 hops |                          10 hops |                     9 steps | `Full input horizon (T)` |

Assumptions: Cheb `k_s = 3` = max 2 hops; NNConv/GAT = 1 hop per layer. Temporal RF: `1 + 2 * dynamic_st_blocks * (k_t - 1)`; current value = 9.

V2.2 also includes the following stability changes:

1. PRENORM is an independent calibration process with no gradients or parameter updates. When calibration uses the full training set, `PreNormLayer` accumulates the mean and variance instead of overwriting them after each batch.
2. NNConv, ChebConv, and GAT use normalized residual blocks, such as `x = LayerNorm(x + alpha * GraphConv(x))`. The residual branch uses a linear projection when the input and output dimensions differ.
3. NNConv uses a degree-normalized sum or mean for neighbor aggregation to limit layer-by-layer growth in activation magnitude.

### 2.2 Multi-Task Learning (V4)

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

V4 returns `STGCNOutput` with UC logits `[B, G, T]` and normalized LMP predictions `[B, N, T]`. Both heads share V1's encoders and bus fusion; the UC head selects generator buses, while the LMP head covers every bus.

`lmp_target [B, N, T]` stays in original price units through processing and batching. Multitask training requires labels in both splits. Processing computes `lmp_normalization` from training labels only; the model checkpoint stores its mean and standard deviation. Recover prices with `lmp_pred * lmp_std + lmp_mean`.

#### 2.2.4 Training Process

The current `multitask` training loop updates the shared encoders and both task heads jointly from the start. It uses binary cross-entropy with logits for UC and Huber loss against normalized LMP labels:

```text
L = L_uc_BCE + lambda_lmp * L_lmp_Huber
```

`lambda_lmp` and the Huber `delta` are configurable and both default to `1.0`. Each epoch logs UC loss, normalized LMP loss, and their weighted total for the training and held-out test splits. The best checkpoint is selected by test-split UC loss. There is no separate validation split, pretraining stage, automatic `lambda_lmp` tuning, shared-encoder gradient-scale logging, or LMP MAE/RMSE reporting in the current pipeline.

## 3. Planned Changes

### 3.1 Edge Set Attention (implemented in V3)

使用 [ESA](https://github.com/davidbuterez/edge-set-attention) 替换现有空间处理，保留时间与空间交替聚合。

#### 3.1.1 模型结构设计

保留母线时序主干，空间信息聚合采用 ESA（边集合注意力），将空间算子统一替换为 `ESASpatial`。显式保留每条线路的表示，结合线路属性学习线路之间的关联。

```text
静态：机组特征 → 母线映射 → ESASpatial
动态：[时间卷积 → ESASpatial（含全局交互）→ 时间卷积] ×2
      → 全周期压缩
融合：拼接静态与动态表示 → ESASpatial
      → 选取机组母线 → 时间展开 → UC logits
```

在时空模块内部引入远距离全局信息交互。动态 ESA 在每个时刻独立计算，接收时间卷积编码的历史信息，通过边 token 的全局交互汇集同一时刻全系统的负荷与报价信息，再交给后续时间卷积建模全系统供需变化及其跨时段影响。保留时间与空间交替聚合、全周期压缩与输出头；最终预测仍依赖完整输入周期。

#### 3.1.2 模型接口设计

`ESASpatial`：母线端点特征与线路属性 → 边 token → ESA encoder self-attention → 按目标母线汇总 → 节点残差与归一化。

全局信息交互采用 encoder self-attention，以边 token 为 query，输出保留逐边对应关系。局部交互可使用拓扑 mask；时空模块中的全局交互层不使用限制远距离连接的拓扑 mask，但始终保留断线及 padding 的有效性 mask，并隔离不同样本与不同时刻的 token。

原版边模式没有现成的母线级输出接口，需新增边到母线读出，首版采用有效入边均值。保留双向边，断线边不得参与注意力或读出，孤立母线通过残差保留自身特征。动态与融合接口需补传线路属性。

#### 3.1.3 V3 实现审查与待定改动

**现状**：`lib/models/esa/` 已按 3.1.1–3.1.2 实现，`tests/test_esa.py` 通过。每个 `ESASpatial` 只有一层全局 self-attention，采用 Post-LN，dropout 为 0。

**1. 拓扑 mask（需要）**

- 原版 ESA 的节点任务用 NSA：节点作 token，M 层按节点邻接做 mask（如 `M S M S M`），不加 PMA，也不用线路属性；原版边模式只支持图级任务。
- 原版边模式的 M 层 mask：两条边共享任一端点即相邻（src–src、tgt–tgt、交叉均算）。
- 方案：沿用边模式的 mask 并**保留对角线**，避免邻边全断时整行被 mask 而输出 NaN；层序列可配置（如 `MS`）。动态分支保留 S 层，以满足全局交互要求。
- 显存：拓扑在样本间共享，`[E, E]` mask 只构造一次，再与各样本的 `edge_mask` 广播；不要展开为 `[B·T, E, E]`（case2383 在 B=8 时约 6.4 GB）。

**2. 单向边**

- case2383 的 5792 条有向边 = 2896 条线路 × 2；后半段是前半段的翻转，属性和 mask 与前半段相同。双向的两份 token 基本冗余，不是必需的。
- 收益：注意力计算量和 mask 约降为 1/4，token 激活显存约减半。
- 配套改动：读出改为按两个端点汇总，并区分起点、终点投影（`W_src`、`W_tgt`）；否则只作起点的母线收不到该线路的信息。
- 只在 V3 内部取 `edge_index[:, :E/2]`，数据和 V1/V2 不变。按位置去重，不按 (u,v) 对去重（case2383 有 10 对并行线路）；加断言检查后半段是前半段的翻转。

**3. 显存**：case2383 一次前向加反向的峰值：B=1 为 1.93 GB，B=4 为 7.6 GB；B=8 在 11 GB 的 2080 Ti 上 OOM。改用单向边后需要重新测。

**4. LN 与 dropout**

- Post-LN `LN(x + f(x))`：每层输出尺度稳定，但梯度要穿过每层 LN，层数多时不稳定（原版用 Admin 初始化缓解）。
- Pre-LN `x + f(LN(x))`：恒等残差通路，深层训练更稳，需要一个最终 LN；是原版默认。
- 建议：堆叠 `MS` 后改用 Pre-LN，并把 dropout 暴露到 `STGCNV3Config`（训练集只有 800 个样本）。

**5. 与 V1 的小差异**

- 多机母线：V1 用赋值，只保留一台机组（CUDA 上保留哪台不确定）；V3 用求和，但成本、最小启停时间等特征相加没有物理意义。case2383 没有多机母线，两者等价；其他算例需要检查，必要时改为 sum + 机组数，或 sum/mean/max 拼接。
- PreNorm 位置：V1 在母线级归一化，统计量被约 86% 的无机组零行主导；V3 在机组级归一化，统计量更干净，但无机组母线的 0 与"平均水平的机组"无法区分，可加"是否有机组"指示特征。

**待确认问题**

1. M 层方案：采用边模式 mask（共享母线即相邻，保留对角线），层序列可配置，静态、融合、动态默认都用 `MS`，是否同意？还是改用原版 NSA 节点方案（会丢掉线路 token 和线路属性）？
2. 单向边：在 V3 内部只取单向边，读出改为按两个端点汇总并区分起点、终点投影，是否同意？
3. 显存：如果改用单向边后，case2383 在 B=8 下仍然 OOM，选哪种处理方式：给 esa 单独调小 batch、gradient checkpointing，还是 AMP 混合精度？
4. LN 与 dropout：是否改用 Pre-LN，并把 dropout 暴露到 `STGCNV3Config`？
5. 静态特征：是否给母线加"是否有机组"指示特征？多机母线的聚合要不要从求和改为 sum + 机组数，或 sum/mean/max 拼接？

### 3.2 学习线路拥堵任务

TBD

### 3.3 模型训练与样本生成同步进行

TBD

### 3.4 Topo & Scuc Together

TBD

## 4. Architecture Decisions

TBD
