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

### 2.3 STGCN V2.2 (`stgcn_v2.2`)

V2.2 changes network depth. Current receptive-field baseline（for case 2383):

| Model | Parameters (case2383 SCUC) |    Static encoder |                             Dynamic encoder |                Fusion layer | Max static-to-output spatial RF | Max dynamic-to-output spatial RF | Temporal RF before collapse |          Final temporal RF |
| ----- | -------------------------: | ----------------: | ------------------------------------------: | --------------------------: | ------------------------------: | -------------------------------: | --------------------------: | -------------------------: |
| V1    |                    751,169 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops | Cheb (`k_s = 3`) = 2 hops |                          4 hops |                           6 hops |                     9 steps | Full input horizon (`T`) |
| V2.0  |                    734,913 | 2 NNConv = 2 hops |                   2 Cheb ST blocks = 4 hops |               1 GAT = 1 hop |                          3 hops |                           5 hops |                     9 steps | Full input horizon (`T`) |
| V2.1  |                    718,785 | 2 NNConv = 2 hops |                    2 GAT ST blocks = 2 hops |               1 GAT = 1 hop |                          3 hops |                           3 hops |                     9 steps | Full input horizon (`T`) |
| V2.2  |                  1,846,529 | 6 NNConv = 6 hops | 2 ST blocks, each with 2 ChebConv = 8 hops |             2 GAT = 2 hops |                          8 hops |                          10 hops |                     9 steps | `Full input horizon (T)` |

Assumptions: Cheb `k_s = 3` = max 2 hops; NNConv/GAT = 1 hop per layer.
Temporal RF: `1 + 2 * dynamic_st_blocks * (k_t - 1)`; current value = 9.

V2.2 also includes the following stability changes:

1. PRENORM is an independent calibration process with no gradients or parameter
   updates. When calibration uses the full training set, `PreNormLayer`
   accumulates the mean and variance instead of overwriting them after each
   batch.
2. NNConv, ChebConv, and GAT use normalized residual blocks, such as
   `x = LayerNorm(x + alpha * GraphConv(x))`. The residual branch uses a linear
   projection when the input and output dimensions differ.
3. NNConv uses a degree-normalized sum or mean for neighbor aggregation to limit
   layer-by-layer growth in activation magnitude.

### 2.4 Multi-Task Learning

#### 2.4.1 Generating Local Marginal Price Labels

`lib/lmp_sover.py` fixes the sample's commitment, startup, and shutdown states and solves a dense LP
with Gurobi, supporting samples generated with either dense or lazy constraints.
`fixed_commitment_lp_v1` assumes all units are initially off, no balance slack, tolerances of
`1e-8`, and only accepts `OPTIMAL`; the same solve produces dispatch and short-term LMP:

`LMP = balance dual + PTDFᵀ × sum of line upper/lower-bound duals + contingency term`

Non-SCUC models omit the last term. The formula requires finite-difference validation;
changes to these conventions require a rule-version update.

#### 2.4.2 Adding Labels to Existing Samples

`label_pricing.py` calls `lib/label_pricing.py` to add:

| Field                                    | Content                                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| `lmp_target`                           | `[N,T]`, currency/MWh                                  |
| `p_target`                             | `[G,T]` unrounded dispatch, MW                         |
| `pricing_obj` / `pricing_solve_time` | LP objective (including fixed costs) / runtime (seconds) |
| `pricing_metadata`                     | `pricing_rule`, `uc_sol_sha256`, `uc_type`         |

Run in the `lu_uc` environment; the case and UC type are inferred from the path:

```bash
python label_pricing.py --sample-dir ../GridUC-Graph/data/case5/samples/tcuc
```

By default, all samples are processed and valid labels are skipped; `--sample-id 1` selects one
sample, and `--overwrite` forces recomputation. Validated results are saved atomically, preserving
the original scenario; the hash covers only the commitment matrix. Failures leave the original
file intact and processing continues; failed matrix inversion does not fall back to a pseudoinverse.
The console and logs under `runs/lmp_labels/casexx/` record status, runtime, and the failure list
without tracebacks; any failure results in a nonzero exit code. Integration with new-sample
generation and training data remains pending.

## 3. Planned Changes

### 3.1 Node Features

The results in `runs/test/case2383/logs/test/test_scuc_stgcn_v2.log` show two
distinct behaviors. For typical cases, the selected threshold is relatively
low, many variables can be fixed, and prediction accuracy is below 100%. In a
small number of cases, however, the threshold reaches 1, very few variables are
fixed, and the reported accuracy is 100%.

One possible explanation is that these exceptional cases contain only one or
two incorrectly predicted variables, and the model assigns very high confidence
to those predictions. This suggests that each variable has a different level of
importance to solving the SCUC problem. Fixing a less important variable
incorrectly may not significantly affect the result. However, fixing a critical
variable to the wrong value may make the entire problem infeasible. Therefore,
critical variables should not be fixed.

This leads to two questions for further discussion: Which variables are critical,
and how can they be excluded from variable fixing?

Possible approaches to excluding critical variables from variable fixing:

- Modify the training labels by setting the labels of critical variables to 0.5.
- Add a regularization term to the loss function.
- Modify the confidence calculation.

Possible ways to identify critical variables:

- Use node features (physical information).
- Train another neural network specifically for this task, potentially in
  combination with multi-task learning.
- Investigate whether critical variables correspond to particular operating
  samples, such as specific load, maintenance, or contingency conditions.

Two questions remain open:

- How should a critical variable be defined?
  - A variable whose incorrect fixing makes an otherwise feasible SCUC problem infeasible.
  - Or variables whose incorrect fixing significantly degrades solution quality.
- How can reliable labels for critical variables be generated?
- Test the core hypothesis by progressively unfixing incorrectly fixed variables
  in the exceptional samples. If too many variables must be unfixed, the instability
  is more likely caused by a combination of variables.

### 3.2 多任务学习

第一个辅助任务将是**节点边际电价（LMP）**预测。
标签不能直接从原始 UC 求解中读取：UC 是一个 MILP，
包含整数变量的模型所给出的对偶值不能作为有效的市场价格。
因此，这里采用的初始定义是：将样本中的最优开停机决策固定后，
通过连续经济调度定价求解得到的短期 LMP。它表示在保持开停机决策不变的情况下，
某个母线在某个时段额外供应 1 MW 负荷的边际成本；它并不是允许开停机决策本身
发生变化时，通过有限差分计算得到的边际成本。

#### 3.2.1 处理后的数据与模型接口约定

扩展 `STGCNInput`、`concat_stgcn_inputs`、`STGCNDataset`、collate 函数
以及处理后数据的验证逻辑，加入：

- `lmp_target [B, N, T]` 及可选的有效性掩码；
- `p_target [B, G, T]` 及可选的有效性掩码。

提升处理后数据/检查点的数据模式版本，并且仅在显式单任务模式下允许读取旧的
仅含 UC 的处理后数据文件。所有目标归一化统计量都只能根据训练集计算，并保存
在检查点/清单文件中。在样本 pickle 中保留原始 LMP 值。由于拥塞或供给稀缺时
LMP 可能呈现重尾分布，首先采用基于训练集的稳健截断加标准化，并以物理单位
再次报告指标；不要根据测试集计算截断阈值。

网络应保留共享的静态/动态编码器，并提供独立的任务头：

```text
共享母线表示 [B, N, H]
|- UC 任务头       -> uc_logits [B, G, T]
|- LMP 任务头      -> lmp_pred  [B, N, T]
`- 发电出力任务头  -> p_pred    [B, G, T]   （首次实验中可选）
```

使用结构化输出对象，而不是改变当前单一输出张量的含义。LMP 任务头必须保持
母线级别；使用 `gen_bus` 进行选择会错误地丢弃没有发电机的母线上的电价。
UC 和发电出力任务头可以通过 `gen_bus` 将共享母线表示映射到发电机。

使用如下加权损失进行训练：

```text
L = L_uc_BCE + lambda_lmp * L_lmp_Huber
                 + lambda_p * L_power_Huber
```

其中，回归损失使用归一化后的目标和掩码计算。分别记录各项原始损失、加权贡献
以及共享编码器的梯度尺度；否则，表面上改善的总损失可能掩盖主要 UC 任务的
性能下降。先仅使用 UC+LMP，在验证数据上调整 `lambda_lmp`，并将添加发电出力
任务头作为一项独立的消融实验。

#### 3.2.2 验证与逐步实施

在生成全部标签之前，使用一个小规模算例和固定的样本子集检查：

- 定价 LP 的状态为 `OPTIMAL`，且调度出力满足功率平衡、发电机、
  爬坡、基态潮流和预想故障潮流约束；
- LMP 的形状严格为 `[N, T]`，发电出力的形状严格为 `[G, T]`，所有值均为有限值，
  且单位一致；
- 无拥塞时段所有母线的 LMP 在容差范围内相同；
- 对于选定的 `(母线, 时段)` 对，在保持相同开停机决策的情况下，将负荷增加
  一个很小的 `epsilon`，定价目标函数值的变化约为
  `LMP[bus,period] * epsilon`（在远离基发生变化的位置使用单侧测试）；
- 连续两次补充标签具有幂等性，写入中断不会损坏样本，
  且数据处理保留样本数量和原始划分；
- 短程过拟合测试能够同时降低 UC 和 LMP 损失，随后使用 UC 准确率/变量固定指标
  以及 LMP MAE/RMSE，对仅使用 UC 与使用 UC+LMP 进行比较。

按以下顺序逐步实施：定价/单元测试、case5 标签补充与训练冒烟测试、
一个较大算例的样本子集，然后是全部已有数据集。如果定价规则或有限差分验证
发生变化，应停止批量生成，而不是静默接受标签。

### 3.3 Topo & Scuc Together

TBD

## 4. Architecture Decisions

TBD
