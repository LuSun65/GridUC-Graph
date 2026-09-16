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

### 3.1 Edge Set Attention

计划采用 [davidbuterez/edge-set-attention](https://github.com/davidbuterez/edge-set-attention)
仓库中的 Edge Set Attention（ESA）模型。可以复用其核心模块，但不能只加一条
`import` 就直接训练现有 UC 任务；仍需适配数据、输出与训练流程。

主要有三处适配：

- **输入适配**：ESA 接收节点特征、边索引、边特征和批次信息。现有的静态机组
  特征、动态时序特征以及检修线路 `edge_mask`，需要转换并接入它的计算流程。
- **输出适配**：原版边模式主要面向图级预测，通过池化生成整张图的表示；本项目
  需要输出每台机组、每个时段的 `logits [B, G, T]`，因此需要设计对应的预测头。
  原仓库的节点级任务使用 NSA，而不是边模式 ESA。
  参见[仓库说明](https://github.com/davidbuterez/edge-set-attention#node-level-tasks)。
- **训练适配**：把模型接到现有 UC 标签、损失函数、训练和评估流程中。原仓库的
  `Estimator` 已封装自己的训练逻辑，直接套用也需要调整。
  参见[源码](https://github.com/davidbuterez/edge-set-attention/blob/main/esa/models.py)。

建议复用 ESA 核心网络，在本项目中增加一个适配模型，沿用现有训练框架。
下一步先确定它是替换整个 STGCN，还是仅替换其中的空间特征提取部分；这会决定
时序信息如何处理，以及 ESA 的结果如何转成机组预测。具体接入方案与实现细节
后续讨论。

### 3.2 多任务学习

第一个辅助任务将是**节点边际电价（LMP）**预测。
标签不能直接从原始 UC 求解中读取：UC 是一个 MILP，
包含整数变量的模型所给出的对偶值不能作为有效的市场价格。
因此，这里采用的初始定义是：将样本中的最优开停机决策固定后，
通过连续经济调度定价求解得到的短期 LMP。它表示在保持开停机决策不变的情况下，
某个母线在某个时段额外供应 1 MW 负荷的边际成本；它并不是允许开停机决策本身
发生变化时，通过有限差分计算得到的边际成本。

#### 3.2.1 定价标签生成加速计划

目标是在保持固定开停机定价模型、完整网络约束和求解精度的前提下，降低
`label_pricing.py` 在大规模系统上的标签生成时间。本节为待实施计划。
保留 `FeasibilityTol = OptimalityTol = 1e-8`、仅接受 `OPTIMAL`、无平衡松弛、
初始机组全停及目标函数包含固定成本的约定，不通过放宽容差、减少故障场景或
提前停止求解来加速。实际求解的是连续 LP，调整 `MIPGap` 不适用于此阶段。

**先建立耗时基线。** 将单样本总耗时拆分为加载、PTDF 计算、建模、固定变量与
LP 复制、优化、标签提取和保存，并记录峰值内存。现有 `pricing_solve_time`
仅记录 Gurobi `Runtime`，不能代表端到端耗时。使用固定的 case5、case118 及
更大系统样本子集比较，覆盖 TCUC、TOPO、SCUC 和不同故障集合；在完成测量前
不预设加速倍数。

按以下优先级逐步实施，每一步单独比较性能和标签质量：

1. **缓存 PTDF。** 当前每个样本都会重新计算基础及各故障场景的 PTDF，而负荷
   和报价变化不影响 PTDF。TCUC 复用相同网络的基础矩阵；TOPO 按检修线路缓存；
   SCUC 按故障线路缓存，再按该样本 `cc_monitor` 的顺序组合。缓存键包含网络
   连接、电抗、参考节点和故障处理规则，不能只使用算例名称。容量及故障后容量
   单独更新或纳入对应缓存键。采用进程内缓存并设置内存上限，不必将 PTDF 写入
   样本 pickle；保留当前矩阵计算与截断规则。
2. **批量访问 Gurobi 属性。** 保存变量和约束引用，批量设置 `z/u/v` 的上下界，
   批量读取出力 `X` 及约束对偶值 `Pi`，替代逐元素按名称查询。SCUC 线路上下限
   对偶值数量为 `2 * L * T * (1 + K)`，其中 `L` 为线路数、`T` 为时段数、
   `K` 为故障数。必须保持索引顺序与 LMP 公式一致；`relax()` 复制后的引用需
   对应新模型，不能沿用原模型对象。
3. **受控的样本间多进程并行。** 每个进程使用独立 Gurobi 环境，每个样本只
   分配给一个进程。联合限制进程数、Gurobi 线程数和 BLAS 线程数，根据峰值
   内存及许可证能力选择并发量。保留原子保存、有效标签跳过和失败汇总行为，
   使用集中日志或唯一日志文件名，避免当前秒级时间戳导致并发日志重名。
4. **复用定价 LP 模板。** 按相同网络、变量结构和故障集合复用模型，逐样本
   更新负荷、报价、容量及固定状态，减少完整 UC 重建和 `relax()` 复制。
   当前报价位于 `pwl_cost` 约束系数中，不能仅更新目标函数及 RHS。也可评估
   直接建立固定状态的连续模型，但须保留原有逻辑约束的可行性验证、爬坡约束
   和固定成本，避免接受原模型本应拒绝的开停机序列。

若上述措施后仍有明显瓶颈，再评估以下方案：

- **稀疏 PTDF 计算：** 以稀疏矩阵分解和线性方程求解替代稠密矩阵显式求逆。
  保持现有软故障语义（故障线路电抗加倍、容量减半及既有容量倍率），不能
  直接替换为完全断线模型；检查数值误差及现有 `1e-6` 截断阈值附近的变化。
- **LP 算法与基热启动：** 在相同精度下比较单纯形、障碍法及基复用，评估
  热启动与预求解的配合。收益以样本子集实测为准。
- **精确约束生成：** 从部分线路约束开始，每轮求解后检查全部基态和故障
  潮流，补入违反约束，直至完整约束集均满足要求。不能直接使用当前依赖
  `MIPSOL` 的 UC lazy 回调；需设计 LP 外层迭代、完整约束验证和对偶值映射，
  并评估与当前 dense 定价规则及版本的兼容性。

**质量与验收标准。** 保持最优目标值和可行性并不保证 LMP 逐元素完全相同：
LP 存在多重最优解时，算法、热启动或模型表示变化可能选择不同的最优出力或
对偶解。优先完成缓存和批量属性访问等不改变模型的优化；后续方案分别检查
完整约束残差、目标值、出力和 LMP 差异，并按 3.2.4 进行有限差分验证。
对差异区分数值误差、合法的多重最优解及实现错误，不仅凭 `OPTIMAL` 判定
标签一致。报告单样本耗时、批量吞吐量、峰值内存和质量对比；定价约定变化时
更新规则版本，验证通过后再用于全部数据集。

参考：[Gurobi 批量模型求解建议](https://support.gurobi.com/hc/en-us/articles/25414264687121-Solving-batches-of-models-with-short-model-runtimes-Efficient-API-usage-parameter-tuning-and-deployment-considerations)、
[多进程环境管理](https://support.gurobi.com/hc/en-us/articles/360043111231-How-do-I-use-multiprocessing-in-Python-with-Gurobi)、
[LP 热启动](https://doc.gurobi.com/projects/optimizer/en/current/features/warmstart.html)。

#### 3.2.2 定价 LP 不可行诊断与修复计划

本节为待实施方案，不代表已修改求解器或确认失败根因。
`runs/lmp_labels/case118/tcuc_20260915_201314.log` 中，1000 个样本有 456 个
保存成功、544 个失败；失败均为 `status=3, sol_count=0`，即定价 LP 不可行。
每个失败在处理时和末尾汇总时各记录一次，因此 ERROR 行数不等于失败样本数。

若原 UC 与定价 LP 的数据、约束完全一致，且保存的整数方案确实可行，固定该
方案后的 LP 应当可行。当前需要排查的是历史采样模式、整数取整及数值容差，
不能仅凭状态码认定线路约束或启停方案错误。当前两个项目的 `uc_model.py`
内容一致，但旧样本没有记录生成时的求解模式、模型版本和容差，当前代码不能
完全证明历史配置。定价强制使用 dense 约束，历史样本若使用 none 模式则可能
缺少线路限制；若使用 lazy 模式，还需核对 `lazy_tol=1e-4` 与定价容差的差异。

**先补充诊断信息。** 在 `lib/lmp_sover.py` 的 `_solve_pricing_lp()` 中，针对
`GRB.INFEASIBLE` 增加可选 IIS 诊断。先选少量失败样本（如 0、1、4），调用
`computeIIS()`，记录样本 ID、约束名称以及被标记的变量上下界；必要时导出
冲突模型。IIS 是不可约冲突约束集，移除其中任一成员可消除该集合自身的冲突，
但它不一定是数量最少的冲突集合，也不保证模型没有其他冲突。它只能定位矛盾，
不会自动修复模型；出现线路约束并不意味着应该放宽线路限额。

按以下顺序判别并修复：

1. **数值容差对照。** 保持场景、固定状态和完整约束不变，仅比较
   `FeasibilityTol=1e-8` 与 `1e-6` 的结果，检查可行解的实际约束残差和模型
   尺度。只有违反量处于微小数值范围，才支持数值边界问题的判断；放宽后
   成功本身不足以证明标签可靠。明显的容量缺口或线路越限不能靠放宽容差处理。
2. **固定启停方案检查。** 若 IIS 涉及 `balance`、`pmax/pmin` 和固定状态，先
   核对每时段 `sum(Pmin * z) <= demand.sum() <= sum(Pmax * z)`；这是必要条件，
   还须检查爬坡、最小开停时间和输电限制。保持同一场景与完整约束，释放启停
   变量重新求 UC：若恢复可行，在确认模型定义正确后重新生成启停方案及关联
   标签；若仍不可行，则继续检查场景数据和模型，不将问题仅归因于旧方案。
3. **线路模型核查。** 若 IIS 涉及 `flow_ub/flow_lb`，核对线路容量、单位、
   PTDF、母线索引及历史采样模式。数据或构造错误应修正；若线路定义正确而
   旧样本未包含这些限制，应在完整模型下重新求 UC。若释放启停后仍不可行，
   检查该负荷场景是否超出网络供电能力。临时去掉线路限制只能辅助定位，不能
   作为删除约束或接受定价标签的依据。

**元数据、标签与日志。** 后续保存采样求解模式、模型版本和实际容差，记录
定价容差与诊断结果。若改变定价规则或容差，更新 `PRICING_RULE`，重新生成
受影响的旧标签，避免已有 456 个成功标签与新标准混用。失败样本继续保持
原文件不变；日志保留逐样本错误、最终数量与失败 ID 汇总，取消末尾逐条重复
的 ERROR。修复后按 3.2.4 检查完整约束残差和 LMP 有限差分，再推广到全量。

#### 3.2.3 处理后的数据与模型接口约定

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

#### 3.2.4 验证与逐步实施

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
