# Network Changes

## 1. Previous Architecture

The current STGCN implementation is defined mainly in `lib/models/v1/stgcn.py`, with version-specific modules under `lib/models/v1/` and shared layers under `lib/models/common/`.

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

The static branch is implemented by `StaticModule` in `lib/models/v1/module_static.py`. It converts generator-level static features into bus-level static graph embeddings.

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

The dynamic branch is implemented by `DyncModule` in `lib/models/v1/module_dync.py`. It converts bus-level time-series features into one dynamic embedding per bus.

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

The fusion branch is implemented by `FusionModule` in `lib/models/v1/module_fusion.py`. It combines static and dynamic bus embeddings, then converts bus-level features back to generator-level time predictions.

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

### 2.2 Multi-Task Learning

#### 2.2.1 Generating Local Marginal Price Labels

`lib/lmp_sover.py` fixes the sample's commitment, startup, and shutdown states and solves a dense LP with Gurobi, supporting samples generated with either dense or lazy constraints. `fixed_commitment_lp_v1` assumes all units are initially off, no balance slack, tolerances of `1e-6`, and only accepts `OPTIMAL`; the same solve produces dispatch and short-term LMP:

`LMP = balance dual + PTDFᵀ × sum of line upper/lower-bound duals + contingency term`

Non-SCUC models omit the last term. The formula requires finite-difference validation; changes to these conventions require a rule-version update.

#### 2.2.2 Adding Labels to Existing Samples

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

By default, all samples are processed and valid labels are skipped; `--sample-id 1` selects one sample, and `--overwrite` forces recomputation. Validated results are saved atomically, preserving the original scenario; the hash covers only the commitment matrix. Failures leave the original file intact and processing continues; failed matrix inversion does not fall back to a pseudoinverse. The console and logs under `runs/lmp_labels/casexx/` record status, runtime, and the failure list without tracebacks; any failure results in a nonzero exit code. Integration with new-sample generation and training data remains pending.

## 3. Planned Changes

### 3.1 Edge Set Attention

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

### 3.2 多任务学习

第一个辅助任务将是**节点边际电价（LMP）**预测。标签不能直接从原始 UC 求解中读取：UC 是一个 MILP，包含整数变量的模型所给出的对偶值不能作为有效的市场价格。因此，这里采用的初始定义是：将样本中的最优开停机决策固定后，通过连续经济调度定价求解得到的短期 LMP。它表示在保持开停机决策不变的情况下，某个母线在某个时段额外供应 1 MW 负荷的边际成本；它并不是允许开停机决策本身发生变化时，通过有限差分计算得到的边际成本。

#### 3.2.1 模型接口

扩展 `STGCNInput`、`concat_stgcn_inputs`、`STGCNDataset`、collate 函数以及处理后数据的验证逻辑，加入：

- `lmp_target [B, N, T]`，所有节点、时段的电价标签均参与损失和统计计算；
- 检查所选任务的标签是否齐全，缺失时提示补充。
- 为处理后的数据增加格式版本，并随模型保存任务配置和归一化参数。
- 仅用训练集计算电价均值和标准差，供训练、验证、测试及预测统一使用。
- 根据训练集分布和验证效果决定是否截断极端电价，默认保留。
- 将预测值还原到原始电价单位，与原始标签比较误差。

网络应保留共享的静态/动态编码器，并提供独立的任务头：

```text
共享母线表示 [B, N, H]
|- UC 任务头       -> uc_logits [B, G, T]
`- LMP 任务头      -> lmp_pred  [B, N, T]
```

使用结构化输出对象，而不是改变当前单一输出张量的含义。LMP 任务头必须保持母线级别；使用 `gen_bus` 进行选择会错误地丢弃没有发电机的母线上的电价。UC 任务头通过 `gen_bus` 将共享母线表示映射到发电机。

#### 3.2.2 训练流程

预训练阶段可仅用 UC 的 BCE 损失更新共享编码器和 UC 头，也可用 UC+LMP 加权损失共同训练共享编码器与两个任务头；随后进入联合训练阶段，共享编码器与两个任务头共同更新，使用：

```text
L = L_uc_BCE + lambda_lmp * L_lmp_Huber
```

在验证集上调整 `lambda_lmp`，分别记录各项原始损失、加权贡献以及共享编码器的梯度尺度，避免总损失的改善掩盖主要 UC 任务的性能下降。

#### 3.2.3 定价标签生成加速计划

本节为待实施计划，尚未实现并行或测得加速比。**首选最小方案：样本级多进程 + 每进程线程限额，先测 2 个 worker，再决定是否增加到 4 个。** 目标是提高批量标签吞吐量；单样本运行不受益于样本并行。

**当前依据与原方案比较。** `label_pricing.py` 用串行循环调用 `label_sample()`；每个样本独立加载、重算 PTDF、建立 dense UC、固定启停状态、复制为 LP、求解并保存。代码未显式设置 Gurobi `Threads/Method`，NumPy 底层也可能多线程，因此不能把整个流程视作单线程，更不能仅凭 CPU 利用率推断瓶颈。case2383 的 `tcuc_v2_20260921_201015.log` 中样本 8–11 的端到端耗时约为 48–50 秒，但尚无分阶段计时。

| 方案                      | 主要收益                                   | 改动与约束                                                     | 本次优先级             |
| ------------------------- | ------------------------------------------ | -------------------------------------------------------------- | ---------------------- |
| 样本并行（原第 3 项）     | 同时推进独立样本，覆盖建模、求解和结果处理 | 改动集中在调度与线程配置；峰值内存随并发增加                   | 第一阶段，最小可用方案 |
| PTDF 缓存（原第 1 项）    | 避免相同网络重复矩阵计算                   | 需要可靠缓存键和内存上限；不同拓扑命中率不确定                 | PTDF 耗时占比高时再做  |
| 批量属性访问（原第 2 项） | 减少逐变量/约束 Python 调用                | 需保持名称、索引和 relax 后模型引用一致；收益取决于耗时占比    | 属性读写成为瓶颈时再做 |
| LP 模板复用（原第 4 项）  | 减少重建和复制模型                         | 必须完整更新负荷、报价、容量、固定状态及成本系数，验证成本最高 | 暂缓                   |

**第一阶段实现范围。**

1. 在 `label_pricing.py` 增加 `--workers`（默认 1）和 `--solver-threads`（并行时默认 1），采用 `ProcessPoolExecutor` 和 `spawn`。父进程分发唯一文件路径及简单参数；子进程内调用现有 `label_sample()`，沿用每个样本独立创建并释放 Gurobi 环境的方式，不跨进程传递模型或完整样本对象。`--workers 1` 保留串行路径，单文件模式只启动一个执行单元。
2. 将求解线程参数传到 `solve_pricing()`，在定价 LP 优化前显式设置 `Threads`。并行模式下，在导入 NumPy、启动子进程之前设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免多个 worker 各自争用所有核心。记录实际线程配置；第一阶段不调整 `Method`。
3. 子进程就地原子保存，只向父进程返回样本 ID、状态、耗时及错误信息。父进程按完成顺序统一写日志和统计，日志名加入 PID 或唯一运行 ID，避免并发启动重名。保留有效标签跳过、`--overwrite`、失败样本汇总及失败时非零退出码；任务异常由父进程记录，进程池异常时报告未完成 ID。一次运行内同一文件只分发一次，也不与其他运行重叠写入同一批样本。
4. 先测单样本峰值内存，再启动 2 个 worker。设可用 CPU 数为 C、每 worker 求解线程数为 T、并发数为 W，控制 `W × T ≤ C`；以 `W × 单 worker 峰值内存` 加父进程开销估算总内存，并预留至少 25% 可用内存。case2383、case6515 和 SCUC 不直接按 CPU 核数开满；还需确认当前 Gurobi 许可允许所选并发量。

**最小验收与配置选择。** 在独立样本副本上固定选取 12–20 个代表性样本，各配置使用同一输入并强制重新计算，避免有效标签跳过造成虚假提速。比较当前串行配置、`W=1/T=1`、`W=2/T=1`；内存充足且吞吐改善时再测 `W=4/T=1`。分别评估 tcuc/topo/scuc，记录包含进程启动、加载、建模、求解和保存的总墙钟时间、成功样本/秒、失败数及全部 worker 的峰值总内存。`pricing_solve_time` 仍只表示 Gurobi 求解时间，不能拿它代替端到端耗时。选择吞吐量明显提高且无内存压力的最小 W；若并行收益不足或内存受限，回到串行并补充 PTDF、建模、LP 复制、优化、结果提取与读写的分阶段计时，再决定下一步。当前没有依据承诺 2 倍或 4 倍加速。

保持现有固定开停机、完整网络约束、无平衡松弛、初始全停及固定成本约定。当前代码为 `FeasibilityTol = OptimalityTol = 1e-6`（原计划写为 `1e-8`，此处与代码同步），本次加速不调整容差，仅接受 `OPTIMAL`。串行与并行比较完整约束残差、目标值、出力及 LMP，并按 3.2.5 验证有限差分；多重最优解可能导致合法差异，不能只凭 `OPTIMAL` 判定一致，也不要求逐位相同。验证跳过、覆盖、失败文件不变及中断后可重跑。纯调度优化不修改 `PRICING_RULE`；若后续改变定价约定，则更新规则版本并重新验证。

**后续按瓶颈选做。** PTDF 缓存优先使用每进程有容量上限的缓存，键包含网络连接及顺序、电抗、参考节点、故障规则和截断规则；故障集合保持顺序。只缓存 PTDF 时，线路容量及故障后限额仍根据当前样本计算，不复用旧容量。LP 模板复用需完整更新包括 `pwl_cost` 在内的系数，并保留逻辑可行性检查、爬坡约束和固定成本。稀疏 PTDF、算法/基热启动和精确约束生成均留待后续独立评估；约束生成必须检查全部基态与故障潮流、正确映射对偶值，不能直接复用 UC 的 `MIPSOL` 回调。

#### 3.2.4 验证与逐步实施

在生成全部标签之前，使用一个小规模算例和固定的样本子集检查：

- 定价 LP 的状态为 `OPTIMAL`，且调度出力满足功率平衡、发电机、爬坡、基态潮流和预想故障潮流约束；
- LMP 的形状严格为 `[N, T]`，发电出力的形状严格为 `[G, T]`，所有值均为有限值，且单位一致；
- 无拥塞时段所有母线的 LMP 在容差范围内相同；
- 对于选定的 `(母线, 时段)` 对，在保持相同开停机决策的情况下，将负荷增加一个很小的 `epsilon`，定价目标函数值的变化约为 `LMP[bus,period] * epsilon`（在远离基发生变化的位置使用单侧测试）；
- 连续两次补充标签具有幂等性，写入中断不会损坏样本，且数据处理保留样本数量和原始划分；
- 短程过拟合测试能够同时降低 UC 和 LMP 损失，随后使用 UC 准确率/变量固定指标以及 LMP MAE/RMSE，对仅使用 UC 与使用 UC+LMP 进行比较。

按以下顺序逐步实施：定价/单元测试、case5 标签补充与训练冒烟测试、一个较大算例的样本子集，然后是全部已有数据集。如果定价规则或有限差分验证发生变化，应停止批量生成，而不是静默接受标签。

### 3.3 学习线路拥堵任务

TBD

### 3.4 模型训练与样本生成同步进行

TBD

### 3.5 Topo & Scuc Together

TBD

## 4. Architecture Decisions

TBD
