# 模型版本与实验管理方案

## 1. 目标

本项目将按照 `doc/NETWORK_CHANGES.md` 逐步修改 STGCN，包括图注意力、节点特征和多任务学习等内容。版本管理需要满足以下要求：

- 当前实现作为 V1 基线保持不变，并且能够随时复现。
- 后续开发全部在 V2 分支和独立工作目录中进行。
- V2 可以复用行为完全一致的现有 layer 和 block；需要改变内部逻辑时，应新增 V2 实现，避免改变 V1 行为。
- V1 与 V2 的 processed data、checkpoint、result 和 log 互不覆盖。
- V1 与 V2 可以使用相同的数据划分和测试场景进行公平比较。
- 每项网络改动都能单独评估，避免无法区分收益来自架构、特征还是训练配置。

整体采用四层隔离：

```text
Git 基线
  ├── main：冻结 V1
  └── feature/stgcn-v2：后续开发

源代码
  ├── 原有 STGCN 和 layers：V1 兼容实现
  └── 新增 V2 模型和 V2 专用模块

数据与产物
  ├── case、原始 samples：按条件共享
  └── processed、checkpoint、result、log：按实验隔离

实验
  ├── 固定测试集
  ├── 实验 manifest
  └── V1、V2 同条件对比
```

## 2. 当前状态与主要风险

制定本方案时，仓库状态如下：

- `main` 位于初始提交 `318c3ec`，适合作为 V1 基线。
- `doc/NETWORK_CHANGES.md` 尚未被 Git 跟踪。
- 当前生成数据位于 `data/`，体积较大，不适合为 V2 整体复制。
- 当前模型路径为 `data/<case>/model/<uc>_<model>.pt`。
- 当前结果路径为 `data/<case>/result/<uc>_<model>.pkl`。
- 当前日志路径也不包含架构版本或实验编号，部分日志采用覆盖模式。
- 当前测试场景在每次测试运行时重新生成；同一次运行中的模型共享场景，但跨版本、跨时间比较缺少固定 benchmark。
- 当前 checkpoint 只保存 `config` 和 `state_dict`，缺少架构版本、数据版本和源码提交信息。

因此，仅创建一个 Git 分支并不能保证 V1 与 V2 互不影响。源码、运行产物和对比数据必须同时隔离。

## 3. Git 分支和工作目录

### 3.1 冻结 V1

将当前基线提交设置为带说明的 tag：

```bash
git tag -a model-v1.0.0 318c3ec -m "Baseline STGCN v1"
```

`main` 用于保存 V1 基线。除非明确决定发布一个 V1 修复版本，否则不在 `main` 上进行 V2 开发。

### 3.2 创建 V2 worktree

建议使用独立 worktree，而不是在同一个目录中频繁切换分支：

```bash
git worktree add ../GridUC-Graph-v2 -b feature/stgcn-v2 main
```

两个目录的职责为：

```text
GridUC-Graph/       # main，V1 基线，原则上只读
GridUC-Graph-v2/    # feature/stgcn-v2，后续开发和实验
```

当前未跟踪的 `doc/NETWORK_CHANGES.md` 应在执行上述操作时复制或移动到 V2 worktree，并在 V2 分支提交。不要为了保存规划文档而改变 V1 基线提交。

### 3.3 提交粒度

建议按职责拆分提交，例如：

```text
docs: add network evolution and version management plans
test: add baseline model contract tests
refactor: add model registry and artifact namespace
feat: add fusion graph attention model
test: add v1-v2 comparison workflow
```

每个提交只处理一类问题。不要在同一个提交中同时进行目录重构、网络修改、特征修改和训练参数调整。

## 4. 模型源码隔离

### 4.1 保留 V1 接口

原有 `lib/stgcn.py` 和现有 layers 作为 V1 兼容实现保留。V2 顶层模型及行为不同的模块使用新文件：

```text
lib/
├── stgcn.py                         # V1，保持向后兼容
├── stgcn_v2.py                      # V2 顶层模型
└── layers/
    ├── module_static.py             # 行为相同时可共享
    ├── module_dync.py               # 行为相同时可共享
    ├── module_fusion.py             # V1，保持不变
    ├── layer_graph_attention.py     # V2 新增
    └── module_fusion_attention.py   # V2 新增
```

文件名最终可根据实现细化，但必须能从名称区分共享实现、V1 兼容实现和 V2 专用实现。

### 4.2 复用原则

- 行为和接口完全一致：直接复用现有 layer 或 block。
- 接口相似但计算细节不同：新增 V2 layer/module。
- 只有超参数不同：可以配置化，但新参数不得改变 V1 默认行为。
- 避免在共享模块中堆积大量 `if version == ...` 分支。
- 实现 V2 时发现的 V1 bug 不应顺手修改；应单独记录、测试并决定是否发布 V1 修复版本。

### 4.3 模型注册

trainer 和 tester 应通过明确的模型名称加载不同架构：

```python
MODEL_CLASSES = {
    "mlp": MLP,
    "stgcn_v1": STGCN,
    "stgcn_v2": STGCNV2,
}
```

迁移期间可保留旧名称作为兼容别名：

```python
"stgcn": STGCN  # legacy alias，等价于 stgcn_v1
```

这样 V1 和 V2 可以在一次测试运行中同时加载，而不需要切换 Git 分支来比较结果。

## 5. 数据和实验产物隔离

### 5.1 输入与输出使用不同根目录

V2 分支的新实验框架应引入两个独立概念：

- `input_root`：case 文件和可共享的原始 samples。
- `artifact_root`：当前实验的 processed data、checkpoint、result 和 log。

`artifact_root` 只在 V2 分支实现，不要求修改 `main`。新实验框架同时注册并运行 V1 兼容架构和 V2 架构，两者的输出通过不同 `run_id` 写入 V2 worktree 的实验目录。这里的 V1 指 V1 模型架构，而不是在冻结的 `main` worktree 中运行新的实验基础设施。

V2 worktree 可以从 V1 目录只读访问已有的大体积输入和 legacy 产物，但所有新输出必须写入 V2 的 `artifact_root`。`main` worktree 及其现有 `data/<case>/model`、`data/<case>/result` 和 `log/` 保持只读。不要把整个 V1 `data/` 以可写 symlink 连接到 V2，否则仍有误覆盖 V1 产物的风险。

### 5.2 实验目录

Git branch 和 worktree 已经标识源码基线，具体架构则由运行配置和 manifest 记录，因此实验编号不再机械重复架构版本和日期。新派生产物使用两级编号：

- `comparison_id`：表示一组使用相同 benchmark 和对比条件的实验。
- `run_id`：表示该组中的一次具体运行，用于区分方案、随机种子或训练配置。

V1 兼容架构和 V2 架构使用相同的 `comparison_id`，由 V2 实验框架写入同一个 `artifact_root`，再以不同 `run_id` 隔离。冻结的 V1 worktree 不需要实现或维护这套目录。

```text
artifacts/
└── comparisons/
    └── <comparison_id>/
        └── runs/
            └── <run_id>/
                ├── manifest.json
                ├── processed/
                │   └── <case>/<uc>.pt
                ├── checkpoints/
                │   └── <case>/<uc>/<model>.pt
                ├── results/
                │   └── <case>/<uc>/<test_suite>/<model>.pkl
                └── logs/
                    └── <case>/...
```

编号应简短、稳定且能说明对比目的。日期不作为核心编号；需要记录时间时，将其作为运行元数据保存。例如：

```text
comparison_id = fusion-attention_case118_tcuc

V2 artifact_root（运行 V1 兼容架构）:
  run_id = baseline_seed26

V2 artifact_root（运行 V2 架构）:
  run_id = fusion-attn_seed26
  run_id = fusion-attn_seed27
```

当一次运行同时覆盖多个 case 或 UC 类型时，可从 `comparison_id` 省略相应字段，由下层 `<case>/<uc>` 目录区分。`run_id` 只加入实际会区分运行的部分；架构类型已经由运行配置和 manifest 表达时，不必机械加入 V1/V2 字样。

现有 `data/<case>/model`、`data/<case>/result` 和 `log/` 视为 legacy 产物，不移动、不重命名、不覆盖。

### 5.3 数据共享规则

| 内容                 | 是否共享 | 说明                                     |
| -------------------- | -------: | ---------------------------------------- |
| `data/case/*.xlsx` |       是 | 原始问题定义，不应由模型实验修改         |
| raw samples          | 通常可以 | 仅当采样逻辑、标签定义和所需字段未变化   |
| processed tensors    |   视情况 | 节点特征或输入 schema 变化时必须重新生成 |
| checkpoint           |       否 | 与模型架构、配置和训练过程绑定           |
| result               |       否 | 与模型、测试集和求解参数绑定             |
| log                  |       否 | 现有日志存在覆盖行为                     |
| 固定测试集           |       是 | V1/V2 应共享同一个 benchmark             |

## 6. 实验 manifest 与 checkpoint

### 6.1 Manifest

每个实验目录必须包含 manifest，至少记录：

```text
comparison_id
run_id
git_commit
model_type
architecture_version
dataset_schema_version
case
uc_type
sample_solver
test_solver
random_seed
train/test split
epochs
batch_size
learning_rate
model config
checkpoint path
test suite id
created_at
```

运行摘要和论文表格应以 manifest 与结构化 result 为依据，不依赖日志文件推断实验条件。`manifest.json` 的具体格式、存放层级和生成方式后续单独讨论。

### 6.2 Checkpoint schema

现有 V1 checkpoint 格式继续由 legacy loader 支持。V2 checkpoint 建议使用显式版本：

```python
{
    "checkpoint_schema": 2,
    "model_type": "stgcn_v2",
    "architecture_version": "2.0",
    "config": {...},
    "state_dict": {...},
    "dataset_schema_version": "...",
    "git_commit": "...",
}
```

新 checkpoint 中的配置优先保存为普通字典，降低 dataclass 所在模块路径变化造成的反序列化风险。

加载时应验证：

- 模型类型是否匹配。
- checkpoint schema 是否受支持。
- 输入特征维度是否与 dataset schema 匹配。
- 必需元数据是否完整。

## 7. 公平比较

### 7.1 固定 benchmark

当前测试场景会在每次运行时重新生成。为了保证跨版本结果可比，应生成带元数据的固定测试集：

```text
benchmarks/
└── <case>/
    └── <uc>/
        └── suite_seed1000000_n20_<solver>.pkl
```

测试集或其配套 manifest 应记录：

- 随机种子及范围。
- `case` 和 `uc_type`。
- ground-truth solver 模式。
- Gurobi time limit、MIP gap 等参数。
- 请求的测试数和成功生成的实际测试数。
- 场景数据或可验证的数据指纹。

### 7.2 控制变量

比较 V1 和 V2 时，以下条件必须一致：

- raw samples 和 train/test split。
- 随机种子。
- batch size、epoch、优化器和学习率。
- 测试场景。
- ground-truth solver 与 accelerated solver 模式。
- threshold ladder。
- time limit 和 MIP gap。

如果某项必须不同，应在 manifest 和结果报告中明确标注，不把该实验描述为单一架构对比。

### 7.3 建议指标

至少保存以下指标：

- train/test BCE loss。
- fix ratio。
- fix accuracy。
- feasibility rate。
- objective gap。
- NN inference time。
- solver time。
- total solve time。
- speedup。
- 模型参数量。
- 峰值显存或内存占用。

## 8. 网络修改实施顺序

按照 `NETWORK_CHANGES.md`，建议一次只改变一个主要因素。

### E0：V1 基线复现

- 不修改网络架构。
- 建立模型注册、实验目录、manifest 和固定测试集。
- 在新实验系统中运行 `stgcn_v1`。
- 确认结果与 legacy 流程在允许误差内一致。

E0 是后续所有实验的前置条件。

### E1：Fusion Graph Attention

- 仅替换 Fusion 中的 `ChebConvLayer`。
- Static、Dynamic、输入特征和训练参数保持不变。
- 优先在 case5 完成 shape、数值和端到端验证，再运行 case118。
- 单独统计 attention 的参数量、推理耗时和显存开销。

这是风险最低且最容易解释的首个 V2 架构实验。

### E2：Dynamic Graph Attention

- 在独立实验中替换 `STConvBlock` 中的空间卷积。
- 不与 Fusion Attention 的首次验证混在同一实验中。
- 重点检查 `[B * T, N, H]` 下的内存和运行时间。
- 在 case5、case118 验证后，再决定是否运行 case2383 和 case6515。

### E3：Node Features

- 提升 `dataset_schema_version`。
- 在新实验目录中重新生成 processed tensors。
- 保留旧 processed tensors。
- 同时比较“V1 架构 + 新特征”和“V2 架构 + 新特征”，区分特征收益与架构收益。

### E4：Multi-task Learning

- 单独升级输出、标签和 checkpoint schema。
- UC logits、LMP、power output 使用清晰的独立 head。
- 为每个任务记录 loss、权重和评价指标。
- 不把多任务改动混入首次 attention 实验。

## 9. 实施前的保护测试

在修改网络前，应先建立以下自动化测试：

- 输入输出 shape 合同，输出保持 `[B, G, T]`。
- `edge_mask=0` 的线路不产生消息。
- generator-to-bus 映射正确。
- 现有 V1 checkpoint 可以加载和推理。
- 固定 seed 下 V1 输出可复现。
- V1/V2 能被同一个 trainer 和 tester 注册、调用。
- 不同 `comparison_id` 或 `run_id` 生成不同产物路径。
- V2 运行不会改变 legacy checkpoint/result 的内容、校验值和修改时间。
- Attention 在无入边节点、线路全部 masked 等边界情况下不产生 NaN/Inf。
- case5 能完成一次采样后处理、训练、加载、测试和汇总 smoke test。

对神经网络数值结果的测试应使用合理容差，不依赖跨设备完全逐位一致。

## 10. 发布与验收条件

V2 阶段性版本合并或打 tag 前，至少满足：

- `main` 和 `model-v1.0.0` 未被改变。
- V1 legacy checkpoint 仍可加载。
- V1 回归测试通过。
- V1/V2 的 processed、checkpoint、result 和 log 路径完全隔离。
- V1/V2 使用同一个固定测试集完成比较。
- Fusion Attention 在 case5 完成端到端测试。
- case118 至少完成一次受控对比。
- 每个实验具有完整 manifest，可由 Git commit 和配置复现。
- `NETWORK_CHANGES.md` 中的 Completed Changes 和 Architecture Decisions 已同步更新。

可以按阶段发布 tag，例如：

```text
model-v1.0.0                 # 当前基线
model-v2.0.0-fusion-attn     # Fusion Attention
model-v2.1.0-dynamic-attn    # Dynamic Attention
model-v2.2.0-node-features   # 新节点特征
model-v3.0.0-multitask       # 输出接口发生较大变化
```

## 11. 后续操作原则

完成本方案的初始化后，后续操作遵守以下约定：

1. V2 开发只在 `feature/stgcn-v2` 对应 worktree 中进行。
2. V1 源码和 legacy 产物默认只读。
3. 新实验必须先指定稳定的 `comparison_id`，每次具体运行必须指定在该对比组内唯一的 `run_id`。
4. 新架构先通过 case5，再运行更大的 case。
5. 每次实验只引入一个主要变量。
6. 任何会改变输入张量含义的修改都提升 `dataset_schema_version`。
7. 任何不兼容的 checkpoint 变化都提升 `checkpoint_schema`。
8. 对比报告必须引用固定测试集和完整 manifest。
9. 不以复制、重命名临时文件代替正式的实验命名空间。
10. 架构决策、已完成改动和实验结论及时回写项目文档。

方案的核心是：使用 Git tag 和 worktree 保护源码基线，使用模型注册表保持运行兼容，使用 comparison/run namespace 保护派生产物，使用固定 benchmark 保证 V1/V2 比较公平。
