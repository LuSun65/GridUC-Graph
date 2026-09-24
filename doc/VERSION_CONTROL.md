# 模型版本与实验管理方案

## 1. 目标

本项目将按照 `doc/NETWORK_CHANGES.md` 逐步修改 STGCN，包括图注意力、节点特征和多任务学习等内容。版本管理需要满足以下要求：

- 当前实现作为 V1 基线保持不变，并且能够随时复现。
- 后续开发全部在 V2 分支和独立工作目录中进行。
- V2 可以复用行为完全一致的现有 layer 和 block；需要改变内部逻辑时，应新增 V2 实现，避免改变 V1 行为。
- V1 与 V2 可以只读共享兼容的 raw samples 和 processed data；新产物不得覆盖已有内容。
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
  ├── case、raw samples、processed：兼容时只读共享
  └── checkpoint、result、log：按运行隔离

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

### 3.4 本地测试与运行入口

`tests/` 只用于本地验证，不纳入 Git 版本控制。`.gitignore` 必须保留对应规则；已经被 Git 跟踪的测试文件应仅从索引移除，不能因此删除本地副本。

仓库根目录的 `run_case*.py` 继续由 Git 跟踪，但其中经常包含设备编号、运行阶段、样本数量和 `run_id` 等机器或单次实验相关参数。除非是在有意更新公共默认配置，否则不要提交这类本地调整。可复用的功能实现与实验约定应写入 `lib/` 和 `doc/` 中，运行入口中也不得保存凭据或其他敏感信息。

## 4. 模型源码隔离

### 4.1 按模型家族组织源码

模型统一从 `lib.models` 导入，按模型家族集中存放；旧顶层模型路径及 `lib/layers/` 已移除，不提供兼容导出。

```text
lib/models/
├── model_input.py       # 共享输入 dataclass
├── model_registry.py    # 统一模型注册
├── mlp.py
├── common/              # 跨版本共享基础层
├── stgcn/               # V1：原始 STGCN
├── stgcn_attn/          # V2、V2.1、V2.2：Attention 系列
├── esa/                 # V3：ESA 接口，计算尚未实现
└── multitask/           # V4：共享 V1 编码器的 UC/LMP 多任务模型
```

配置 dataclass 与实现同文件。复用的 V1 分支可直接从 `stgcn/` 导入，共享算子从 `common/` 导入。数据加载器和模型直接使用 `model_input.py` 中的共享输入类型。预处理数据按当前代码生成，不提供旧类路径兼容。

### 4.2 复用原则

- 行为和接口完全一致：直接复用现有 layer 或 block。
- 接口相似但计算细节不同：新增 V2 layer/module。
- 只有超参数不同：可以配置化，但新参数不得改变 V1 默认行为。
- 避免在共享模块中堆积大量 `if version == ...` 分支。
- 实现 V2 时发现的 V1 bug 不应顺手修改；应单独记录、测试并决定是否发布 V1 修复版本。

### 4.3 模型注册

注册表位于 `lib/models/model_registry.py`，对应规则如下：

| 版本 | 目录 | 注册名称 | 别名 |
|---|---|---|---|
| 基线 | `mlp.py` | `mlp` | — |
| V1 | `stgcn/` | `stgcn_v1` | `stgcn` |
| V2.0 | `stgcn_attn/` | `stgcn_v2` | `stgcn_attn`、`stgcn-v2` |
| V2.1 | `stgcn_attn/` | `stgcn_v2.1` | `stgcn-v2.1` |
| V2.2 | `stgcn_attn/` | `stgcn_v2.2` | `stgcn-v2.2` |
| V3 | `esa/` | `stgcn_v3` | `esa` |
| V4 | `multitask/` | `stgcn_v1_mtl` | `multitask`、`stgcn_v4` |

`stgcn_attn` 固定对应 V2.0；V2.1、V2.2 需明确使用对应的注册名称或别名。

目录改名不改变模型计算、现有注册名称或检查点架构版本。V4 是模型家族编号；
其 `architecture_version` 仍为 `1.0`，保持现有多任务检查点的元数据兼容。
Python 导入使用新目录路径；不提供旧 `lib.models.v1/v2/v3` 包路径兼容。
以配置字典和 `state_dict` 保存的检查点不依赖这些旧包路径；直接 pickle
旧配置对象或整个模型的产物需要在旧环境转存为配置字典和 `state_dict`。
产物文件名仍使用调用方传入的模型名称，保存和加载应使用同一名称；
`stgcn_v4` 别名不会自动查找名为 `stgcn_v1_mtl.pt` 的文件。

## 5. 数据和实验产物隔离

### 5.1 输入与输出使用不同根目录

V2 分支的新实验框架只需区分输入与运行产物：

- `input_root`：case 文件、raw samples，以及可复用的 processed data。
- `output_root`：各 `run_id` 目录的可写父目录。

新实验框架同时注册并运行 V1 兼容架构和 V2 架构。两者可以通过配置直接读取同一份 V1 processed data；只有输入特征、标签、样本集合、数据划分或处理逻辑发生变化时才重新运行 process。训练及测试输出通过不同 `run_id` 写入 V2 worktree 的实验目录。这里的 V1 指 V1 模型架构，而不是在冻结的 `main` worktree 中运行新的实验基础设施。

V2 worktree 可以从 V1 目录只读访问已有的大体积输入和 legacy processed data；所有新文件写入 `<output_root>/runs/<run_id>/<case>/`。`main` worktree 及其现有 `data/<case>/processed`、`data/<case>/model`、`data/<case>/result` 和 `log/` 保持只读。不要把整个 V1 `data/` 以可写 symlink 连接到 V2，否则仍有误覆盖 V1 数据和产物的风险。

### 5.2 实验目录

Git branch 和 worktree 已经标识源码基线，具体架构、数据和训练条件由 manifest 记录。目录只使用一个 `run_id`，其唯一职责是防止产物覆盖，不再把模型、case、seed 等信息重复编码进多级 ID。

```text
<output_root>/
└── runs/
    └── <run_id>/
        └── <case>/
            ├── processed/                 # 仅在本次需要重新 process 时存在
            │   └── <uc>.pt
            ├── checkpoints/
            │   └── <uc>/<model>.pt
            ├── benchmarks/
            │   └── <uc>_<solver>_seed<seed>_n<count>.pkl
            ├── logs/
            │   └── <stage>/...
            └── results/
                └── <uc>/<model>.pkl
```

`run_id` 只需简短且唯一，例如：

```text
run_id = r001
run_id = r002
run_id = r003
```

若希望便于人工浏览，也可以使用 `fusion-attn-01` 这类简短名称，但不要求在名称中完整复述架构、case、UC 类型、seed 和日期。需要把若干运行归为一组对比时，在 manifest 中填写可选的 `comparison_group`；它只是报告分组标签，不参与目录寻址，也不要求全局唯一。

现有 `data/<case>/model`、`data/<case>/result` 和 `log/` 视为 legacy 产物，不移动、不重命名、不覆盖。

### 5.3 数据共享规则

| 内容                 | 是否共享 | 说明                                   |
| -------------------- | -------: | -------------------------------------- |
| `data/case/*.xlsx` |       是 | 原始问题定义，不应由模型实验修改       |
| raw samples          | 通常可以 | 仅当采样逻辑、标签定义和所需字段未变化 |
| processed tensors    | 条件共享 | 输入和处理方式未变时直接只读复用       |
| checkpoint           |       否 | 与模型架构、配置和训练过程绑定         |
| result               |       否 | 与模型、测试集和求解参数绑定           |
| log                  |       否 | 现有日志存在覆盖行为                   |
| 固定测试集           |       是 | V1/V2 应共享同一个 benchmark           |

### 5.4 数据管理预案

当前阶段不建立独立的数据版本系统。V1、V2 表示模型架构，不表示数据归属；默认继续只读复用现有 raw samples 和 processed data，并在 manifest 中记录实际读取路径、样本数量和 train/test split。

后续需要增加训练数据时，按以下简单规则处理：

- 尚未用于正式对比的数据可以原地补齐，但运行开始后不再改变本次使用的样本集合。
- 已用于正式结果的数据应保留；新增样本放入新的目录或使用明确的文件名后缀，避免旧实验无法复现。
- 只有输入特征、标签、样本集合、数据划分或 process 逻辑改变时，才重新生成 processed data。
- V1/V2 公平对比应读取相同的 raw samples 或同一份 processed data，并使用相同的数据划分。
- 如果未来出现多个长期并存、容易混淆的数据版本，再引入统一的数据版本号或内容校验值；当前不预先实现 `dataset_id`、`dataset_schema_version` 或多层 fingerprint。

V1 legacy processed 缺少完整元数据时，可以先检查样本数、tensor shape 和划分是否符合预期，再作为只读输入使用。新生成的 processed data 默认保存在生成它的 run 目录中，其他 run 可以通过 `processed_path` 只读引用；不得写回或覆盖 V1 文件。

## 6. 实验 manifest 与 checkpoint

### 6.1 Manifest

每个实验目录必须包含 manifest，至少记录：

```text
run_id
comparison_group（可选）
git_commit
model_type
architecture_version
input_root
processed_path
sample_count
sample_id_range
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

`run_id` 负责唯一定位产物，`comparison_group` 仅在需要汇总若干运行时填写。`sample_count` 应记录实际参与处理的样本数，而不只记录请求数量。`processed_path` 可以指向 V1 的只读 legacy 文件，也可以指向某个 run 新生成的文件；路径之外不强制维护额外的数据 ID 或指纹。

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
    "input_shapes": {...},
    "git_commit": "...",
}
```

新 checkpoint 中的配置优先保存为普通字典，降低 dataclass 所在模块路径变化造成的反序列化风险。

加载时应验证：

- 模型类型是否匹配。
- checkpoint schema 是否受支持。
- 输入特征维度是否与 checkpoint 记录匹配。
- 必需元数据是否完整。

## 7. 公平比较

### 7.1 固定 benchmark

测试阶段使用带元数据的固定测试集，以保证同一 `run_id` 内不同模型以及跨次调用的结果可比。首次运行时生成并保存，后续调用只有在 case、UC 类型、solver 模式、随机种子、请求数量和 solver 配置全部匹配时才复用：

```text
<output_root>/runs/<run_id>/<case>/benchmarks/
└── <uc>_<solver>_seed<seed>_n<count>.pkl
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

- 在新的 run 目录中重新生成 processed tensors，并在 manifest 中记录来源与处理说明。
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
- 不同 `run_id` 生成不同产物路径。
- V2 运行不会改变 legacy checkpoint/result 的内容、校验值和修改时间。
- Attention 在无入边节点、线路全部 masked 等边界情况下不产生 NaN/Inf。
- case5 能完成一次采样后处理、训练、加载、测试和汇总 smoke test。

对神经网络数值结果的测试应使用合理容差，不依赖跨设备完全逐位一致。

## 10. 发布与验收条件

V2 阶段性版本合并或打 tag 前，至少满足：

- `main` 和 `model-v1.0.0` 未被改变。
- V1 legacy checkpoint 仍可加载。
- V1 回归测试通过。
- 共享的 processed data 已检查样本数、shape 和数据划分并保持只读；不兼容的数据重新生成且不覆盖旧文件。
- V1/V2 的 checkpoint、result 和 log 路径完全隔离。
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
model-v3.0.0-esa             # ESA（实现后发布）
model-v4.0.0-multitask       # UC/LMP 多任务模型家族
```

## 11. 后续操作原则

完成本方案的初始化后，后续操作遵守以下约定：

1. V2 开发只在 `feature/stgcn-v2` 对应 worktree 中进行。
2. V1 源码和 legacy 产物默认只读。
3. 每次具体运行必须指定唯一的 `run_id`；只有需要汇总对比时才设置可选的 `comparison_group`。
4. 新架构先通过 case5，再运行更大的 case。
5. 每次实验只引入一个主要变量。
6. 正式实验使用的数据不再原地修改；确需扩充时保留旧文件并记录新路径和样本范围。
7. 输入特征、标签、样本集合、划分或 process 逻辑改变时重新生成 processed data，否则优先只读复用。
8. 任何不兼容的 checkpoint 变化都提升 `checkpoint_schema`。
9. V1/V2 架构对比必须使用相同的数据、样本划分和固定测试集。
10. 对比报告必须引用固定测试集和完整 manifest。
11. 不以复制、重命名临时文件代替正式的运行目录。
12. 架构决策、已完成改动和实验结论及时回写项目文档。

方案的核心是：使用 Git tag 和 worktree 保护源码基线，使用模型注册表保持运行兼容，以单一 `run_id` 隔离产物，优先只读复用兼容数据，并使用固定 benchmark 保证 V1/V2 比较公平。数据版本体系作为未来确有需要时再启用的预案，而不是当前开发的前置负担。
