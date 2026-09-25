# GE-Sampling — Learning-Accelerated Unit Commitment

Use machine learning (STGCN / MLP) to predict the on/off schedule of unit commitment (UC), fix the high-confidence 0/1 variables up front, and thereby accelerate the Gurobi MILP solve.

Three UC problems are supported (`uc_type`):

| uc_type | Meaning |
|---|---|
| `tcuc` | UC with transmission constraints |
| `topo` | UC with one randomly selected monitored line under maintenance (topology change) |
| `scuc` | UC with N-1 security constraints |

Three ways of modeling the transmission / security constraints are supported (`mode`, see SOLVE_MODES below):

| mode | Meaning |
|---|---|
| `none` | No line flow constraints at all — only unit constraints and power balance |
| `lazy` | Flow constraints added on demand via a Gurobi lazy callback (cuts are globally valid, and already-added ones are not re-added) |
| `dense` | All flow constraints laid out at once at model build time |

`lazy` and `dense` enforce **exactly the same** feasible region (both cover only the lines where `line_monitor` is true), they differ only in how the constraints reach Gurobi, so their optimal objective values must agree; `none` is a relaxation, so its objective can never be higher than the other two.

On large cases the solve times differ greatly: `dense` may fail to converge for a long time on the case2383 SCUC, while `lazy` is the only viable choice on large cases.

## Project structure

```
GE-Sampling/
├── doc/README.md        # this document
├── doc/requirements.txt # dependencies
├── lib/                 # all core logic
│   ├── case_loader.py   # read UCData from data/case/*.xlsx
│   ├── case_modifier.py # random scenario generation (load / offer price / maintenance line / N-1 monitoring)
│   ├── ptdf_solver.py   # PTDF / LODF / post-contingency PTDF computation
│   ├── uc_model.py      # Gurobi modeling and solving: build_uc(data, uc_type, mode) + solve_uc
│   ├── sampler.py       # sampling: sample_uc for a single sample, run_sampling for batches (multiprocessing / resumable)
│   ├── data_loader.py   # UCData -> STGCNInput conversion, chunked preprocessing, DataLoader
│   ├── models/          # model families: stgcn/, stgcn_attn/, esa/, multitask/
│   │   └── mlp.py       # MLP baseline model
│   ├── trainer.py       # unified training entry train_model("stgcn"|"mlp", ...)
│   ├── tester.py        # test case generation, accelerated-solve evaluation, result storage, summary table printing
│   ├── runner.py        # pipeline orchestration: run_all(case, stages=[...], ...)
│   └── toolkit.py       # logging, timing, random seeds, pkl.gz I/O
├── run_case5.py         # per-case entry script (the only entry point, see below)
├── run_case118.py
├── run_case2383.py
├── run_case6515.py
├── data/                # default input_root: case definitions and reusable raw samples
├── log/                 # per-case stage logs (gitignored)
└── .gitignore
```

There is no CLI. The whole pipeline runs inside a single Python process by calling `lib.runner.run_all` from `run_<case>.py`.

## Model names and directories

| Registry name and filename suffix | Source directory | Description |
|---|---|---|
| `mlp` | `lib/models/mlp.py` | MLP baseline |
| `stgcn` | `lib/models/stgcn/` | Original STGCN |
| `stgcn_attn_0` | `lib/models/stgcn_attn/` | Fusion attention |
| `stgcn_attn_1` | `lib/models/stgcn_attn/` | Dynamic and fusion attention |
| `stgcn_attn_2` | `lib/models/stgcn_attn/` | Deeper attention network with residual blocks |
| `esa` | `lib/models/esa/` | ESA interface; computation is not implemented |
| `multitask` | `lib/models/multitask/` | Shared STGCN encoder with UC/LMP heads |

Use these exact names in configuration and model/result/log filenames. Source
packages are unchanged; the attention models still share `stgcn_attn/`.
Historical model names are recognized only when reading old checkpoint metadata,
not as names for new calls. Existing checkpoint, result, and log bytes remain
unchanged; only filenames are renamed. Checkpoint `architecture_version` values
are retained for compatibility and do not define registry names.

## Environment

```bash
pip install -r doc/requirements.txt
```

A valid Gurobi license is required. Verified setups:

| Platform | Configuration |
|---|---|
| macOS | Apple M3 Pro / 18GB / macOS 15.7.7 / torch 2.11.0 (`device="mps"`) |
| Linux | Xeon Gold 5217 ×2 (32 threads) / 187GB / Ubuntu 24.04.4 / Quadro RTX 6000 24GB / torch 2.11.0+cu130 (`device="cuda"`) |

Both use gurobipy 13.0.1. Training and testing of the large cases (case2383 / case6515) is best run on the GPU machine: one STGCN epoch on case6515 takes about 38s on CUDA versus about 650s on MPS.

## How to run

One script per case — edit the parameters at the top of the script and execute it:

```bash
python run_case5.py       # case5: small case, fast
python run_case118.py     # case118
python run_case2383.py    # case2383
python run_case6515.py    # case6515: large case, slow to sample and to solve
```

The checked-in case scripts resolve `input_root` to the sibling frozen V1
worktree (`../GridUC-Graph/data`) and write each run directly below this V2
worktree. Adjust `V1_INPUT_ROOT` if the two worktrees are stored elsewhere.

Each script is just a single `run_all` call:

```python
from lib.runner import run_all

if __name__ == "__main__":
    run_all(
        case="case118",
        stages=["sample", "process", "train", "test", "summary"],
        n_samples=1000,
        n_test=20,
        epochs=50,
        batch_size=32,
        device="mps",
        sample_solver="dense",
        test_solver="dense",
        workers=None,
        input_root="data",
        output_root=".",
    )
```

### run_all parameters

| Parameter | Default | Description |
|---|---|---|
| `case` | required | Case name, e.g. `case5` / `case118` / `case2383` / `case6515` |
| `stages` | all five | Any subset of `sample` / `process` / `train` / `test` / `summary`, executed in the given order |
| `n_samples` | 1000 | Number of samples (indexed 0 ~ n_samples−1) |
| `n_test` | 20 | Number of test cases |
| `epochs` / `batch_size` | 20 / 32 | Training hyperparameters |
| `device` | `cpu` | `cpu` / `cuda` / `mps` |
| `train_devices` | none | Optional device list such as `("cuda:0", "cuda:1", "cuda:2", "cuda:3")`; training tasks are split across one spawned process per device |
| `workers` | no parallelism | `"auto"` (all cores) or an integer (number of processes) — **effective only in the sample stage** |
| `sample_solver` | `dense` | The mode used to solve for training labels in the sample stage |
| `test_solver` | `dense` | The mode used in the test stage, for both ground truth and the accelerated solve |
| `uc_types` | all three | A subset of `("tcuc", "topo", "scuc")` |
| `models` | both | A subset of `("stgcn", "mlp")` |
| `input_root` | `data` | Read source for `case/*.xlsx` and reusable `<case>/samples/<uc>/*.pkl` |
| `output_root` | `.` | Writable project root for models, results, benchmarks, and logs |
| `process_chunk_size` | `200` | Number of samples converted per processing chunk |
| `train_ratio` | `0.8` | Sequential train/test split used when processing |
| `processed_path` | none | Optional path template such as `/v1/data/{case}/processed/{uc}.pt`; used when the process stage is omitted |

`stages` is the mechanism for resuming: if the models are already trained and you only want to re-evaluate, pass `stages=["test", "summary"]`.

Each task (case × uc_type × model) runs in its own try/except — a failure is recorded rather than aborting the run, and the list of failures is printed at the end.

### The five stages

**sample** — Generate UC samples together with their optimal solutions, one `pkl.gz` per sample. With `skip_existing=True`, re-running only fills in the missing samples. A failed single-sample solve affects only that sample.

**process** — Raw samples → training tensors `.pt` under `input_root/<case>/processed/`. Chunk size is 200 by default and the train/test split is a sequential 0.8 cut. The process stage overwrites the matching `<uc>.pt`; omit this stage and set `processed_path` to reuse a file elsewhere.

**train** — The best model by test loss is saved at `data/<case>/model/<uc>_<model>.pt`, with checkpoint structure `{config, state_dict}`. When `train_devices` is set, the `(uc_type, model)` tasks are assigned round-robin to those devices; each device has one process and runs its assigned queue sequentially.

The two models differ in size by three orders of magnitude: on case6515 the MLP's first layer flattens the input into 682,320 dimensions fully connected to 512, for 3.6×10⁸ parameters (1.45GB); the STGCN shares weights over nodes and edges, for 7.5×10⁵ parameters (3.0MB).

**test** — Load or generate `n_test` ground truth test cases (the same set is reused by every model under that uc_type, to keep the comparison fair), then evaluate model by model. Generated cases, including their ground truth, are saved below `data/<case>/benchmarks/` and reused by later test runs when the UC type, solver mode, seed, requested count, and solver configuration match. Training and testing share `<output_root>/data/<case>/model/<uc>_<model>.pt`. Results are stored at `<output_root>/data/<case>/result/<uc>_<model>.pkl`. Model names are preserved exactly as supplied.

**summary** — Reprint the summary table from stored results, so format changes need no re-testing (`print_case_summary` in `lib/tester.py`).

### Fixing method (threshold ramp-up)

The model outputs an on-probability for every (unit, time period). Given a threshold `t`, `prob ≥ t` is fixed to on, `prob ≤ 1−t` is fixed to off, and the rest is left to the solver; the fixing scheme is first repaired against the minimum up/down time constraints, then hard-fixed and solved.

The threshold ramps from 0.95 to 1.0 in steps of 0.01 — the higher the threshold, the fewer variables fixed, the looser the constraints, and the easier feasibility becomes. **The first threshold that yields a feasible solution is the one adopted, and its solve time is the reported time** (earlier failed attempts are not timed, but they do accumulate in `total_solve_time`). If even 1.0 is infeasible, the prediction is abandoned and the original UC is solved with nothing fixed (necessarily feasible); the per-sample log then shows `threshold=none`.

> These solves at different thresholds are independent, so in production they could run in parallel and take the fastest feasible solution; this repository runs them serially because of limited compute (see `run_test` in `lib/tester.py`). On large cases, rebuilding the model per threshold dominates the runtime.

The default Gurobi parameters are defined by `UCConfig` in `lib/uc_model.py` and shared by the sampling and testing stages. Pass a customized `UCConfig` to override them for a specific run.

### Test case generation

Test cases are cached as validated benchmark files. Seeds increment from `TEST_SEED_BASE`; the complete `UCData`, ground-truth solution, objective, solve time, successful seed list, security formulation, and solver configuration are persisted. A later test with matching metadata loads the benchmark and skips generation. Within a run all models therefore share exactly the same cases and ground truth, even when tested in separate invocations.

If a case fails to solve within the time limit during generation, `FAILED to solve, skipped` is printed and it is skipped, so the actual number of test cases for that uc_type will be lower than `n_test`.

### The two solver switches

| Parameter | Role |
|---|---|
| `sample_solver` | Which mode the sample stage uses to solve for training labels |
| `test_solver` | The mode used in the test stage for the ground truth **and** for the NN-fixed accelerated solve (they must match, otherwise the speedup is meaningless) |

Crossing them is a supported experiment (e.g. train on `none` labels and evaluate against `dense` ground truth): the mode only changes the MILP's constraint set, not the `UCData` fields, the feature tensors, or the model architecture, so a model trained under any mode can be tested under any mode.

Models, results, benchmarks, and logs use stable paths by case and model. Copy artifacts explicitly when a separate snapshot is needed.

Before a sample is written to disk, the dense PTDF matrices are stripped out (a single matrix reaches GB scale on large cases); on read-back, `restore_ptdf` deterministically rebuilds them from the topology and the random draws (`maintenance_line` / `cc_monitor`).

## Summary table

`print_case_summary` groups by uc_type, with one MILP baseline row plus one row per model in each group:

```
 UC       Method      Avg.gap(%)   Gap STD   Speedup  Threshold  Fix ratio(%)  Fix acc.(%)
 tcuc     MILP                 -         -         -          -             -            -
          MLP             1.0393    0.1974     4.73x     0.9560         83.25        99.71
          STGCN           0.0483    0.1674     1.32x     0.9715         71.80        99.94
```

`Threshold`, `Fix ratio(%)`, and `Fix acc.(%)` are averages over the test samples. A fallback sample whose threshold is `none` is excluded from the threshold average; its zero fix ratio is retained, while its undefined fix accuracy is excluded. The per-sample values remain available in the test stage logs.

## Logs

Every stage writes logs below `<output_root>/log/<case>/<stage>/` and to the terminal. `sample` and `process` both write into `sample/`. Train and test logs overwrite the same case/model log on each run; copy a log explicitly when you need to retain a snapshot.

## Input and output directories

Inputs and generated artifacts have separate roots. `input_root` may point at the V1 worktree's data directory and can be mounted read-only when the `sample` stage is not used:

```
<input_root>/
├── case/<case>.xlsx
└── <case>/
    ├── samples/<uc>/<sid>.pkl
    └── processed/<uc>.pt             # optional reusable legacy input

<output_root>/
├── data/<case>/
│   ├── model/<uc>_<model>.pt
│   ├── result/<uc>_<model>.pkl
│   └── benchmarks/<uc>_<solver>_seed<seed>_n<count>.pkl
└── log/<case>/<stage>/...
```

The `sample` and `process` stages write reusable samples and processed tensors under `input_root`, so that root must be writable when either stage runs. Processing writes `input_root/<case>/processed/<uc>.pt`, atomically replacing a same-name file. When `process` is omitted, training reads that path by default or the explicit `processed_path` template. The loader performs lightweight checks of train/test entries, tensor batch dimensions, graph fields, and—when raw samples are available—the total sample count. Ordinary logs use `<output_root>/log/<case>/<stage>/`. LMP label logs are stored at `log/<case>/lmp_labels/`; IIS diagnostic files keep their existing locations.

When moving between machines: `processed/` is generated deterministically from `samples/`, so just re-run the `process` stage instead of transferring it; MLP checkpoints can reach GB scale on large cases. Checkpoint loading uses `map_location`, so weights trained on CUDA load directly on mps/cpu.

## .gitignore rules

- `data/` and `log/`: local inputs, models, results, and logs are not committed.
- Python caches (`__pycache__/` etc.), virtual environments (`.venv/` etc.), IDE configs (`.vscode/`, `.idea/`, `.claude/`), and macOS metadata (`.DS_Store` etc.) are all ignored.
