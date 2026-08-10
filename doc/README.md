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
│   ├── stgcn.py         # STGCN model (layers/ holds its submodules)
│   ├── mlp.py           # MLP baseline model
│   ├── trainer.py       # unified training entry train_model("stgcn"|"mlp", ...)
│   ├── tester.py        # test case generation, accelerated-solve evaluation, result storage, summary table printing
│   ├── runner.py        # pipeline orchestration: run_all(case, stages=[...], ...)
│   └── toolkit.py       # logging, timing, random seeds, pkl.gz I/O
├── run_case5.py         # per-case entry script (the only entry point, see below)
├── run_case118.py
├── run_case2383.py
├── run_case6515.py
├── data/                # data: data/case holds the source xlsx, data/<case>/ holds each case's generated artifacts (gitignored)
├── log/                 # run logs (gitignored)
└── .gitignore
```

There is no CLI. The whole pipeline runs inside a single Python process by calling `lib.runner.run_all` from `run_<case>.py`.

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
| `workers` | no parallelism | `"auto"` (all cores) or an integer (number of processes) — **effective only in the sample stage** |
| `sample_solver` | `dense` | The mode used to solve for training labels in the sample stage |
| `test_solver` | `dense` | The mode used in the test stage, for both ground truth and the accelerated solve |
| `uc_types` | all three | A subset of `("tcuc", "topo", "scuc")` |
| `models` | both | A subset of `("stgcn", "mlp")` |

`stages` is the mechanism for resuming: if the models are already trained and you only want to re-evaluate, pass `stages=["test", "summary"]`.

Each task (case × uc_type × model) runs in its own try/except — a failure is recorded rather than aborting the run, and the list of failures is printed at the end.

### The five stages

**sample** — Generate UC samples together with their optimal solutions, one `pkl.gz` per sample. With `skip_existing=True`, re-running only fills in the missing samples. A failed single-sample solve affects only that sample.

**process** — Raw samples → training tensors `.pt`. Chunk size is 200 (to avoid memory blow-up on large cases); the train/test split is a sequential 0.8 cut.

**train** — The best model by test loss is saved to `data/<case>/model/<uc>_<model>.pt`, with checkpoint structure `{config, state_dict}`.

The two models differ in size by three orders of magnitude: on case6515 the MLP's first layer flattens the input into 682,320 dimensions fully connected to 512, for 3.6×10⁸ parameters (1.45GB); the STGCN shares weights over nodes and edges, for 7.5×10⁵ parameters (3.0MB).

**test** — First generate `n_test` ground truth test cases (the same set is reused by every model under that uc_type, to keep the comparison fair), then evaluate model by model. Results are stored in `data/<case>/result/<uc>_<model>.pkl`.

**summary** — Reprint the summary table from stored results, so format changes need no re-testing (`print_case_summary` in `lib/tester.py`).

### Fixing method (threshold ramp-up)

The model outputs an on-probability for every (unit, time period). Given a threshold `t`, `prob ≥ t` is fixed to on, `prob ≤ 1−t` is fixed to off, and the rest is left to the solver; the fixing scheme is first repaired against the minimum up/down time constraints, then hard-fixed and solved.

The threshold ramps from 0.95 to 1.0 in steps of 0.01 — the higher the threshold, the fewer variables fixed, the looser the constraints, and the easier feasibility becomes. **The first threshold that yields a feasible solution is the one adopted, and its solve time is the reported time** (earlier failed attempts are not timed, but they do accumulate in `total_solve_time`). If even 1.0 is infeasible, the prediction is abandoned and the original UC is solved with nothing fixed (necessarily feasible); the per-sample log then shows `threshold=none`.

> These solves at different thresholds are independent, so in production they could run in parallel and take the fastest feasible solution; this repository runs them serially because of limited compute (see `run_test` in `lib/tester.py`). On large cases, rebuilding the model per threshold dominates the runtime.

The Gurobi parameters are hard-coded in `lib/tester.py` as `time_limit=3600.0` and `mip_gap=1e-3`, and identically in the sample stage (`lib/sampler.py`).

### Test case generation

Test cases are **not cached**: seeds increment from `TEST_SEED_BASE` and every test stage re-solves them on the spot. The reason is that the ground truth depends on the security formulation used, and that information is not persisted, so a cache could not be validated. Within a single run all models share the same set of cases — that is the precondition for model-to-model comparison to mean anything.

If a case fails to solve within the time limit during generation, `FAILED to solve, skipped` is printed and it is skipped, so the actual number of test cases for that uc_type will be lower than `n_test`.

### The two solver switches

| Parameter | Role |
|---|---|
| `sample_solver` | Which mode the sample stage uses to solve for training labels |
| `test_solver` | The mode used in the test stage for the ground truth **and** for the NN-fixed accelerated solve (they must match, otherwise the speedup is meaningless) |

Crossing them is a supported experiment (e.g. train on `none` labels and evaluate against `dense` ground truth): the mode only changes the MILP's constraint set, not the `UCData` fields, the feature tensors, or the model architecture, so a model trained under any mode can be tested under any mode.

**File names do not encode the mode**: the samples / processed / model / result paths are distinguished only by `case` and `uc_type`. The cost is that re-running with a different mode overwrites in place — to keep results for comparison, copy `data/<case>` aside yourself.

Before a sample is written to disk, the dense PTDF matrices are stripped out (a single matrix reaches GB scale on large cases); on read-back, `restore_ptdf` deterministically rebuilds them from the topology and the random draws (`maintenance_line` / `cc_monitor`).

## Summary table

`print_case_summary` groups by uc_type, with one MILP baseline row plus one row per model in each group:

```
 UC       Method      Avg.cost($)  Avg.gap(%)   Gap STD  Avg.time(s)  Time STD   Speedup
 tcuc     MILP         16,536,102           -         -        13.41      2.15         -
          MLP          16,708,010      1.0393    0.1974         2.84      0.76     4.73x
          STGCN        16,544,085      0.0483    0.1674        10.19      3.51     1.32x
```

The per-sample details (including the threshold actually used for each sample, plus fix_ratio and fix_accuracy) live in the test stage logs, not in the summary table.

`Time STD` is worth watching: larger than the mean means the configuration is highly unstable across samples, which usually indicates that some samples degenerated into solving the original problem with nothing fixed.

## Logs

Every stage writes logs automatically to `log/<case>/<stage>/` (and also to the terminal). `sample` and `process` both belong to the sampling stage and both write into `sample/`:

```
log/<case>/sample/sample_<uc>.log         # append mode (multiprocessing-safe)
log/<case>/sample/process_<uc>.log
log/<case>/train/train_<uc>_<model>.log   # overwrite mode
log/<case>/test/gen_<uc>.log              # test case generation
log/<case>/test/test_<uc>_<model>.log
log/<case>/summary/summary.log
```

Note that the train/test logs are in **overwrite** mode, so re-running wipes the previous records. Copy them aside first if you want to keep them.

## Data directory

All generated artifacts are collected under one `data/<case>/` subtree (mirroring `log/<case>/`), so deleting, copying, or backing up a case touches a single folder:

```
data/case/                              # case definition xlsx (the only data checked into git)
data/<case>/
├── samples/<uc>/<sid>.pkl              # raw samples, one pkl.gz per sample
├── processed/<uc>.pt                   # preprocessed tensors
├── processed/.tmp/                     # temporary chunk files from process (cleaned up automatically)
├── model/<uc>_<stgcn|mlp>.pt           # trained models
└── result/<uc>_<model>.pkl             # test results (pkl.gz), the data source for summary
```

When moving between machines: `processed/` is generated deterministically from `samples/`, so just re-run the `process` stage instead of transferring it; the MLP checkpoints in `model/` reach GB scale on large cases. Checkpoint loading uses `map_location`, so weights trained on CUDA load directly on mps/cpu.

## .gitignore rules

- `log/`: no run logs are committed.
- `data/*` except `data/case/`: each case subtree (`data/<case>/`, holding samples, tensors, models, results) is regenerable and is not committed; the case definitions are kept.
- Python caches (`__pycache__/` etc.), virtual environments (`.venv/` etc.), IDE configs (`.vscode/`, `.idea/`, `.claude/`), and macOS metadata (`.DS_Store` etc.) are all ignored.
