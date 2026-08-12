---
name: manage-model-versions
description: Manage isolated model versions and reproducible experiments in the GridUC-Graph repository. Use when Codex needs to plan or initialize a new STGCN/model version, create a baseline tag or development worktree, add versioned model registration, separate datasets/checkpoints/results/logs, compare V1 and V2 fairly, audit version isolation, or prepare a model-version release.
---

# Manage Model Versions

Protect the current model while making new architectures independently runnable and comparable.

## Load the project policy

Read `doc/VERSION_CONTROL.md` completely before taking project actions. Also read `doc/NETWORK_CHANGES.md` when the task changes model architecture, features, targets, or experiment sequencing.

Treat those documents as the project policy. If repository state conflicts with a recorded assumption, report the difference and use the observed state.

## Select the workflow

Classify the request before acting:

- **Plan or audit:** inspect and report only. Do not create branches, tags, worktrees, commits, files, or artifacts.
- **Initialize a version:** establish the baseline, development branch/worktree, and initial isolation scaffolding requested by the user.
- **Develop a version:** implement only in the new-version code path and namespace.
- **Run a comparison:** hold data, seeds, solver settings, and benchmark cases constant; write outputs to separate experiment namespaces.
- **Release a version:** verify provenance, compatibility, tests, documentation, and a clean intended diff before tagging or publishing.

Do not infer permission to commit, tag, push, delete, move large data, or overwrite artifacts merely because this skill was invoked. Perform those actions when the user's requested workflow clearly includes them; otherwise prepare the changes and report the remaining operation.

## Run the preflight

Before any mutation, inspect at least:

```bash
git status --short --branch
git branch -vv
git tag --list
git worktree list
git log --oneline --decorate -12
```

Also locate current model registries and all functions that construct processed-data, checkpoint, result, benchmark, and log paths. Use `rg` rather than assuming the documentation still matches the code.

Preserve all pre-existing tracked and untracked changes. Never treat an untracked planning document as part of the baseline commit unless the user requests that scope. Record the exact baseline commit before creating a version.

## Initialize an isolated version

Follow this order:

1. Verify that the intended V1 baseline commit is known and reproducible.
2. Create an annotated baseline tag only when included in the requested operation.
3. Create the new version from that exact commit, preferably in a separate Git worktree.
4. Keep the baseline worktree on `main` and perform subsequent changes in the new-version worktree.
5. Transfer intended untracked documents explicitly; do not silently omit or absorb them.
6. Verify `git status`, branch, commit, and worktree paths in both locations.

Never duplicate the full ignored `data/` tree merely to initialize a worktree. Separate shared inputs from versioned outputs instead.

## Isolate model code

Keep the legacy model import and checkpoint path compatible by default.

- Reuse an existing layer only when its behavior is unchanged.
- Add a new layer or module when computation semantics differ.
- Do not add widespread version conditionals to shared modules.
- Register explicit names such as `stgcn_v1` and `stgcn_v2`.
- Keep `stgcn` as a temporary V1 alias when old scripts or checkpoints require it.
- Do not fix an unrelated V1 bug inside a V2 architecture commit. Isolate and test that correction separately.

For architecture work, change one major factor per experiment in this order unless the user specifies otherwise: baseline reproduction, Fusion attention, Dynamic attention, node features, then multi-task outputs.

## Isolate data and artifacts

Distinguish shared immutable inputs from experiment-owned outputs:

- Share case definitions and raw samples only when their schema and label semantics remain valid.
- Regenerate processed tensors whenever feature meaning or tensor schema changes.
- Namespace processed tensors, checkpoints, results, and logs by `experiment_id`.
- Never overwrite legacy paths under `data/<case>/model`, `data/<case>/result`, or existing logs.
- Avoid a writable symlink from a new worktree to the whole legacy `data/` directory.

Require a stable, filesystem-safe `experiment_id`. Record an experiment manifest containing the Git commit, model and schema versions, case and UC type, seeds, split, solvers, training settings, checkpoint, benchmark suite, and creation time.

Version new checkpoints explicitly. Preserve a legacy loader for existing `{config, state_dict}` checkpoints, and validate model type and dataset schema before loading new checkpoints.

## Make comparisons reproducible

Use one fixed benchmark suite for every model in a comparison. Keep constant:

- raw samples and train/test split;
- random seeds;
- optimizer and training hyperparameters;
- test cases;
- ground-truth and accelerated solver modes;
- threshold ladder, time limit, and MIP gap.

If inputs or features differ, add the necessary controls, such as V1 with old features, V1 with new features, and V2 with new features. Do not attribute a combined change solely to the architecture.

Store structured metrics and provenance with the result. Do not rely on overwrite-mode logs as the experiment record.

## Validate before completion

Choose checks proportional to the change, including:

- V1 checkpoint load and inference compatibility;
- input/output shape contracts;
- edge-mask and generator-to-bus behavior;
- distinct paths for distinct experiment IDs;
- unchanged hashes and modification times for protected legacy artifacts;
- no NaN/Inf in attention edge cases;
- a case5 end-to-end smoke test before larger cases;
- a controlled case118 comparison before claiming an improvement.

For Git/worktree initialization, verify references and status without modifying model or data files. For source changes, run focused tests first, then the smallest meaningful end-to-end test.

## Report the result

State:

- baseline commit and tag, if any;
- active branch and worktree;
- files and registries changed;
- experiment and artifact namespaces;
- compatibility guarantees;
- validation performed and its outcome;
- operations intentionally not performed, such as commit, tag, push, training, or large-case evaluation.

Use clickable repository file links in the final handoff. Never claim V1/V2 independence until both source isolation and artifact-path isolation have been verified.
