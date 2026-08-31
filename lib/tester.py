import os
import time
import traceback
from dataclasses import asdict
import numpy as np
import torch

from lib.case_loader import UCData, load_case
from lib.sampler import sample_uc
from lib.uc_model import (UCConfig, UCResult, build_uc, solve_uc, add_fix_uc,
                          reset_fix_uc, dispose_uc)
from lib.data_loader import to_stgcn_input
from lib.trainer import load_model
from lib.toolkit import save_pkl, load_pkl, set_rnd_seed
from lib.experiment import ExperimentPaths

# Test scenarios live in their own seed range so they can never collide with the
# training samples drawn at sampler.SEED_BASE + sid.
TEST_SEED_BASE = 10 ** 6
BENCHMARK_SCHEMA = 1


def repair_pre_z_mintime(pre_z: np.ndarray, uc_data: UCData) -> np.ndarray:
    pre_z = pre_z.copy()
    G, T = pre_z.shape

    changed = True
    while changed:
        changed = False
        for g in range(G):
            min_up = int(uc_data.gen_min_up[g])
            min_down = int(uc_data.gen_min_down[g])
            z = pre_z[g]

            for t in range(T):
                if z[t] == -1:
                    continue

                prev = 0
                for s in range(t - 1, -1, -1):
                    if z[s] != -1:
                        prev = int(z[s])
                        break

                if z[t] == 1 and prev == 0:
                    for s in range(t, min(t + min_up, T)):
                        if z[s] == 0:
                            pre_z[g, s] = -1
                            changed = True

                if z[t] == 0 and prev == 1:
                    for s in range(t, min(t + min_down, T)):
                        if z[s] == 1:
                            pre_z[g, s] = -1
                            changed = True

    return pre_z


def probs_to_fix(probs: np.ndarray, threshold: float) -> np.ndarray:
    pre_z = np.full(probs.shape, -1, dtype=int)
    on = threshold if threshold != 0.5 else 0.50001
    off = 1 - on
    pre_z[probs >= on] = 1
    pre_z[probs <= off] = 0
    return pre_z


def threshold_ladder(start: float = 0.95, step: float = 0.01) -> list:
    ladder = []
    t = start
    while t < 1.0 - 1e-9:
        ladder.append(round(t, 4))
        t += step
    ladder.append(1.0)
    return ladder


def get_probs(model, uc_data: UCData, uc_type: str, device: str = "cpu"):
    data = to_stgcn_input(uc_data, uc_type).to(device)
    t0 = time.time()
    with torch.no_grad():
        probs = torch.sigmoid(model(data))
    nn_time = time.time() - t0
    return probs[0].cpu().numpy(), nn_time


def compare_results(uc_data: UCData, pre_z: np.ndarray, result: UCResult) -> dict:
    gt_z = uc_data.uc_sol
    gt_obj = uc_data.uc_obj
    gt_sol_time = uc_data.uc_sol_time

    fixed_mask = (pre_z >= 0)
    fix_ratio = fixed_mask.mean()
    fix_accuracy = (pre_z[fixed_mask] == gt_z[fixed_mask]).mean() if fixed_mask.any() else float("nan")

    speedup = gt_sol_time / result.solve_time if result.solve_time > 0 else float("inf")
    obj_gap = (result.obj - gt_obj) / abs(gt_obj) if gt_obj != 0 else float("nan")

    return {
        "gt_obj": gt_obj,
        "gt_z": gt_z,
        "gt_sol_time": gt_sol_time,
        "res_obj": result.obj,
        "res_z": result.z,
        "res_sol_time": result.solve_time,
        "speedup": speedup,
        "obj_gap": obj_gap,
        "fix_ratio": fix_ratio,
        "fix_accuracy": fix_accuracy,
    }


def run_test(uc_data: UCData, uc_type: str, test_id: int, model,
             mode: str = "dense",
             threshold_start: float = 0.95, threshold_step: float = 0.01,
             device: str = "cpu", config: UCConfig = None) -> dict:
    if config is None:
        config = UCConfig()

    probs, nn_time = get_probs(model, uc_data, uc_type, device)

    # Escalating fixed-threshold solve.
    # Walk thresholds from low to high: a higher threshold fixes fewer variables
    # (prob >= t for "on", prob <= 1 - t for "off"), so the problem is looser and
    # more likely to stay feasible. The first threshold that gives a feasible
    # solution wins, and its solve time is the one reported (failed attempts are
    # accumulated separately in "total_solve_time").
    # NOTE: these per-threshold solves are independent; in production they can run
    # in parallel and take the fastest feasible one. Here they run serially, over a
    # single Gurobi model whose z bounds are reset between attempts — rebuilding
    # the model per threshold dominates the runtime on the large cases.
    uc_model = build_uc(uc_data, uc_type, mode=mode)
    chosen_t = None
    best_result = None
    best_pre_z = None
    total_solve_time = 0.0
    try:
        for t in threshold_ladder(threshold_start, threshold_step):
            pre_z = repair_pre_z_mintime(probs_to_fix(probs, t), uc_data)
            reset_fix_uc(uc_model)
            add_fix_uc(uc_model, pre_z)
            result = solve_uc(uc_model, config, dispose=False)
            total_solve_time += result.solve_time
            if result.success:
                chosen_t, best_result, best_pre_z = t, result, pre_z
                break

        # Fallback: every threshold infeasible -> drop the prediction and solve the
        # original UC with no variables fixed (always feasible).
        if best_result is None:
            best_pre_z = np.full(probs.shape, -1, dtype=int)
            reset_fix_uc(uc_model)
            best_result = solve_uc(uc_model, config, dispose=False)
            total_solve_time += best_result.solve_time
    finally:
        dispose_uc(uc_model)

    best_result.solve_time += nn_time
    cmp = compare_results(uc_data, best_pre_z, best_result)
    cmp["threshold"] = chosen_t
    cmp["nn_time"] = nn_time
    cmp["total_solve_time"] = total_solve_time + nn_time
    print(f"[test {test_id}] threshold={chosen_t if chosen_t is not None else 'none'}"
          f"  obj_gap={cmp['obj_gap']:.4%}  speedup={cmp['speedup']:.2f}x"
          f"  fix_ratio={cmp['fix_ratio']:.2%}  fix_acc={cmp['fix_accuracy']:.2%}")
    return cmp


def result_path(model_type: str, casename: str, uc_type: str,
                paths: ExperimentPaths):
    return paths.result_path(casename, uc_type, model_type)


def save_results(res: dict, paths: ExperimentPaths, path=None):
    if path is None:
        path = result_path(res["model"], res["casename"], res["uc_type"], paths)
    save_pkl(res, path)
    print(f"[SAVED] results -> {path}")


def load_results(model_type: str, casename: str, uc_type: str,
                 paths: ExperimentPaths) -> dict:
    path = result_path(model_type, casename, uc_type, paths)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    return load_pkl(path)


def _generate_test_cases_with_seeds(
    casename: str,
    uc_type: str,
    n_test: int,
    paths: ExperimentPaths,
    mode: str = "dense",
    config: UCConfig = None,
) -> tuple:
    """Draw fresh random scenarios and solve them to optimality as ground truth.

    The caller decides whether these fresh cases need to be persisted. Seeds of
    successful solves are returned with the cases so a benchmark can record
    exactly which requested scenarios were retained.
    """
    if config is None:
        config = UCConfig()

    case = load_case(casename, data_dir=paths.case_dir())
    test_cases = []
    successful_seeds = []
    for i in range(n_test):
        seed = TEST_SEED_BASE + i
        set_rnd_seed(seed)
        uc_data, result = sample_uc(case, uc_type, config, mode=mode, attach_result=True)
        if not result.success:
            print(f"[gen] test case {i + 1}/{n_test} FAILED to solve, skipped")
            continue
        test_cases.append(uc_data)
        successful_seeds.append(seed)
        print(f"[gen] test case {i + 1}/{n_test}  obj={uc_data.uc_obj:.2f}"
              f"  time={uc_data.uc_sol_time:.2f}s")

    return test_cases, successful_seeds


def generate_test_cases(casename: str, uc_type: str, n_test: int,
                        paths: ExperimentPaths,
                        mode: str = "dense", config: UCConfig = None) -> list:
    """Generate fresh test cases without reading or writing a benchmark."""
    test_cases, _ = _generate_test_cases_with_seeds(
        casename, uc_type, n_test, paths, mode=mode, config=config
    )
    return test_cases


def _validate_benchmark(payload: dict, path, casename: str, uc_type: str,
                        n_test: int, mode: str, config: UCConfig) -> list:
    expected = {
        "benchmark_schema": BENCHMARK_SCHEMA,
        "casename": casename,
        "uc_type": uc_type,
        "solver_mode": mode,
        "solver_config": asdict(config),
        "seed_base": TEST_SEED_BASE,
        "requested_n_test": n_test,
    }
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark format: {path}")
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"Benchmark metadata mismatch for {name!r} at {path}: "
                f"expected {value!r}, found {payload.get(name)!r}"
            )

    test_cases = payload.get("test_cases")
    successful_seeds = payload.get("successful_seeds")
    if not isinstance(test_cases, list) or not isinstance(successful_seeds, list):
        raise ValueError(f"Benchmark is missing test cases or seeds: {path}")
    if len(test_cases) != len(successful_seeds):
        raise ValueError(f"Benchmark case/seed count mismatch: {path}")
    if payload.get("actual_n_test") != len(test_cases):
        raise ValueError(f"Benchmark actual_n_test is inconsistent: {path}")
    for index, case in enumerate(test_cases):
        if (getattr(case, "uc_sol", None) is None
                or getattr(case, "uc_obj", None) is None
                or getattr(case, "uc_sol_time", None) is None):
            raise ValueError(
                f"Benchmark case {index} has incomplete ground truth: {path}"
            )
    return test_cases


def load_or_generate_test_cases(
    casename: str,
    uc_type: str,
    n_test: int,
    paths: ExperimentPaths,
    mode: str = "dense",
    config: UCConfig = None,
) -> list:
    """Load a matching benchmark, or generate and atomically save one."""
    if config is None:
        config = UCConfig()
    path = paths.benchmark_path(
        casename, uc_type, mode, TEST_SEED_BASE, n_test
    )
    if path.exists():
        test_cases = _validate_benchmark(
            load_pkl(path), path, casename, uc_type, n_test, mode, config
        )
        print(f"[benchmark] loaded {len(test_cases)} test cases <- {path}")
        return test_cases

    test_cases, successful_seeds = _generate_test_cases_with_seeds(
        casename, uc_type, n_test, paths, mode=mode, config=config
    )
    payload = {
        "benchmark_schema": BENCHMARK_SCHEMA,
        "casename": casename,
        "uc_type": uc_type,
        "solver_mode": mode,
        "solver_config": asdict(config),
        "seed_base": TEST_SEED_BASE,
        "requested_n_test": n_test,
        "successful_seeds": successful_seeds,
        "actual_n_test": len(test_cases),
        "test_cases": test_cases,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_pkl(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[benchmark] saved {len(test_cases)} test cases -> {path}")
    return test_cases


def run_testing(model_type: str, casename: str, uc_type: str,
                test_cases: list, paths: ExperimentPaths, mode: str = "dense",
                threshold_start: float = 0.95, threshold_step: float = 0.01,
                device: str = "cpu", config: UCConfig = None) -> dict:
    model = load_model(model_type, casename, uc_type, paths, device=device)
    if config is None:
        config = UCConfig()

    cmp_list = []
    n_error = 0
    for test_id, uc_data in enumerate(test_cases):
        try:
            cmp = run_test(uc_data, uc_type, test_id, model, mode=mode,
                           threshold_start=threshold_start, threshold_step=threshold_step,
                           device=device, config=config)
            cmp_list.append(cmp)
        except Exception:
            n_error += 1
            print(f"[test {test_id}] ERROR:\n{traceback.format_exc()}")

    res = {
        "model": model_type,
        "casename": casename,
        "uc_type": uc_type,
        "threshold_start": threshold_start,
        "threshold_step": threshold_step,
        "n_test": len(test_cases),
        "n_error": n_error,
        "cmp_list": cmp_list,
    }
    save_results(res, paths)
    return res


def print_case_summary(casename, uc_types=("tcuc", "topo", "scuc"),
                       models=("mlp", "stgcn"), *, paths: ExperimentPaths):
    all_res = {}
    for uc in uc_types:
        all_res[uc] = {}
        for m in models:
            try:
                all_res[uc][m] = load_results(m, casename, uc, paths)
            except FileNotFoundError:
                pass

    header = (f" {'UC':<8} {'Method':<12}"
              f" {'Avg.gap(%)':>11} {'Gap STD':>9} {'Speedup':>9}"
              f" {'Threshold':>10} {'Fix ratio(%)':>13} {'Fix acc.(%)':>12}")
    w = len(header)
    sep = "-" * w

    print()
    print("=" * w)
    print(f"  SUMMARY | case = {casename}")
    print("=" * w)
    print(header)
    print(sep)

    for uc in uc_types:
        if not all_res[uc]:
            print(f" {uc:<8}  (no results)")
            print(sep)
            continue

        gt_times = []
        for res in all_res[uc].values():
            for c in res["cmp_list"]:
                gt_times.append(c["gt_sol_time"])

        milp_avg_time = np.mean(gt_times)

        print(f" {uc:<8} {'MILP':<12}"
              f" {'-':>11} {'-':>9} {'-':>9}"
              f" {'-':>10} {'-':>13} {'-':>12}")

        for m in models:
            if m not in all_res[uc]:
                continue
            cmp_list = all_res[uc][m]["cmp_list"]
            if not cmp_list:
                continue

            gaps = [c["obj_gap"] * 100 for c in cmp_list]
            times = [c["res_sol_time"] for c in cmp_list]
            thresholds = [c.get("threshold") for c in cmp_list
                          if c.get("threshold") is not None]
            fix_ratios = [c.get("fix_ratio") for c in cmp_list
                          if c.get("fix_ratio") is not None]
            fix_accuracies = [c.get("fix_accuracy") for c in cmp_list
                              if c.get("fix_accuracy") is not None
                              and np.isfinite(c.get("fix_accuracy"))]

            avg_time = np.mean(times)
            speedup = milp_avg_time / avg_time if avg_time > 0 else float("inf")
            threshold_text = f"{np.mean(thresholds):.4f}" if thresholds else "-"
            fix_ratio_text = f"{np.mean(fix_ratios) * 100:.2f}" if fix_ratios else "-"
            fix_accuracy_text = (f"{np.mean(fix_accuracies) * 100:.2f}"
                                 if fix_accuracies else "-")

            print(f" {'':8} {m.upper():<12}"
                  f" {np.mean(gaps):>11.4f} {np.std(gaps):>9.4f}"
                  f" {speedup:>8.2f}x"
                  f" {threshold_text:>10} {fix_ratio_text:>13}"
                  f" {fix_accuracy_text:>12}")

        print(sep)

    print()
