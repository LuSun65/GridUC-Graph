import os
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from lib.toolkit import setlog
from lib.sampler import sample_dir, run_sampling
from lib.data_loader import process_data
from lib.trainer import train_model
from lib.models.model_registry import canonical_model_type
from lib.tester import (
    load_or_generate_test_cases,
    run_testing,
    print_case_summary,
)
from lib.uc_model import SOLVE_MODES
from lib.experiment import ExperimentPaths


def _check_samples(case, uc_type, paths):
    d = sample_dir(case, uc_type, paths)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"No samples found: {d}")
    good = [f for f in os.listdir(d)
            if f.endswith(".pkl") and os.path.getsize(os.path.join(d, f)) > 0]
    if not good:
        raise FileNotFoundError(f"No usable samples in {d} (directory empty or all files 0 bytes)")


def _check_processed(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")


def _processed_source(paths, configured_path, case, uc_type, generated):
    if generated:
        return paths.processed_path(case, uc_type)
    if configured_path is None:
        return paths.default_input_processed_path(case, uc_type)
    return os.fspath(configured_path).format(case=case, uc=uc_type)


def _sample_count(case, uc_type, paths):
    directory = sample_dir(case, uc_type, paths)
    if not os.path.isdir(directory):
        return None
    return sum(name.endswith(".pkl") for name in os.listdir(directory))


def _check_model(case, uc_type, model_type, paths):
    p = paths.test_checkpoint_path(case, uc_type, model_type)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model not found: {p}")


def _run_task(label, fn):
    print(f"\n[task] {label}")
    t0 = time.time()
    try:
        fn()
        print(f"[task] {label} OK ({time.time() - t0:.1f}s)")
        return True
    except Exception:
        print(f"[task] {label} FAILED ({time.time() - t0:.1f}s)")
        traceback.print_exc()
        return False


def run_sample(case, n_samples, uc_types, paths, mode="dense", workers=None):
    failed = []
    for uc in uc_types:
        ok = _run_task(f"sample {case}/{uc}", lambda uc=uc: (
            run_sampling(casename=case, uc_type=uc, start=0, end=n_samples - 1,
                         paths=paths, mode=mode, skip_existing=True, workers=workers)
        ))
        if not ok:
            failed.append(f"sample {case}/{uc}")
    return failed


def run_process(case, uc_types, paths, chunk_size, train_ratio):
    failed = []
    for uc in uc_types:
        _check_samples(case, uc, paths)
        ok = _run_task(f"process {case}/{uc}", lambda uc=uc: (
            setlog(paths.log_path(case, "sample", f"process_{uc}.log")),
            process_data(
                casename=case, uc_type=uc, paths=paths,
                chunk_size=chunk_size, train_ratio=train_ratio, save=True,
            ),
        ))
        if not ok:
            failed.append(f"process {case}/{uc}")
    return failed


def run_train(case, uc_types, models, paths, configured_processed_path,
              generated_processed,
              epochs=20, device="cpu", batch_size=32, random_seed=42,
              lambda_lmp=1.0, huber_delta=1.0):
    failed = []
    for uc in uc_types:
        source = _processed_source(
            paths, configured_processed_path, case, uc, generated_processed
        )
        _check_processed(source)
        expected_count = _sample_count(case, uc, paths)
        for m in models:
            ok = _run_task(f"train {case}/{uc}/{m}", lambda uc=uc, m=m: (
                setlog(paths.log_path(case, "train", f"train_{uc}_{m}.log"),
                       overwrite=True),
                train_model(model_type=m, casename=case, uc_type=uc,
                            paths=paths, epochs=epochs, device=device,
                            batch_size=batch_size,
                            random_seed=random_seed,
                            lambda_lmp=lambda_lmp, huber_delta=huber_delta,
                            processed_path=source,
                            expected_sample_count=expected_count),
            ))
            if not ok:
                failed.append(f"train {case}/{uc}/{m}")
    return failed


def _run_train_on_device(device, tasks, case, paths, epochs, batch_size,
                         random_seed, lambda_lmp=1.0, huber_delta=1.0):
    """Run one queue of training tasks sequentially in a spawned GPU process."""
    failed = []
    for uc, model, source, expected_count in tasks:
        label = f"train {case}/{uc}/{model} on {device}"
        ok = _run_task(label, lambda uc=uc, model=model, source=source,
                       expected_count=expected_count: (
            setlog(paths.log_path(case, "train", f"train_{uc}_{model}.log"),
                   overwrite=True),
            train_model(
                model_type=model, casename=case, uc_type=uc, paths=paths,
                epochs=epochs, device=device, batch_size=batch_size,
                random_seed=random_seed, processed_path=source,
                lambda_lmp=lambda_lmp, huber_delta=huber_delta,
                expected_sample_count=expected_count,
            ),
        ))
        if not ok:
            failed.append(f"train {case}/{uc}/{model}")
    return failed


def run_train_parallel(case, uc_types, models, paths,
                       configured_processed_path, generated_processed,
                       devices, epochs=20, batch_size=32, random_seed=42,
                       lambda_lmp=1.0, huber_delta=1.0):
    """Distribute training tasks over devices, with one process per device."""
    devices = tuple(devices)
    if not devices:
        raise ValueError("train_devices must contain at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError(f"train_devices contains duplicates: {devices}")

    tasks = []
    for uc in uc_types:
        source = _processed_source(
            paths, configured_processed_path, case, uc, generated_processed
        )
        _check_processed(source)
        expected_count = _sample_count(case, uc, paths)
        for model in models:
            tasks.append((uc, model, source, expected_count))

    queues = [[] for _ in devices]
    for index, task in enumerate(tasks):
        queues[index % len(devices)].append(task)

    print("\n[train] parallel GPU assignment:")
    for device, queue in zip(devices, queues):
        labels = ", ".join(f"{uc}/{model}" for uc, model, _, _ in queue)
        print(f"  {device}: {labels or '(idle)'}")

    failed = []
    # CUDA must not be initialized before a fork. Spawn gives every GPU worker
    # a clean CUDA runtime, and each worker owns its device queue exclusively.
    context = mp.get_context("spawn")
    active = [(device, queue) for device, queue in zip(devices, queues) if queue]
    with ProcessPoolExecutor(max_workers=len(active), mp_context=context) as pool:
        futures = {
            pool.submit(
                _run_train_on_device, device, queue, case, paths, epochs,
                batch_size, random_seed, lambda_lmp, huber_delta,
            ): (device, queue)
            for device, queue in active
        }
        for future in as_completed(futures):
            device, queue = futures[future]
            try:
                failed.extend(future.result())
            except Exception:
                print(f"[train] worker for {device} FAILED")
                traceback.print_exc()
                failed.extend(
                    f"train {case}/{uc}/{model}"
                    for uc, model, _, _ in queue
                )
    return failed


def run_test(case, uc_types, models, paths, n_test=20, mode="dense", device="cpu"):
    failed = []
    for uc in uc_types:
        setlog(paths.log_path(case, "test", f"gen_{uc}.log"), overwrite=True)
        print(f"\n[gen] generating {n_test} test cases for {case}/{uc}")
        test_cases = load_or_generate_test_cases(
            case, uc, n_test, paths, mode=mode
        )
        for m in models:
            _check_model(case, uc, m, paths)
            ok = _run_task(f"test {case}/{uc}/{m}", lambda uc=uc, m=m: (
                setlog(paths.log_path(case, "test", f"test_{uc}_{m}.log"),
                       overwrite=True),
                run_testing(model_type=m, casename=case, uc_type=uc,
                            test_cases=test_cases, paths=paths, mode=mode,
                            device=device),
            ))
            if not ok:
                failed.append(f"test {case}/{uc}/{m}")
    return failed


def run_summary(case, paths, uc_types=("tcuc", "topo", "scuc"),
                models=("mlp", "stgcn_v1")):
    failed = []
    ok = _run_task(f"summary {case}", lambda: (
        setlog(paths.log_path(case, "summary", "summary.log"), overwrite=True),
        print_case_summary(case, uc_types=uc_types, models=models, paths=paths),
    ))
    if not ok:
        failed.append(f"summary {case}")
    return failed


def run_all(case, stages=("sample", "process", "train", "test", "summary"),
            n_samples=1000, n_test=20, epochs=20, batch_size=32,
            device="cpu", workers=None, random_seed=42,
            sample_solver="dense", test_solver="dense",
            uc_types=("tcuc", "topo", "scuc"),
            models=("stgcn_v2", "stgcn_v1", "mlp"), *,
            input_root="data", output_root=".", run_id: str,
            processed_path=None, process_chunk_size=200, train_ratio=0.8,
            train_devices=None, checkpoint_run_id=None,
            lambda_lmp=1.0, huber_delta=1.0):
    """
    sample_solver / test_solver: transmission-security formulation used when
    generating training labels and when testing, respectively. One of
    "none" (no line limits), "lazy" (callback cuts), "dense" (all limits up
    front). The two are independent — crossing them (e.g. train on "none"
    labels, test against "dense" ground truth) is a supported experiment;
    the sample/model/test data structures do not depend on the choice.
    """
    for m in (sample_solver, test_solver):
        if m not in SOLVE_MODES:
            raise ValueError(f"Unknown solve mode: {m!r}, expected one of {SOLVE_MODES}")
    for model_type in models:
        canonical_model_type(model_type)

    paths = ExperimentPaths(
        input_root=input_root,
        output_root=output_root,
        run_id=run_id,
        checkpoint_run_id=checkpoint_run_id,
    )
    print(f"[paths] input_root={paths.input_root}")
    print(f"[paths] run_root={paths.run_root}")
    print(f"[paths] checkpoint_run_root={paths.checkpoint_run_root}")
    generated_processed = "process" in stages

    all_failed = []
    for stage in stages:
        print(f"\n{'=' * 80}")
        print(f"  STAGE: {stage}")
        print(f"{'=' * 80}")
        if stage == "sample":
            failed = run_sample(case, n_samples, uc_types, paths, sample_solver, workers)
        elif stage == "process":
            failed = run_process(
                case, uc_types, paths, process_chunk_size, train_ratio,
            )
        elif stage == "train":
            if train_devices is None:
                failed = run_train(
                    case, uc_types, models, paths, processed_path,
                    generated_processed,
                    epochs, device, batch_size, random_seed, lambda_lmp, huber_delta,
                )
            else:
                failed = run_train_parallel(
                    case, uc_types, models, paths, processed_path,
                    generated_processed, train_devices,
                    epochs, batch_size, random_seed, lambda_lmp, huber_delta,
                )
        elif stage == "test":
            failed = run_test(case, uc_types, models, paths, n_test, test_solver, device)
        elif stage == "summary":
            failed = run_summary(case, paths, uc_types, models)
        else:
            print(f"Unknown stage: {stage}")
            continue
        all_failed.extend(failed)

    print(f"\n{'=' * 80}")
    if all_failed:
        print(f"{len(all_failed)} tasks failed:")
        for f in all_failed:
            print(f"  {f}")
    else:
        print("all tasks succeeded")
    print("=" * 80)
    return all_failed
