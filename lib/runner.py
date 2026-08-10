import os
import time
import traceback

from lib.toolkit import setlog
from lib.sampler import sample_dir, run_sampling
from lib.data_loader import processed_path, process_data
from lib.trainer import model_path, train_model
from lib.tester import generate_test_cases, run_testing, print_case_summary
from lib.uc_model import SOLVE_MODES


def _check_samples(case, uc_type):
    d = sample_dir(case, uc_type)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"No samples found: {d}")
    good = [f for f in os.listdir(d)
            if f.endswith(".pkl") and os.path.getsize(os.path.join(d, f)) > 0]
    if not good:
        raise FileNotFoundError(f"No usable samples in {d} (directory empty or all files 0 bytes)")


def _check_processed(case, uc_type):
    p = processed_path(case, uc_type)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Processed data not found: {p}")


def _check_model(case, uc_type, model_type):
    p = model_path(model_type, case, uc_type)
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


def run_sample(case, n_samples, uc_types, mode="dense", workers=None):
    failed = []
    for uc in uc_types:
        ok = _run_task(f"sample {case}/{uc}", lambda uc=uc: (
            run_sampling(casename=case, uc_type=uc, start=0, end=n_samples - 1,
                         mode=mode, skip_existing=True, workers=workers)
        ))
        if not ok:
            failed.append(f"sample {case}/{uc}")
    return failed


def run_process(case, uc_types):
    failed = []
    for uc in uc_types:
        _check_samples(case, uc)
        ok = _run_task(f"process {case}/{uc}", lambda uc=uc: (
            setlog(f"log/{case}/sample/process_{uc}.log"),
            process_data(casename=case, uc_type=uc, save=True),
        ))
        if not ok:
            failed.append(f"process {case}/{uc}")
    return failed


def run_train(case, uc_types, models, epochs=20, device="cpu", batch_size=32):
    failed = []
    for uc in uc_types:
        _check_processed(case, uc)
        for m in models:
            ok = _run_task(f"train {case}/{uc}/{m}", lambda uc=uc, m=m: (
                setlog(f"log/{case}/train/train_{uc}_{m}.log", overwrite=True),
                train_model(model_type=m, casename=case, uc_type=uc,
                            epochs=epochs, device=device, batch_size=batch_size),
            ))
            if not ok:
                failed.append(f"train {case}/{uc}/{m}")
    return failed


def run_test(case, uc_types, models, n_test=20, mode="dense", device="cpu"):
    failed = []
    for uc in uc_types:
        setlog(f"log/{case}/test/gen_{uc}.log", overwrite=True)
        print(f"\n[gen] generating {n_test} test cases for {case}/{uc}")
        test_cases = generate_test_cases(case, uc, n_test, mode=mode)
        for m in models:
            _check_model(case, uc, m)
            ok = _run_task(f"test {case}/{uc}/{m}", lambda uc=uc, m=m: (
                setlog(f"log/{case}/test/test_{uc}_{m}.log", overwrite=True),
                run_testing(model_type=m, casename=case, uc_type=uc,
                            test_cases=test_cases, mode=mode, device=device),
            ))
            if not ok:
                failed.append(f"test {case}/{uc}/{m}")
    return failed


def run_summary(case):
    failed = []
    ok = _run_task(f"summary {case}", lambda: (
        setlog(f"log/{case}/summary/summary.log", overwrite=True),
        print_case_summary(case),
    ))
    if not ok:
        failed.append(f"summary {case}")
    return failed


def run_all(case, stages=("sample", "process", "train", "test", "summary"),
            n_samples=1000, n_test=20, epochs=20, batch_size=32,
            device="cpu", workers=None,
            sample_solver="dense", test_solver="dense",
            uc_types=("tcuc", "topo", "scuc"),
            models=("stgcn", "mlp")):
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

    all_failed = []
    for stage in stages:
        print(f"\n{'=' * 80}")
        print(f"  STAGE: {stage}")
        print(f"{'=' * 80}")
        if stage == "sample":
            failed = run_sample(case, n_samples, uc_types, sample_solver, workers)
        elif stage == "process":
            failed = run_process(case, uc_types)
        elif stage == "train":
            failed = run_train(case, uc_types, models, epochs, device, batch_size)
        elif stage == "test":
            failed = run_test(case, uc_types, models, n_test, test_solver, device)
        elif stage == "summary":
            failed = run_summary(case)
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
