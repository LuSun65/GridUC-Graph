import os
import traceback
from multiprocessing import Pool, cpu_count

from lib.case_loader import UCData, load_case
from lib.case_modifier import case_tcuc_rnd, case_topo_rnd, case_scuc_rnd
from lib.uc_model import build_uc, solve_uc, UCConfig
from lib.toolkit import tic, toc, setlog, set_rnd_seed, save_pkl

CASE_MODIFIERS = {
    "tcuc": case_tcuc_rnd,
    "topo": case_topo_rnd,
    "scuc": case_scuc_rnd,
}

# Per-sample seed offset, so a sample id maps to the same scenario whether it was
# drawn serially or by a pool worker (fork would otherwise clone one RNG state).
SEED_BASE = 26


def sample_uc(input_uc: UCData, uc_type: str, config: UCConfig, mode: str = "dense",
              attach_result: bool = False) -> UCData:
    uc_data = CASE_MODIFIERS[uc_type](input_uc)
    uc_model = build_uc(uc_data, uc_type, mode=mode)
    result = solve_uc(uc_model, config)
    uc_data.uc_sol = result.z
    if attach_result:
        uc_data.uc_obj = result.obj
        uc_data.uc_sol_time = result.solve_time
    return uc_data, result


def sample_dir(casename: str, uc_type: str) -> str:
    return f"data/{casename}/samples/{uc_type}"


def sample_path(casename: str, uc_type: str, sid: int) -> str:
    return f"{sample_dir(casename, uc_type)}/{sid}.pkl"


def _sample_one(case: UCData, casename: str, uc_type: str, sid: int,
                config: UCConfig, mode: str):
    try:
        t0 = tic()
        sample, result = sample_uc(case, uc_type, config, mode=mode)
        elapsed = toc(t0)
        if not result.success:
            # An infeasible/timed-out solve yields an all-zero schedule; keeping it
            # would poison the training set with a bogus "everything off" label.
            print(f"[{uc_type}] sample {sid} FAILED (not saved), time={elapsed:.2f}s")
            return
        print(f"[{uc_type}] sample {sid} SUCCESS, time={elapsed:.2f}s")
        save_pkl(sample, sample_path(casename, uc_type, sid))
    except Exception:
        print(f"[{uc_type}] sample {sid} ERROR:\n{traceback.format_exc()}")


def _sample_worker(args):
    casename, uc_type, sid, config, mode = args
    setlog(f"log/{casename}/sample/sample_{uc_type}.log")
    set_rnd_seed(SEED_BASE + sid)
    case = load_case(casename)
    _sample_one(case, casename, uc_type, sid, config, mode)


def run_sampling(casename: str, uc_type: str, start: int, end: int,
                 mode: str = "dense", workers=None, skip_existing: bool = False,
                 time_limit: float = 3600.0, mip_gap: float = 1e-3, verbose: int = 0):
    setlog(f"log/{casename}/sample/sample_{uc_type}.log")
    config = UCConfig(verbose=verbose, mip_gap=mip_gap, time_limit=time_limit)

    sids = list(range(start, end + 1))
    if skip_existing:
        done = {s for s in sids if os.path.exists(sample_path(casename, uc_type, s))}
        sids = [s for s in sids if s not in done]
        if done:
            print(f"[{uc_type}] skip {len(done)} existing samples")
    if not sids:
        print(f"[{uc_type}] nothing to do")
        return

    print(f"[{uc_type}] sampling {casename}: {len(sids)} samples, workers={workers}")

    if workers is None:
        case = load_case(casename)
        for sid in sids:
            set_rnd_seed(SEED_BASE + sid)
            _sample_one(case, casename, uc_type, sid, config, mode)
    else:
        if workers == "auto":
            workers = cpu_count()
        tasks = [(casename, uc_type, sid, config, mode) for sid in sids]
        with Pool(int(workers)) as pool:
            pool.map(_sample_worker, tasks)
