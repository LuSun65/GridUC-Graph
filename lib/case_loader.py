from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import pandas as pd


@dataclass
class UCData:
    """
    Full data container for a unit-commitment case.

    Base fields are populated by load_case(); reserved fields (demand,
    load_lds_idx, bid_lds_idx) are left as None and filled when a concrete
    scenario is generated. PTDF matrices are never stored here: they are
    derived from the topology and recomputed in build_uc().

    All indices are 0-based.
    """

    case_name: str

    # --- dimensions ---
    num_gen: int
    num_line: int
    num_bus: int
    num_period: int

    # --- generator (length num_gen) ---
    gen_bus: np.ndarray
    gen_pmax: np.ndarray
    gen_pmin: np.ndarray
    gen_ramp_up: np.ndarray
    gen_ramp_down: np.ndarray
    gen_min_up: np.ndarray
    gen_min_down: np.ndarray
    gen_cost: np.ndarray
    gen_startup_cost: np.ndarray

    # --- line (length num_line) ---
    line_from: np.ndarray
    line_to: np.ndarray
    line_reactance: np.ndarray
    line_fmax: np.ndarray

    # --- load (length num_bus) ---
    nodal_peak: np.ndarray

    # --- bid: piecewise-linear cost ---
    bid_mw: np.ndarray       # [num_gen, num_segments + 1] MW breakpoints
    bid_price: np.ndarray    # [num_gen, num_segments] marginal prices per segment
    bid_no_load: np.ndarray  # [num_gen] no-load cost

    # --- loadshape pool ---
    lds_pool: np.ndarray  # [num_lds, num_period] load-shape ratio profiles

    # --- dynamic contingency monitor ---
    dync_monitor: list  # list of dicts, each with 'lds_id' (int) and 'lines' (list[int])

    # --- reserved fields (filled when generating a scenario) ---
    demand: Optional[np.ndarray] = None          # [num_bus, num_period]
    # [num_bus] index into lds_pool, per-node loadshape selection for time-varying demand
    load_lds_idx: Optional[np.ndarray] = None
    # [num_gen] index into lds_pool, per-generator loadshape selection for time-varying bids
    bid_lds_idx: Optional[np.ndarray] = None
    # line index under maintenance; changes the topology the N-0 flows are computed on
    maintenance_line: Optional[int] = None
    # contingency list: line indices taken out one at a time for the N-1 constraints (SCUC)
    cc_monitor: Optional[list] = None
    # UC solution: commitment schedule [num_gen, num_period]
    uc_sol: Optional[np.ndarray] = None
    # ground-truth objective / solve time, attached when sampling test cases
    uc_obj: Optional[float] = None
    uc_sol_time: Optional[float] = None
    # Fixed-commitment pricing labels; old samples remain readable without them.
    lmp_target: Optional[np.ndarray] = None  # [num_bus, num_period], currency/MWh
    p_target: Optional[np.ndarray] = None    # [num_gen, num_period], MW
    pricing_obj: Optional[float] = None
    pricing_solve_time: Optional[float] = None
    pricing_metadata: Optional[dict] = None


def load_case(case_name: str, data_dir: str = "data/case") -> UCData:
    """
    Load a UC case from a single Excel file.

    Usage:
        data = load_case("case5")
        data = load_case("case118", data_dir="/path/to/cases")

    The Excel file must contain sheets:
        gen, line, load, bid_mw, bid_price, lds_pool, dync_monitor
    """
    path = os.path.join(data_dir, f"{case_name}.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case file not found: {path}")

    sheets = pd.read_excel(path, sheet_name=None)

    # --- gen ---
    gen = sheets["gen"]
    num_gen = len(gen)

    # --- line ---
    line = sheets["line"]
    num_line = len(line)

    # --- load ---
    load = sheets["load"]
    num_bus = len(load)

    # --- lds_pool ---
    lds = sheets["lds_pool"]
    lds_pool = lds.drop(columns=["id"]).values.astype(float)
    num_period = lds_pool.shape[1]

    # --- bid ---
    bid_mw_df = sheets["bid_mw"]
    bid_price_df = sheets["bid_price"]
    bid_mw = bid_mw_df.drop(columns=["id"]).values.astype(float)
    bid_no_load = bid_price_df["c0"].values.astype(float)
    bid_price = bid_price_df.drop(columns=["id", "c0"]).values.astype(float)

    # --- dync_monitor (all indices to 0-based) ---
    dm = sheets["dync_monitor"]
    line_cols = [c for c in dm.columns if c.startswith("line_")]
    dync_monitor = []
    for _, row in dm.iterrows():
        lines = [int(row[c]) - 1 for c in line_cols if pd.notna(row[c])]
        dync_monitor.append({
            "lds_id": int(row["lds_id"]) - 1,
            "lines": lines,
        })

    return UCData(
        case_name=case_name,
        num_gen=num_gen,
        num_line=num_line,
        num_bus=num_bus,
        num_period=num_period,
        # gen (0-based bus)
        gen_bus=gen["bus"].values.astype(int) - 1,
        gen_pmax=gen["pmax"].values.astype(float),
        gen_pmin=gen["pmin"].values.astype(float),
        gen_ramp_up=gen["rampup"].values.astype(float),
        gen_ramp_down=gen["rampdown"].values.astype(float),
        gen_min_up=gen["tminup"].values.astype(int),
        gen_min_down=gen["tmindown"].values.astype(int),
        gen_cost=gen["gencost"].values.astype(float),
        gen_startup_cost=gen["startcost"].values.astype(float),
        # line (0-based from/to)
        line_from=line["from"].values.astype(int) - 1,
        line_to=line["to"].values.astype(int) - 1,
        line_reactance=line["x"].values.astype(float),
        line_fmax=line["fmax"].values.astype(float),
        # load
        nodal_peak=load["peak"].values.astype(float),
        # bid
        bid_mw=bid_mw,
        bid_price=bid_price,
        bid_no_load=bid_no_load,
        # lds
        lds_pool=lds_pool,
        # dync_monitor
        dync_monitor=dync_monitor,
    )
