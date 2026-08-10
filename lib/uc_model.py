import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Optional
from lib.case_loader import UCData
from lib.ptdf_solver import compute_ptdf, compute_post_ptdf

# Transmission-security handling. "lazy" and "dense" enforce the identical
# constraint set — every line, at every period — and differ only in how it
# reaches Gurobi; "none" drops the flow limits altogether.
SOLVE_MODES = ("none", "lazy", "dense")


@dataclass
class UCModel:
    env: gp.Env
    model: gp.Model
    z: gp.MVar
    p: gp.MVar
    mode: str = "none"
    demands: Optional[np.ndarray] = None
    # N-0: [num_line, num_bus] / [num_line]
    ptdf: Optional[np.ndarray] = None
    fmax: Optional[np.ndarray] = None
    # N-1, stacked over cc_monitor: [n_cc * num_line, num_bus] / [n_cc * num_line]
    post_ptdf: Optional[np.ndarray] = None
    post_fmax: Optional[np.ndarray] = None
    gen_bus: Optional[np.ndarray] = None


@dataclass
class UCResult:
    z: np.ndarray
    p: np.ndarray
    obj: float
    solve_time: float
    success: bool
    mip_gap: float = 0.0


@dataclass
class UCConfig:
    time_limit: float = 3600.0
    mip_gap: float = 1e-3
    verbose: int = 0
    lazy_tol: float = 1e-4


def init_uc_grb() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    return env


def build_basic_uc(data: UCData) -> UCModel:
    if data.demand is None:
        raise ValueError("UCData.demand must be set before building the UC model.")

    G = data.num_gen
    T = data.num_period
    NS = data.bid_price.shape[1]

    total_demand = data.demand.sum(axis=0)
    seg_width = np.diff(data.bid_mw, axis=1)

    env = init_uc_grb()
    model = gp.Model("uc_model", env=env)

    z = model.addMVar((G, T), vtype=GRB.BINARY, name="z")
    u = model.addMVar((G, T), vtype=GRB.BINARY, name="u")
    v = model.addMVar((G, T), vtype=GRB.BINARY, name="v")
    p = model.addMVar((G, T), lb=0, name="p")
    c = model.addMVar((G, T), lb=0, name="c")
    dp = model.addMVar((G, NS, T), lb=0, name="dp")

    obj = c.sum() + (data.gen_startup_cost[:, None] * u).sum()
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstr(
        c == data.bid_no_load[:, None] * z + (data.bid_price * dp).sum(axis=1),
        name="pwl_cost",
    )
    model.addConstr(
        p == data.bid_mw[:, 0:1] * z + dp.sum(axis=1),
        name="pwl_link",
    )
    model.addConstr(
        dp <= seg_width[:, :, None] * z[:, None, :],
        name="dp_upper",
    )

    model.addConstr(p >= z * data.gen_pmin[:, None], name="pmin")
    model.addConstr(p <= z * data.gen_pmax[:, None], name="pmax")
    model.addConstr(p.sum(axis=0) == total_demand, name="balance")

    model.addConstr(
        p[:, 1:] - p[:, :-1] <= data.gen_ramp_up[:, None],
        name="ramp_up",
    )
    model.addConstr(
        p[:, :-1] - p[:, 1:] <= data.gen_ramp_down[:, None],
        name="ramp_down",
    )

    model.addConstr(u + v <= 1, name="logic_uv")
    model.addConstr(
        z[:, 1:] - z[:, :-1] == u[:, 1:] - v[:, 1:],
        name="transition",
    )
    model.addConstr(z[:, 0] == u[:, 0], name="transition_init")

    for g in range(G):
        t_up = int(data.gen_min_up[g])
        t_down = int(data.gen_min_down[g])
        for t in range(T):
            if t_up > 1:
                e = min(t + t_up, T)
                model.addConstr(z[g, t:e].sum() >= u[g, t] * (e - t), name=f"up_{g}_{t}")
            if t_down > 1:
                e = min(t + t_down, T)
                model.addConstr((1 - z[g, t:e]).sum() >= v[g, t] * (e - t), name=f"dn_{g}_{t}")

    model.update()
    return UCModel(env=env, model=model, z=z, p=p)


def add_fix_uc(uc: UCModel, pre_z: np.ndarray):
    """Hard-fix the commitment variables the prediction is confident about."""
    uc.z[pre_z == 1].LB = 1.0
    uc.z[pre_z == 0].UB = 0.0
    uc.model.update()


def reset_fix_uc(uc: UCModel):
    """Release every commitment variable back to a free binary."""
    uc.z.LB = 0.0
    uc.z.UB = 1.0
    uc.model.update()


def _add_dense_security(uc: UCModel):
    """Enumerate every flow limit up front."""
    ptdf_gen = uc.ptdf[:, uc.gen_bus]
    rhs = uc.ptdf @ uc.demands
    uc.model.addConstr(ptdf_gen @ uc.p <= uc.fmax[:, None] + rhs, name="flow_ub")
    uc.model.addConstr(ptdf_gen @ uc.p >= -uc.fmax[:, None] + rhs, name="flow_lb")

    if uc.post_ptdf is not None:
        sc_gen = uc.post_ptdf[:, uc.gen_bus]
        sc_rhs = uc.post_ptdf @ uc.demands
        uc.model.addConstr(sc_gen @ uc.p <= uc.post_fmax[:, None] + sc_rhs, name="sc_flow_ub")
        uc.model.addConstr(sc_gen @ uc.p >= -uc.post_fmax[:, None] + sc_rhs, name="sc_flow_lb")


def build_uc(data: UCData, uc_type: str, mode: str = "dense") -> UCModel:
    if uc_type not in ("tcuc", "topo", "scuc"):
        raise ValueError(f"Unknown uc_type: {uc_type!r}, expected 'tcuc', 'topo', or 'scuc'")
    if mode not in SOLVE_MODES:
        raise ValueError(f"Unknown solve mode: {mode!r}, expected one of {SOLVE_MODES}")
    if uc_type == "scuc" and not data.cc_monitor:
        raise ValueError("cc_monitor must be set before building SCUC.")

    uc = build_basic_uc(data)
    uc.mode = mode
    uc.gen_bus = data.gen_bus
    uc.demands = data.demand

    if mode != "none":
        # PTDFs are derived from the topology, never stored on UCData: N-0 on the
        # base grid, or on the maintenance-outage grid when one line is out.
        if data.maintenance_line is None:
            uc.ptdf = compute_ptdf(data)
        else:
            uc.ptdf, _ = compute_post_ptdf(data, outage_line=data.maintenance_line)
        uc.fmax = np.asarray(data.line_fmax, float)
        if uc_type == "scuc":
            # N-1: one soft outage per contingency line, stacked
            post = [compute_post_ptdf(data, outage_line=k) for k in data.cc_monitor]
            uc.post_ptdf = np.vstack([pp for pp, _ in post])
            uc.post_fmax = np.concatenate([pf for _, pf in post])
        if mode == "dense":
            _add_dense_security(uc)

    uc.model.update()
    return uc


def _extract_result(uc: UCModel, dispose: bool = True) -> UCResult:
    actual_gap = 0.0
    if uc.model.SolCount > 0 and abs(uc.model.ObjVal) > 1e-10:
        actual_gap = abs(uc.model.ObjVal - uc.model.ObjBound) / abs(uc.model.ObjVal)

    if (uc.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]
            and uc.model.SolCount > 0):
        result = UCResult(
            z=np.round(uc.z.X).astype(int),
            p=np.round(uc.p.X, 4),
            obj=np.round(uc.model.ObjVal, 2),
            solve_time=np.round(uc.model.Runtime, 4),
            success=True,
            mip_gap=actual_gap,
        )
    else:
        result = UCResult(
            z=np.zeros(uc.z.shape, dtype=int),
            p=np.zeros(uc.p.shape, dtype=float),
            obj=float("inf"),
            solve_time=np.round(uc.model.Runtime, 4),
            success=False,
            mip_gap=float("inf"),
        )

    if dispose:
        dispose_uc(uc)
    return result


def dispose_uc(uc: UCModel):
    uc.model.dispose()
    uc.env.dispose()


def _solve_lazy(uc: UCModel, config: UCConfig):
    uc.model.Params.LazyConstraints = 1

    p = uc.p
    tol = config.lazy_tol
    T = uc.demands.shape[1]

    ptdf_gen = uc.ptdf[:, uc.gen_bus]
    fmax = uc.fmax
    rhs = uc.ptdf @ uc.demands

    has_security = uc.post_ptdf is not None
    if has_security:
        sc_ptdf_gen = uc.post_ptdf[:, uc.gen_bus]
        sc_fmax = uc.post_fmax
        sc_rhs = uc.post_ptdf @ uc.demands

    # A lazy cut is globally valid, so adding it twice only bloats the model.
    added = set()

    def _cut(m, key, a_row, lhs_var_row, bound):
        if key in added:
            return
        added.add(key)
        m.cbLazy(a_row @ lhs_var_row <= bound if key[3] else a_row @ lhs_var_row >= bound)

    def cb(m, where):
        if where != GRB.Callback.MIPSOL:
            return
        p_vals = m.cbGetSolution(p)

        flow = ptdf_gen @ p_vals - rhs
        for l, t in zip(*np.where(flow - fmax[:, None] > tol)):
            _cut(m, (0, l, t, True), ptdf_gen[l], p[:, t], float(fmax[l] + rhs[l, t]))
        for l, t in zip(*np.where(-fmax[:, None] - flow > tol)):
            _cut(m, (0, l, t, False), ptdf_gen[l], p[:, t], float(-fmax[l] + rhs[l, t]))

        if has_security:
            sc_flow = sc_ptdf_gen @ p_vals - sc_rhs
            for l, t in zip(*np.where(sc_flow - sc_fmax[:, None] > tol)):
                _cut(m, (1, l, t, True), sc_ptdf_gen[l], p[:, t], float(sc_fmax[l] + sc_rhs[l, t]))
            for l, t in zip(*np.where(-sc_fmax[:, None] - sc_flow > tol)):
                _cut(m, (1, l, t, False), sc_ptdf_gen[l], p[:, t], float(-sc_fmax[l] + sc_rhs[l, t]))

    uc.model.optimize(cb)


def solve_uc(uc: UCModel, config: UCConfig, dispose: bool = True) -> UCResult:
    uc.model.Params.TimeLimit = config.time_limit
    uc.model.Params.MIPGap = config.mip_gap
    uc.model.Params.OutputFlag = config.verbose

    if uc.mode == "lazy":
        _solve_lazy(uc, config)
    else:
        uc.model.optimize()

    return _extract_result(uc, dispose=dispose)
