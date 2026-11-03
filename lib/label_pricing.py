"""Add fixed-commitment pricing labels to existing samples in place."""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

from lib.lmp_sover import solve_pricing, line_binding, PRICING_FEASIBILITY_TOL, PRICING_OPTIMALITY_TOL
from lib.toolkit import load_pkl


TARGET_FIELDS = (
    "p_target", "lmp_target", "lmp_energy_target", "lmp_congestion_target",
    "shadow_price_target", "line_flow_target", "line_binding_target",
    "pricing_obj", "pricing_solve_time", "pricing_metadata",
)
REQUIRED_FIELDS = TARGET_FIELDS + ("uc_sol",)


def has_pricing_labels(data) -> bool:
    """Presence only: complete labels are recomputed exclusively with overwrite."""
    return all(getattr(data, name, None) is not None for name in REQUIRED_FIELDS)


def infer_uc_type(data) -> str:
    if getattr(data, "cc_monitor", None):
        return "scuc"
    return "topo" if getattr(data, "maintenance_line", None) is not None else "tcuc"


def valid_pricing_labels(data, uc_type: str) -> bool:
    """Validate newly generated results, never decide whether labels are stale."""
    try:
        if not has_pricing_labels(data) or infer_uc_type(data) != uc_type:
            return False
        G, N, L, T = data.num_gen, data.num_bus, data.num_line, data.num_period
        S = 1 + (len(data.cc_monitor) if uc_type == "scuc" else 0)
        for name, shape in (
            ("uc_sol", (G, T)), ("p_target", (G, T)), ("lmp_target", (N, T)),
            ("lmp_energy_target", (T,)), ("lmp_congestion_target", (N, T)),
            ("line_flow_target", (S, L, T)), ("line_binding_target", (S, L, T, 2)),
        ):
            value = getattr(data, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or not np.isfinite(value).all():
                return False
        if not np.isin(data.uc_sol, (0, 1)).all():
            return False
        if data.line_flow_target.dtype != np.float64 or data.line_binding_target.dtype != np.bool_:
            return False
        for name in ("pricing_obj", "pricing_solve_time"):
            value = getattr(data, name)
            if not np.isscalar(value) or not np.isfinite(value):
                return False
        if data.pricing_solve_time < 0:
            return False
        meta = data.pricing_metadata
        if (meta["uc_type"] != uc_type or meta["lp_status"] != 2
                or meta["ref_bus"] != 0 or meta["period_hours"] != 1.0
                or tuple(meta["bound_order"]) != ("upper", "lower")
                or len(meta["scenarios"]) != S
                or not np.isfinite(meta["max_network_violation"])
                or not 0 <= meta["max_network_violation"] <= PRICING_FEASIBILITY_TOL):
            return False
        for key in ("solver_version", "solver_parameters", "index_convention", "flow_direction",
                    "constraint_generation_rounds", "network_constraints_added"):
            if key not in meta:
                return False
        tol = meta["binding_tol_mw"]
        if not np.isfinite(tol) or tol < 0:
            return False
        if not np.allclose(data.lmp_target, data.lmp_energy_target[None, :] + data.lmp_congestion_target,
                           rtol=1e-9, atol=1e-7):
            return False
        shadow = data.shadow_price_target
        if tuple(shadow["shape"]) != (S, L, T, 2):
            return False
        idx, val = shadow["indices"], shadow["values"]
        if (not isinstance(idx, np.ndarray) or idx.ndim != 2 or idx.shape[1] != 4
                or not np.issubdtype(idx.dtype, np.integer)
                or not isinstance(val, np.ndarray) or val.shape != (len(idx),)
                or not np.isfinite(val).all() or np.any(val < -PRICING_OPTIMALITY_TOL)
                or np.any(idx < 0) or np.any(idx >= np.array([S, L, T, 2]))
                or len(np.unique(idx, axis=0)) != len(idx)):
            return False
        for scenario in range(S):
            limits = np.array(data.line_fmax, dtype=float)
            outage = data.maintenance_line if scenario == 0 else data.cc_monitor[scenario - 1]
            expected = {"outage_line": outage, "capacity_factor": 1.0 if scenario == 0 else 1.05,
                        "outage_capacity_factor": 1.0 if scenario == 0 else 0.5}
            if meta["scenarios"][scenario] != expected:
                return False
            if scenario:
                limits *= 1.05
                limits[outage] *= 0.5
            if not np.array_equal(data.line_binding_target[scenario],
                                  line_binding(data.line_flow_target[scenario], limits, tol)):
                return False
        p = data.p_target
        atol = 2 * PRICING_FEASIBILITY_TOL
        if (not np.allclose(p.sum(axis=0), data.demand.sum(axis=0), rtol=0, atol=atol)
                or np.any(p < data.uc_sol * data.gen_pmin[:, None] - atol)
                or np.any(p > data.uc_sol * data.gen_pmax[:, None] + atol)
                or np.any(np.diff(p, axis=1) > data.gen_ramp_up[:, None] + atol)
                or np.any(-np.diff(p, axis=1) > data.gen_ramp_down[:, None] + atol)):
            return False
        return True
    except (TypeError, ValueError, AttributeError, KeyError, IndexError, RuntimeError):
        return False


def _atomic_save(data, path: Path) -> None:
    # The temporary file must share the destination filesystem for atomic replace.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def label_sample(path, uc_type: str, overwrite: bool = False) -> str:
    """Return 'saved' or 'skipped'; failures leave the original file untouched."""
    path = Path(path)
    data = load_pkl(path)
    if infer_uc_type(data) != uc_type:
        raise ValueError("UC type does not match sample scenario data")
    if not overwrite and has_pricing_labels(data):
        return "skipped"
    result = solve_pricing(data, data.uc_sol, uc_type)
    for name in TARGET_FIELDS:
        setattr(data, name, getattr(result, name))
    if not valid_pricing_labels(data, uc_type):
        raise ValueError("Pricing result does not satisfy the label contract")
    _atomic_save(data, path)
    return "saved"
