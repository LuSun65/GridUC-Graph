"""Continuous pricing solve for a fixed unit-commitment schedule."""

from dataclasses import dataclass
import hashlib

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from lib.case_loader import UCData
from lib.uc_model import UCModel, build_uc


PRICING_FEASIBILITY_TOL = 1e-4
PRICING_OPTIMALITY_TOL = 1e-6
# Contract: dense fixed-commitment Gurobi LP, initially all off, no balance
# slack, OPTIMAL only, tolerances above, power in MW and LMP in currency/MWh.
# Bump this version whenever these pricing conventions change.
PRICING_RULE = "fixed_commitment_lp_v1"


def commitment_hash(commitment: np.ndarray) -> str:
    """Hash binary commitments independently of their original numpy dtype."""
    values = np.ascontiguousarray(np.rint(commitment), dtype=np.uint8)
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


class PricingSolveError(RuntimeError):
    """Raised when a reliable optimal pricing solution cannot be produced."""


@dataclass
class PricingResult:
    """Targets obtained from one internally consistent pricing LP solution."""

    p_target: np.ndarray
    lmp_target: np.ndarray
    obj: float
    solve_time: float
    metadata: dict


def _validate_uc_solution(data: UCData, uc_sol: np.ndarray, uc_type: str) -> np.ndarray:
    """Validate and normalize the saved commitment schedule."""
    if uc_type not in ("tcuc", "topo", "scuc"):
        raise ValueError(
            f"Unknown uc_type: {uc_type!r}, expected 'tcuc', 'topo', or 'scuc'"
        )

    commitment = np.asarray(uc_sol, dtype=float)
    expected_shape = (data.num_gen, data.num_period)
    if commitment.shape != expected_shape:
        raise ValueError(
            f"uc_sol has shape {commitment.shape}, expected {expected_shape}"
        )
    if not np.isfinite(commitment).all():
        raise ValueError("uc_sol contains non-finite values")

    rounded = np.rint(commitment)
    if not np.allclose(commitment, rounded, rtol=0.0, atol=1e-8):
        raise ValueError("uc_sol must contain only binary values")
    if not np.isin(rounded, (0.0, 1.0)).all():
        raise ValueError("uc_sol must contain only binary values")
    return rounded.astype(int)


def _derive_startup_shutdown(commitment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive startup and shutdown schedules, assuming every unit starts off."""
    startup = np.zeros_like(commitment)
    shutdown = np.zeros_like(commitment)

    startup[:, 0] = commitment[:, 0]
    changes = commitment[:, 1:] - commitment[:, :-1]
    startup[:, 1:] = np.maximum(changes, 0)
    shutdown[:, 1:] = np.maximum(-changes, 0)
    return startup, shutdown


def _build_dense_pricing_uc(data: UCData, uc_type: str) -> UCModel:
    """Rebuild the sampled scenario with every transmission constraint present."""
    return build_uc(data, uc_type, mode="dense")


def _fix_matrix_variable(model: gp.Model, name: str, values: np.ndarray) -> None:
    """Fix every member of a named two-dimensional Gurobi variable array."""
    for row, col in np.ndindex(values.shape):
        variable = model.getVarByName(f"{name}[{row},{col}]")
        if variable is None:
            raise PricingSolveError(f"Pricing model is missing variable {name}[{row},{col}]")
        value = float(values[row, col])
        variable.LB = value
        variable.UB = value


def _fix_commitment_state(
    uc: UCModel,
    commitment: np.ndarray,
    startup: np.ndarray,
    shutdown: np.ndarray,
) -> None:
    """Fix commitment, startup, and shutdown decisions to the saved schedule."""
    _fix_matrix_variable(uc.model, "z", commitment)
    _fix_matrix_variable(uc.model, "u", startup)
    _fix_matrix_variable(uc.model, "v", shutdown)
    uc.model.update()


def _relax_to_pricing_lp(uc: UCModel) -> gp.Model:
    """Create the continuous LP after all integer decisions have been fixed."""
    uc.model.update()
    pricing_lp = uc.model.relax()
    pricing_lp.ModelName = "pricing_lp"
    pricing_lp.update()
    return pricing_lp


def _solve_pricing_lp(pricing_lp: gp.Model) -> None:
    """Solve with strict LP tolerances and reject every non-optimal outcome."""
    pricing_lp.Params.FeasibilityTol = PRICING_FEASIBILITY_TOL
    pricing_lp.Params.OptimalityTol = PRICING_OPTIMALITY_TOL
    pricing_lp.optimize()

    if pricing_lp.Status != GRB.OPTIMAL:
        raise PricingSolveError(
            "Pricing LP did not reach OPTIMAL status "
            f"(status={pricing_lp.Status}, sol_count={pricing_lp.SolCount})"
        )
    if not np.isfinite(pricing_lp.ObjVal):
        raise PricingSolveError("Pricing LP returned a non-finite objective value")


def _extract_variable_values(
    model: gp.Model, name: str, shape: tuple[int, int]
) -> np.ndarray:
    """Read an unrounded two-dimensional variable array from an optimal LP."""
    values = np.empty(shape, dtype=float)
    for row, col in np.ndindex(shape):
        variable = model.getVarByName(f"{name}[{row},{col}]")
        if variable is None:
            raise PricingSolveError(f"Pricing LP is missing variable {name}[{row},{col}]")
        values[row, col] = variable.X
    if not np.isfinite(values).all():
        raise PricingSolveError(f"Pricing LP returned non-finite values for {name}")
    return values


def _extract_constraint_duals(
    model: gp.Model, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Read the dual values of a named Gurobi constraint array."""
    duals = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        suffix = ",".join(str(i) for i in index)
        constraint = model.getConstrByName(f"{name}[{suffix}]")
        if constraint is None:
            raise PricingSolveError(
                f"Pricing LP is missing constraint {name}[{suffix}]"
            )
        duals[index] = constraint.Pi
    if not np.isfinite(duals).all():
        raise PricingSolveError(f"Pricing LP returned non-finite duals for {name}")
    return duals


def _calculate_lmp(uc: UCModel, pricing_lp: gp.Model) -> np.ndarray:
    """Calculate nodal prices from balance and dense line-constraint duals."""
    num_period = uc.demands.shape[1]
    balance = _extract_constraint_duals(pricing_lp, "balance", (num_period,))

    flow_shape = (uc.ptdf.shape[0], num_period)
    flow_ub = _extract_constraint_duals(pricing_lp, "flow_ub", flow_shape)
    flow_lb = _extract_constraint_duals(pricing_lp, "flow_lb", flow_shape)
    lmp = balance[None, :] + uc.ptdf.T @ (flow_ub + flow_lb)

    if uc.post_ptdf is not None:
        security_shape = (uc.post_ptdf.shape[0], num_period)
        security_ub = _extract_constraint_duals(
            pricing_lp, "sc_flow_ub", security_shape
        )
        security_lb = _extract_constraint_duals(
            pricing_lp, "sc_flow_lb", security_shape
        )
        lmp += uc.post_ptdf.T @ (security_ub + security_lb)

    if not np.isfinite(lmp).all():
        raise PricingSolveError("Pricing LP produced non-finite LMP values")
    return lmp


def _dispose_pricing_models(uc: UCModel | None, pricing_lp: gp.Model | None) -> None:
    """Release both Gurobi models and their shared environment."""
    if pricing_lp is not None:
        pricing_lp.dispose()
    if uc is not None:
        uc.model.dispose()
        uc.env.dispose()


def solve_pricing(data: UCData, uc_sol: np.ndarray, uc_type: str) -> PricingResult:
    """Build and solve a dense fixed-commitment LP and return power/LMP targets.

    Any invalid input, non-optimal solver status, missing model component, or
    non-finite result raises an exception directly.
    """
    commitment = _validate_uc_solution(data, uc_sol, uc_type)
    startup, shutdown = _derive_startup_shutdown(commitment)

    uc = None
    pricing_lp = None
    try:
        uc = _build_dense_pricing_uc(data, uc_type)
        _fix_commitment_state(uc, commitment, startup, shutdown)
        pricing_lp = _relax_to_pricing_lp(uc)
        _solve_pricing_lp(pricing_lp)

        p_target = _extract_variable_values(
            pricing_lp, "p", (data.num_gen, data.num_period)
        )
        lmp_target = _calculate_lmp(uc, pricing_lp)
        return PricingResult(
            p_target=p_target,
            lmp_target=lmp_target,
            obj=float(pricing_lp.ObjVal),
            solve_time=float(pricing_lp.Runtime),
            metadata={
                "pricing_rule": PRICING_RULE,
                "uc_sol_sha256": commitment_hash(commitment),
                "uc_type": uc_type,
            },
        )
    finally:
        _dispose_pricing_models(uc, pricing_lp)
