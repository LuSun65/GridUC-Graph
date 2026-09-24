"""Continuous pricing solve for a fixed unit-commitment schedule."""

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from lib.case_loader import UCData
from lib.uc_model import UCModel, build_uc, init_uc_grb
from lib.ptdf_solver import compute_ptdf, compute_post_ptdf


PRICING_FEASIBILITY_TOL = 1e-6
PRICING_OPTIMALITY_TOL = 1e-6
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


def _validate_fixed_transitions(data, commitment, startup, shutdown):
    # Preserve the original truncated minimum-up/down constraints, even though
    # their now-constant rows do not need to be sent to the LP solver.
    for g in range(data.num_gen):
        for t in range(data.num_period):
            for duration, state, trigger in (
                (int(data.gen_min_up[g]), commitment[g], startup[g, t]),
                (int(data.gen_min_down[g]), 1 - commitment[g], shutdown[g, t]),
            ):
                end = min(t + duration, data.num_period)
                if duration > 1 and state[t:end].sum() < trigger * (end - t):
                    raise PricingSolveError(f"Invalid minimum up/down schedule at generator {g}, period {t}")


def _build_fixed_lp(model, data, commitment, startup):
    G, T = commitment.shape
    p = model.addMVar((G, T), lb=0, name="p")
    c = model.addMVar((G, T), lb=0, name="c")
    dp = model.addMVar((G, data.bid_price.shape[1], T), lb=0, name="dp")
    model.setObjective(c.sum() + float((data.gen_startup_cost[:, None] * startup).sum()), GRB.MINIMIZE)
    model.addConstr(c == data.bid_no_load[:, None] * commitment + (data.bid_price * dp).sum(axis=1), name="pwl_cost")
    model.addConstr(p == data.bid_mw[:, 0:1] * commitment + dp.sum(axis=1), name="pwl_link")
    model.addConstr(dp <= np.diff(data.bid_mw, axis=1)[:, :, None] * commitment[:, None, :], name="dp_upper")
    model.addConstr(p >= commitment * data.gen_pmin[:, None], name="pmin")
    model.addConstr(p <= commitment * data.gen_pmax[:, None], name="pmax")
    balance = model.addConstr(p.sum(axis=0) == data.demand.sum(axis=0), name="balance")
    model.addConstr(p[:, 1:] - p[:, :-1] <= data.gen_ramp_up[:, None], name="ramp_up")
    model.addConstr(p[:, :-1] - p[:, 1:] <= data.gen_ramp_down[:, None], name="ramp_down")
    return p, balance


def _scenario_ptdf(data, outage):
    if outage is None:
        return compute_ptdf(data)
    return compute_post_ptdf(data, outage_line=outage)[0]


def solve_pricing(data: UCData, uc_sol: np.ndarray, uc_type: str) -> PricingResult:
    """Solve a direct continuous LP, separating every network scenario each round.

    Omitted rows have zero duals. Labels are returned only after an optimal LP
    solution passes the full network check at the pricing feasibility tolerance.
    """
    commitment = _validate_uc_solution(data, uc_sol, uc_type)
    startup, shutdown = _derive_startup_shutdown(commitment)
    _validate_fixed_transitions(data, commitment, startup, shutdown)
    if data.demand is None:
        raise ValueError("UCData.demand must be set")
    if uc_type == "scuc" and not data.cc_monitor:
        raise ValueError("cc_monitor must be set before pricing SCUC")

    # Match build_uc: maintenance affects N-0 PTDF but not N-0 ratings;
    # contingency scenarios are each derived from the original network.
    scenarios = [(data.maintenance_line, np.asarray(data.line_fmax, float))]
    if uc_type == "scuc":
        for outage in data.cc_monitor:
            limits = np.array(data.line_fmax, dtype=float) * 1.05
            limits[outage] *= 0.5
            scenarios.append((outage, limits))
    # Only generator columns and demand products survive scenario preparation.
    networks = []
    for outage, limits in scenarios:
        matrix = _scenario_ptdf(data, outage)
        networks.append((matrix[:, data.gen_bus].copy(), matrix @ data.demand, limits))
    del matrix

    with init_uc_grb() as env, gp.Model("pricing_lp", env=env) as model:
        p, balance = _build_fixed_lp(model, data, commitment, startup)
        added = {}
        runtime = 0.0
        rounds = 0
        while True:
            _solve_pricing_lp(model)
            runtime += model.Runtime
            rounds += 1
            values = p.X
            if not np.isfinite(values).all():
                raise PricingSolveError("Non-finite pricing dispatch")
            pending = []
            max_violation = 0.0
            for scenario, (coeff, rhs, limits) in enumerate(networks):
                flow = coeff @ values - rhs
                if not np.isfinite(flow).all():
                    raise PricingSolveError("Non-finite network flows")
                for upper, violation in ((True, flow - limits[:, None]), (False, -limits[:, None] - flow)):
                    max_violation = max(max_violation, float(violation.max()))
                    for line, period in zip(*np.where(violation > PRICING_FEASIBILITY_TOL)):
                        key = (scenario, int(line), int(period), upper)
                        if key not in added:
                            pending.append(key)
            if not pending:
                if max_violation > PRICING_FEASIBILITY_TOL:
                    raise PricingSolveError(f"Existing network rows violate pricing tolerance: {max_violation:g}")
                break
            for scenario, line, period, upper in pending:
                coeff, rhs, limits = networks[scenario]
                expression = coeff[line] @ p[:, period]
                bound = float(rhs[line, period] + (limits[line] if upper else -limits[line]))
                added[scenario, line, period, upper] = model.addConstr(
                    expression <= bound if upper else expression >= bound,
                    name=f"network_{scenario}_{line}_{period}_{int(upper)}",
                )

        lmp = np.broadcast_to(balance.Pi, (data.num_bus, data.num_period)).copy()
        # Reconstruct full bus sensitivities one scenario at a time, only for
        # scenarios with nonzero duals. No stacked contingency PTDF is retained.
        for scenario, (outage, _) in enumerate(scenarios):
            active = [(line, period, constraint.Pi)
                      for (s, line, period, _), constraint in added.items()
                      if s == scenario and constraint.Pi != 0]
            if active:
                matrix = _scenario_ptdf(data, outage)
                for line, period, dual in active:
                    lmp[:, period] += matrix[line] * dual
        if not np.isfinite(lmp).all():
            raise PricingSolveError("Non-finite pricing LMP")
        return PricingResult(
            p_target=values, lmp_target=lmp, obj=float(model.ObjVal),
            solve_time=float(runtime),
            metadata={"uc_type": uc_type,
                      "constraint_generation_rounds": rounds,
                      "network_constraints_added": len(added),
                      "max_network_violation": max_violation},
        )
