"""Discrete lot-allocation solvers for exact and heuristic baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from ._result import make_result
from .portfolio_model import (
    discrete_constraints_hold,
    lot_bounds,
    lots_to_weights,
    objective_value,
)
from .schemas import (
    InfeasibleProblemError,
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverUnavailableError,
)


def enumerate_feasible_lots(problem: PortfolioProblem, units: int) -> Iterator[np.ndarray]:
    """Yield each hard-feasible integer composition exactly once."""
    low, high = lot_bounds(problem, units)
    if np.any(low > high) or low.sum() > units or high.sum() < units:
        return
    n = problem.n
    suffix_low = np.zeros(n + 1, dtype=int)
    suffix_high = np.zeros(n + 1, dtype=int)
    for i in range(n - 1, -1, -1):
        suffix_low[i] = suffix_low[i + 1] + low[i]
        suffix_high[i] = suffix_high[i + 1] + high[i]
    current = np.zeros(n, dtype=int)

    def recurse(i: int, remaining: int) -> Iterator[np.ndarray]:
        if i == n:
            if remaining == 0 and discrete_constraints_hold(current, problem, units):
                yield current.copy()
            return
        lo = max(int(low[i]), remaining - int(suffix_high[i + 1]))
        hi = min(int(high[i]), remaining - int(suffix_low[i + 1]))
        for value in range(lo, hi + 1):
            current[i] = value
            yield from recurse(i + 1, remaining - value)
        current[i] = 0

    yield from recurse(0, int(units))


def _first_feasible_lots(problem: PortfolioProblem, units: int) -> np.ndarray:
    try:
        return next(enumerate_feasible_lots(problem, units))
    except StopIteration as exc:
        raise InfeasibleProblemError(
            f"no feasible discrete allocation exists at M={units}; change the grid or bounds"
        ) from exc


def solve_discrete_enumeration(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 10,
) -> SolveResult:
    """Prove the tiny-instance optimum by exhaustive feasible enumeration."""
    preferences = preferences or Preferences()
    start = time.perf_counter()
    best_lots: np.ndarray | None = None
    best_objective = np.inf
    feasible_candidates = 0
    for lots in enumerate_feasible_lots(problem, units):
        feasible_candidates += 1
        weights = lots_to_weights(lots, problem, units)
        value = objective_value(weights, problem, preferences)
        if value < best_objective - 1e-15:
            best_objective = value
            best_lots = lots.copy()
    runtime = time.perf_counter() - start
    if best_lots is None:
        raise InfeasibleProblemError(f"no feasible discrete allocation exists at M={units}")
    return make_result(
        method="exact_enumeration",
        model_type="discrete",
        weights=lots_to_weights(best_lots, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status="optimal",
        success=True,
        optimal=True,
        units=units,
        metadata={
            "lots": best_lots.tolist(),
            "feasible_candidates": feasible_candidates,
        },
    )


def _local_improve(
    initial: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
    units: int,
    max_iterations: int,
) -> tuple[np.ndarray, int, int]:
    """Best-improvement one-lot swap search with hard feasibility."""
    current = np.asarray(initial, dtype=int).copy()
    current_value = objective_value(lots_to_weights(current, problem, units), problem, preferences)
    evaluations = 0
    iterations = 0
    for _ in range(max_iterations):
        best_trial: np.ndarray | None = None
        best_value = current_value
        for donor in range(problem.n):
            if current[donor] <= 0:
                continue
            for receiver in range(problem.n):
                if receiver == donor:
                    continue
                trial = current.copy()
                trial[donor] -= 1
                trial[receiver] += 1
                if not discrete_constraints_hold(trial, problem, units):
                    continue
                evaluations += 1
                value = objective_value(
                    lots_to_weights(trial, problem, units), problem, preferences
                )
                if value < best_value - 1e-13:
                    best_value = value
                    best_trial = trial
        if best_trial is None:
            break
        current = best_trial
        current_value = best_value
        iterations += 1
    return current, iterations, evaluations


def solve_discrete_local_search(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    max_iterations: int = 500,
    initial_lots: np.ndarray | None = None,
) -> SolveResult:
    """Run deterministic one-lot best-improvement local search."""
    preferences = preferences or Preferences()
    start = time.perf_counter()
    initial = (
        _first_feasible_lots(problem, units)
        if initial_lots is None
        else np.asarray(initial_lots, dtype=int)
    )
    if not discrete_constraints_hold(initial, problem, units):
        raise ValueError("initial_lots is not hard-feasible")
    best, iterations, evaluations = _local_improve(
        initial, problem, preferences, units, max_iterations
    )
    runtime = time.perf_counter() - start
    return make_result(
        method="swap_local_search",
        model_type="discrete",
        weights=lots_to_weights(best, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status="locally_optimal",
        success=True,
        optimal=False,
        units=units,
        metadata={
            "lots": best.tolist(),
            "iterations": iterations,
            "objective_evaluations": evaluations,
        },
    )


def solve_discrete_annealing(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    seed: int = 0,
    n_iterations: int = 20_000,
    initial_temperature: float = 0.02,
    final_temperature: float = 1e-5,
    local_polish: bool = True,
) -> SolveResult:
    """Budget-preserving simulated annealing followed by optional swap polish."""
    if n_iterations <= 0 or initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("annealing iterations and temperatures must be positive")
    preferences = preferences or Preferences()
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    current = _first_feasible_lots(problem, units)

    # Randomize the deterministic feasible start without violating hard constraints.
    for _ in range(10 * problem.n):
        donor, receiver = rng.choice(problem.n, size=2, replace=False)
        trial = current.copy()
        trial[donor] -= 1
        trial[receiver] += 1
        if discrete_constraints_hold(trial, problem, units):
            current = trial

    current_value = objective_value(
        lots_to_weights(current, problem, units), problem, preferences
    )
    best = current.copy()
    best_value = current_value
    accepted = 0
    feasible_trials = 0
    ratio = final_temperature / initial_temperature

    for step in range(n_iterations):
        donor, receiver = rng.choice(problem.n, size=2, replace=False)
        trial = current.copy()
        trial[donor] -= 1
        trial[receiver] += 1
        if not discrete_constraints_hold(trial, problem, units):
            continue
        feasible_trials += 1
        trial_value = objective_value(lots_to_weights(trial, problem, units), problem, preferences)
        delta = trial_value - current_value
        fraction = step / max(n_iterations - 1, 1)
        temperature = initial_temperature * ratio**fraction
        if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
            current = trial
            current_value = trial_value
            accepted += 1
            if current_value < best_value:
                best = current.copy()
                best_value = current_value

    polish_iterations = 0
    polish_evaluations = 0
    if local_polish:
        best, polish_iterations, polish_evaluations = _local_improve(
            best, problem, preferences, units, max_iterations=500
        )
    runtime = time.perf_counter() - start
    return make_result(
        method="simulated_annealing_swap",
        model_type="discrete",
        weights=lots_to_weights(best, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status="heuristic_complete",
        success=True,
        optimal=False,
        units=units,
        seed=seed,
        metadata={
            "lots": best.tolist(),
            "iterations": n_iterations,
            "feasible_trials": feasible_trials,
            "accepted_moves": accepted,
            "polish_iterations": polish_iterations,
            "polish_evaluations": polish_evaluations,
        },
    )


def solve_discrete_gurobi(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    time_limit: float | None = None,
    mip_gap: float = 1e-9,
    output: bool = False,
) -> SolveResult:
    """Solve the integer-lot MIQP directly with optional Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("gurobipy is not installed; install the 'gurobi' extra") from exc

    preferences = preferences or Preferences()
    low, high = lot_bounds(problem, units)
    if np.any(low > high):
        raise InfeasibleProblemError("discrete lot bounds are infeasible")
    lot_size = problem.budget / units
    start = time.perf_counter()
    try:
        model = gp.Model("vanguard_discrete")
        model.Params.OutputFlag = int(output)
        model.Params.MIPGap = float(mip_gap)
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)
        q = model.addMVar(problem.n, vtype=GRB.INTEGER, lb=low, ub=high, name="q")
        t = model.addMVar(problem.n, lb=0.0, name="t")
        model.addConstr(q.sum() == units, name="budget_lots")
        model.addConstr(lot_size * (problem.A @ q) >= problem.group_lower, name="group_lower")
        model.addConstr(lot_size * (problem.A @ q) <= problem.group_upper, name="group_upper")
        model.addConstr(t >= lot_size * q - problem.w0, name="turnover_plus")
        model.addConstr(t >= problem.w0 - lot_size * q, name="turnover_minus")
        if problem.target_return is not None:
            model.addConstr(lot_size * (problem.mu @ q) >= problem.target_return)
        if problem.max_turnover is not None:
            model.addConstr(t.sum() <= problem.max_turnover)
        model.setObjective(
            preferences.lambda_risk * lot_size**2 * (q @ problem.cov @ q)
            - preferences.lambda_return * lot_size * (problem.mu @ q)
            - preferences.lambda_income * lot_size * (problem.y @ q)
            + preferences.lambda_cost * (problem.c @ t),
            GRB.MINIMIZE,
        )
        model.optimize()
    except gp.GurobiError as exc:  # pragma: no cover - license dependent
        raise SolverUnavailableError(f"Gurobi could not start or solve: {exc}") from exc
    runtime = time.perf_counter() - start
    success = model.SolCount > 0
    optimal = model.Status == GRB.OPTIMAL
    lots = np.rint(np.asarray(q.X)).astype(int) if success else np.zeros(problem.n, dtype=int)
    weights = lots_to_weights(lots, problem, units) if success else np.full(problem.n, np.nan)
    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    metadata: dict[str, Any] = {
        "lots": lots.tolist(),
        "solver_runtime": float(model.Runtime),
        "nodes": float(model.NodeCount),
    }
    if success:
        metadata["native_objective"] = float(model.ObjVal)
        try:
            metadata["best_bound"] = float(model.ObjBound)
            metadata["reported_mip_gap"] = float(model.MIPGap)
        except gp.GurobiError:
            pass
    return make_result(
        method="gurobi_miqp",
        model_type="discrete",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=status_names.get(model.Status, f"status_{model.Status}"),
        success=success,
        optimal=optimal,
        units=units,
        metadata=metadata,
    )


def solve_discrete_cvxpy(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    solver_name: str = "SCIP",
) -> SolveResult:
    """Solve the MIQP through an installed CVXPY MIP-capable backend."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("CVXPY is not installed; install the 'qp' extra") from exc

    solver_name = solver_name.upper()
    if solver_name not in cp.installed_solvers():
        raise SolverUnavailableError(f"CVXPY solver {solver_name!r} is not installed")
    preferences = preferences or Preferences()
    low, high = lot_bounds(problem, units)
    lot_size = problem.budget / units
    q = cp.Variable(problem.n, integer=True, name="q")
    t = cp.Variable(problem.n, nonneg=True, name="t")
    w = lot_size * q
    objective = cp.Minimize(
        preferences.lambda_risk * cp.quad_form(w, cp.psd_wrap(problem.cov))
        - preferences.lambda_return * problem.mu @ w
        - preferences.lambda_income * problem.y @ w
        + preferences.lambda_cost * problem.c @ t
    )
    constraints = [
        q >= low,
        q <= high,
        cp.sum(q) == units,
        problem.A @ w >= problem.group_lower,
        problem.A @ w <= problem.group_upper,
        t >= w - problem.w0,
        t >= problem.w0 - w,
    ]
    if problem.target_return is not None:
        constraints.append(problem.mu @ w >= problem.target_return)
    if problem.max_turnover is not None:
        constraints.append(cp.sum(t) <= problem.max_turnover)
    model = cp.Problem(objective, constraints)
    start = time.perf_counter()
    try:
        model.solve(solver=solver_name, verbose=False)
    except cp.error.SolverError as exc:  # pragma: no cover - solver-specific
        raise SolverUnavailableError(f"CVXPY/{solver_name} could not run: {exc}") from exc
    runtime = time.perf_counter() - start
    success = model.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and q.value is not None
    lots = np.rint(np.asarray(q.value)).astype(int) if success else np.zeros(problem.n, dtype=int)
    weights = lots_to_weights(lots, problem, units) if success else np.full(problem.n, np.nan)
    return make_result(
        method=f"cvxpy_{solver_name.lower()}_miqp",
        model_type="discrete",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=str(model.status),
        success=success,
        optimal=model.status == cp.OPTIMAL,
        units=units,
        metadata={
            "lots": lots.tolist(),
            "solver_name": solver_name,
            "native_objective": float(model.value) if model.value is not None else None,
            "solver_runtime": getattr(model.solver_stats, "solve_time", None),
        },
    )


def solve_discrete(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    backend: str = "enumeration",
    **kwargs: Any,
) -> SolveResult:
    name = backend.strip()
    lower = name.lower()
    if lower in {"enumeration", "exact", "brute"}:
        return solve_discrete_enumeration(problem, preferences, units=units, **kwargs)
    if lower in {"local_search", "swap", "swap_local_search"}:
        return solve_discrete_local_search(problem, preferences, units=units, **kwargs)
    if lower in {"annealing", "anneal", "simulated_annealing"}:
        return solve_discrete_annealing(problem, preferences, units=units, **kwargs)
    if lower == "gurobi":
        return solve_discrete_gurobi(problem, preferences, units=units, **kwargs)
    if lower.startswith("cvxpy:"):
        return solve_discrete_cvxpy(
            problem,
            preferences,
            units=units,
            solver_name=name.split(":", 1)[1],
            **kwargs,
        )
    raise ValueError(f"unknown discrete backend {backend!r}")


@dataclass
class MeanVarianceDiscreteOptimizer:
    problem: PortfolioProblem
    preferences: Preferences = Preferences()
    units: int = 20
    method: str = "annealing"
    seed: int = 0

    def solve(self, **kwargs: Any) -> SolveResult:
        if self.method.lower() in {"annealing", "anneal", "simulated_annealing"}:
            kwargs.setdefault("seed", self.seed)
        return solve_discrete(
            self.problem,
            self.preferences,
            units=self.units,
            backend=self.method,
            **kwargs,
        )


def mean_variance_discrete(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    n_lots: int | None = None,
    units: int | None = None,
    method: str = "annealing",
    **kwargs: Any,
) -> SolveResult:
    """Compatibility wrapper accepting either ``n_lots`` or ``units``."""
    resolved_units = units if units is not None else (n_lots if n_lots is not None else 20)
    return solve_discrete(
        problem,
        preferences,
        units=resolved_units,
        backend=method,
        **kwargs,
    )


__all__ = [
    "MeanVarianceDiscreteOptimizer",
    "enumerate_feasible_lots",
    "mean_variance_discrete",
    "solve_discrete",
    "solve_discrete_annealing",
    "solve_discrete_cvxpy",
    "solve_discrete_enumeration",
    "solve_discrete_gurobi",
    "solve_discrete_local_search",
]

