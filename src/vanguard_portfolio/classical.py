"""Classical portfolio optimizers.

Two reference optimizers are implemented, matching Sections 6-8 of
`docs/mathematical_model.md`:

* :func:solve_continuous - the continuous mean-variance quadratic program
  solved with SciPy's SLSQP routine. This is the ground-truth reference.
* :func:solve_discrete - the discrete allocation-unit model solved exactly by
  enumeration. This is the reference the QUBO/quantum solvers must match.

Both optimizers minimise the same objective (Section 6)::

    lambda_risk   * w^T Sigma w
  - lambda_return * mu^T w
  - lambda_income * y^T w
  + lambda_cost   * sum_i c_i |w_i - w0_i|

subject to the hard constraints (budget, per-asset bounds, group limits and
long-only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .data_generation import PortfolioProblem
from .metrics import constraint_report, portfolio_metrics


@dataclass
class Preferences:
    """Objective weights (the `lambda` coefficients of Section 6)."""

    lambda_return: float = 1.0
    lambda_risk: float = 5.0
    lambda_income: float = 0.0
    lambda_cost: float = 1.0


# Tunable investor goals (Deliverable 5): growth, income, drawdown control
# and cost sensitivity, plus a balanced baseline.
PRESETS: dict[str, Preferences] = {
    "balanced": Preferences(lambda_return=1.0, lambda_risk=5.0, lambda_income=0.0, lambda_cost=1.0),
    "growth": Preferences(lambda_return=2.0, lambda_risk=3.0, lambda_income=0.0, lambda_cost=0.5),
    "income": Preferences(lambda_return=0.5, lambda_risk=4.0, lambda_income=2.0, lambda_cost=1.0),
    "drawdown_control": Preferences(lambda_return=0.5, lambda_risk=15.0, lambda_income=0.0, lambda_cost=1.0),
    "cost_sensitive": Preferences(lambda_return=1.0, lambda_risk=5.0, lambda_income=0.0, lambda_cost=10.0),
}


@dataclass
class SolveResult:
    """Outcome of a single optimizer run."""

    method: str
    weights: np.ndarray
    objective: float
    runtime: float
    feasible: bool
    breaches: int
    max_violation: float
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True


def objective_value(w: np.ndarray, problem: PortfolioProblem, prefs: Preferences) -> float:
    """Evaluate the scalar objective of Section 6 for allocation `w`."""
    risk = w @ problem.cov @ w
    ret = problem.mu @ w
    inc = problem.y @ w
    cost = np.sum(problem.c * np.abs(w - problem.w0))
    return float(
        prefs.lambda_risk * risk
        - prefs.lambda_return * ret
        - prefs.lambda_income * inc
        + prefs.lambda_cost * cost
    )


def _build_result(
    method: str,
    w: np.ndarray,
    problem: PortfolioProblem,
    prefs: Preferences,
    runtime: float,
    success: bool = True,
) -> SolveResult:
    report = constraint_report(w, problem)
    return SolveResult(
        method=method,
        weights=w,
        objective=objective_value(w, problem, prefs),
        runtime=runtime,
        feasible=report.feasible,
        breaches=report.breaches,
        max_violation=report.max_violation,
        metrics=portfolio_metrics(w, problem),
        success=success,
    )


def solve_continuous(problem: PortfolioProblem, prefs: Preferences | None = None) -> SolveResult:
    """Solve the continuous mean-variance QP of Section 6/7 with SLSQP.

    The decision vector is augmented as `z = [w, t]` where `t_i` models the
    absolute allocation change `|w_i - w0_i|` so that the transaction-cost
    term is linear (Section 6).
    """
    prefs = prefs or Preferences()
    n = problem.n

    def objective(z: np.ndarray) -> float:
        w = z[:n]
        t = z[n:]
        risk = w @ problem.cov @ w
        ret = problem.mu @ w
        inc = problem.y @ w
        cost = problem.c @ t
        return (
            prefs.lambda_risk * risk
            - prefs.lambda_return * ret
            - prefs.lambda_income * inc
            + prefs.lambda_cost * cost
        )

    constraints: list[dict] = [
        # Full investment: sum(w) = 1.
        {"type": "eq", "fun": lambda z: np.sum(z[:n]) - 1.0},
    ]
    # Group exposure limits: L_g <= A_g w <= U_g.
    for g in range(problem.num_groups):
        a_g = problem.A[g]
        low = problem.group_lower[g]
        high = problem.group_upper[g]
        constraints.append({"type": "ineq", "fun": (lambda z, a=a_g, low=low: a @ z[:n] - low)})
        constraints.append({"type": "ineq", "fun": (lambda z, a=a_g, high=high: high - a @ z[:n])})
    # Turnover linearisation: t_i >= w_i - w0_i and t_i >= w0_i - w_i.
    constraints.append({"type": "ineq", "fun": lambda z: z[n:] - (z[:n] - problem.w0)})
    constraints.append({"type": "ineq", "fun": lambda z: z[n:] - (problem.w0 - z[:n])})

    # Bounds: w_i in [l_i, u_i], t_i in [0, 1].
    bounds = [(problem.lower[i], problem.upper[i]) for i in range(n)]
    bounds += [(0.0, 1.0) for _ in range(n)]

    z0 = np.concatenate([problem.w0, np.zeros(n)])

    start = time.perf_counter()
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    runtime = time.perf_counter() - start

    w = np.clip(result.x[:n], problem.lower, problem.upper)
    w = w / w.sum()  # renormalise away tiny numerical drift
    return _build_result("continuous", w, problem, prefs, runtime, success=bool(result.success))


def _integer_allocations(
    problem: PortfolioProblem,
    units: int,
) -> "list[np.ndarray]":
    """Enumerate every feasible integer allocation `q` with `sum q = units`.

    Per-asset bounds (Section 8) prune the search; group limits are checked on
    completed vectors.
    """
    n = problem.n
    q_low = np.ceil(problem.lower * units - 1e-9).astype(int)
    q_high = np.floor(problem.upper * units + 1e-9).astype(int)
    q_low = np.clip(q_low, 0, units)
    q_high = np.clip(q_high, 0, units)

    results: list[np.ndarray] = []
    current = np.zeros(n, dtype=int)

    # Suffix maxima to prune branches that can no longer reach `units`.
    suffix_max = np.zeros(n + 1, dtype=int)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + q_high[i]

    def recurse(i: int, remaining: int) -> None:
        if i == n:
            if remaining == 0:
                results.append(current.copy())
            return
        # Feasible range for asset i given remaining budget and future capacity.
        lo = max(q_low[i], remaining - suffix_max[i + 1])
        hi = min(q_high[i], remaining)
        for value in range(lo, hi + 1):
            current[i] = value
            recurse(i + 1, remaining - value)
        current[i] = 0

    recurse(0, units)
    return results


def solve_discrete(
    problem: PortfolioProblem,
    prefs: Preferences | None = None,
    units: int = 10,
) -> SolveResult:
    """Solve the discrete allocation-unit model of Section 8 by exact enumeration.

    `units` is `M` in the model, so each unit represents `1/M` of the
    portfolio. Group limits are enforced as hard constraints; the feasible
    allocation with the lowest objective is returned.
    """
    prefs = prefs or Preferences()

    start = time.perf_counter()
    best_w: np.ndarray | None = None
    best_obj = np.inf

    group_low = np.ceil(problem.group_lower * units - 1e-9)
    group_high = np.floor(problem.group_upper * units + 1e-9)

    for q in _integer_allocations(problem, units):
        exposure = problem.A @ q
        if np.any(exposure < group_low - 1e-9) or np.any(exposure > group_high + 1e-9):
            continue
        w = q / units
        obj = objective_value(w, problem, prefs)
        if obj < best_obj:
            best_obj = obj
            best_w = w

    runtime = time.perf_counter() - start

    if best_w is None:
        # No feasible integer allocation exists for this granularity.
        return SolveResult(
            method=f"discrete(M={units})",
            weights=np.full(problem.n, np.nan),
            objective=np.inf,
            runtime=runtime,
            feasible=False,
            breaches=-1,
            max_violation=np.inf,
            metrics={},
            success=False,
        )

    return _build_result(f"discrete(M={units})", best_w, problem, prefs, runtime)