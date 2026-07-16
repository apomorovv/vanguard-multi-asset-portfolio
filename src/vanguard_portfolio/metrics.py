"""Portfolio metrics and constraint checking.

Every solver in the project is scored with the functions in this module so
that the comparison is fair. The formulas follow Section 5 and Section 11 of
`docs/mathematical_model.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data_generation import PortfolioProblem


def expected_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Portfolio expected return `R(w) = mu^T w`."""
    return float(mu @ w)


def variance(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio variance `V(w) = w^T Sigma w`."""
    return float(w @ cov @ w)


def volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility `sigma_p(w) = sqrt(w^T Sigma w)`."""
    return float(np.sqrt(max(variance(w, cov), 0.0)))


def income(w: np.ndarray, y: np.ndarray) -> float:
    """Portfolio income yield `Y(w) = y^T w`."""
    return float(y @ w)


def turnover(w: np.ndarray, w0: np.ndarray) -> float:
    """Turnover `T(w) = sum_i |w_i - w0_i|`."""
    return float(np.sum(np.abs(w - w0)))


def transaction_cost(w: np.ndarray, w0: np.ndarray, c: np.ndarray) -> float:
    """Estimated transaction cost `C(w) = sum_i c_i |w_i - w0_i|`."""
    return float(np.sum(c * np.abs(w - w0)))


@dataclass
class ConstraintReport:
    """Result of checking every hard constraint for an allocation."""

    feasible: bool
    breaches: int
    max_violation: float
    details: list[str] = field(default_factory=list)


def constraint_report(
    w: np.ndarray,
    problem: PortfolioProblem,
    tol: float = 1e-6,
) -> ConstraintReport:
    """Check the hard constraints from Section 10 of the model.

    Hard constraints: full investment, long-only, per-asset bounds and
    asset-group exposure limits. A breach is recorded whenever a constraint
    is violated by more than `tol`.
    """
    details: list[str] = []
    violations: list[float] = []

    # Full investment: sum(w) == 1.
    budget_gap = abs(float(np.sum(w)) - 1.0)
    if budget_gap > tol:
        details.append(f"budget: sum(w)={np.sum(w):.6f} (gap {budget_gap:.2e})")
        violations.append(budget_gap)

    # Long-only: w_i >= 0.
    for i, wi in enumerate(w):
        if wi < -tol:
            name = problem.asset_names[i]
            details.append(f"long-only: {name} weight {wi:.6f} < 0")
            violations.append(-wi)

    # Per-asset bounds: l_i <= w_i <= u_i.
    for i, wi in enumerate(w):
        name = problem.asset_names[i]
        if wi < problem.lower[i] - tol:
            gap = problem.lower[i] - wi
            details.append(f"lower bound: {name} {wi:.6f} < {problem.lower[i]:.4f}")
            violations.append(gap)
        if wi > problem.upper[i] + tol:
            gap = wi - problem.upper[i]
            details.append(f"upper bound: {name} {wi:.6f} > {problem.upper[i]:.4f}")
            violations.append(gap)

    # Group exposure limits: L_g <= sum_i a_gi w_i <= U_g.
    group_exposure = problem.A @ w
    for g, exposure in enumerate(group_exposure):
        name = problem.group_names[g]
        if exposure < problem.group_lower[g] - tol:
            gap = problem.group_lower[g] - exposure
            details.append(
                f"group lower: {name} {exposure:.6f} < {problem.group_lower[g]:.4f}"
            )
            violations.append(gap)
        if exposure > problem.group_upper[g] + tol:
            gap = exposure - problem.group_upper[g]
            details.append(
                f"group upper: {name} {exposure:.6f} > {problem.group_upper[g]:.4f}"
            )
            violations.append(gap)

    max_violation = float(max(violations)) if violations else 0.0
    return ConstraintReport(
        feasible=len(violations) == 0,
        breaches=len(violations),
        max_violation=max_violation,
        details=details,
    )


def portfolio_metrics(w: np.ndarray, problem: PortfolioProblem) -> dict[str, float]:
    """Compute the full metric set from Section 11 for allocation `w`."""
    return {
        "expected_return": expected_return(w, problem.mu),
        "variance": variance(w, problem.cov),
        "volatility": volatility(w, problem.cov),
        "income": income(w, problem.y),
        "turnover": turnover(w, problem.w0),
        "transaction_cost": transaction_cost(w, problem.w0, problem.c),
    }