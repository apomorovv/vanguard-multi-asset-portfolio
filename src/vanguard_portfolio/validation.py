"""Independent hard-constraint validation for decoded portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .portfolio_model import empirical_cvar, turnover
from .schemas import PortfolioConstraints, PortfolioProblem


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    sense: str
    lhs: float
    rhs: float
    violation: float
    slack: float


@dataclass
class ConstraintReport:
    feasible: bool
    breaches: int
    max_violation: float
    checks: list[ConstraintCheck] = field(default_factory=list)

    @property
    def details(self) -> list[str]:
        return [
            f"{check.name}: {check.lhs:.8g} {check.sense} {check.rhs:.8g} "
            f"(violation={check.violation:.3e})"
            for check in self.checks
            if check.violation > 0.0
        ]


def validate_weights(
    weights: np.ndarray,
    problem: PortfolioProblem,
    *,
    units: int | None = None,
    constraints: PortfolioConstraints | None = None,
    support_tol: float = 1e-8,
    tol: float = 1e-7,
) -> ConstraintReport:
    """Check every hard constraint without modifying the candidate weights."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (problem.n,) or not np.all(np.isfinite(w)):
        return ConstraintReport(False, 1, np.inf, [])

    checks: list[ConstraintCheck] = []

    def equal(name: str, lhs: float, rhs: float) -> None:
        violation = abs(lhs - rhs)
        checks.append(ConstraintCheck(name, "=", lhs, rhs, violation, -violation))

    def lower(name: str, lhs: float, rhs: float) -> None:
        checks.append(ConstraintCheck(name, ">=", lhs, rhs, max(rhs - lhs, 0.0), lhs - rhs))

    def upper(name: str, lhs: float, rhs: float) -> None:
        checks.append(ConstraintCheck(name, "<=", lhs, rhs, max(lhs - rhs, 0.0), rhs - lhs))

    equal("budget", float(w.sum()), problem.budget)
    for i, name in enumerate(problem.asset_names):
        lower(f"asset_lower:{name}", float(w[i]), float(problem.lower[i]))
        upper(f"asset_upper:{name}", float(w[i]), float(problem.upper[i]))

    exposure = problem.A @ w
    for g, name in enumerate(problem.group_names):
        lower(f"group_lower:{name}", float(exposure[g]), float(problem.group_lower[g]))
        upper(f"group_upper:{name}", float(exposure[g]), float(problem.group_upper[g]))

    if problem.target_return is not None:
        lower("target_return", float(problem.mu @ w), problem.target_return)
    if problem.max_turnover is not None:
        upper("max_turnover", turnover(w, problem), problem.max_turnover)

    if units is not None:
        if units <= 0:
            raise ValueError("units must be positive")
        lot_size = problem.budget / units
        for i, name in enumerate(problem.asset_names):
            closest = round(w[i] / lot_size) * lot_size
            equal(f"lot_grid:{name}", float(w[i]), float(closest))

    if constraints is not None:
        constraints.validate_for(problem)
        active = w > float(support_tol)
        eligible = constraints.eligible_mask(problem.n)
        for index in np.flatnonzero(~eligible):
            upper(f"eligible:{problem.asset_names[index]}", float(w[index]), 0.0)
        for index in constraints.mandatory_assets:
            lower(
                f"mandatory:{problem.asset_names[index]}",
                float(w[index]),
                max(constraints.minimum_active_weight, support_tol),
            )
        if constraints.exact_cardinality is not None:
            equal(
                "exact_cardinality",
                float(np.count_nonzero(active)),
                float(constraints.exact_cardinality),
            )
        if constraints.minimum_active_weight > 0.0:
            for index in np.flatnonzero(active):
                lower(
                    f"minimum_active_weight:{problem.asset_names[index]}",
                    float(w[index]),
                    constraints.minimum_active_weight,
                )
        if constraints.maximum_weights is not None:
            for index, name in enumerate(problem.asset_names):
                upper(
                    f"implementation_upper:{name}",
                    float(w[index]),
                    float(constraints.maximum_weights[index]),
                )
        if constraints.minimum_income is not None:
            lower("minimum_income", float(problem.y @ w), constraints.minimum_income)
        if constraints.factor_lower is not None:
            factor_exposure = problem.factor_loadings.T @ w
            for index, name in enumerate(problem.factor_names):
                lower(
                    f"factor_lower:{name}",
                    float(factor_exposure[index]),
                    float(constraints.factor_lower[index]),
                )
                upper(
                    f"factor_upper:{name}",
                    float(factor_exposure[index]),
                    float(constraints.factor_upper[index]),
                )
        if constraints.stress_scenarios is not None:
            stress_returns = constraints.stress_scenarios @ w
            for index, value in enumerate(stress_returns):
                lower(
                    f"stress_floor:{index}",
                    float(value),
                    float(constraints.stress_floors[index]),
                )
        if constraints.maximum_cvar is not None:
            upper(
                f"cvar_{constraints.cvar_alpha:.3f}",
                empirical_cvar(w, constraints.scenario_returns, constraints.cvar_alpha),
                constraints.maximum_cvar,
            )

    violations = [check.violation for check in checks]
    breaches = sum(violation > tol for violation in violations)
    return ConstraintReport(
        feasible=breaches == 0,
        breaches=breaches,
        max_violation=float(max(violations, default=0.0)),
        checks=checks,
    )


def constraint_report(
    weights: np.ndarray,
    problem: PortfolioProblem,
    tol: float = 1e-7,
) -> ConstraintReport:
    """Backward-compatible alias for the original public function."""
    return validate_weights(weights, problem, tol=tol)


def signed_constraint_slacks(
    weights: np.ndarray,
    problem: PortfolioProblem,
) -> dict[str, float]:
    """Return signed slacks; nonnegative means satisfied."""
    return {check.name: check.slack for check in validate_weights(weights, problem, tol=0.0).checks}


__all__ = [
    "ConstraintCheck",
    "ConstraintReport",
    "constraint_report",
    "signed_constraint_slacks",
    "validate_weights",
]
