"""Canonical objective and algebraic model builders.

The direct evaluator is the source of truth.  Solver-native models are only
alternative representations of these same equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .schemas import PortfolioProblem, Preferences


def expected_return(weights: np.ndarray, problem: PortfolioProblem) -> float:
    return float(problem.mu @ np.asarray(weights, dtype=float))


def variance(weights: np.ndarray, problem: PortfolioProblem) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w @ problem.cov @ w)


def volatility(weights: np.ndarray, problem: PortfolioProblem) -> float:
    return float(np.sqrt(max(variance(weights, problem), 0.0)))


def income_yield(weights: np.ndarray, problem: PortfolioProblem) -> float:
    return float(problem.y @ np.asarray(weights, dtype=float))


def turnover(weights: np.ndarray, problem: PortfolioProblem) -> float:
    return float(np.sum(np.abs(np.asarray(weights, dtype=float) - problem.w0)))


def transaction_cost(weights: np.ndarray, problem: PortfolioProblem) -> float:
    return float(problem.c @ np.abs(np.asarray(weights, dtype=float) - problem.w0))


def objective_breakdown(
    weights: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
) -> dict[str, float]:
    """Return signed terms whose sum is the canonical minimization objective."""
    return {
        "risk_term": preferences.lambda_risk * variance(weights, problem),
        "return_term": -preferences.lambda_return * expected_return(weights, problem),
        "income_term": -preferences.lambda_income * income_yield(weights, problem),
        "cost_term": preferences.lambda_cost * transaction_cost(weights, problem),
    }


def objective_value(
    weights: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
) -> float:
    return float(sum(objective_breakdown(weights, problem, preferences).values()))


@dataclass(frozen=True)
class QPData:
    """OSQP-form continuous model: min 1/2 x'Px + q'x, l <= Ax <= u."""

    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    row_names: tuple[str, ...]


def build_continuous_qp(
    problem: PortfolioProblem,
    preferences: Preferences,
) -> QPData:
    """Build the exact continuous QP in a backend-neutral matrix form.

    The decision is ``x = [w, t]``.  The turnover epigraph uses
    ``t >= w-w0`` and ``t >= w0-w``.  Bounds are represented as rows of
    ``A`` so the same object can feed OSQP, SciPy, and feasibility checks.
    """
    n = problem.n
    dim = 2 * n

    values, vectors = np.linalg.eigh(problem.cov)
    cov_psd = (vectors * np.maximum(values, 0.0)) @ vectors.T
    P = np.zeros((dim, dim), dtype=float)
    P[:n, :n] = 2.0 * preferences.lambda_risk * cov_psd
    q = np.concatenate(
        [
            -preferences.lambda_return * problem.mu
            - preferences.lambda_income * problem.y,
            preferences.lambda_cost * problem.c,
        ]
    )

    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []
    names: list[str] = []

    def add(row: np.ndarray, low: float, high: float, name: str) -> None:
        rows.append(np.asarray(row, dtype=float))
        lows.append(float(low))
        highs.append(float(high))
        names.append(name)

    for i, asset in enumerate(problem.asset_names):
        row = np.zeros(dim)
        row[i] = 1.0
        add(row, problem.lower[i], problem.upper[i], f"asset:{asset}")
    for i, asset in enumerate(problem.asset_names):
        row = np.zeros(dim)
        row[n + i] = 1.0
        add(row, 0.0, np.inf, f"turnover_aux:{asset}")

    budget = np.concatenate([np.ones(n), np.zeros(n)])
    add(budget, problem.budget, problem.budget, "budget")

    for g, group in enumerate(problem.group_names):
        row = np.concatenate([problem.A[g], np.zeros(n)])
        add(row, problem.group_lower[g], problem.group_upper[g], f"group:{group}")

    for i, asset in enumerate(problem.asset_names):
        row = np.zeros(dim)
        row[i] = -1.0
        row[n + i] = 1.0
        add(row, -problem.w0[i], np.inf, f"abs_plus:{asset}")

        row = np.zeros(dim)
        row[i] = 1.0
        row[n + i] = 1.0
        add(row, problem.w0[i], np.inf, f"abs_minus:{asset}")

    if problem.target_return is not None:
        add(
            np.concatenate([problem.mu, np.zeros(n)]),
            problem.target_return,
            np.inf,
            "target_return",
        )
    if problem.max_turnover is not None:
        add(
            np.concatenate([np.zeros(n), np.ones(n)]),
            -np.inf,
            problem.max_turnover,
            "max_turnover",
        )

    return QPData(
        P=sparse.csc_matrix(np.triu(P)),
        q=q,
        A=sparse.csc_matrix(np.vstack(rows)),
        lower=np.asarray(lows),
        upper=np.asarray(highs),
        row_names=tuple(names),
    )


def lot_bounds(problem: PortfolioProblem, units: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert weight bounds to exact integer-lot bounds.

    A lower bound rounds upward and an upper bound rounds downward.  Reversing
    those directions would admit allocations that violate the original model.
    """
    if not isinstance(units, (int, np.integer)) or units <= 0:
        raise ValueError("units must be a positive integer")
    lot_size = problem.budget / int(units)
    low = np.ceil(problem.lower / lot_size - 1e-12).astype(int)
    high = np.floor(problem.upper / lot_size + 1e-12).astype(int)
    return np.clip(low, 0, units), np.clip(high, 0, units)


def lots_to_weights(lots: np.ndarray, problem: PortfolioProblem, units: int) -> np.ndarray:
    lots = np.asarray(lots, dtype=int).reshape(-1)
    if lots.shape != (problem.n,):
        raise ValueError(f"lots must have shape ({problem.n},)")
    return (problem.budget / int(units)) * lots.astype(float)


def discrete_constraints_hold(
    lots: np.ndarray,
    problem: PortfolioProblem,
    units: int,
    tol: float = 1e-10,
) -> bool:
    lots = np.asarray(lots, dtype=int)
    low, high = lot_bounds(problem, units)
    if lots.shape != (problem.n,) or lots.sum() != units:
        return False
    if np.any(lots < low) or np.any(lots > high):
        return False
    weights = lots_to_weights(lots, problem, units)
    group = problem.A @ weights
    if np.any(group < problem.group_lower - tol) or np.any(group > problem.group_upper + tol):
        return False
    if problem.target_return is not None and problem.mu @ weights < problem.target_return - tol:
        return False
    if problem.max_turnover is not None and turnover(weights, problem) > problem.max_turnover + tol:
        return False
    return True


__all__ = [
    "QPData",
    "build_continuous_qp",
    "discrete_constraints_hold",
    "expected_return",
    "income_yield",
    "lot_bounds",
    "lots_to_weights",
    "objective_breakdown",
    "objective_value",
    "transaction_cost",
    "turnover",
    "variance",
    "volatility",
]


