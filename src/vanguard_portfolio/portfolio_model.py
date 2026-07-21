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

    # PortfolioProblem has already validated symmetry and PSD.  Repeating a
    # dense eigendecomposition here would add O(n^3) work for every backend.
    # Build the solver matrix directly without allocating a dense 2n-by-2n
    # block containing mostly zeros.
    risk_block = sparse.csc_matrix(2.0 * preferences.lambda_risk * problem.cov)
    zero_block = sparse.csc_matrix((n, n), dtype=float)
    P = sparse.block_diag((risk_block, zero_block), format="csc")
    q = np.concatenate(
        [
            -preferences.lambda_return * problem.mu
            - preferences.lambda_income * problem.y,
            preferences.lambda_cost * problem.c,
        ]
    )

    blocks: list[sparse.spmatrix] = []
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    names: list[str] = []

    def add_block(
        matrix: sparse.spmatrix,
        low: np.ndarray | float,
        high: np.ndarray | float,
        row_names: list[str],
    ) -> None:
        rows = matrix.shape[0]
        blocks.append(matrix.tocsr())
        lows.append(np.broadcast_to(np.asarray(low, dtype=float), (rows,)).copy())
        highs.append(np.broadcast_to(np.asarray(high, dtype=float), (rows,)).copy())
        names.extend(row_names)

    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, n), dtype=float)
    add_block(
        sparse.hstack((identity, zeros), format="csr"),
        problem.lower,
        problem.upper,
        [f"asset:{asset}" for asset in problem.asset_names],
    )
    add_block(
        sparse.hstack((zeros, identity), format="csr"),
        0.0,
        np.inf,
        [f"turnover_aux:{asset}" for asset in problem.asset_names],
    )

    budget = sparse.csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, dim)
    )
    add_block(budget, problem.budget, problem.budget, ["budget"])

    group_matrix = sparse.hstack(
        (sparse.csr_matrix(problem.A), sparse.csr_matrix((problem.num_groups, n))),
        format="csr",
    )
    add_block(
        group_matrix,
        problem.group_lower,
        problem.group_upper,
        [f"group:{group}" for group in problem.group_names],
    )

    add_block(
        sparse.hstack((-identity, identity), format="csr"),
        -problem.w0,
        np.inf,
        [f"abs_plus:{asset}" for asset in problem.asset_names],
    )
    add_block(
        sparse.hstack((identity, identity), format="csr"),
        problem.w0,
        np.inf,
        [f"abs_minus:{asset}" for asset in problem.asset_names],
    )

    if problem.target_return is not None:
        add_block(
            sparse.csr_matrix(np.concatenate([problem.mu, np.zeros(n)])[None, :]),
            problem.target_return,
            np.inf,
            ["target_return"],
        )
    if problem.max_turnover is not None:
        add_block(
            sparse.csr_matrix(np.concatenate([np.zeros(n), np.ones(n)])[None, :]),
            -np.inf,
            problem.max_turnover,
            ["max_turnover"],
        )

    return QPData(
        P=sparse.triu(P, format="csc"),
        q=q,
        A=sparse.vstack(blocks, format="csc"),
        lower=np.concatenate(lows),
        upper=np.concatenate(highs),
        row_names=tuple(names),
    )


def swap_objective_delta(
    weights: np.ndarray,
    cov_times_weights: np.ndarray,
    donor: int,
    receiver: int,
    problem: PortfolioProblem,
    preferences: Preferences,
    units: int,
) -> float:
    """Exact objective change for moving one lot from donor to receiver.

    The calculation is O(1) once ``cov_times_weights = cov @ weights`` is
    available.  This avoids an O(n^2) quadratic-form evaluation for every
    local-search or annealing proposal.
    """
    if donor == receiver:
        return 0.0
    w = np.asarray(weights, dtype=float)
    cov_w = np.asarray(cov_times_weights, dtype=float)
    lot_size = problem.budget / int(units)

    variance_delta = (
        2.0 * lot_size * (cov_w[receiver] - cov_w[donor])
        + lot_size**2
        * (
            problem.cov[receiver, receiver]
            + problem.cov[donor, donor]
            - 2.0 * problem.cov[donor, receiver]
        )
    )
    linear_delta = (
        -preferences.lambda_return
        * lot_size
        * (problem.mu[receiver] - problem.mu[donor])
        - preferences.lambda_income
        * lot_size
        * (problem.y[receiver] - problem.y[donor])
    )

    before_cost = (
        problem.c[donor] * abs(w[donor] - problem.w0[donor])
        + problem.c[receiver] * abs(w[receiver] - problem.w0[receiver])
    )
    after_cost = (
        problem.c[donor]
        * abs(w[donor] - lot_size - problem.w0[donor])
        + problem.c[receiver]
        * abs(w[receiver] + lot_size - problem.w0[receiver])
    )
    return float(
        preferences.lambda_risk * variance_delta
        + linear_delta
        + preferences.lambda_cost * (after_cost - before_cost)
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
    "swap_objective_delta",
    "variance",
    "volatility",
]
