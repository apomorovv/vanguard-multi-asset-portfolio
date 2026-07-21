"""Common financial and solver-comparison metrics."""

from __future__ import annotations

import numpy as np

from .portfolio_model import (
    expected_return as _expected_return,
    income_yield as _income_yield,
    transaction_cost as _transaction_cost,
    turnover as _turnover,
    variance as _variance,
    volatility as _volatility,
)
from .schemas import PortfolioProblem
from .validation import ConstraintReport, constraint_report


def expected_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.asarray(mu) @ np.asarray(w))


def variance(w: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(w)
    return float(w @ np.asarray(cov) @ w)


def volatility(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(variance(w, cov), 0.0)))


def income(w: np.ndarray, y: np.ndarray) -> float:
    return float(np.asarray(y) @ np.asarray(w))


def turnover(w: np.ndarray, w0: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w) - np.asarray(w0))))


def transaction_cost(w: np.ndarray, w0: np.ndarray, c: np.ndarray) -> float:
    return float(np.asarray(c) @ np.abs(np.asarray(w) - np.asarray(w0)))


def portfolio_metrics(w: np.ndarray, problem: PortfolioProblem) -> dict[str, float]:
    w = np.asarray(w, dtype=float)
    concentration = float(w @ w)
    return {
        "expected_return": _expected_return(w, problem),
        "variance": _variance(w, problem),
        "volatility": _volatility(w, problem),
        "income": _income_yield(w, problem),
        "turnover": _turnover(w, problem),
        "transaction_cost": _transaction_cost(w, problem),
        "concentration_hhi": concentration,
        "effective_holdings": float(1.0 / concentration) if concentration > 0 else np.inf,
    }


def objective_gap(value: float, reference: float) -> tuple[float, float]:
    """Return absolute and scale-safe relative gaps for minimization."""
    absolute = float(value - reference)
    relative = absolute / max(abs(reference), 1e-12)
    return absolute, float(relative)


def allocation_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


__all__ = [
    "ConstraintReport",
    "allocation_l1",
    "constraint_report",
    "expected_return",
    "income",
    "objective_gap",
    "portfolio_metrics",
    "transaction_cost",
    "turnover",
    "variance",
    "volatility",
]
