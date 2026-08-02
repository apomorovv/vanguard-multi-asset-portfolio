"""Common financial and solver-comparison metrics."""

from __future__ import annotations

import numpy as np

from .portfolio_model import (
    empirical_cvar as _empirical_cvar,
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
        "support_size": float(np.count_nonzero(np.abs(w) > 1e-8)),
        "return_to_risk": float(_expected_return(w, problem) / _volatility(w, problem))
        if _volatility(w, problem) > 0
        else np.inf,
    }


def wealth_path(weights: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
    scenarios = np.asarray(realized_returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if scenarios.ndim != 2 or scenarios.shape[1] != w.size:
        raise ValueError("realized_returns must have shape (periods, n_assets)")
    portfolio_returns = scenarios @ w
    return np.concatenate([[1.0], np.cumprod(1.0 + portfolio_returns)])


def maximum_drawdown(wealth: np.ndarray) -> float:
    values = np.asarray(wealth, dtype=float).reshape(-1)
    peaks = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / np.maximum(peaks, 1e-15)
    return float(np.max(drawdowns, initial=0.0))


def backtest_metrics(
    weights: np.ndarray,
    realized_returns: np.ndarray,
    *,
    periods_per_year: int = 12,
    cvar_alpha: float = 0.95,
) -> dict[str, float]:
    scenarios = np.asarray(realized_returns, dtype=float)
    portfolio_returns = scenarios @ np.asarray(weights, dtype=float)
    wealth = wealth_path(weights, scenarios)
    annual_return = float(np.mean(portfolio_returns) * periods_per_year)
    annual_volatility = float(np.std(portfolio_returns, ddof=1) * np.sqrt(periods_per_year))
    return {
        "realized_annual_return": annual_return,
        "realized_annual_volatility": annual_volatility,
        "realized_return_to_risk": annual_return / annual_volatility
        if annual_volatility > 0
        else np.inf,
        "maximum_drawdown": maximum_drawdown(wealth),
        "empirical_cvar": _empirical_cvar(weights, scenarios, cvar_alpha),
        "terminal_wealth": float(wealth[-1]),
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
    "backtest_metrics",
    "constraint_report",
    "expected_return",
    "income",
    "objective_gap",
    "portfolio_metrics",
    "transaction_cost",
    "turnover",
    "variance",
    "volatility",
    "maximum_drawdown",
    "wealth_path",
]
