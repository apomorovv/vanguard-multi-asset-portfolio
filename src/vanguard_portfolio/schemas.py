"""Canonical data structures shared by every portfolio solver.

There must be exactly one problem schema and one scalar objective definition in
the repository.  Classical, QUBO, and quantum modules import these structures
instead of maintaining solver-specific copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


class PortfolioError(RuntimeError):
    """Base class for domain-specific portfolio errors."""


class InfeasibleProblemError(PortfolioError):
    """Raised when the requested hard constraints have no feasible allocation."""


class SolverUnavailableError(PortfolioError):
    """Raised when an optional solver package or license is unavailable."""


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


@dataclass
class PortfolioProblem:
    """All data and hard constraints for one portfolio instance.

    Parameters use annual decimal units: ``0.07`` means seven percent.  Group
    membership is encoded by ``asset_group[i]`` and exposed as the matrix
    :attr:`A`.  ``target_return`` and ``max_turnover`` are optional hard
    guardrails; leaving them as ``None`` recovers the original baseline model.
    """

    asset_names: list[str]
    group_names: list[str]
    asset_group: list[int]
    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    cov: np.ndarray
    y: np.ndarray
    c: np.ndarray
    w0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    group_lower: np.ndarray
    group_upper: np.ndarray
    budget: float = 1.0
    target_return: float | None = None
    max_turnover: float | None = None

    def __post_init__(self) -> None:
        self.asset_names = [str(name) for name in self.asset_names]
        self.group_names = [str(name) for name in self.group_names]
        self.asset_group = [int(group) for group in self.asset_group]
        n = len(self.asset_names)
        g = len(self.group_names)

        if n == 0:
            raise ValueError("at least one asset is required")
        if len(set(self.asset_names)) != n:
            raise ValueError("asset_names must be unique")
        if len(set(self.group_names)) != g:
            raise ValueError("group_names must be unique")
        if len(self.asset_group) != n:
            raise ValueError("asset_group must contain one group index per asset")
        if g == 0 or any(group < 0 or group >= g for group in self.asset_group):
            raise ValueError("every asset_group index must identify an existing group")

        for name in ("mu", "sigma", "y", "c", "w0", "lower", "upper"):
            setattr(self, name, _vector(getattr(self, name), n, name))
        self.group_lower = _vector(self.group_lower, g, "group_lower")
        self.group_upper = _vector(self.group_upper, g, "group_upper")

        self.corr = np.asarray(self.corr, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        for name, matrix in (("corr", self.corr), ("cov", self.cov)):
            if matrix.shape != (n, n):
                raise ValueError(f"{name} must have shape ({n}, {n}); got {matrix.shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} contains non-finite values")
            if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
                raise ValueError(f"{name} must be symmetric")
        self.corr = 0.5 * (self.corr + self.corr.T)
        self.cov = 0.5 * (self.cov + self.cov.T)

        if np.any(self.sigma < 0):
            raise ValueError("sigma must be nonnegative")
        if np.any(self.c < 0):
            raise ValueError("transaction-cost coefficients c must be nonnegative")
        if not np.allclose(np.diag(self.corr), 1.0, atol=1e-8):
            raise ValueError("corr must have a unit diagonal")
        if np.min(np.linalg.eigvalsh(self.cov)) < -1e-9:
            raise ValueError("cov must be positive semidefinite")
        expected_cov = self.corr * np.outer(self.sigma, self.sigma)
        if not np.allclose(self.cov, expected_cov, atol=1e-9, rtol=1e-7):
            raise ValueError("cov must equal corr * outer(sigma, sigma)")

        self.budget = float(self.budget)
        if not np.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("budget must be a finite positive number")
        if np.any(self.lower < -1e-12):
            raise ValueError("the baseline is long-only, so lower bounds must be nonnegative")
        if np.any(self.lower > self.upper + 1e-12):
            raise ValueError("lower bounds must not exceed upper bounds")
        if self.lower.sum() > self.budget + 1e-12 or self.upper.sum() < self.budget - 1e-12:
            raise ValueError("asset bounds cannot satisfy the budget")
        if np.any(self.group_lower < -1e-12):
            raise ValueError("group lower bounds must be nonnegative")
        if np.any(self.group_lower > self.group_upper + 1e-12):
            raise ValueError("group lower bounds must not exceed group upper bounds")

        if self.target_return is not None:
            self.target_return = float(self.target_return)
            if not np.isfinite(self.target_return):
                raise ValueError("target_return must be finite")
        if self.max_turnover is not None:
            self.max_turnover = float(self.max_turnover)
            if not np.isfinite(self.max_turnover) or self.max_turnover < 0:
                raise ValueError("max_turnover must be finite and nonnegative")

    @property
    def n(self) -> int:
        return len(self.asset_names)

    @property
    def num_groups(self) -> int:
        return len(self.group_names)

    @property
    def A(self) -> np.ndarray:
        matrix = np.zeros((self.num_groups, self.n), dtype=float)
        matrix[np.asarray(self.asset_group, dtype=int), np.arange(self.n)] = 1.0
        return matrix

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "asset_names": list(self.asset_names),
            "group_names": list(self.group_names),
            "asset_group": list(self.asset_group),
            "budget": self.budget,
            "target_return": self.target_return,
            "max_turnover": self.max_turnover,
        }
        for name in (
            "mu",
            "sigma",
            "corr",
            "cov",
            "y",
            "c",
            "w0",
            "lower",
            "upper",
            "group_lower",
            "group_upper",
        ):
            result[name] = getattr(self, name).tolist()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortfolioProblem":
        return cls(**dict(data))


@dataclass(frozen=True)
class Preferences:
    """Nonnegative coefficients in the canonical minimization objective."""

    lambda_return: float = 1.0
    lambda_risk: float = 5.0
    lambda_income: float = 0.0
    lambda_cost: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lambda_return", "lambda_risk", "lambda_income", "lambda_cost"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_return": self.lambda_return,
            "lambda_risk": self.lambda_risk,
            "lambda_income": self.lambda_income,
            "lambda_cost": self.lambda_cost,
        }


@dataclass
class SolveResult:
    """Normalized output returned by every classical backend."""

    method: str
    model_type: str
    weights: np.ndarray
    objective: float
    runtime: float
    status: str
    success: bool
    optimal: bool
    feasible: bool
    breaches: int
    max_violation: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    units: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "method": self.method,
            "model_type": self.model_type,
            "objective": float(self.objective),
            "runtime_seconds": float(self.runtime),
            "status": self.status,
            "success": bool(self.success),
            "optimal": bool(self.optimal),
            "feasible": bool(self.feasible),
            "breaches": int(self.breaches),
            "max_violation": float(self.max_violation),
            "units": self.units,
            "seed": self.seed,
        }
        record.update({key: float(value) for key, value in self.metrics.items()})
        return record


