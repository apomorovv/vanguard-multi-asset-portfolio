"""Canonical data structures shared by every portfolio solver.

There must be exactly one problem schema and one scalar objective definition in
the repository.  Classical, QUBO, and quantum modules import these structures
instead of maintaining solver-specific copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _smallest_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    """Return the smallest eigenvalue without a full spectrum for large inputs.

    A complete dense eigendecomposition is reliable for small validation
    instances but becomes an avoidable cubic setup cost for large universes.
    ARPACK only requests the smallest algebraic eigenvalue for matrices above
    the cutoff.  A deterministic dense fallback preserves strict validation if
    the iterative method does not converge.
    """
    n = matrix.shape[0]
    if n <= 384:
        return float(np.linalg.eigvalsh(matrix)[0])

    try:
        from scipy.sparse.linalg import (
            ArpackError,
            ArpackNoConvergence,
            LinearOperator,
            eigsh,
        )

        operator = LinearOperator(
            shape=matrix.shape,
            matvec=lambda vector: matrix @ vector,
            rmatvec=lambda vector: matrix @ vector,
            dtype=matrix.dtype,
        )
        value = eigsh(
            operator,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1e-7,
            maxiter=max(2_000, 5 * n),
            v0=np.random.default_rng(0).normal(size=n),
        )[0]
        if not np.isfinite(value):
            raise RuntimeError("iterative PSD validation returned a non-finite eigenvalue")
        return float(value)
    except (
        ArpackError,
        ArpackNoConvergence,
        np.linalg.LinAlgError,
        ValueError,
        RuntimeError,
    ):
        from scipy.linalg import eigh

        return float(
            eigh(
                matrix,
                subset_by_index=[0, 0],
                check_finite=False,
                eigvals_only=True,
            )[0]
        )


class PortfolioError(RuntimeError):
    """Base class for domain-specific portfolio errors."""


class InfeasibleProblemError(PortfolioError):
    """Raised when the requested hard constraints have no feasible allocation."""


class SolverUnavailableError(PortfolioError):
    """Raised when an optional solver package or license is unavailable."""


class SolverSkippedError(PortfolioError):
    """Raised when an explicit safety guard intentionally skips a solver."""


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
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
    corr: np.ndarray | None
    cov: np.ndarray | None
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
    factor_names: list[str] | None = None
    factor_loadings: np.ndarray | None = None
    factor_cov: np.ndarray | None = None
    idiosyncratic_var: np.ndarray | None = None
    _A: np.ndarray = field(init=False, repr=False)

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

        if np.any(self.sigma < 0):
            raise ValueError("sigma must be nonnegative")
        if np.any(self.c < 0):
            raise ValueError("transaction-cost coefficients c must be nonnegative")

        complete_factor_hint = all(
            value is not None
            for value in (
                self.factor_loadings,
                self.factor_cov,
                self.idiosyncratic_var,
            )
        )
        factor_parts = (
            self.factor_loadings,
            self.factor_cov,
            self.idiosyncratic_var,
        )
        if any(value is not None for value in factor_parts):
            if not all(value is not None for value in factor_parts):
                raise ValueError(
                    "factor_loadings, factor_cov, and idiosyncratic_var must be supplied together"
                )
            loadings = np.asarray(self.factor_loadings, dtype=float)
            if loadings.ndim != 2 or loadings.shape[0] != n or loadings.shape[1] == 0:
                raise ValueError("factor_loadings must have shape (n_assets, n_factors)")
            k = int(loadings.shape[1])
            self.factor_loadings = _matrix(loadings, (n, k), "factor_loadings")
            self.factor_cov = _matrix(self.factor_cov, (k, k), "factor_cov")
            self.factor_cov = 0.5 * (self.factor_cov + self.factor_cov.T)
            self.idiosyncratic_var = _vector(
                self.idiosyncratic_var, n, "idiosyncratic_var"
            )
            if np.any(self.idiosyncratic_var < -1e-12):
                raise ValueError("idiosyncratic_var must be nonnegative")
            if _smallest_symmetric_eigenvalue(self.factor_cov) < -1e-9:
                raise ValueError("factor_cov must be positive semidefinite")
            if self.factor_names is None:
                self.factor_names = [f"Factor_{index}" for index in range(k)]
            else:
                self.factor_names = [str(name) for name in self.factor_names]
                if len(self.factor_names) != k or len(set(self.factor_names)) != k:
                    raise ValueError("factor_names must contain one unique name per factor")
            factor_variance = np.einsum(
                "ij,jk,ik->i",
                self.factor_loadings,
                self.factor_cov,
                self.factor_loadings,
                optimize=True,
            )
            factor_variance += self.idiosyncratic_var
            if not np.allclose(
                factor_variance,
                np.square(self.sigma),
                atol=1e-10,
                rtol=1e-7,
            ):
                raise ValueError("factor model diagonal must equal sigma squared")
        elif self.factor_names not in (None, []):
            raise ValueError("factor_names requires a complete factor model")
        else:
            self.factor_names = None

        if (self.corr is None) != (self.cov is None):
            raise ValueError("corr and cov must either both be supplied or both be omitted")
        if self.corr is None:
            if not complete_factor_hint:
                raise ValueError("corr and cov may be omitted only for a complete factor model")
        else:
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
            if not np.allclose(np.diag(self.corr), 1.0, atol=1e-8):
                raise ValueError("corr must have a unit diagonal")
            if not complete_factor_hint and _smallest_symmetric_eigenvalue(self.cov) < -1e-9:
                raise ValueError("cov must be positive semidefinite")
            expected_cov = self.corr * np.outer(self.sigma, self.sigma)
            if not np.allclose(self.cov, expected_cov, atol=1e-9, rtol=1e-7):
                raise ValueError("cov must equal corr * outer(sigma, sigma)")
            if complete_factor_hint:
                reconstructed = (
                    self.factor_loadings @ self.factor_cov @ self.factor_loadings.T
                    + np.diag(self.idiosyncratic_var)
                )
                if not np.allclose(reconstructed, self.cov, atol=1e-9, rtol=1e-7):
                    raise ValueError("factor model must reconstruct cov")

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

        self._A = np.zeros((g, n), dtype=float)
        self._A[np.asarray(self.asset_group, dtype=int), np.arange(n)] = 1.0

    @property
    def n(self) -> int:
        return len(self.asset_names)

    @property
    def num_groups(self) -> int:
        return len(self.group_names)

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def has_factor_model(self) -> bool:
        return self.factor_loadings is not None

    @property
    def has_dense_covariance(self) -> bool:
        return self.cov is not None

    @property
    def num_factors(self) -> int:
        return 0 if self.factor_loadings is None else int(self.factor_loadings.shape[1])

    def covariance_matvec(self, weights: np.ndarray) -> np.ndarray:
        """Return ``Sigma @ weights`` without requiring a dense covariance matrix."""
        vector = _vector(weights, self.n, "weights")
        if self.has_factor_model:
            exposure = self.factor_loadings.T @ vector
            return (
                self.factor_loadings @ (self.factor_cov @ exposure)
                + self.idiosyncratic_var * vector
            )
        return self.cov @ vector

    def covariance_submatrix(
        self,
        rows: np.ndarray | list[int] | tuple[int, ...],
        columns: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Return selected covariance entries using factor algebra when needed."""
        row_index = np.asarray(rows, dtype=int).reshape(-1)
        column_index = row_index if columns is None else np.asarray(columns, dtype=int).reshape(-1)
        if np.any(row_index < 0) or np.any(row_index >= self.n):
            raise IndexError("covariance row index is out of range")
        if np.any(column_index < 0) or np.any(column_index >= self.n):
            raise IndexError("covariance column index is out of range")
        if self.has_dense_covariance:
            return self.cov[np.ix_(row_index, column_index)].copy()
        block = (
            self.factor_loadings[row_index]
            @ self.factor_cov
            @ self.factor_loadings[column_index].T
        )
        matches = row_index[:, None] == column_index[None, :]
        if np.any(matches):
            block = block + matches * self.idiosyncratic_var[row_index, None]
        return np.asarray(block, dtype=float)

    def correlation_submatrix(
        self,
        rows: np.ndarray | list[int] | tuple[int, ...],
        columns: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Return selected correlations without materializing the full matrix."""
        row_index = np.asarray(rows, dtype=int).reshape(-1)
        column_index = row_index if columns is None else np.asarray(columns, dtype=int).reshape(-1)
        covariance = self.covariance_submatrix(row_index, column_index)
        scale = np.outer(self.sigma[row_index], self.sigma[column_index])
        correlation = covariance / np.maximum(scale, 1e-15)
        return np.clip(correlation, -1.0, 1.0)

    def dense_covariance_bytes(self) -> int:
        """Bytes required for one dense float64 covariance matrix."""
        return int(self.n) * int(self.n) * np.dtype(float).itemsize

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
            "y",
            "c",
            "w0",
            "lower",
            "upper",
            "group_lower",
            "group_upper",
        ):
            result[name] = getattr(self, name).tolist()
        result["corr"] = None if self.corr is None else self.corr.tolist()
        result["cov"] = None if self.cov is None else self.cov.tolist()
        if self.has_factor_model:
            result.update(
                {
                    "factor_names": list(self.factor_names or []),
                    "factor_loadings": self.factor_loadings.tolist(),
                    "factor_cov": self.factor_cov.tolist(),
                    "idiosyncratic_var": self.idiosyncratic_var.tolist(),
                }
            )
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortfolioProblem:
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


@dataclass(frozen=True)
class PortfolioConstraints:
    """Optional implementation-aware guardrails shared by hybrid solvers.

    The base :class:`PortfolioProblem` always owns budget, asset/group bounds,
    return, and turnover.  This object adds sparse-support and scenario rules
    without changing the legacy continuous and equal-lot baselines.
    """

    exact_cardinality: int | None = None
    minimum_active_weight: float = 0.0
    eligible_assets: tuple[int, ...] | None = None
    mandatory_assets: tuple[int, ...] = ()
    minimum_income: float | None = None
    maximum_weights: np.ndarray | None = None
    factor_lower: np.ndarray | None = None
    factor_upper: np.ndarray | None = None
    stress_scenarios: np.ndarray | None = None
    stress_floors: np.ndarray | None = None
    scenario_returns: np.ndarray | None = None
    maximum_cvar: float | None = None
    cvar_alpha: float = 0.95

    def __post_init__(self) -> None:
        if self.exact_cardinality is not None and int(self.exact_cardinality) <= 0:
            raise ValueError("exact_cardinality must be positive")
        if self.exact_cardinality is not None:
            object.__setattr__(self, "exact_cardinality", int(self.exact_cardinality))
        minimum = float(self.minimum_active_weight)
        if not np.isfinite(minimum) or minimum < 0:
            raise ValueError("minimum_active_weight must be finite and nonnegative")
        object.__setattr__(self, "minimum_active_weight", minimum)
        if self.minimum_income is not None:
            value = float(self.minimum_income)
            if not np.isfinite(value):
                raise ValueError("minimum_income must be finite")
            object.__setattr__(self, "minimum_income", value)
        alpha = float(self.cvar_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("cvar_alpha must be strictly between zero and one")
        object.__setattr__(self, "cvar_alpha", alpha)
        if self.maximum_cvar is not None:
            value = float(self.maximum_cvar)
            if not np.isfinite(value):
                raise ValueError("maximum_cvar must be finite")
            object.__setattr__(self, "maximum_cvar", value)
        for name in ("eligible_assets", "mandatory_assets"):
            value = getattr(self, name)
            if value is None:
                continue
            indices = tuple(sorted({int(index) for index in value}))
            if any(index < 0 for index in indices):
                raise ValueError(f"{name} cannot contain negative indices")
            object.__setattr__(self, name, indices)
        for name in (
            "maximum_weights",
            "factor_lower",
            "factor_upper",
            "stress_scenarios",
            "stress_floors",
            "scenario_returns",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains non-finite values")
                object.__setattr__(self, name, array.copy())

    def validate_for(self, problem: PortfolioProblem) -> PortfolioConstraints:
        n = problem.n
        eligible = set(range(n)) if self.eligible_assets is None else set(self.eligible_assets)
        mandatory = set(self.mandatory_assets)
        if any(index >= n for index in eligible | mandatory):
            raise ValueError("eligible/mandatory asset index is out of range")
        if not mandatory.issubset(eligible):
            raise ValueError("mandatory assets must also be eligible")
        if self.exact_cardinality is not None:
            if self.exact_cardinality > len(eligible):
                raise ValueError("exact_cardinality exceeds the eligible universe")
            if self.exact_cardinality < len(mandatory):
                raise ValueError("exact_cardinality is smaller than the mandatory set")
            if self.minimum_active_weight * self.exact_cardinality > problem.budget + 1e-12:
                raise ValueError("minimum active weights exceed the budget")
        if self.maximum_weights is not None:
            if self.maximum_weights.shape != (n,):
                raise ValueError(f"maximum_weights must have shape ({n},)")
            if np.any(self.maximum_weights < -1e-12):
                raise ValueError("maximum_weights must be nonnegative")
        factor_arrays = (self.factor_lower, self.factor_upper)
        if any(value is not None for value in factor_arrays):
            if not problem.has_factor_model:
                raise ValueError("factor bounds require factor data on PortfolioProblem")
            if not all(value is not None for value in factor_arrays):
                raise ValueError("factor_lower and factor_upper must be supplied together")
            expected = (problem.num_factors,)
            if self.factor_lower.shape != expected or self.factor_upper.shape != expected:
                raise ValueError(f"factor bounds must have shape {expected}")
            if np.any(self.factor_lower > self.factor_upper + 1e-12):
                raise ValueError("factor_lower must not exceed factor_upper")
        stress_arrays = (self.stress_scenarios, self.stress_floors)
        if any(value is not None for value in stress_arrays):
            if not all(value is not None for value in stress_arrays):
                raise ValueError("stress_scenarios and stress_floors must be supplied together")
            if self.stress_scenarios.ndim != 2 or self.stress_scenarios.shape[1] != n:
                raise ValueError("stress_scenarios must have shape (n_stresses, n_assets)")
            if self.stress_floors.shape != (self.stress_scenarios.shape[0],):
                raise ValueError("stress_floors must contain one floor per stress scenario")
        if self.maximum_cvar is not None:
            if self.scenario_returns is None:
                raise ValueError("maximum_cvar requires scenario_returns")
            if self.scenario_returns.ndim != 2 or self.scenario_returns.shape[1] != n:
                raise ValueError("scenario_returns must have shape (n_scenarios, n_assets)")
            if self.scenario_returns.shape[0] == 0:
                raise ValueError("maximum_cvar requires at least one return scenario")
        elif self.scenario_returns is not None and (
            self.scenario_returns.ndim != 2 or self.scenario_returns.shape[1] != n
        ):
            raise ValueError("scenario_returns must have shape (n_scenarios, n_assets)")
        return self

    def eligible_mask(self, n_assets: int) -> np.ndarray:
        mask = np.zeros(n_assets, dtype=bool)
        if self.eligible_assets is None:
            mask[:] = True
        else:
            mask[np.asarray(self.eligible_assets, dtype=int)] = True
        return mask

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "exact_cardinality": self.exact_cardinality,
            "minimum_active_weight": self.minimum_active_weight,
            "eligible_assets": None
            if self.eligible_assets is None
            else list(self.eligible_assets),
            "mandatory_assets": list(self.mandatory_assets),
            "minimum_income": self.minimum_income,
            "maximum_cvar": self.maximum_cvar,
            "cvar_alpha": self.cvar_alpha,
        }
        for name in (
            "maximum_weights",
            "factor_lower",
            "factor_upper",
            "stress_scenarios",
            "stress_floors",
            "scenario_returns",
        ):
            value = getattr(self, name)
            result[name] = None if value is None else value.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PortfolioConstraints:
        return cls() if data is None else cls(**dict(data))


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
    run_id: str = ""
    repetition: int = 0
    objective_terms: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    units: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "method": self.method,
            "model_type": self.model_type,
            "repetition": int(self.repetition),
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
        record.update({key: float(value) for key, value in self.objective_terms.items()})
        record.update({key: float(value) for key, value in self.metrics.items()})
        return record
