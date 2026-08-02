"""Deterministic synthetic and scalable factor-model portfolio data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import PortfolioProblem


def is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and np.allclose(matrix, matrix.T, atol=1e-12)
        and np.min(np.linalg.eigvalsh(matrix)) >= -tol
    )


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Return a symmetric PSD correlation matrix by eigenvalue clipping."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    psd = (vectors * np.maximum(values, floor)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(psd), floor))
    corr = psd / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def generate_synthetic_universe() -> PortfolioProblem:
    """Build the deterministic six-asset instance used in tests and examples."""
    asset_names = [
        "US_Equity",
        "Intl_Equity",
        "Gov_Bonds",
        "Corp_Bonds",
        "Commodities",
        "Cash",
    ]
    group_names = ["Equity", "FixedIncome", "Alternatives", "Cash"]
    asset_group = [0, 0, 1, 1, 2, 3]

    mu = np.array([0.070, 0.080, 0.020, 0.035, 0.050, 0.015])
    sigma = np.array([0.160, 0.180, 0.050, 0.070, 0.200, 0.005])
    y = np.array([0.018, 0.025, 0.030, 0.040, 0.000, 0.015])
    c = np.array([0.0010, 0.0015, 0.0008, 0.0010, 0.0020, 0.0001])
    w0 = np.array([0.30, 0.20, 0.20, 0.15, 0.05, 0.10])
    lower = np.zeros(6)
    upper = np.array([0.50, 0.40, 0.50, 0.40, 0.20, 0.30])
    corr = np.array(
        [
            [1.00, 0.80, -0.20, 0.10, 0.30, 0.00],
            [0.80, 1.00, -0.15, 0.10, 0.35, 0.00],
            [-0.20, -0.15, 1.00, 0.60, -0.10, 0.05],
            [0.10, 0.10, 0.60, 1.00, 0.05, 0.02],
            [0.30, 0.35, -0.10, 0.05, 1.00, 0.00],
            [0.00, 0.00, 0.05, 0.02, 0.00, 1.00],
        ]
    )
    if not is_psd(corr):
        corr = nearest_correlation(corr)
    cov = corr * np.outer(sigma, sigma)

    return PortfolioProblem(
        asset_names=asset_names,
        group_names=group_names,
        asset_group=asset_group,
        mu=mu,
        sigma=sigma,
        corr=corr,
        cov=cov,
        y=y,
        c=c,
        w0=w0,
        lower=lower,
        upper=upper,
        group_lower=np.array([0.30, 0.20, 0.00, 0.02]),
        group_upper=np.array([0.70, 0.60, 0.20, 0.30]),
    )


def generate_factor_universe(
    n_assets: int = 25,
    n_groups: int = 5,
    n_factors: int = 4,
    seed: int = 0,
) -> PortfolioProblem:
    """Generate a reproducible PSD factor-model instance for scaling tests."""
    if n_assets < 2 or not 1 <= n_groups <= n_assets:
        raise ValueError("require n_assets >= 2 and 1 <= n_groups <= n_assets")
    rng = np.random.default_rng(seed)
    n_factors = max(1, min(int(n_factors), n_assets))
    asset_group = (np.arange(n_assets) % n_groups).tolist()
    rng.shuffle(asset_group)

    loadings = rng.normal(0.0, 0.18, size=(n_assets, n_factors))
    idiosyncratic = rng.uniform(0.05, 0.15, size=n_assets)
    raw_cov = loadings @ loadings.T + np.diag(idiosyncratic**2)
    raw_sigma = np.sqrt(np.diag(raw_cov))
    desired_sigma = rng.uniform(0.05, 0.22, size=n_assets)
    scale = desired_sigma / raw_sigma
    scaled_loadings = loadings * scale[:, None]
    idiosyncratic_var = (idiosyncratic * scale) ** 2
    factor_cov = np.eye(n_factors)
    cov = scaled_loadings @ factor_cov @ scaled_loadings.T + np.diag(idiosyncratic_var)
    corr = cov / np.outer(desired_sigma, desired_sigma)
    # BB' + D is positive definite by construction, and positive diagonal
    # scaling preserves PSD.  A full eigen-projection here used to impose an
    # unnecessary O(n^3) cost on every large generated universe.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(desired_sigma, desired_sigma)

    w0 = np.full(n_assets, 1.0 / n_assets)
    upper = np.full(n_assets, max(0.20, 2.5 / n_assets))
    upper = np.minimum(upper, 1.0)
    group_exposure = np.bincount(asset_group, weights=w0, minlength=n_groups)
    group_lower = np.maximum(0.0, group_exposure - 0.10)
    group_upper = np.minimum(1.0, group_exposure + 0.15)

    return PortfolioProblem(
        asset_names=[f"Asset_{i:03d}" for i in range(n_assets)],
        group_names=[f"Group_{g}" for g in range(n_groups)],
        asset_group=asset_group,
        mu=rng.uniform(0.015, 0.105, size=n_assets),
        sigma=desired_sigma,
        corr=corr,
        cov=cov,
        y=rng.uniform(0.0, 0.045, size=n_assets),
        c=rng.uniform(0.0001, 0.0030, size=n_assets),
        w0=w0,
        lower=np.zeros(n_assets),
        upper=upper,
        group_lower=group_lower,
        group_upper=group_upper,
        factor_names=[f"Factor_{index}" for index in range(n_factors)],
        factor_loadings=scaled_loadings,
        factor_cov=factor_cov,
        idiosyncratic_var=idiosyncratic_var,
    )


def generate_return_scenarios(
    problem: PortfolioProblem,
    n_scenarios: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Draw reproducible one-period multivariate-normal return scenarios.

    Synthetic scenarios are deliberately separate from the canonical problem
    so CVaR remains an optional, data-dependent guardrail rather than a hidden
    default assumption.
    """
    if int(n_scenarios) <= 1:
        raise ValueError("n_scenarios must exceed one")
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(
        mean=problem.mu,
        cov=problem.cov,
        size=int(n_scenarios),
        method="cholesky",
    )


def generate_backtest_returns(
    problem: PortfolioProblem,
    periods: int = 120,
    periods_per_year: int = 12,
    seed: int = 1,
) -> np.ndarray:
    """Generate an independent synthetic out-of-sample path for demonstrations."""
    if int(periods) <= 1 or int(periods_per_year) <= 0:
        raise ValueError("periods must exceed one and periods_per_year must be positive")
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(
        mean=problem.mu / int(periods_per_year),
        cov=problem.cov / int(periods_per_year),
        size=int(periods),
        method="cholesky",
    )


def save_problem(problem: PortfolioProblem, destination: str | Path) -> Path:
    """Save a problem as JSON; a directory target gets the standard filename."""
    path = Path(destination)
    if path.suffix.lower() != ".json":
        path = path / "synthetic_universe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(problem.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_problem(path: str | Path) -> PortfolioProblem:
    return PortfolioProblem.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "PortfolioProblem",
    "generate_factor_universe",
    "generate_backtest_returns",
    "generate_return_scenarios",
    "generate_synthetic_universe",
    "is_psd",
    "load_problem",
    "nearest_correlation",
    "save_problem",
]
