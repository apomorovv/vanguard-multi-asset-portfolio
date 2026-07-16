"""Synthetic multi-asset universe.

This module builds a small, fully synthetic set of asset-class inputs that
matches the notation used in `docs/mathematical_model.md`:

* `mu`   annual expected return            (Section 3)
* `sigma` annual volatility                 (Section 3)
* `corr` correlation matrix                 (Section 3)
* `cov`  covariance matrix `Sigma`        (Section 3)
* `y`    annual income yield               (Section 3)
* `c`    transaction-cost coefficient      (Section 3)
* `w0`   current allocation                (Section 3)
* `l,u`  per-asset allocation bounds       (Section 3)
* group membership `A` and group limits `L_g, U_g` (Section 3)

No real or confidential data is used. All numbers are illustrative and
deterministic so that every solver is compared on an identical problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class PortfolioProblem:
    """Container for every parameter of the portfolio-construction model."""

    asset_names: list[str]
    group_names: list[str]
    # index of the group each asset belongs to (length n)
    asset_group: list[int]

    mu: np.ndarray          # expected return, shape (n,)
    sigma: np.ndarray       # volatility, shape (n,)
    corr: np.ndarray        # correlation matrix, shape (n, n)
    cov: np.ndarray         # covariance matrix Sigma, shape (n, n)
    y: np.ndarray           # income yield, shape (n,)
    c: np.ndarray           # transaction cost, shape (n,)
    w0: np.ndarray          # current allocation, shape (n,)
    lower: np.ndarray       # per-asset lower bound l_i, shape (n,)
    upper: np.ndarray       # per-asset upper bound u_i, shape (n,)

    group_lower: np.ndarray  # group lower limit L_g, shape (G,)
    group_upper: np.ndarray  # group upper limit U_g, shape (G,)

    @property
    def n(self) -> int:
        """Number of assets."""
        return len(self.asset_names)

    @property
    def num_groups(self) -> int:
        """Number of asset groups."""
        return len(self.group_names)

    @property
    def A(self) -> np.ndarray:
        """Group-membership matrix `a_{gi}` of shape (G, n)."""
        mat = np.zeros((self.num_groups, self.n))
        for i, g in enumerate(self.asset_group):
            mat[g, i] = 1.0
        return mat


def _is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True when `matrix` is symmetric positive semidefinite."""
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    return bool(np.min(eigenvalues) >= -tol)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues to obtain the nearest PSD matrix."""
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def generate_synthetic_universe() -> PortfolioProblem:
    """Build the deterministic six-asset synthetic universe.

    The asset classes mirror the example in the mathematical model: US
    equity, international equity, government bonds, corporate bonds,
    commodities and cash, grouped into equity, fixed income, alternatives
    and cash.
    """

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

    lower = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    upper = np.array([0.50, 0.40, 0.50, 0.40, 0.20, 0.30])

    # Correlation matrix (symmetric, unit diagonal).
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

    cov = corr * np.outer(sigma, sigma)  # Sigma_ij = rho_ij * sigma_i * sigma_j
    if not _is_psd(cov):
        cov = _nearest_psd(cov)

    # Group limits L_g, U_g for equity, fixed income, alternatives, cash.
    group_lower = np.array([0.30, 0.20, 0.00, 0.02])
    group_upper = np.array([0.70, 0.60, 0.20, 0.30])

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
        group_lower=group_lower,
        group_upper=group_upper,
    )


def save_problem(problem: PortfolioProblem, directory: str | Path) -> Path:
    """Serialise a :class:PortfolioProblem to a JSON file.

    Returns the path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "synthetic_universe.json"

    payload = asdict(problem)
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_problem(path: str | Path) -> PortfolioProblem:
    """Load a :class:PortfolioProblem previously written by :func:save_problem."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    array_keys = {
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
    }
    for key in array_keys:
        data[key] = np.array(data[key])
    return PortfolioProblem(**data)