"""Classical continuous mean-variance portfolio optimizer.

This module implements the baseline mean-variance (Markowitz) optimizer used as
the classical reference/validator for the Multi-Asset Portfolio Construction
challenge. Weights are treated as continuous real numbers.

The objective maximizes the mean-variance utility (optionally net of linear
transaction costs):

    U(w) = mu^T w  -  (gamma / 2) * w^T Sigma w  -  lambda_c * c^T |w - w_prev|

subject to a budget constraint, per-asset bounds, and optional linear sector
exposure limits and a minimum target return.

The shared :class:PortfolioProblem container defined here is also reused by the
discrete optimizer in `classical_discrete.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

try:  # scipy is used for the constrained nonlinear solve
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - surfaced only if scipy missing
    raise ImportError(
        "classical_continuous requires scipy. Install it with pip install scipy."
    ) from exc


# ---------------------------------------------------------------------------
# Shared problem definition
# ---------------------------------------------------------------------------
@dataclass
class PortfolioProblem:
    """Container describing a multi-asset mean-variance problem.

    Parameters
    ----------
    expected_returns:
        Expected return per asset, `mu` with shape `(n,)`.
    covariance:
        Asset return covariance matrix `Sigma` with shape `(n, n)`. Must be
        symmetric positive semi-definite.
    risk_aversion:
        Risk-aversion coefficient `gamma` (>= 0). Higher values penalize
        variance more heavily.
    transaction_cost:
        Optional per-asset linear cost applied to turnover `|w - w_prev|`.
    prev_weights:
        Current/previous holdings `w_prev` used for the turnover/cost term.
        Defaults to all-zeros (i.e. building from cash).
    cost_aversion:
        Multiplier `lambda_c` on the transaction-cost term.
    lower_bounds / upper_bounds:
        Per-asset weight bounds. Scalars are broadcast to every asset. Defaults
        to a long-only `[0, 1]` box.
    budget:
        Sum-of-weights target (fully invested = `1.0`).
    sector_map:
        Optional mapping from asset index to a sector label.
    sector_limits:
        Optional mapping from sector label to a maximum aggregate exposure.
    asset_names:
        Optional human-readable names for reporting.
    """

    expected_returns: np.ndarray
    covariance: np.ndarray
    risk_aversion: float = 1.0
    transaction_cost: Optional[np.ndarray] = None
    prev_weights: Optional[np.ndarray] = None
    cost_aversion: float = 1.0
    lower_bounds: np.ndarray = 0.0
    upper_bounds: np.ndarray = 1.0
    budget: float = 1.0
    sector_map: Optional[Sequence] = None
    sector_limits: Optional[Mapping[object, float]] = None
    asset_names: Optional[List[str]] = None

    # Normalized/validated arrays are populated in __post_init__.
    def __post_init__(self) -> None:
        self.expected_returns = np.asarray(self.expected_returns, dtype=float).ravel()
        self.covariance = np.asarray(self.covariance, dtype=float)
        n = self.expected_returns.size

        if self.covariance.shape != (n, n):
            raise ValueError(
                f"covariance must be ({n}, {n}); got {self.covariance.shape}"
            )
        # Symmetrize to guard against tiny numerical asymmetries.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        if self.prev_weights is None:
            self.prev_weights = np.zeros(n)
        else:
            self.prev_weights = np.asarray(self.prev_weights, dtype=float).ravel()
            if self.prev_weights.size != n:
                raise ValueError("prev_weights must match the number of assets")

        if self.transaction_cost is None:
            self.transaction_cost = np.zeros(n)
        else:
            self.transaction_cost = np.broadcast_to(
                np.asarray(self.transaction_cost, dtype=float), (n,)
            ).copy()

        self.lower_bounds = np.broadcast_to(
            np.asarray(self.lower_bounds, dtype=float), (n,)
        ).copy()
        self.upper_bounds = np.broadcast_to(
            np.asarray(self.upper_bounds, dtype=float), (n,)
        ).copy()

        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("lower_bounds must not exceed upper_bounds")

        if self.asset_names is None:
            self.asset_names = [f"asset_{i}" for i in range(n)]

    @property
    def n_assets(self) -> int:
        return self.expected_returns.size

    # -- objective components -------------------------------------------------
    def expected_return(self, w: np.ndarray) -> float:
        return float(self.expected_returns @ w)

    def variance(self, w: np.ndarray) -> float:
        return float(w @ self.covariance @ w)

    def turnover(self, w: np.ndarray) -> float:
        return float(np.sum(np.abs(w - self.prev_weights)))

    def cost(self, w: np.ndarray) -> float:
        return float(self.transaction_cost @ np.abs(w - self.prev_weights))

    def utility(self, w: np.ndarray) -> float:
        """Mean-variance utility net of transaction cost (higher is better)."""
        return (
            self.expected_return(w)
            - 0.5 * self.risk_aversion * self.variance(w)
            - self.cost_aversion * self.cost(w)
        )


# ---------------------------------------------------------------------------
# Continuous optimizer
# ---------------------------------------------------------------------------
@dataclass
class MeanVarianceContinuousOptimizer:
    """Constrained continuous mean-variance optimizer (SLSQP).

    The optimizer minimizes `-utility(w)` subject to the budget equality,
    per-asset bounds, optional sector-exposure inequalities, and an optional
    minimum target return.
    """

    problem: PortfolioProblem
    target_return: Optional[float] = None
    max_iter: int = 500
    tol: float = 1e-9

    def _bounds(self):
        p = self.problem
        return list(zip(p.lower_bounds, p.upper_bounds))

    def _constraints(self) -> List[dict]:
        p = self.problem
        cons: List[dict] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - p.budget}
        ]

        if self.target_return is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w: p.expected_return(w) - self.target_return,
                }
            )

        if p.sector_map is not None and p.sector_limits:
            sector_map = np.asarray(p.sector_map)
            for sector, limit in p.sector_limits.items():
                mask = (sector_map == sector).astype(float)
                # exposure <= limit  ->  limit - mask^T w >= 0
                cons.append(
                    {
                        "type": "ineq",
                        "fun": (lambda w, m=mask, lim=limit: lim - float(m @ w)),
                    }
                )
        return cons

    def _initial_guess(self) -> np.ndarray:
        p = self.problem
        # Feasible warm start: clip an equal-weight vector into the box, then
        # rescale toward the budget.
        w0 = np.clip(np.full(p.n_assets, p.budget / p.n_assets),
                     p.lower_bounds, p.upper_bounds)
        total = w0.sum()
        if total > 0:
            w0 = w0 * (p.budget / total)
            w0 = np.clip(w0, p.lower_bounds, p.upper_bounds)
        return w0

    def solve(self, w0: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Solve the problem and return an allocation with diagnostics."""
        p = self.problem
        x0 = np.asarray(w0, dtype=float) if w0 is not None else self._initial_guess()

        def neg_utility(w: np.ndarray) -> float:
            return -p.utility(w)

        res = minimize(
            neg_utility,
            x0,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        w = res.x
        return {
            "weights": w,
            "utility": p.utility(w),
            "expected_return": p.expected_return(w),
            "variance": p.variance(w),
            "volatility": float(np.sqrt(max(p.variance(w), 0.0))),
            "turnover": p.turnover(w),
            "cost": p.cost(w),
            "success": bool(res.success),
            "message": res.message,
            "n_iter": int(res.get("nit", 0)),
        }


def mean_variance_continuous(
    problem: PortfolioProblem,
    target_return: Optional[float] = None,
    **kwargs,
) -> Dict[str, object]:
    """Convenience wrapper: build the optimizer and return its solution."""
    return MeanVarianceContinuousOptimizer(
        problem=problem, target_return=target_return, **kwargs
    ).solve()


def _format_result(problem: PortfolioProblem, result: Dict[str, object]) -> str:
    """Render a solver result as a readable report."""
    lines = ["Continuous mean-variance allocation", "-" * 40]
    for name, w in zip(problem.asset_names, result["weights"]):
        lines.append(f"  {name:<12} {w:>8.2%}")
    lines.append("-" * 40)
    lines.append(f"  Expected return {result['expected_return']:>8.2%}")
    lines.append(f"  Volatility      {result['volatility']:>8.2%}")
    lines.append(f"  Variance        {result['variance']:>8.4f}")
    lines.append(f"  Turnover        {result['turnover']:>8.2%}")
    lines.append(f"  Cost            {result['cost']:>8.4f}")
    lines.append(f"  Utility         {result['utility']:>8.4f}")
    lines.append(f"  Solver success  {str(result['success']):>8}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Minimal demo so the module can be run directly to eyeball results:
    #   python -m src.classical_continuous      (from the project root)
    demo_mu = np.array([0.10, 0.07, 0.03])
    demo_cov = np.diag([0.04, 0.02, 0.005])
    demo_problem = PortfolioProblem(
        demo_mu,
        demo_cov,
        risk_aversion=3.0,
        asset_names=["equities", "credit", "govt_bonds"],
    )
    demo_result = mean_variance_continuous(demo_problem)
    print(_format_result(demo_problem, demo_result))