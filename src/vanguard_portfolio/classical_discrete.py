"""Classical discrete mean-variance portfolio optimizer.

This is the discrete counterpart to :mod:classical_continuous. Instead of
continuous real-valued weights it allocates a fixed number of integer "lots"
across assets, which keeps the budget constraint satisfied by construction
and mirrors the binary/integer decision variables required by the quantum
formulation.

Weight encoding
---------------
The budget is split into `n_lots` equal units. Asset `i` receives an integer
number of lots `k_i` with `sum_i k_i == n_lots`. The resulting weight is::

    w_i = (budget / n_lots) * k_i

Per-asset bounds are translated into integer lot bounds. This lattice
formulation is directly QUBO-friendly (a lot is a group of binary variables) and
is used as the classical validator for the discrete/quantum solvers.

Two solvers are provided:

* `brute` - exhaustively enumerates every feasible lot allocation. Exact, but
  only tractable for small `n_assets` / `n_lots`.
* `anneal` - simulated annealing that moves one lot between assets each step,
  preserving the budget automatically. Scales to larger problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Optional

import numpy as np

from .classical_continuous import PortfolioProblem


def _lot_bounds(problem: PortfolioProblem, n_lots: int) -> tuple[np.ndarray, np.ndarray]:
    """Translate continuous weight bounds into integer lot bounds."""
    lot_size = problem.budget / n_lots
    lo = np.floor(problem.lower_bounds / lot_size + 1e-9).astype(int)
    hi = np.ceil(problem.upper_bounds / lot_size - 1e-9).astype(int)
    lo = np.clip(lo, 0, n_lots)
    hi = np.clip(hi, 0, n_lots)
    hi = np.maximum(hi, lo)
    return lo, hi


def _lots_to_weights(lots: np.ndarray, problem: PortfolioProblem, n_lots: int) -> np.ndarray:
    return (problem.budget / n_lots) * lots.astype(float)


def _sector_penalty(weights: np.ndarray, problem: PortfolioProblem) -> float:
    """Squared-violation penalty for exceeded sector-exposure limits."""
    if problem.sector_map is None or not problem.sector_limits:
        return 0.0
    sector_map = np.asarray(problem.sector_map)
    penalty = 0.0
    for sector, limit in problem.sector_limits.items():
        exposure = float(weights[sector_map == sector].sum())
        if exposure > limit:
            penalty += (exposure - limit) ** 2
    return penalty


@dataclass
class MeanVarianceDiscreteOptimizer:
    """Discrete (integer-lot) mean-variance optimizer.

    Parameters
    ----------
    problem:
        The shared :class:PortfolioProblem definition.
    n_lots:
        Number of discrete units the budget is divided into. The weight
        resolution is `budget / n_lots`.
    method:
        `"brute"` for exhaustive search or `"anneal"` for simulated
        annealing.
    penalty_weight:
        Multiplier on the soft sector-exposure penalty added to the (negative)
        utility during search.
    """

    problem: PortfolioProblem
    n_lots: int = 20
    method: str = "anneal"
    penalty_weight: float = 100.0
    # Simulated-annealing controls
    n_iter: int = 20_000
    init_temp: float = 1.0
    final_temp: float = 1e-3
    seed: Optional[int] = None

    def _score(self, lots: np.ndarray) -> float:
        """Objective to maximize: utility minus soft sector penalty."""
        w = _lots_to_weights(lots, self.problem, self.n_lots)
        return self.problem.utility(w) - self.penalty_weight * _sector_penalty(w, self.problem)

    # -- brute-force ----------------------------------------------------------
    def _solve_brute(self) -> np.ndarray:
        p = self.problem
        n = p.n_assets
        lo, hi = _lot_bounds(p, self.n_lots)

        best_lots: Optional[np.ndarray] = None
        best_score = -np.inf

        # Enumerate all compositions of n_lots into n non-negative integers by
        # choosing n-1 "dividers" among the lot positions (stars and bars).
        for dividers in combinations_with_replacement(range(n), self.n_lots):
            lots = np.bincount(dividers, minlength=n)
            if np.any(lots < lo) or np.any(lots > hi):
                continue
            score = self._score(lots)
            if score > best_score:
                best_score = score
                best_lots = lots

        if best_lots is None:
            raise ValueError(
                "No feasible lot allocation satisfies the per-asset bounds; "
                "relax bounds or increase n_lots."
            )
        return best_lots

    # -- simulated annealing --------------------------------------------------
    def _feasible_start(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        p = self.problem
        n = p.n_assets
        if int(lo.sum()) > self.n_lots or int(hi.sum()) < self.n_lots:
            raise ValueError(
                "Per-asset lot bounds are infeasible for the requested n_lots."
            )
        lots = lo.copy()
        remaining = self.n_lots - int(lots.sum())
        # Distribute the remaining lots respecting upper bounds.
        headroom = hi - lots
        order = np.argsort(-p.expected_returns)  # seed toward higher-return assets
        for i in order:
            if remaining <= 0:
                break
            add = min(headroom[i], remaining)
            lots[i] += add
            remaining -= add
        if remaining > 0:  # pragma: no cover - guarded by feasibility check above
            raise ValueError("Could not place all lots within bounds.")
        return lots

    def _solve_anneal(self) -> np.ndarray:
        p = self.problem
        n = p.n_assets
        rng = np.random.default_rng(self.seed)
        lo, hi = _lot_bounds(p, self.n_lots)

        lots = self._feasible_start(lo, hi)
        score = self._score(lots)
        best_lots, best_score = lots.copy(), score

        temps = np.geomspace(self.init_temp, self.final_temp, self.n_iter)
        for temp in temps:
            # Move one lot from a donor (has >lo) to a receiver (has <hi).
            donors = np.flatnonzero(lots > lo)
            receivers = np.flatnonzero(lots < hi)
            if donors.size == 0 or receivers.size == 0:
                break
            i = rng.choice(donors)
            j = rng.choice(receivers)
            if i == j:
                continue

            trial = lots.copy()
            trial[i] -= 1
            trial[j] += 1
            trial_score = self._score(trial)

            delta = trial_score - score
            if delta >= 0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
                lots, score = trial, trial_score
                if score > best_score:
                    best_lots, best_score = lots.copy(), score

        return best_lots

    # -- public API -----------------------------------------------------------
    def solve(self) -> Dict[str, object]:
        if self.method == "brute":
            lots = self._solve_brute()
        elif self.method == "anneal":
            lots = self._solve_anneal()
        else:
            raise ValueError(f"Unknown method: {self.method!r} (use 'brute' or 'anneal')")

        p = self.problem
        w = _lots_to_weights(lots, p, self.n_lots)
        return {
            "weights": w,
            "lots": lots,
            "utility": p.utility(w),
            "expected_return": p.expected_return(w),
            "variance": p.variance(w),
            "volatility": float(np.sqrt(max(p.variance(w), 0.0))),
            "turnover": p.turnover(w),
            "cost": p.cost(w),
            "sector_penalty": _sector_penalty(w, p),
            "n_lots": self.n_lots,
            "method": self.method,
        }


def mean_variance_discrete(
    problem: PortfolioProblem,
    n_lots: int = 20,
    method: str = "anneal",
    **kwargs,
) -> Dict[str, object]:
    """Convenience wrapper: build the discrete optimizer and return its solution."""
    return MeanVarianceDiscreteOptimizer(
        problem=problem, n_lots=n_lots, method=method, **kwargs
    ).solve()


def _format_result(problem: PortfolioProblem, result: Dict[str, object]) -> str:
    """Render a discrete solver result as a readable report."""
    lines = [
        f"Discrete mean-variance allocation "
        f"(method={result['method']}, n_lots={result['n_lots']})",
        "-" * 46,
    ]
    for name, w, k in zip(problem.asset_names, result["weights"], result["lots"]):
        lines.append(f"  {name:<12} {w:>8.2%}   ({int(k)} lots)")
    lines.append("-" * 46)
    lines.append(f"  Expected return {result['expected_return']:>8.2%}")
    lines.append(f"  Volatility      {result['volatility']:>8.2%}")
    lines.append(f"  Variance        {result['variance']:>8.4f}")
    lines.append(f"  Turnover        {result['turnover']:>8.2%}")
    lines.append(f"  Cost            {result['cost']:>8.4f}")
    lines.append(f"  Sector penalty  {result['sector_penalty']:>8.4f}")
    lines.append(f"  Utility         {result['utility']:>8.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Minimal demo so the module can be run directly to eyeball results:
    #   python -m src.classical_discrete       (from the project root)
    demo_mu = np.array([0.10, 0.07, 0.03])
    demo_cov = np.diag([0.04, 0.02, 0.005])
    demo_problem = PortfolioProblem(
        demo_mu,
        demo_cov,
        risk_aversion=3.0,
        asset_names=["equities", "credit", "govt_bonds"],
    )
    # Exhaustive (exact) and simulated-annealing solutions for comparison.
    brute_result = mean_variance_discrete(demo_problem, n_lots=10, method="brute")
    anneal_result = mean_variance_discrete(
        demo_problem, n_lots=10, method="anneal", seed=0
    )
    print(_format_result(demo_problem, brute_result))
    print()
    print(_format_result(demo_problem, anneal_result))