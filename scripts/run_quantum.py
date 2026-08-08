"""Run the quantum (QAOA + PCE) discrete portfolio optimizer and compare it
against the classical discrete solver, both reading the SAME shared data
file (data/portfolio_problem.json) -- not any self-generated synthetic data.

Usage (from the repository root)::

    python scripts/run_quantum.py
    python scripts/run_quantum.py --n-lots 20 --risk-aversion 3.0

------------------------------------------------------------------------
Data / API mismatches found while wiring this up -- please read
------------------------------------------------------------------------
1. `classical_discrete.py` (as given to me) does
   `from .classical_continuous import PortfolioProblem` and exposes
   `MeanVarianceDiscreteOptimizer` / `mean_variance_discrete`. But
   `run_classical.py` (also as given to me) instead imports
   `from src.vanguard_portfolio.classical import PRESETS, SolveResult,
   solve_continuous, solve_discrete` and `from
   src.vanguard_portfolio.data_generation import
   generate_synthetic_universe, save_problem` -- a *different* module
   (`classical.py`, singular) and a *different* function/class API
   than what's actually in `classical_discrete.py`. These two files
   can't both be current against the same codebase as given. This script
   is built against the concrete implementation you've sent me
   (`classical_discrete.py` + `classical_continuous.PortfolioProblem`).
   If `classical.py` / `data_generation.py` / `PRESETS` are the ones
   that are actually current, send them over and I'll re-point this
   script (and swap `PRESETS["balanced"]` in for the `--risk-aversion`/
   `--cost-aversion` CLI flags below).

2. [RESOLVED] `PortfolioProblem`'s real constructor uses `prev_weights`
   (not `w0`) and `transaction_cost` (not a bare "c"), and its utility is

       U(w) = mu.w - 0.5*risk_aversion*variance(w) - cost_aversion*cost(w)
       cost(w) = transaction_cost . |w - prev_weights|

   Notably there's NO income/yield term. The shared JSON's `"y"` field is
   loaded below but deliberately NOT passed into the objective anywhere
   (classical or quantum) since `PortfolioProblem.utility()` doesn't use
   it -- including it would silently optimize for something the real
   objective doesn't reward. `load_portfolio_problem()` now passes
   `transaction_cost=c` and `prev_weights=w0` explicitly; earlier this
   passed neither, so the classical solver was silently running with
   zero transaction costs.

3. `classical_discrete.py`'s soft sector penalty
   (`_sector_penalty`/`sector_limits`) is one-sided: it only penalizes
   `exposure > limit`, i.e. an upper bound. The shared JSON data has
   *two*-sided group bounds (`group_lower` AND `group_upper`, e.g. Equity
   in [0.30, 0.70]). I've passed `group_upper` through to
   `PortfolioProblem.sector_limits` (so the classical solver's existing
   upper-bound soft penalty is at least partially wired up), but
   `group_lower` has NO effect on the classical solver as given -- it
   only constrains the quantum solver (which enforces both sides as HARD
   constraints, matching the original quantum file's group-exposure
   design). If you want the classical solver to also respect
   `group_lower`, that needs a small change to `_sector_penalty` (in
   `classical_discrete.py`) to support a `(lower, upper)` tuple instead
   of a single upper limit.

4. `risk_aversion` and `cost_aversion` are not present in the shared
   JSON at all. Defaulted to `risk_aversion=3.0` (matching the demo in
   both `classical_continuous.py`'s and `classical_discrete.py`'s
   `__main__` blocks) and `cost_aversion=1.0` (matching
   `PortfolioProblem`'s own default). `budget` also isn't in the JSON;
   defaulted to `1.0`. All three are CLI flags below -- override them, or
   tell me your `PRESETS["balanced"]` values and I'll hard-code them.

5. Asset naming: the original standalone quantum file used
   `"US Equity"`, `"International Equity"`, etc. (spaces, longer names).
   The shared JSON uses `"US_Equity"`, `"Intl_Equity"`, etc. This script
   uses the JSON's names throughout (per your instruction to read off the
   shared data) -- just flagging in case anything else in the repo still
   references the old names.

6. The original quantum file's noise model
   (`NoiseModel.from_backend(FakeMarrakesh())`) was computed but never
   actually passed to the estimator (`"noise_model": None` was hardcoded
   instead) -- so the original ran noiseless despite the surrounding
   commentary about a noisy backend. That's a latent bug in the source
   file, not something I introduced. Preserved as `apply_noise=False` by
   default in `quantum_discrete.py`; pass `apply_noise=True` if you
   actually want the noise model applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vanguard_portfolio.classical_continuous import PortfolioProblem
from src.vanguard_portfolio.classical_discrete import mean_variance_discrete
from src.vanguard_portfolio.quantum_discrete import (
    _format_result,
    quantum_mean_variance_discrete,
)


def load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_portfolio_problem(data: dict, risk_aversion: float, budget: float,
                            cost_aversion: float) -> PortfolioProblem:
    """Build the shared PortfolioProblem from the JSON data."""
    mu = np.array(data["mu"], dtype=float)
    cov = np.array(data["cov"], dtype=float)
    lower = np.array(data["lower"], dtype=float)
    upper = np.array(data["upper"], dtype=float)
    w0 = np.array(data["w0"], dtype=float)
    c = np.array(data["c"], dtype=float)
    asset_group = data["asset_group"]
    sector_limits = {g: data["group_upper"][g] for g in set(asset_group)}

    return PortfolioProblem(
        mu,
        cov,
        risk_aversion=risk_aversion,
        asset_names=data["asset_names"],
        budget=budget,
        lower_bounds=lower,
        upper_bounds=upper,
        sector_map=asset_group,
        sector_limits=sector_limits,
        transaction_cost=c,
        prev_weights=w0,
        cost_aversion=cost_aversion,
    )

def _print_allocation_table(title: str, asset_names, weights, lots=None) -> None:
    print(f"\nAllocation - {title}")
    print("-" * 60)
    for i, name in enumerate(asset_names):
        w = weights[i]
        bar = "#" * int(round(w * 40))
        lot_str = f"  ({int(lots[i])} lots)" if lots is not None else ""
        print(f"  {name:<14} {w:6.1%} {bar}{lot_str}")


def _print_comparison(classical_result: dict, quantum_result: dict) -> None:
    header = f"{'method':<22}{'utility':>12}{'return':>9}{'volat.':>9}{'turnover':>10}{'constraints':>14}"
    print("\nSolver comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'classical:' + classical_result['method']:<22}"
          f"{classical_result['utility']:>12.5f}"
          f"{classical_result['expected_return']:>9.2%}"
          f"{classical_result['volatility']:>9.2%}"
          f"{classical_result['turnover']:>10.3f}"
          f"{'n/a (soft)':>14}")
    print(f"{'quantum:qaoa_pce':<22}"
          f"{quantum_result['utility']:>12.5f}"
          f"{quantum_result['expected_return']:>9.2%}"
          f"{quantum_result['volatility']:>9.2%}"
          f"{quantum_result['turnover']:>10.3f}"
          f"{str(quantum_result['n_constraints_satisfied']) + '/' + str(quantum_result['n_constraints']):>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "synthetic_universe.json")
    parser.add_argument("--n-lots", type=int, default=20)
    parser.add_argument("--risk-aversion", type=float, default=3.0,
                         help="Not present in the shared JSON -- see mismatch note #4.")
    parser.add_argument("--budget", type=float, default=1.0,
                         help="Not present in the shared JSON -- see mismatch note #4.")
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-aversion", type=float, default=1.0,
                     help="Not present in the shared JSON -- see mismatch note #4.")
    args = parser.parse_args()

    data = load_data(args.data)
    print(f"Loaded shared portfolio data from: {args.data.relative_to(ROOT)}")
    print(f"Assets: {', '.join(data['asset_names'])}")
    print(f"Groups: {', '.join(data['group_names'])}")

    problem = load_portfolio_problem(data, args.risk_aversion, args.budget, args.cost_aversion)
    mu = np.array(data["mu"], dtype=float)
    cov = np.array(data["cov"], dtype=float)
    y = np.array(data["y"], dtype=float)
    c = np.array(data["c"], dtype=float)
    w0 = np.array(data["w0"], dtype=float)

    # Classical baseline: brute force is exact and tractable at this scale
    # (6 assets, n_lots<=~20 -> a few tens of thousands of compositions);
    # fall back to simulated annealing for larger n_lots.
    classical_method = "brute" if args.n_lots <= 25 else "anneal"
    print(f"\nSolving classical discrete baseline (method={classical_method})...")
    classical_result = mean_variance_discrete(
        problem, n_lots=args.n_lots, method=classical_method,
        seed=args.seed if classical_method == "anneal" else None,
    )
    _print_allocation_table(f"classical:{classical_method}", data["asset_names"],
                             classical_result["weights"], classical_result["lots"])

    print(f"\nSolving quantum discrete allocation (QAOA + PCE, "
          f"{args.n_restarts} restarts)...")
    quantum_result = quantum_mean_variance_discrete(
        problem, mu, cov, c, w0,
        group_map=data["asset_group"],
        group_lower=data["group_lower"],
        group_upper=data["group_upper"],
        group_names=data["group_names"],
        n_lots=args.n_lots,
        n_restarts=args.n_restarts,
        reps=args.reps,
        seed=args.seed,
    )
    _print_allocation_table("quantum:qaoa_pce", data["asset_names"],
                             quantum_result["weights"], quantum_result["lots"])
    print()
    print(_format_result(problem, quantum_result))

    _print_comparison(classical_result, quantum_result)

    gap = quantum_result["utility"] - classical_result["utility"]
    print(f"\nQuantum vs classical utility gap: {gap:+.5f} "
          f"({'quantum ahead' if gap > 0 else 'classical ahead' if gap < 0 else 'tied'})")

    if quantum_result["n_constraints_satisfied"] < quantum_result["n_constraints"]:
        failed = [name for name, ok, _ in quantum_result["constraint_report"] if not ok]
        print(f"\n{len(failed)} quantum constraint(s) violated: {', '.join(failed)}")
    else:
        print("\nAll quantum hard constraints satisfied exactly.")


if __name__ == "__main__":
    main()