"""Run the quantum VQE portfolio optimizer and compare it against the
classical continuous and discrete baselines.

Usage (from the repository root)::

    python scripts/run_VQE.py

NOTE ON `run_classical.py`
---------------------------
`run_classical.py` imports `PRESETS`, `SolveResult`, `solve_continuous`, and
`solve_discrete` from `src.vanguard_portfolio.classical`, and
`generate_synthetic_universe`/`save_problem` from
`src.vanguard_portfolio.data_generation`. Neither module (nor those names)
exists in `classical_continuous.py` / `classical_discrete.py` -- those files
define `PortfolioProblem`, `mean_variance_continuous`, and
`mean_variance_discrete` instead, and no data-generation module was
provided. This script is written against the API that actually exists, and
loads `synthetic_universe.json` directly instead of regenerating it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the `src` package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vanguard_portfolio.classical_continuous import mean_variance_continuous
from src.vanguard_portfolio.classical_discrete import mean_variance_discrete
from src.vanguard_portfolio.quantum_vqe_solver import (
    N_LOTS_DEFAULT,
    load_universe_from_json,
    run_vqe,
)

DATA_PATH = ROOT / "data" / "synthetic" / "synthetic_universe.json"


def _print_allocation(problem, name: str, weights: np.ndarray) -> None:
    print(f"\nAllocation - {name}")
    print("-" * 60)
    for asset_name, w in zip(problem.asset_names, weights):
        bar = "#" * int(round(w * 40))
        print(f"  {asset_name:<12} {w:6.1%} {bar}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Place synthetic_universe.json there "
            "(e.g. copy the uploaded file into data/synthetic/)."
        )

    problem = load_universe_from_json(DATA_PATH)
    print(f"Assets: {', '.join(problem.asset_names)}")
    print(f"Groups: {', '.join(problem.group_names)}")

    n_lots = N_LOTS_DEFAULT
    print(f"\nDiscretization: n_lots={n_lots}")

    print("\n" + "=" * 70)
    print("Classical continuous baseline (real-valued weights)")
    print("=" * 70)
    cont_result = mean_variance_continuous(problem)
    _print_allocation(problem, "continuous", cont_result["weights"])
    print(f"\n  utility={cont_result['utility']:.5f}  success={cont_result['success']}")

    print("\n" + "=" * 70)
    print(f"Classical discrete baseline (exact brute force, n_lots={n_lots})")
    print("=" * 70)
    brute_result = mean_variance_discrete(problem, n_lots=n_lots, method="brute")
    _print_allocation(problem, "discrete-brute (exact)", brute_result["weights"])
    print(f"\n  utility={brute_result['utility']:.5f}  "
          f"sector_penalty={brute_result['sector_penalty']:.6f}")

    print("\n" + "=" * 70)
    print(f"Classical discrete baseline (simulated annealing, n_lots={n_lots})")
    print("=" * 70)
    anneal_result = mean_variance_discrete(problem, n_lots=n_lots, method="anneal", seed=0)
    _print_allocation(problem, "discrete-anneal", anneal_result["weights"])
    print(f"\n  utility={anneal_result['utility']:.5f}  "
          f"sector_penalty={anneal_result['sector_penalty']:.6f}")

    print("\n" + "=" * 70)
    print("Quantum VQE (PSO -> NFT, adaptive CVaR, HNDC-1 ansatz)")
    print("=" * 70)
    vqe_result = run_vqe(problem, n_lots=n_lots)
    print(f"\nOptimization complete: {vqe_result['total_evals']} total evaluations")
    _print_allocation(problem, "vqe-raw", vqe_result["weights_raw"])
    _print_allocation(problem, "vqe-postprocessed", vqe_result["weights_pp"])
    print(f"\n  budget_ok raw/pp   = {vqe_result['budget_ok_raw']}/{vqe_result['budget_ok_pp']}")
    print(f"  bounds_ok raw/pp   = {vqe_result['bounds_ok_raw']}/{vqe_result['bounds_ok_pp']}")

    print("\nSolver comparison")
    header = f"{'method':<20}{'utility':>12}{'sector_pen':>12}{'feasible':>10}"
    print(header)
    print("-" * len(header))
    rows = [
        ("continuous", cont_result["utility"], 0.0, True),
        ("discrete-brute", brute_result["utility"], brute_result["sector_penalty"], True),
        ("discrete-anneal", anneal_result["utility"], anneal_result["sector_penalty"], True),
        ("vqe-raw", vqe_result["utility_raw"], vqe_result["sector_penalty_raw"],
         vqe_result["budget_ok_raw"] and vqe_result["bounds_ok_raw"]),
        ("vqe-postprocessed", vqe_result["utility_pp"], vqe_result["sector_penalty_pp"],
         vqe_result["budget_ok_pp"] and vqe_result["bounds_ok_pp"]),
    ]
    for name, util, sec_pen, feasible in rows:
        print(f"{name:<20}{util:>12.5f}{sec_pen:>12.6f}{str(feasible):>10}")

    if vqe_result["budget_ok_pp"] and vqe_result["bounds_ok_pp"]:
        gap = 100 * (brute_result["utility"] - vqe_result["utility_pp"]) / abs(brute_result["utility"])
        print(f"\nVQE postprocessed utility gap vs. exact discrete optimum: {gap:.2f}%")
    else:
        print(
            "\nVQE postprocessed solution is not feasible (budget/bounds violated) -- "
            "gap-vs-optimum is not meaningful until that's resolved (see "
            "sector_penalty / budget_ok / bounds_ok above)."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()