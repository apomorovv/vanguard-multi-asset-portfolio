"""Run the PCE-compressed quantum VQE portfolio optimizer and compare it
against the classical continuous and discrete baselines.

Usage (from the repository root)::

    python scripts/run_vqe_pce.py

This runner mirrors `run_quantum_vqe.py` but uses the Pauli Correlation Encoding
(PCE) version of the VQE solver, which compresses the logical lot-bit
register before optimization.
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
from src.vanguard_portfolio.quantum_vqe_pce_solver import (
    N_LOTS_DEFAULT,
    VQEPCEConfig,
    run_pce_vqe,
)
from src.vanguard_portfolio.quantum_vqe_solver import load_universe_from_json

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
    print(
        f"\n  utility={cont_result['utility']:.5f}  "
        f"success={cont_result['success']}"
    )

    print("\n" + "=" * 70)
    print(f"Classical discrete baseline (exact brute force, n_lots={n_lots})")
    print("=" * 70)
    brute_result = mean_variance_discrete(
        problem,
        n_lots=n_lots,
        method="brute",
    )
    _print_allocation(problem, "discrete-brute (exact)", brute_result["weights"])
    print(
        f"\n  utility={brute_result['utility']:.5f}  "
        f"sector_penalty={brute_result['sector_penalty']:.6f}"
    )

    print("\n" + "=" * 70)
    print(f"Classical discrete baseline (simulated annealing, n_lots={n_lots})")
    print("=" * 70)
    anneal_result = mean_variance_discrete(
        problem,
        n_lots=n_lots,
        method="anneal",
        seed=0,
    )
    _print_allocation(problem, "discrete-anneal", anneal_result["weights"])
    print(
        f"\n  utility={anneal_result['utility']:.5f}  "
        f"sector_penalty={anneal_result['sector_penalty']:.6f}"
    )

    print("\n" + "=" * 70)
    print("Quantum VQE + PCE (compressed register, adaptive CVaR, PSO -> NFT)")
    print("=" * 70)

    cfg = VQEPCEConfig(n_lots=n_lots)
    pce_result = run_pce_vqe(problem, config=cfg)

    print(f"\nOptimization complete: {pce_result['total_evals']} total evaluations")
    print(
        f"PCE compression: {pce_result['n_vars']} logical lot-bits "
        f"-> {pce_result['n_qubits']} physical qubits"
    )
    print(
        f"Final answer chosen from top-{pce_result['final_top_k']} most-frequent "
        f"states per basis group ({pce_result['final_pool_shape'][0]}x"
        f"{pce_result['final_pool_shape'][1]}x{pce_result['final_pool_shape'][2]} = "
        f"{pce_result['final_pool_size']} candidates), not from raw exhaustive "
        f"coverage -- see quantum_vqe_pce_solver.py's module docstring "
        f"('Fixed bug: final answer was independent of training') if this "
        f"number looks surprising."
    )

    _print_allocation(problem, "pce-vqe-raw", pce_result["weights_raw"])
    _print_allocation(problem, "pce-vqe-postprocessed", pce_result["weights_pp"])

    print(
        f"\n  budget_ok raw/pp   = "
        f"{pce_result['budget_ok_raw']}/{pce_result['budget_ok_pp']}"
    )
    print(
        f"  bounds_ok raw/pp   = "
        f"{pce_result['bounds_ok_raw']}/{pce_result['bounds_ok_pp']}"
    )

    stats = pce_result["feasibility_stats"]
    print("\nFinal candidate pool feasibility")
    print(
        f"  budget_ok : {stats['frac_budget_ok']:.1%}\n"
        f"  bounds_ok : {stats['frac_bounds_ok']:.1%}\n"
        f"  sector_ok : {stats['frac_sector_ok']:.1%}\n"
        f"  all_ok    : {stats['frac_all_ok']:.1%}"
    )

    print("\nSolver comparison")
    header = f"{'method':<22}{'utility':>12}{'sector_pen':>12}{'feasible':>10}"
    print(header)
    print("-" * len(header))

    rows = [
        ("continuous",
         cont_result["utility"],
         0.0,
         True),
        ("discrete-brute",
         brute_result["utility"],
         brute_result["sector_penalty"],
         True),
        ("discrete-anneal",
         anneal_result["utility"],
         anneal_result["sector_penalty"],
         True),
        ("pce-vqe-raw",
         pce_result["utility_raw"],
         pce_result["sector_penalty_raw"],
         pce_result["budget_ok_raw"] and pce_result["bounds_ok_raw"]),
        ("pce-vqe-postprocessed",
         pce_result["utility_pp"],
         pce_result["sector_penalty_pp"],
         pce_result["budget_ok_pp"] and pce_result["bounds_ok_pp"]),
    ]

    for name, util, sec_pen, feasible in rows:
        print(
            f"{name:<22}"
            f"{util:>12.5f}"
            f"{sec_pen:>12.6f}"
            f"{str(feasible):>10}"
        )

    if pce_result["budget_ok_pp"] and pce_result["bounds_ok_pp"]:
        gap = (
            100
            * (brute_result["utility"] - pce_result["utility_pp"])
            / abs(brute_result["utility"])
        )
        print(
            f"\nPCE-VQE postprocessed utility gap vs. "
            f"exact discrete optimum: {gap:.2f}%"
        )
    else:
        print(
            "\nPCE-VQE postprocessed solution is not feasible "
            "(budget/bounds violated) -- gap vs. optimum is not meaningful."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()