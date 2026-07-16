"""Run the classical portfolio optimizers and display the results.

Usage (from the repository root)::

    python scripts/run_classical.py

The script:

1. Builds the synthetic asset universe and saves it to `data/synthetic`.
2. Solves the continuous and discrete classical models for every investor
   preset.
3. Prints an allocation table and a solver-comparison table.
4. Saves a bar chart of the balanced allocation to `results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the `src` package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vanguard_portfolio.classical import PRESETS, SolveResult, solve_continuous, solve_discrete
from src.vanguard_portfolio.data_generation import generate_synthetic_universe, save_problem


def _format_weights(names: list[str], w: np.ndarray) -> str:
    return "  ".join(f"{name}={weight:5.1%}" for name, weight in zip(names, w))


def _print_allocation_table(problem, result: SolveResult) -> None:
    print(f"\nAllocation - {result.method}")
    print("-" * 60)
    for name, weight in zip(problem.asset_names, result.weights):
        bar = "#" * int(round(weight * 40))
        print(f"  {name:<12} {weight:6.1%} {bar}")


def _print_comparison(results: list[SolveResult]) -> None:
    header = (
        f"{'method':<16}{'objective':>12}{'return':>9}{'volat.':>9}"
        f"{'income':>9}{'turnover':>10}{'cost':>9}{'breaches':>10}{'runtime_s':>11}"
    )
    print("\nSolver comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metrics
        print(
            f"{r.method:<16}{r.objective:>12.5f}{m.get('expected_return', float('nan')):>9.2%}"
            f"{m.get('volatility', float('nan')):>9.2%}{m.get('income', float('nan')):>9.2%}"
            f"{m.get('turnover', float('nan')):>10.3f}{m.get('transaction_cost', float('nan')):>9.4f}"
            f"{r.breaches:>10}{r.runtime:>11.4f}"
        )


def _save_plot(problem, result: SolveResult, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available - skipping chart)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(problem.n)
    width = 0.4
    ax.bar(x - width / 2, problem.w0, width, label="current (w0)", color="#B0BEC5")
    ax.bar(x + width / 2, result.weights, width, label=result.method, color="#1E88E5")
    ax.set_xticks(x)
    ax.set_xticklabels(problem.asset_names, rotation=30, ha="right")
    ax.set_ylabel("allocation")
    ax.set_title("Recommended vs current allocation (balanced preset)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "allocation_balanced.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved chart: {path.relative_to(ROOT)}")


def main() -> None:
    problem = generate_synthetic_universe()
    data_path = save_problem(problem, ROOT / "data/synthetic")
    print(f"Synthetic universe saved to: {data_path.relative_to(ROOT)}")
    print(f"Assets: {', '.join(problem.asset_names)}")

    # Detailed view of the balanced preset with both solvers.
    balanced = PRESETS["balanced"]
    cont = solve_continuous(problem, balanced)
    disc = solve_discrete(problem, balanced, units=10)

    _print_allocation_table(problem, cont)
    _print_allocation_table(problem, disc)
    print(
        f"\nContinuous feasible={cont.feasible} | "
        f"Discrete feasible={disc.feasible} | "
        f"discrete-vs-continuous L1 distance="
        f"{np.sum(np.abs(disc.weights - cont.weights)):.4f}"
    )

    # Compare every investor preset (continuous solver).
    preset_results: list[SolveResult] = []
    for name, prefs in PRESETS.items():
        result = solve_continuous(problem, prefs)
        result.method = f"cont:{name}"
        preset_results.append(result)
    _print_comparison(preset_results)

    _save_plot(problem, cont, ROOT / "results")


if __name__ == "__main__":
    main()