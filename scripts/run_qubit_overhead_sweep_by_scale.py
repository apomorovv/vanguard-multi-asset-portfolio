"""Run the small qubit-overhead sweep (see qubit_overhead_sweep.py) across
several asset-count scales, generating a fresh synthetic universe at each
scale via generate_synthetic_universe.py.

ASSUMPTION -- please adjust if you had different asset counts in mind: the
request that prompted this script didn't name specific counts, so it
defaults to [6, 10, 15, 20, 25]: 6 matches the original fixed-size dataset,
15 matches generate_synthetic_universe.py's own usage example, and 20/25
extend a bit further out. Override with --asset-counts.

Usage (from the repository root)::

    python scripts/run_qubit_overhead_sweep_by_scale.py
    python scripts/run_qubit_overhead_sweep_by_scale.py --asset-counts 6 12 18 24
    python scripts/run_qubit_overhead_sweep_by_scale.py --overheads 0 1 2 --n-lots 10

COST WARNING
------------
This runs one full PortfolioVQEPCESolver.run() (PSO -> NFT -> audit) per
(asset_count, overhead) pair -- e.g. the default 5 asset counts x 6
overheads (0..5) is 30 full solver runs. main() prints the total run
count up front so you can gauge this before it starts; narrow
--asset-counts / --overheads first if 30 runs is more than you want.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_uni_scale import generate_synthetic_universe  # noqa: E402

from src.vanguard_portfolio.quantum_vqe_solver import load_universe_from_json  # noqa: E402
from src.vanguard_portfolio.quantum_vqe_pce_solver import VQEPCEConfig  # noqa: E402
from src.vanguard_portfolio.qubit_overhead_sweep import (  # noqa: E402
    DEFAULT_OVERHEADS,
    sweep_qubit_overhead,
)

DEFAULT_ASSET_COUNTS = [6, 10, 15, 20, 30]
DEFAULT_N_LOTS = 20
DATA_DIR = ROOT / "data" / "synthetic"


def _build_problem(n_assets: int, seed: int):
    """Generate (and persist, same as generate_synthetic_universe.py's own
    CLI would) a synthetic universe at this asset count, then load it into
    a PortfolioProblem the same way run_VQE.py does."""
    data = generate_synthetic_universe(n_assets, seed=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"synthetic_universe_{n_assets}.json"
    path.write_text(json.dumps(data, indent=2))
    return load_universe_from_json(path), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-counts", type=int, nargs="+", default=DEFAULT_ASSET_COUNTS)
    parser.add_argument("--overheads", type=int, nargs="+", default=list(DEFAULT_OVERHEADS))
    parser.add_argument("--n-lots", type=int, default=DEFAULT_N_LOTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_runs = len(args.asset_counts) * len(args.overheads)
    print(
        f"Sweeping asset_counts={args.asset_counts} x overheads={args.overheads} "
        f"-> {n_runs} total solver runs, n_lots={args.n_lots}, seed={args.seed}\n"
    )

    all_results: dict[int, list[dict]] = {}
    for n_assets in args.asset_counts:
        print(f"\n{'#' * 70}\n# n_assets = {n_assets}\n{'#' * 70}")
        problem, path = _build_problem(n_assets, seed=args.seed)
        print(f"Universe written to {path.relative_to(ROOT)}")

        cfg = VQEPCEConfig(n_lots=args.n_lots)
        results = sweep_qubit_overhead(
            problem, overheads=args.overheads, n_lots=args.n_lots, base_config=cfg
        )
        all_results[n_assets] = results

    _print_combined_summary(all_results)


def _print_combined_summary(all_results: dict) -> None:
    header = (
        f"{'n_assets':>9}{'overhead':>9}{'n_qubits':>9}"
        f"{'utility_pp':>12}{'feasible':>10}"
    )
    print("\n" + "=" * 70)
    print("Combined summary: asset count x qubit overhead")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for n_assets, results in all_results.items():
        for r in results:
            feasible = r["budget_ok_pp"] and r["bounds_ok_pp"]
            print(
                f"{n_assets:>9}{r['overhead']:>9}{r['n_qubits']:>9}"
                f"{r['utility_pp']:>12.5f}{str(feasible):>10}"
            )

    print("\nBest feasible overhead per asset count:")
    for n_assets, results in all_results.items():
        feasible_results = [r for r in results if r["budget_ok_pp"] and r["bounds_ok_pp"]]
        if not feasible_results:
            print(f"  n_assets={n_assets:>3}: no feasible result in this sweep")
            continue
        best = max(feasible_results, key=lambda r: r["utility_pp"])
        print(
            f"  n_assets={n_assets:>3}: best overhead=+{best['overhead']} "
            f"(n_qubits={best['n_qubits']}, PCE min={best['pce_min_qubits']}), "
            f"utility_pp={best['utility_pp']:.5f}"
        )

    print(
        "\nWatch how the PCE-minimum qubit count itself grows with n_assets "
        "(reported per-row above) -- that's the scaling curve PCE is meant "
        "to flatten relative to the uncompressed quantum_vqe_solver, where "
        "qubit count grows linearly with n_assets x bits_per_asset instead."
    )


if __name__ == "__main__":
    main()
