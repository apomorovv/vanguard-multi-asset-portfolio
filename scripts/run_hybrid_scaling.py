#!/usr/bin/env python3
"""Benchmark the final hybrid architecture from tiny to 2,000 assets."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vanguard_portfolio.data_generation import generate_factor_universe
from vanguard_portfolio.hybrid import HybridConfig, run_hybrid_optimizer
from vanguard_portfolio.quantum_solver import XYQAOAConfig
from vanguard_portfolio.schemas import PortfolioConstraints, Preferences


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    successful = [row for row in rows if row.get("success")]
    methods = sorted({str(row["method"]) for row in successful})
    colors = ["#0B5CAD", "#00A6A6", "#F28E2B", "#7B61A8", "#D1495B"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for index, method in enumerate(methods):
        group = sorted(
            (row for row in successful if row["method"] == method),
            key=lambda row: int(row["n_assets"]),
        )
        ax.plot(
            [int(row["n_assets"]) for row in group],
            [float(row["runtime_seconds"]) for row in group],
            marker="o",
            linewidth=1.8,
            color=colors[index % len(colors)],
            label=method.replace("_", " "),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Asset universe size")
    ax.set_ylabel("End-to-end seconds")
    ax.set_title("Hybrid solver scaling with a fixed-size quantum window")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "scaling_runtime.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "scaling_runtime.pdf", bbox_inches="tight")
    plt.close(fig)

    best_rows = [row for row in successful if row["method"] == "best_valid"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].plot(
        [int(row["n_assets"]) for row in best_rows],
        [float(row["time_to_first_valid_seconds"]) for row in best_rows],
        marker="o",
        color="#00A6A6",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Asset universe size")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Time to first valid exact-K portfolio")
    axes[0].grid(alpha=0.25, which="both")
    axes[1].plot(
        [int(row["n_assets"]) for row in best_rows],
        [int(row["oracle_calls"]) for row in best_rows],
        marker="s",
        color="#7B61A8",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Asset universe size")
    axes[1].set_ylabel("Unique allocation-oracle calls")
    axes[1].set_title("Search work remains window-controlled")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output / "scaling_feasibility_and_oracle.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "scaling_feasibility_and_oracle.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 250, 500, 1000, 2000],
    )
    parser.add_argument("--cardinality", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--backend", default="osqp")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--quantum", action="store_true")
    parser.add_argument("--aer-gpu", action="store_true")
    parser.add_argument("--gurobi", action="store_true")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, default=ROOT / "results/hybrid_scaling")
    args = parser.parse_args(argv)
    rows: list[dict[str, object]] = []
    for size in args.sizes:
        groups = min(10, max(2, size // 10))
        cardinality = min(int(args.cardinality), size - 1)
        cardinality = max(groups, cardinality)
        for repetition in range(args.repetitions):
            seed = args.seed + 1000 * size + repetition
            try:
                problem = generate_factor_universe(
                    n_assets=size,
                    n_groups=groups,
                    n_factors=min(12, max(2, size // 20)),
                    seed=seed,
                )
                constraints = PortfolioConstraints(
                    exact_cardinality=cardinality,
                    minimum_active_weight=min(0.005, 0.5 / cardinality),
                    maximum_weights=np.full(size, max(0.04, 1.5 / cardinality)),
                )
                config = HybridConfig(
                    iterations=1,
                    window_size=min(args.window_size, size),
                    allocation_backend=args.backend,
                    allocation_options={"tol": 1e-7, "max_iter": 500_000},
                    classical_tabu_iterations=35,
                    classical_oracle_candidates=3,
                    run_quantum=args.quantum,
                    run_penalty_qaoa=False,
                    run_gurobi_reference=args.gurobi,
                    seed=seed,
                    quantum=XYQAOAConfig(
                        depth=1,
                        shots=2_048,
                        optimizer_maxiter=35,
                        optimizer_starts=1,
                        seed=seed,
                        backend="aer_gpu" if args.aer_gpu else "subspace",
                    ),
                )
                run = run_hybrid_optimizer(problem, Preferences(), constraints, config)
                for result in run.all_results():
                    rows.append(
                        {
                            "n_assets": size,
                            "cardinality": cardinality,
                            "window_size": min(args.window_size, size),
                            "repetition": repetition,
                            "method": result.method,
                            "runtime_seconds": result.runtime,
                            "objective": result.objective,
                            "objective_gap_to_relaxation": (
                                result.objective - run.relaxation.objective
                            ),
                            "success": result.success,
                            "feasible": result.feasible,
                            "breaches": result.breaches,
                            "time_to_first_valid_seconds": run.timeline[0]["elapsed_seconds"],
                            "oracle_calls": run.oracle_calls,
                            "oracle_cache_hits": run.oracle_cache_hits,
                        }
                    )
                rows.append(
                    {
                        "n_assets": size,
                        "cardinality": cardinality,
                        "window_size": min(args.window_size, size),
                        "repetition": repetition,
                        "method": "best_valid",
                        "runtime_seconds": run.runtime,
                        "objective": run.best.objective,
                        "objective_gap_to_relaxation": (
                            run.best.objective - run.relaxation.objective
                        ),
                        "success": True,
                        "feasible": True,
                        "breaches": 0,
                        "time_to_first_valid_seconds": run.timeline[0]["elapsed_seconds"],
                        "oracle_calls": run.oracle_calls,
                        "oracle_cache_hits": run.oracle_cache_hits,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "n_assets": size,
                        "cardinality": cardinality,
                        "window_size": min(args.window_size, size),
                        "repetition": repetition,
                        "method": "run_failure",
                        "runtime_seconds": "",
                        "objective": "",
                        "objective_gap_to_relaxation": "",
                        "success": False,
                        "feasible": False,
                        "breaches": "",
                        "time_to_first_valid_seconds": "",
                        "oracle_calls": "",
                        "oracle_cache_hits": "",
                        "error": str(exc),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "hybrid_scaling.csv", rows)
    _plot(rows, args.output)
    print(f"Saved {len(rows)} records and four plots to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
