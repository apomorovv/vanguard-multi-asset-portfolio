#!/usr/bin/env python3
"""Run repeated continuous-solver scaling experiments on factor universes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vanguard_portfolio.classical import write_artifact_manifest  # noqa: E402
from vanguard_portfolio.classical_continuous import solve_continuous  # noqa: E402
from vanguard_portfolio.data_generation import generate_factor_universe  # noqa: E402
from vanguard_portfolio.schemas import Preferences, SolverUnavailableError  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _solver_options(backend: str, tolerance: float, time_limit: float) -> dict:
    lower = backend.lower()
    if lower in {"scipy", "slsqp", "scipy_slsqp"}:
        return {"tol": tolerance, "max_iter": 10_000}
    if lower == "osqp":
        return {"tol": tolerance, "max_iter": 500_000}
    if lower == "gurobi":
        return {"time_limit": time_limit, "threads": 0, "seed": 0}
    if lower.startswith("cvxpy:"):
        return {
            "solver_options": {
                "tol_gap_abs": tolerance,
                "tol_feas": tolerance,
            }
        }
    return {}


def _summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for record in records:
        if record["status"].__str__().startswith("skipped:"):
            continue
        groups.setdefault((int(record["n_assets"]), str(record["backend"])), []).append(record)

    rows: list[dict[str, object]] = []
    for (size, backend), runs in sorted(groups.items()):
        runtimes = np.asarray([float(run["runtime_seconds"]) for run in runs])
        objectives = np.asarray(
            [float(run["objective"]) for run in runs if bool(run["feasible"])]
        )
        rows.append(
            {
                "n_assets": size,
                "backend": backend,
                "runs": len(runs),
                "success_rate": np.mean([bool(run["success"]) for run in runs]),
                "feasible_rate": np.mean([bool(run["feasible"]) for run in runs]),
                "median_runtime_seconds": float(np.median(runtimes)),
                "runtime_q1_seconds": float(np.quantile(runtimes, 0.25)),
                "runtime_q3_seconds": float(np.quantile(runtimes, 0.75)),
                "median_objective": float(np.median(objectives)) if objectives.size else "",
            }
        )
    return rows


def _plot_scaling(summary: list[dict[str, object]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for backend in sorted({str(row["backend"]) for row in summary}):
        rows = sorted(
            (row for row in summary if row["backend"] == backend),
            key=lambda row: int(row["n_assets"]),
        )
        sizes = np.asarray([int(row["n_assets"]) for row in rows])
        median = np.asarray([float(row["median_runtime_seconds"]) for row in rows])
        q1 = np.asarray([float(row["runtime_q1_seconds"]) for row in rows])
        q3 = np.asarray([float(row["runtime_q3_seconds"]) for row in rows])
        ax.plot(sizes, median, marker="o", linewidth=1.8, label=backend)
        ax.fill_between(sizes, q1, q3, alpha=0.16)
    ax.set_xlabel("Number of assets")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Continuous solver scaling (median and interquartile range)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50, 100, 250])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["scipy", "osqp", "cvxpy:CLARABEL", "gurobi"],
    )
    parser.add_argument("--instance-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=5)
    parser.add_argument("--n-factors", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--strict-optional", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/scaling/scaling_benchmark.csv",
    )
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    records: list[dict[str, object]] = []
    for size in args.sizes:
        for instance_seed in args.instance_seeds:
            problem = generate_factor_universe(
                n_assets=size,
                n_groups=min(args.n_groups, size),
                n_factors=args.n_factors,
                seed=instance_seed,
            )
            for backend in args.backends:
                for repetition in range(args.repetitions):
                    try:
                        result = solve_continuous(
                            problem,
                            Preferences(),
                            backend=backend,
                            **_solver_options(backend, args.tolerance, args.time_limit),
                        )
                        records.append(
                            {
                                "n_assets": size,
                                "n_groups": problem.num_groups,
                                "n_factors": args.n_factors,
                                "instance_seed": instance_seed,
                                "repetition": repetition,
                                "backend": backend,
                                "method": result.method,
                                "status": result.status,
                                "success": result.success,
                                "optimal": result.optimal,
                                "feasible": result.feasible,
                                "objective": result.objective,
                                "runtime_seconds": result.runtime,
                                "model_build_seconds": result.metadata.get(
                                    "model_build_seconds", ""
                                ),
                                "solve_seconds": result.metadata.get("solve_seconds", ""),
                                "max_violation": result.max_violation,
                                "expected_return": result.metrics.get("expected_return", ""),
                                "volatility": result.metrics.get("volatility", ""),
                            }
                        )
                    except SolverUnavailableError as exc:
                        if args.strict_optional:
                            raise
                        records.append(
                            {
                                "n_assets": size,
                                "n_groups": problem.num_groups,
                                "n_factors": args.n_factors,
                                "instance_seed": instance_seed,
                                "repetition": repetition,
                                "backend": backend,
                                "method": "",
                                "status": f"skipped: {exc}",
                                "success": False,
                                "optimal": False,
                                "feasible": False,
                                "objective": "",
                                "runtime_seconds": "",
                                "model_build_seconds": "",
                                "solve_seconds": "",
                                "max_violation": "",
                                "expected_return": "",
                                "volatility": "",
                            }
                        )
                        break

    _write_csv(args.output, records)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary = _summary(records)
    _write_csv(summary_path, summary)
    plot_path = args.output.with_name(f"{args.output.stem}_runtime.png")
    _plot_scaling(summary, plot_path)
    manifest_path = write_artifact_manifest(
        {"runs": args.output, "summary": summary_path, "runtime_plot": plot_path},
        args.output.parent,
    )
    print(f"Saved {len(records)} runs to {args.output}")
    print(f"Saved aggregate scaling results to {summary_path}")
    print(f"Saved runtime plot to {plot_path}")
    print(f"Saved checksums to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
