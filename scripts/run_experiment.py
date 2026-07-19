#!/usr/bin/env python3
"""Run a reproducible continuous-solver scaling experiment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vanguard_portfolio.classical_continuous import solve_continuous  # noqa: E402
from vanguard_portfolio.data_generation import generate_factor_universe  # noqa: E402
from vanguard_portfolio.schemas import Preferences, SolverUnavailableError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--backends", nargs="+", default=["scipy", "osqp", "gurobi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_benchmark.csv")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    for size in args.sizes:
        problem = generate_factor_universe(
            n_assets=size,
            n_groups=min(5, size),
            seed=args.seed + size,
        )
        for backend in args.backends:
            try:
                result = solve_continuous(problem, Preferences(), backend=backend)
                records.append(
                    {
                        "n_assets": size,
                        "backend": backend,
                        "status": result.status,
                        "feasible": result.feasible,
                        "objective": result.objective,
                        "runtime_seconds": result.runtime,
                        "max_violation": result.max_violation,
                    }
                )
            except SolverUnavailableError as exc:
                records.append(
                    {
                        "n_assets": size,
                        "backend": backend,
                        "status": f"skipped: {exc}",
                        "feasible": False,
                        "objective": "",
                        "runtime_seconds": "",
                        "max_violation": "",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


