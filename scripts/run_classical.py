#!/usr/bin/env python3
"""Run the complete classical benchmark and create tables and figures."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vanguard_portfolio.classical import (  # noqa: E402
    benchmark_solvers,
    preferences_from_config,
    solve_continuous,
    write_artifact_manifest,
    write_benchmark_artifacts,
)
from vanguard_portfolio.data_generation import (  # noqa: E402
    generate_factor_universe,
    generate_synthetic_universe,
    load_problem,
    save_problem,
)
from vanguard_portfolio.plotting import generate_benchmark_plots  # noqa: E402
from vanguard_portfolio.schemas import Preferences  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _print_summary(report) -> None:
    print("\nClassical solver comparison")
    print("=" * 104)
    print(
        f"{'model':<11} {'method':<27} {'runs':>4} {'feasible':>9} "
        f"{'best objective':>16} {'gap':>12} {'median s':>11}"
    )
    print("-" * 104)
    for row in report.summary_records():
        print(
            f"{row['model_type']:<11} {row['method']:<27} {row['runs']:>4d} "
            f"{row['feasible_rate']:>8.0%} {row['best_objective']:>16.9f} "
            f"{row['absolute_gap_to_reference']:>12.3e} "
            f"{row['median_runtime_seconds']:>11.6f}"
        )
    if report.skipped:
        print("\nOptional solvers skipped:")
        for backend, reason in sorted(report.skipped.items()):
            print(f"  - {backend}: {reason}")


def _build_problem(problem_config: dict):
    source = str(problem_config.get("source", "synthetic"))
    if source == "synthetic":
        problem = generate_synthetic_universe()
        save_problem(problem, ROOT / "data/synthetic/synthetic_universe.json")
    elif source == "factor":
        problem = generate_factor_universe(
            n_assets=int(problem_config.get("n_assets", 100)),
            n_groups=int(problem_config.get("n_groups", 5)),
            n_factors=int(problem_config.get("n_factors", 4)),
            seed=int(problem_config.get("seed", 0)),
        )
    else:
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        problem = load_problem(candidate)

    overrides = dict(problem_config.get("overrides", {}))
    allowed = {"target_return", "max_turnover"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported problem override fields: {sorted(unknown)}")
    if overrides:
        problem = replace(problem, **overrides)
    return problem


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="fail instead of skipping an unavailable OSQP/CVXPY/Gurobi backend",
    )
    args = parser.parse_args(argv)
    config = _load_config(args.config)

    problem_config = config.get("problem", {})
    problem = _build_problem(problem_config)

    preferences = preferences_from_config(config.get("preferences", {}))
    discrete = config.get("discrete", {})
    solver_config = config.get("solvers", {})
    benchmark_config = config.get("benchmark", {})
    output_config = config.get("outputs", {})
    output = args.output or ROOT / output_config.get("directory", "results")
    if not output.is_absolute():
        output = ROOT / output

    report = benchmark_solvers(
        problem,
        preferences,
        units=int(discrete.get("units", 20)),
        continuous_backends=solver_config.get("continuous", ["scipy"]),
        discrete_backends=solver_config.get("discrete", ["enumeration"]),
        seeds=discrete.get("seeds", [0]),
        annealing_iterations=int(discrete.get("annealing_iterations", 20_000)),
        missing_optional="error"
        if args.strict_optional
        else solver_config.get("missing_optional", "skip"),
        require_feasible_results=bool(
            benchmark_config.get("require_feasible_results", False)
        ),
        problem_name=str(problem_config.get("name", problem_config.get("source", "portfolio"))),
    )
    resolved_config = deepcopy(config)
    resolved_config["_runtime"] = {
        "config_path": str(args.config.resolve()),
        "output_directory": str(output.resolve()),
        "strict_optional": bool(args.strict_optional),
        "effective_missing_optional": "error"
        if args.strict_optional
        else solver_config.get("missing_optional", "skip"),
    }
    artifacts = write_benchmark_artifacts(
        report,
        output,
        resolved_config=resolved_config,
    )

    sweep_results = None
    risk_values = None
    if output_config.get("make_risk_aversion_sweep", False):
        risk_values = output_config.get(
            "risk_values", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        )
        sweep_backend = str(output_config.get("sweep_backend", "scipy"))
        sweep_options = dict(output_config.get("sweep_options", {}))
        sweep_results = []
        for risk in risk_values:
            sweep_preferences = Preferences(
                lambda_return=preferences.lambda_return,
                lambda_risk=risk,
                lambda_income=preferences.lambda_income,
                lambda_cost=preferences.lambda_cost,
            )
            sweep_results.append(
                solve_continuous(
                    problem,
                    sweep_preferences,
                    backend=sweep_backend,
                    **sweep_options,
                )
            )

    plots = {}
    if output_config.get("make_plots", True):
        plots = generate_benchmark_plots(
            problem,
            report,
            output,
            sweep_results=sweep_results,
            risk_values=risk_values,
        )

    manifest_inputs = {**artifacts, **plots}
    manifest = write_artifact_manifest(manifest_inputs, output)
    artifacts["manifest"] = manifest

    _print_summary(report)
    print(f"\nTables/report: {_display_path(artifacts['report'])}")
    if plots:
        print(f"Graphics: {len(plots)} files in {_display_path(output)}")
    print(f"Manifest: {_display_path(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
