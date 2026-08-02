#!/usr/bin/env python3
"""Run the constraint-safe hybrid portfolio optimizer from YAML."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vanguard_portfolio.classical import preferences_from_config
from vanguard_portfolio.data_generation import (
    generate_backtest_returns,
    generate_factor_universe,
    generate_return_scenarios,
    generate_synthetic_universe,
    load_problem,
)
from vanguard_portfolio.hybrid import HybridConfig, run_hybrid_optimizer
from vanguard_portfolio.presentation import write_hybrid_artifacts
from vanguard_portfolio.quantum_solver import XYQAOAConfig
from vanguard_portfolio.schemas import PortfolioConstraints, PortfolioProblem


def _problem(config: dict[str, Any]) -> PortfolioProblem:
    source = str(config.get("source", "factor")).lower()
    if source == "synthetic":
        problem = generate_synthetic_universe()
    elif source == "factor":
        problem = generate_factor_universe(
            n_assets=int(config.get("n_assets", 100)),
            n_groups=int(config.get("n_groups", 8)),
            n_factors=int(config.get("n_factors", 6)),
            seed=int(config.get("seed", 0)),
            current_cardinality=(
                None
                if config.get("current_cardinality") is None
                else int(config["current_cardinality"])
            ),
            materialize_covariance=bool(config.get("materialize_covariance", True)),
        )
    elif source in {"json", "file"}:
        problem = load_problem(ROOT / str(config["path"]))
    else:
        raise ValueError(f"unknown problem source {source!r}")
    overrides = dict(config.get("overrides", {}))
    if overrides:
        payload = problem.to_dict()
        payload.update(overrides)
        problem = PortfolioProblem.from_dict(payload)
    return problem


def _asset_indices(values: Any, problem: PortfolioProblem) -> tuple[int, ...] | None:
    if values is None:
        return None
    lookup = {name: index for index, name in enumerate(problem.asset_names)}
    result = []
    for value in values:
        if isinstance(value, str) and not value.isdigit():
            if value not in lookup:
                raise ValueError(f"unknown asset name {value!r}")
            result.append(lookup[value])
        else:
            result.append(int(value))
    return tuple(result)


def _constraints(config: dict[str, Any], problem: PortfolioProblem) -> PortfolioConstraints:
    values = dict(config)
    scenario_count = int(values.pop("scenario_count", 0))
    scenario_seed = int(values.pop("scenario_seed", 0))
    if "eligible_assets" in values:
        values["eligible_assets"] = _asset_indices(values["eligible_assets"], problem)
    if "mandatory_assets" in values:
        values["mandatory_assets"] = _asset_indices(values["mandatory_assets"], problem) or ()
    if "maximum_weights" in values and np.isscalar(values["maximum_weights"]):
        values["maximum_weights"] = np.full(problem.n, float(values["maximum_weights"]))
    if "factor_lower" in values and np.isscalar(values["factor_lower"]):
        values["factor_lower"] = np.full(problem.num_factors, float(values["factor_lower"]))
    if "factor_upper" in values and np.isscalar(values["factor_upper"]):
        values["factor_upper"] = np.full(problem.num_factors, float(values["factor_upper"]))
    if values.get("maximum_cvar") is not None and "scenario_returns" not in values:
        if scenario_count <= 1:
            raise ValueError("maximum_cvar requires scenario_count > 1")
        values["scenario_returns"] = generate_return_scenarios(
            problem, scenario_count, scenario_seed
        )
    return PortfolioConstraints.from_dict(values).validate_for(problem)


def _hybrid_config(config: dict[str, Any]) -> HybridConfig:
    values = dict(config)
    quantum = XYQAOAConfig(**dict(values.pop("quantum", {})))
    return HybridConfig(quantum=quantum, **values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final_hybrid.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-gurobi", action="store_true")
    parser.add_argument(
        "--quantum-backend",
        choices=("subspace", "aer_cpu", "aer_gpu", "ibm_runtime"),
        help="Override the quantum sampler configured in YAML.",
    )
    parser.add_argument(
        "--ibm-backend",
        help="IBM QPU name; required when --quantum-backend=ibm_runtime.",
    )
    parser.add_argument("--quantum-shots", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problem = _problem(dict(config.get("problem", {})))
    preferences = preferences_from_config(dict(config.get("preferences", {})))
    constraints = _constraints(dict(config.get("constraints", {})), problem)
    hybrid = _hybrid_config(dict(config.get("hybrid", {})))
    quantum_updates: dict[str, Any] = {}
    if args.quantum_backend is not None:
        quantum_updates["backend"] = args.quantum_backend
    if args.ibm_backend is not None:
        quantum_updates["ibm_backend"] = args.ibm_backend
    if args.quantum_shots is not None:
        quantum_updates["shots"] = args.quantum_shots
    if quantum_updates:
        hybrid = replace(hybrid, quantum=replace(hybrid.quantum, **quantum_updates))
    hybrid_updates: dict[str, Any] = {}
    if args.window_size is not None:
        hybrid_updates["window_size"] = args.window_size
    if args.iterations is not None:
        hybrid_updates["iterations"] = args.iterations
    if hybrid_updates:
        hybrid = replace(hybrid, **hybrid_updates)
    if args.no_quantum:
        hybrid = replace(hybrid, run_quantum=False, run_penalty_qaoa=False)
    if args.no_gurobi:
        hybrid = replace(hybrid, run_gurobi_reference=False)
    if (
        hybrid.run_quantum
        and hybrid.quantum.backend == "ibm_runtime"
        and not hybrid.quantum.ibm_backend
    ):
        parser.error("IBM Runtime sampling requires --ibm-backend or quantum.ibm_backend")
    configured_output = config.get("outputs", {}).get(
        "directory",
        "results/final_hybrid",
    )
    output = args.output or ROOT / str(configured_output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to replace named artifacts")

    run = run_hybrid_optimizer(problem, preferences, constraints, hybrid)
    backtest_config = dict(config.get("backtest", {}))
    realized = generate_backtest_returns(
        problem,
        periods=int(backtest_config.get("periods", 120)),
        periods_per_year=int(backtest_config.get("periods_per_year", 12)),
        seed=int(backtest_config.get("seed", 20260802)),
    )
    artifacts = write_hybrid_artifacts(run, output, realized_returns=realized)
    print(f"Best method: {run.best.method}")
    print(f"Objective: {run.best.objective:.10g}")
    print(f"Hard-constraint breaches: {run.best.breaches}")
    print(f"Selected assets: {np.count_nonzero(run.best.weights > 1e-8)}")
    print(f"Runtime: {run.runtime:.3f} s")
    print(f"Artifacts: {len(artifacts)} files in {output}")
    if run.skipped:
        print("Skipped optional components:")
        for name, reason in run.skipped.items():
            print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
