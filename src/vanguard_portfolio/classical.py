"""Public facade and fair classical-solver benchmark harness."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .classical_continuous import solve_continuous
from .classical_discrete import solve_discrete
from .metrics import objective_gap
from .schemas import (
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverSkippedError,
    SolverUnavailableError,
)
from .validation import validate_weights


PRESETS: dict[str, Preferences] = {
    "balanced": Preferences(1.0, 5.0, 0.0, 1.0),
    "growth": Preferences(2.0, 3.0, 0.0, 0.5),
    "income": Preferences(0.5, 4.0, 2.0, 1.0),
    "drawdown_control": Preferences(0.5, 15.0, 0.0, 1.0),
    "cost_sensitive": Preferences(1.0, 5.0, 0.0, 10.0),
}


def preferences_from_config(config: dict[str, Any]) -> Preferences:
    values = dict(config)
    preset_name = str(values.pop("preset", "balanced"))
    base = PRESETS.get(preset_name)
    if base is None:
        raise ValueError(f"unknown preference preset {preset_name!r}")
    merged = base.to_dict()
    merged.update({key: value for key, value in values.items() if key in merged})
    return Preferences(**merged)


@dataclass(frozen=True)
class SolverSpec:
    """One backend request with independent repetitions and keyword options."""

    name: str
    repetitions: int = 1
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("solver name must not be empty")
        if int(self.repetitions) <= 0:
            raise ValueError("solver repetitions must be positive")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "repetitions", int(self.repetitions))
        object.__setattr__(self, "options", dict(self.options))


def normalize_solver_specs(
    values: Iterable[str | Mapping[str, Any] | SolverSpec],
) -> list[SolverSpec]:
    """Accept concise strings or detailed YAML-style solver mappings."""
    specs: list[SolverSpec] = []
    for value in values:
        if isinstance(value, SolverSpec):
            specs.append(value)
        elif isinstance(value, str):
            specs.append(SolverSpec(value))
        elif isinstance(value, Mapping):
            payload = dict(value)
            name = payload.pop("name", payload.pop("backend", None))
            if name is None:
                raise ValueError("a solver mapping requires a 'name' field")
            repetitions = payload.pop("repetitions", 1)
            options = dict(payload.pop("options", {}))
            if payload:
                unknown = ", ".join(sorted(payload))
                raise ValueError(f"unknown solver specification fields: {unknown}")
            specs.append(SolverSpec(str(name), repetitions=int(repetitions), options=options))
        else:
            raise TypeError(f"unsupported solver specification {value!r}")
    return specs


@dataclass
class BenchmarkReport:
    problem_name: str
    preferences: Preferences
    units: int
    problem: PortfolioProblem | None = None
    results: list[SolveResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    requested_solvers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def reference_objective(self, model_type: str) -> float | None:
        references = [
            result.objective
            for result in self.results
            if result.model_type == model_type
            and result.success
            and result.feasible
            and result.optimal
        ]
        return float(min(references)) if references else None

    def certified_lower_bound(self, model_type: str) -> float | None:
        """Return the tightest available global lower bound for minimization."""
        bounds: list[float] = []
        for result in self.results:
            if result.model_type != model_type:
                continue
            if result.success and result.feasible and result.optimal:
                bounds.append(float(result.objective))
            native_bound = result.metadata.get("best_bound")
            if native_bound is not None and np.isfinite(native_bound):
                bounds.append(float(native_bound))
        return float(max(bounds)) if bounds else None

    def raw_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for result in self.results:
            record = result.to_record()
            reference = self.reference_objective(result.model_type)
            certified_bound = self.certified_lower_bound(result.model_type)
            if reference is not None and np.isfinite(result.objective):
                absolute, relative = objective_gap(result.objective, reference)
                record["absolute_objective_gap"] = absolute
                record["relative_objective_gap"] = relative
            else:
                record["absolute_objective_gap"] = np.nan
                record["relative_objective_gap"] = np.nan
            if certified_bound is not None and np.isfinite(result.objective):
                absolute, relative = objective_gap(result.objective, certified_bound)
                record["absolute_gap_to_certified_bound"] = absolute
                record["relative_gap_to_certified_bound"] = relative
            else:
                record["absolute_gap_to_certified_bound"] = np.nan
                record["relative_gap_to_certified_bound"] = np.nan
            records.append(record)
        return records

    def summary_records(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[SolveResult]] = {}
        for result in self.results:
            groups.setdefault((result.model_type, result.method), []).append(result)

        summary: list[dict[str, Any]] = []
        for (model_type, method), runs in sorted(groups.items()):
            valid = [run for run in runs if run.success and run.feasible]
            objectives = np.asarray([run.objective for run in valid], dtype=float)
            runtimes = np.asarray([run.runtime for run in runs], dtype=float)
            build_times = np.asarray(
                [
                    run.metadata.get("model_build_seconds", np.nan)
                    for run in runs
                ],
                dtype=float,
            )
            solve_times = np.asarray(
                [run.metadata.get("solve_seconds", np.nan) for run in runs],
                dtype=float,
            )
            certified_bounds = np.asarray(
                [
                    run.metadata.get(
                        "best_bound", run.objective if run.optimal else np.nan
                    )
                    for run in runs
                ],
                dtype=float,
            )
            reported_mip_gaps = np.asarray(
                [run.metadata.get("reported_mip_gap", np.nan) for run in runs],
                dtype=float,
            )
            reference = self.reference_objective(model_type)
            certified_bound = self.certified_lower_bound(model_type)
            best = float(np.min(objectives)) if objectives.size else np.nan
            absolute_gap, relative_gap = (
                objective_gap(best, reference)
                if reference is not None and np.isfinite(best)
                else (np.nan, np.nan)
            )
            bound_gap, relative_bound_gap = (
                objective_gap(best, certified_bound)
                if certified_bound is not None and np.isfinite(best)
                else (np.nan, np.nan)
            )
            summary.append(
                {
                    "model_type": model_type,
                    "method": method,
                    "runs": len(runs),
                    "success_rate": len(valid) / len(runs),
                    "feasible_rate": sum(run.feasible for run in runs) / len(runs),
                    "optimal_status_rate": sum(run.optimal for run in runs) / len(runs),
                    "best_objective": best,
                    "median_objective": float(np.median(objectives)) if objectives.size else np.nan,
                    "objective_std": float(np.std(objectives)) if objectives.size else np.nan,
                    "absolute_gap_to_reference": absolute_gap,
                    "relative_gap_to_reference": relative_gap,
                    "absolute_gap_to_certified_bound": bound_gap,
                    "relative_gap_to_certified_bound": relative_bound_gap,
                    "median_runtime_seconds": float(np.median(runtimes)),
                    "runtime_q1_seconds": float(np.quantile(runtimes, 0.25)),
                    "runtime_q3_seconds": float(np.quantile(runtimes, 0.75)),
                    "median_model_build_seconds": float(
                        np.nanmedian(build_times)
                    )
                    if np.any(np.isfinite(build_times))
                    else np.nan,
                    "median_solve_seconds": float(np.nanmedian(solve_times))
                    if np.any(np.isfinite(solve_times))
                    else np.nan,
                    "best_certified_bound": float(np.nanmax(certified_bounds))
                    if np.any(np.isfinite(certified_bounds))
                    else np.nan,
                    "median_reported_mip_gap": float(np.nanmedian(reported_mip_gaps))
                    if np.any(np.isfinite(reported_mip_gaps))
                    else np.nan,
                    "median_expected_return": float(
                        np.median([run.metrics["expected_return"] for run in valid])
                    )
                    if valid
                    else np.nan,
                    "median_volatility": float(
                        np.median([run.metrics["volatility"] for run in valid])
                    )
                    if valid
                    else np.nan,
                    "median_turnover": float(
                        np.median([run.metrics["turnover"] for run in valid])
                    )
                    if valid
                    else np.nan,
                }
            )
        return summary


def benchmark_solvers(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    continuous_backends: Iterable[str | Mapping[str, Any] | SolverSpec] = (
        "scipy",
        "osqp",
        "gurobi",
    ),
    discrete_backends: Iterable[str | Mapping[str, Any] | SolverSpec] = (
        "enumeration",
        "local_search",
        "annealing",
        "gurobi",
    ),
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    annealing_iterations: int = 20_000,
    missing_optional: str = "skip",
    require_feasible_results: bool = False,
    problem_name: str = "portfolio",
) -> BenchmarkReport:
    """Run solver backends with identical data, objective, and hard constraints."""
    preferences = preferences or Preferences()
    if missing_optional not in {"skip", "error"}:
        raise ValueError("missing_optional must be 'skip' or 'error'")
    continuous_specs = normalize_solver_specs(continuous_backends)
    discrete_specs = normalize_solver_specs(discrete_backends)
    report = BenchmarkReport(
        problem_name,
        preferences,
        units,
        problem=problem,
        requested_solvers={
            "continuous": [
                {
                    "name": spec.name,
                    "repetitions": spec.repetitions,
                    "options": spec.options,
                }
                for spec in continuous_specs
            ],
            "discrete": [
                {
                    "name": spec.name,
                    "repetitions": spec.repetitions,
                    "options": spec.options,
                }
                for spec in discrete_specs
            ],
        },
    )
    used_run_ids: set[str] = set()

    def unavailable(key: str, exc: SolverUnavailableError | SolverSkippedError) -> None:
        if missing_optional == "error":
            raise exc
        report.skipped[key] = str(exc)

    def append_result(result: SolveResult, repetition: int) -> None:
        result.repetition = int(repetition)
        seed_label = "none" if result.seed is None else str(result.seed)
        base = (
            f"{result.model_type}__{result.method}__"
            f"rep{repetition:03d}__seed{seed_label}"
        )
        run_id = base
        suffix = 1
        while run_id in used_run_ids:
            suffix += 1
            run_id = f"{base}__run{suffix:02d}"
        used_run_ids.add(run_id)
        result.run_id = run_id
        report.results.append(result)
        if require_feasible_results and not result.feasible:
            raise RuntimeError(
                f"configured solver {result.method!r} returned an infeasible result: "
                f"{result.status}"
            )

    for spec in continuous_specs:
        key = f"continuous:{spec.name}"
        for repetition in range(spec.repetitions):
            try:
                result = solve_continuous(
                    problem,
                    preferences,
                    backend=spec.name,
                    **dict(spec.options),
                )
                append_result(result, repetition)
            except (SolverUnavailableError, SolverSkippedError) as exc:
                unavailable(key, exc)
                break

    seed_values = [int(seed) for seed in seeds]
    if any(
        spec.name.lower() in {"annealing", "anneal", "simulated_annealing"}
        for spec in discrete_specs
    ) and not seed_values:
        raise ValueError("at least one seed is required when annealing is configured")
    for spec in discrete_specs:
        backend = spec.name
        key = f"discrete:{backend}"
        is_annealing = backend.lower() in {
            "annealing",
            "anneal",
            "simulated_annealing",
        }
        failed = False
        for repetition in range(spec.repetitions):
            if is_annealing:
                for seed in seed_values:
                    options = dict(spec.options)
                    options.setdefault("n_iterations", annealing_iterations)
                    options["seed"] = seed
                    try:
                        result = solve_discrete(
                            problem,
                            preferences,
                            units=units,
                            backend=backend,
                            **options,
                        )
                        append_result(result, repetition)
                    except (SolverUnavailableError, SolverSkippedError) as exc:
                        unavailable(key, exc)
                        failed = True
                        break
                if failed:
                    break
            else:
                try:
                    result = solve_discrete(
                        problem,
                        preferences,
                        units=units,
                        backend=backend,
                        **dict(spec.options),
                    )
                    append_result(result, repetition)
                except (SolverUnavailableError, SolverSkippedError) as exc:
                    unavailable(key, exc)
                    failed = True
                    break
        if failed:
            continue
    return report


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _environment_metadata() -> dict[str, Any]:
    distributions = [
        "vanguard-multi-asset-portfolio",
        "numpy",
        "scipy",
        "PyYAML",
        "matplotlib",
        "osqp",
        "cvxpy",
        "clarabel",
        "gurobipy",
        "pyscipopt",
    ]
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }


def _allocation_records(report: BenchmarkReport) -> list[dict[str, Any]]:
    if report.problem is None:
        return []
    problem = report.problem
    records: list[dict[str, Any]] = []
    for result in report.results:
        lot_size = problem.budget / result.units if result.units is not None else None
        for index, (asset, weight) in enumerate(zip(problem.asset_names, result.weights)):
            records.append(
                {
                    "run_id": result.run_id,
                    "model_type": result.model_type,
                    "method": result.method,
                    "repetition": result.repetition,
                    "seed": result.seed,
                    "asset_index": index,
                    "asset": asset,
                    "group": problem.group_names[problem.asset_group[index]],
                    "current_weight": float(problem.w0[index]),
                    "optimized_weight": float(weight),
                    "weight_change": float(weight - problem.w0[index]),
                    "integer_lots": int(round(weight / lot_size))
                    if lot_size is not None and np.isfinite(weight)
                    else None,
                    "asset_lower": float(problem.lower[index]),
                    "asset_upper": float(problem.upper[index]),
                }
            )
    return records


def _constraint_records(report: BenchmarkReport) -> list[dict[str, Any]]:
    if report.problem is None:
        return []
    records: list[dict[str, Any]] = []
    for result in report.results:
        validation = validate_weights(
            result.weights,
            report.problem,
            units=result.units if result.model_type == "discrete" else None,
        )
        for check in validation.checks:
            records.append(
                {
                    "run_id": result.run_id,
                    "model_type": result.model_type,
                    "method": result.method,
                    "repetition": result.repetition,
                    "seed": result.seed,
                    "constraint": check.name,
                    "sense": check.sense,
                    "lhs": check.lhs,
                    "rhs": check.rhs,
                    "slack": check.slack,
                    "violation": check.violation,
                }
            )
    return records


def write_benchmark_artifacts(
    report: BenchmarkReport,
    directory: str | Path,
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a complete, auditable benchmark snapshot."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "benchmark_runs.csv"
    summary_path = destination / "benchmark_summary.csv"
    metadata_path = destination / "benchmark_metadata.json"
    markdown_path = destination / "classical_baseline_report.md"
    allocations_path = destination / "allocation_weights.csv"
    diagnostics_path = destination / "solver_diagnostics.json"
    constraints_path = destination / "constraint_checks.csv"
    problem_path = destination / "problem.json"
    config_path = destination / "resolved_config.yaml"
    raw = report.raw_records()
    summary = report.summary_records()
    _write_csv(raw_path, raw)
    _write_csv(summary_path, summary)
    _write_csv(allocations_path, _allocation_records(report))
    _write_csv(constraints_path, _constraint_records(report))

    problem_payload = report.problem.to_dict() if report.problem is not None else None
    canonical_problem = json.dumps(problem_payload, sort_keys=True, separators=(",", ":"))
    problem_fingerprint = hashlib.sha256(canonical_problem.encode("utf-8")).hexdigest()
    if problem_payload is not None:
        problem_path.write_text(
            json.dumps(problem_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    diagnostics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": result.run_id,
                        "model_type": result.model_type,
                        "method": result.method,
                        "repetition": result.repetition,
                        "seed": result.seed,
                        "status": result.status,
                        "success": result.success,
                        "optimal": result.optimal,
                        "feasible": result.feasible,
                        "metadata": _jsonable(result.metadata),
                    }
                    for result in report.results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if resolved_config is not None:
        config_path.write_text(
            yaml.safe_dump(_jsonable(dict(resolved_config)), sort_keys=False),
            encoding="utf-8",
        )

    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "problem": report.problem_name,
                "problem_fingerprint_sha256": problem_fingerprint,
                "n_assets": report.problem.n if report.problem is not None else None,
                "n_groups": report.problem.num_groups if report.problem is not None else None,
                "units": report.units,
                "preferences": report.preferences.to_dict(),
                "continuous_reference": report.reference_objective("continuous"),
                "discrete_reference": report.reference_objective("discrete"),
                "continuous_certified_lower_bound": report.certified_lower_bound(
                    "continuous"
                ),
                "discrete_certified_lower_bound": report.certified_lower_bound("discrete"),
                "requested_solvers": _jsonable(report.requested_solvers),
                "skipped": report.skipped,
                "environment": _environment_metadata(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Classical baseline benchmark",
        "",
        "All objective values use the same minimization convention. "
        "A zero breach count is required.",
        "Wall-clock runtime includes Python model construction and solver execution.",
        "",
        "| model | method | runs | feasible | best objective | reference/bound gap | "
        "median runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        displayed_gap = row["absolute_gap_to_reference"]
        if not np.isfinite(displayed_gap):
            displayed_gap = row["absolute_gap_to_certified_bound"]
        lines.append(
            f"| {row['model_type']} | {row['method']} | {row['runs']} | "
            f"{row['feasible_rate']:.0%} | {row['best_objective']:.8f} | "
            f"{displayed_gap:.3e} | "
            f"{row['median_runtime_seconds']:.6f} |"
        )
    if report.skipped:
        lines.extend(["", "## Optional backends not run", ""])
        lines.extend(f"- `{name}`: {reason}" for name, reason in sorted(report.skipped.items()))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The continuous optimum is a relaxation and can be better than the discrete optimum.",
            "- Enumeration and an optimal MIQP solve should agree at the same lot resolution.",
            "- Heuristic gaps are measured against an exact/optimal reference, "
            "never against another heuristic.",
            "- A missing commercial license is recorded as skipped; it is never "
            "presented as a failed model.",
            "",
            "## Auditable files",
            "",
            "- `allocation_weights.csv` contains every numeric asset weight and "
            "discrete lot count.",
            "- `constraint_checks.csv` contains every independently recomputed "
            "hard-constraint check.",
            "- `solver_diagnostics.json` preserves native iterations, bounds, "
            "nodes, gaps, and timing phases.",
            "- `problem.json` and `resolved_config.yaml` reconstruct the exact run inputs.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {
        "runs": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "report": markdown_path,
        "allocations": allocations_path,
        "diagnostics": diagnostics_path,
        "constraints": constraints_path,
    }
    if problem_payload is not None:
        artifacts["problem"] = problem_path
    if resolved_config is not None:
        artifacts["config"] = config_path
    return artifacts


def write_artifact_manifest(
    artifacts: Mapping[str, str | Path],
    destination: str | Path,
) -> Path:
    """Write sizes and SHA-256 checksums after tables and plots are complete."""
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_path / "artifact_manifest.json"
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for label, raw_path in sorted(artifacts.items()):
        path = Path(raw_path)
        if not path.is_file() or path in seen or path == manifest_path:
            continue
        seen.add(path)
        records.append(
            {
                "label": label,
                "path": str(path.relative_to(destination_path)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "BenchmarkReport",
    "PRESETS",
    "PortfolioProblem",
    "Preferences",
    "SolveResult",
    "SolverSpec",
    "benchmark_solvers",
    "normalize_solver_specs",
    "preferences_from_config",
    "solve_continuous",
    "solve_discrete",
    "write_artifact_manifest",
    "write_benchmark_artifacts",
]
