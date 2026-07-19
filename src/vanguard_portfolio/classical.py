"""Public facade and classical-solver benchmark harness."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .classical_continuous import solve_continuous
from .classical_discrete import solve_discrete
from .metrics import objective_gap
from .schemas import (
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverUnavailableError,
)


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


@dataclass
class BenchmarkReport:
    problem_name: str
    preferences: Preferences
    units: int
    results: list[SolveResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

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

    def raw_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for result in self.results:
            record = result.to_record()
            reference = self.reference_objective(result.model_type)
            if reference is not None and np.isfinite(result.objective):
                absolute, relative = objective_gap(result.objective, reference)
                record["absolute_objective_gap"] = absolute
                record["relative_objective_gap"] = relative
            else:
                record["absolute_objective_gap"] = np.nan
                record["relative_objective_gap"] = np.nan
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
            reference = self.reference_objective(model_type)
            best = float(np.min(objectives)) if objectives.size else np.nan
            absolute_gap, relative_gap = (
                objective_gap(best, reference)
                if reference is not None and np.isfinite(best)
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
                    "absolute_gap_to_reference": absolute_gap,
                    "relative_gap_to_reference": relative_gap,
                    "median_runtime_seconds": float(np.median(runtimes)),
                    "runtime_q1_seconds": float(np.quantile(runtimes, 0.25)),
                    "runtime_q3_seconds": float(np.quantile(runtimes, 0.75)),
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
    continuous_backends: Iterable[str] = ("scipy", "osqp", "gurobi"),
    discrete_backends: Iterable[str] = (
        "enumeration",
        "local_search",
        "annealing",
        "gurobi",
    ),
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    annealing_iterations: int = 20_000,
    missing_optional: str = "skip",
) -> BenchmarkReport:
    """Run solver backends with identical data, objective, and hard constraints."""
    preferences = preferences or Preferences()
    report = BenchmarkReport("portfolio", preferences, units)

    def unavailable(key: str, exc: SolverUnavailableError) -> None:
        if missing_optional == "error":
            raise exc
        report.skipped[key] = str(exc)

    for backend in continuous_backends:
        key = f"continuous:{backend}"
        try:
            report.results.append(
                solve_continuous(problem, preferences, backend=str(backend))
            )
        except SolverUnavailableError as exc:
            unavailable(key, exc)

    for backend in discrete_backends:
        backend = str(backend)
        key = f"discrete:{backend}"
        if backend.lower() in {"annealing", "anneal", "simulated_annealing"}:
            for seed in seeds:
                try:
                    report.results.append(
                        solve_discrete(
                            problem,
                            preferences,
                            units=units,
                            backend=backend,
                            seed=int(seed),
                            n_iterations=annealing_iterations,
                        )
                    )
                except SolverUnavailableError as exc:
                    unavailable(key, exc)
                    break
        else:
            try:
                report.results.append(
                    solve_discrete(problem, preferences, units=units, backend=backend)
                )
            except SolverUnavailableError as exc:
                unavailable(key, exc)
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


def write_benchmark_artifacts(report: BenchmarkReport, directory: str | Path) -> dict[str, Path]:
    """Write raw runs, aggregate summary, availability, and a Markdown report."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "benchmark_runs.csv"
    summary_path = destination / "benchmark_summary.csv"
    metadata_path = destination / "benchmark_metadata.json"
    markdown_path = destination / "classical_baseline_report.md"
    raw = report.raw_records()
    summary = report.summary_records()
    _write_csv(raw_path, raw)
    _write_csv(summary_path, summary)
    metadata_path.write_text(
        json.dumps(
            {
                "problem": report.problem_name,
                "units": report.units,
                "preferences": report.preferences.to_dict(),
                "continuous_reference": report.reference_objective("continuous"),
                "discrete_reference": report.reference_objective("discrete"),
                "skipped": report.skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Classical baseline benchmark",
        "",
        "All objective values use the same minimization convention. A zero breach count is required.",
        "Wall-clock runtime includes Python model construction and solver execution.",
        "",
        "| model | method | runs | feasible | best objective | gap | median runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model_type']} | {row['method']} | {row['runs']} | "
            f"{row['feasible_rate']:.0%} | {row['best_objective']:.8f} | "
            f"{row['absolute_gap_to_reference']:.3e} | "
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
            "- Heuristic gaps are measured against an exact/optimal reference, never against another heuristic.",
            "- A missing commercial license is recorded as skipped; it is never presented as a failed model.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "runs": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "report": markdown_path,
    }


__all__ = [
    "BenchmarkReport",
    "PRESETS",
    "PortfolioProblem",
    "Preferences",
    "SolveResult",
    "benchmark_solvers",
    "preferences_from_config",
    "solve_continuous",
    "solve_discrete",
    "write_benchmark_artifacts",
]

