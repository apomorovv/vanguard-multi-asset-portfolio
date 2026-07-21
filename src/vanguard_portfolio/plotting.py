"""Publication-ready graphics for the classical benchmark."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .classical import BenchmarkReport
from .schemas import PortfolioProblem, SolveResult
from .validation import signed_constraint_slacks


COLORS = ["#0B5CAD", "#00A6A6", "#F28E2B", "#7B61A8", "#D1495B", "#5C677D"]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _best_by_method(results: Iterable[SolveResult]) -> list[SolveResult]:
    best: dict[str, SolveResult] = {}
    for result in results:
        if not result.success or not result.feasible:
            continue
        if result.method not in best or result.objective < best[result.method].objective:
            best[result.method] = result
    return list(best.values())


def plot_allocations(
    problem: PortfolioProblem,
    results: Iterable[SolveResult],
    path: str | Path,
) -> Path:
    selected = _best_by_method(results)
    series = [("current", problem.w0)] + [(result.method, result.weights) for result in selected]
    max_assets = 30
    if problem.n > max_assets:
        stacked = np.vstack([values for _, values in series])
        importance = np.max(stacked, axis=0)
        chosen = np.argsort(importance)[-max_assets:]
        chosen = chosen[np.argsort(importance[chosen])[::-1]]
        labels = [problem.asset_names[index] for index in chosen] + ["Other assets"]
        compact_series = []
        for label, values in series:
            displayed = values[chosen]
            other = float(np.sum(values) - np.sum(displayed))
            compact_series.append((label, np.concatenate((displayed, [other]))))
        series = compact_series
    else:
        labels = list(problem.asset_names)

    x = np.arange(len(labels))
    width = min(0.8 / len(series), 0.22)
    fig, ax = plt.subplots(figsize=(min(16.0, max(9.0, len(labels) * 0.48)), 5.4))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for j, ((label, values), offset) in enumerate(zip(series, offsets)):
        color = "#AAB2BD" if j == 0 else COLORS[(j - 1) % len(COLORS)]
        ax.bar(x + offset, values, width, label=label.replace("_", " "), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Current and optimized allocations")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, len(series)), fontsize=8)
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_risk_return(results: Iterable[SolveResult], path: str | Path) -> Path:
    selected = [result for result in results if result.success and result.feasible]
    methods = sorted({result.method for result in selected})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for index, method in enumerate(methods):
        group = [result for result in selected if result.method == method]
        ax.scatter(
            [result.metrics["volatility"] for result in group],
            [result.metrics["expected_return"] for result in group],
            label=method.replace("_", " "),
            color=COLORS[index % len(COLORS)],
            marker="o" if group[0].model_type == "continuous" else "s",
            s=58,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
    # Coincident points are an important result (several solvers found the same
    # allocation), but otherwise look like a missing series. Label those groups.
    locations: dict[tuple[float, float], set[str]] = {}
    for result in selected:
        key = (
            round(result.metrics["volatility"], 10),
            round(result.metrics["expected_return"], 10),
        )
        locations.setdefault(key, set()).add(result.method)
    for (volatility_value, return_value), coincident in locations.items():
        if len(coincident) > 1:
            ax.annotate(
                f"{len(coincident)} methods coincide",
                (volatility_value, return_value),
                xytext=(-8, -18),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color="#444444",
            )
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Risk-return comparison")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_runtime(report: BenchmarkReport, path: str | Path) -> Path:
    summary = report.summary_records()
    labels = [
        f"{row['model_type'][0].upper()}: {row['method'].replace('_', ' ')}" for row in summary
    ]
    median = np.asarray([row["median_runtime_seconds"] for row in summary])
    q1 = np.asarray([row["runtime_q1_seconds"] for row in summary])
    q3 = np.asarray([row["runtime_q3_seconds"] for row in summary])
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(8.0, len(summary) * 1.15), 4.8))
    errors = np.vstack([np.maximum(median - q1, 0), np.maximum(q3 - median, 0)])
    ax.bar(x, median, color=[COLORS[i % len(COLORS)] for i in x], yerr=errors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Wall-clock seconds (log scale)")
    ax.set_yscale("log")
    ax.set_title("Solver runtime: model construction plus solve")
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_objective_gap(report: BenchmarkReport, path: str | Path) -> Path:
    summary = report.summary_records()
    labels = [
        f"{row['model_type'][0].upper()}: {row['method'].replace('_', ' ')}" for row in summary
    ]
    gap_values = []
    for row in summary:
        value = float(row["absolute_gap_to_reference"])
        if not np.isfinite(value):
            value = float(row["absolute_gap_to_certified_bound"])
        gap_values.append(max(value, 0.0) if np.isfinite(value) else np.nan)
    gaps = np.asarray(gap_values)
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(8.0, len(summary) * 1.15), 4.8))
    ax.bar(x, gaps, color=[COLORS[i % len(COLORS)] for i in x])
    ax.scatter(x, gaps, color="#1F2933", s=18, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Best objective - optimal reference / certified lower bound")
    ax.set_title("Certified objective gap within each model class")
    ax.grid(axis="y", alpha=0.25)
    finite_gaps = gaps[np.isfinite(gaps)]
    if (
        finite_gaps.size == gaps.size
        and finite_gaps.size
        and np.max(np.abs(finite_gaps)) < 1e-12
    ):
        ax.set_ylim(-0.1, 1.0)
        ax.set_yticks([0.0])
        ax.text(
            0.5,
            0.68,
            "All tested methods matched their reference objective",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#217A67",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E8F5F1", "edgecolor": "#75B8A7"},
        )
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_correlation(problem: PortfolioProblem, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(problem.corr, cmap="RdBu_r", vmin=-1, vmax=1)
    if problem.n <= 60:
        ax.set_xticks(np.arange(problem.n))
        ax.set_yticks(np.arange(problem.n))
        ax.set_xticklabels(problem.asset_names, rotation=40, ha="right")
        ax.set_yticklabels(problem.asset_names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{problem.n} assets (labels hidden for readability)")
    ax.set_title("Asset return correlation")
    fig.colorbar(image, ax=ax, label="Correlation", shrink=0.85)
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_constraint_slacks(
    problem: PortfolioProblem,
    result: SolveResult,
    path: str | Path,
) -> Path:
    slacks = signed_constraint_slacks(result.weights, problem)
    # Budget is an equality and always has zero signed slack; the remaining
    # inequalities show distance to their nearest hard boundary.
    items = [(name, value) for name, value in slacks.items() if name != "budget"]
    max_constraints = 50
    if len(items) > max_constraints:
        # Negative and smallest positive slacks are the violations/binding
        # limits that matter.  Avoid a multi-hundred-inch figure for large n.
        items = sorted(items, key=lambda item: item[1])[:max_constraints]
    labels = [name.replace("_", " ") for name, _ in items]
    values = np.asarray([value for _, value in items])
    y = np.arange(len(items))
    colors = np.where(values >= -1e-8, "#2A9D8F", "#D1495B")
    fig, ax = plt.subplots(figsize=(9.0, max(5.0, len(items) * 0.26)))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Signed slack (negative means violation)")
    ax.set_title(f"Hard-constraint margins: {result.method.replace('_', ' ')}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, Path(path))


def plot_risk_aversion_sweep(
    results: Iterable[SolveResult],
    risk_values: Iterable[float],
    path: str | Path,
) -> Path:
    results = list(results)
    risk_values = list(risk_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = [result.metrics["volatility"] for result in results]
    y = [result.metrics["expected_return"] for result in results]
    ax.plot(x, y, color=COLORS[0], marker="o", linewidth=1.8)
    offsets = [(-6, 10), (-18, -14), (10, 5), (8, 5), (8, 8), (8, -8), (8, -20)]
    for index, (volatility_value, expected, risk) in enumerate(zip(x, y, risk_values)):
        offset = offsets[index] if index < len(offsets) else (6, 6)
        ax.annotate(
            f"λr={risk:g}",
            (volatility_value, expected),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if offset[0] < 0 else "left",
        )
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Continuous mean-variance risk-aversion sweep")
    ax.grid(alpha=0.25)
    ax.margins(x=0.10, y=0.12)
    fig.tight_layout()
    return _save(fig, Path(path))


def generate_benchmark_plots(
    problem: PortfolioProblem,
    report: BenchmarkReport,
    directory: str | Path,
    *,
    sweep_results: Iterable[SolveResult] | None = None,
    risk_values: Iterable[float] | None = None,
) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    successful = [result for result in report.results if result.success and result.feasible]
    paths = {
        "allocations": plot_allocations(
            problem, successful, destination / "allocation_comparison.png"
        ),
        "risk_return": plot_risk_return(successful, destination / "risk_return.png"),
        "runtime": plot_runtime(report, destination / "runtime_comparison.png"),
        "objective_gap": plot_objective_gap(report, destination / "objective_gap.png"),
        "correlation": plot_correlation(problem, destination / "correlation_heatmap.png"),
    }
    if successful:
        best_continuous = min(
            (result for result in successful if result.model_type == "continuous"),
            key=lambda result: result.objective,
            default=min(successful, key=lambda result: result.objective),
        )
        paths["constraint_slacks"] = plot_constraint_slacks(
            problem, best_continuous, destination / "constraint_slacks.png"
        )
    if sweep_results is not None and risk_values is not None:
        paths["risk_sweep"] = plot_risk_aversion_sweep(
            sweep_results, risk_values, destination / "risk_aversion_sweep.png"
        )
    return paths


__all__ = [
    "generate_benchmark_plots",
    "plot_allocations",
    "plot_constraint_slacks",
    "plot_correlation",
    "plot_objective_gap",
    "plot_risk_aversion_sweep",
    "plot_risk_return",
    "plot_runtime",
]
