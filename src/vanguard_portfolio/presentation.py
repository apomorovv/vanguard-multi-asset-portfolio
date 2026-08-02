"""Presentation-ready tables, plots, diagnostics, and human-readable report."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .classical import write_artifact_manifest
from .hybrid import HybridRun
from .metrics import backtest_metrics, wealth_path
from .schemas import SolveResult
from .topology import market_communities
from .validation import validate_weights


COLORS = ["#0B5CAD", "#00A6A6", "#F28E2B", "#7B61A8", "#D1495B", "#5C677D"]
VALIDATION_TOLERANCE = 1e-7


def _label(method: str) -> str:
    replacements = {
        "feasible_initial_portfolio": "Valid initial",
        "classical_enumeration": "Classical exact window",
        "classical_tabu_lns": "Classical tabu/LNS",
        "xy_qaoa_subspace": "XY-QAOA",
        "xy_qaoa_aer_gpu": "XY-QAOA (Aer GPU sample)",
        "xy_qaoa_aer_cpu": "XY-QAOA (Aer CPU sample)",
        "penalty_qaoa_statevector": "Penalty QAOA",
        "gurobi_cardinality_miqp": "Gurobi MIQP",
        "scipy_slsqp": "Continuous relaxation",
        "osqp": "Continuous relaxation",
    }
    return replacements.get(method, method.replace("_", " ").title())


def _best_methods(results: Iterable[SolveResult]) -> list[SolveResult]:
    best: dict[str, SolveResult] = {}
    for result in results:
        if not result.success or not result.feasible:
            continue
        if result.method not in best or result.objective < best[result.method].objective:
            best[result.method] = result
    return sorted(best.values(), key=lambda result: result.objective)


def _unique_method_labels(methods: Iterable[str]) -> list[str]:
    """Make repeated window experiments unambiguous in chart legends."""
    base = [_label(method) for method in methods]
    totals = {name: base.count(name) for name in set(base)}
    seen: dict[str, int] = {}
    labels: list[str] = []
    for name in base:
        seen[name] = seen.get(name, 0) + 1
        labels.append(f"{name} (window {seen[name]})" if totals[name] > 1 else name)
    return labels


def _quantum_execution_rows(run: HybridRun) -> list[dict[str, Any]]:
    """Flatten quantum correctness, device, resource, and timing evidence."""
    rows: list[dict[str, Any]] = []
    for iteration, search in enumerate(run.quantum_searches):
        metadata = search.metadata
        rows.append(
            {
                "iteration": metadata.get("iteration", iteration),
                "method": search.method,
                "parameter_optimizer_backend": metadata.get(
                    "parameter_optimizer_backend", ""
                ),
                "requested_sampler_backend": metadata.get("requested_backend", ""),
                "actual_sampler_backend": metadata.get("backend", ""),
                "execution_device": metadata.get("execution_device", ""),
                "gpu_accelerated": metadata.get("gpu_accelerated", False),
                "device_verification": metadata.get("device_verification", ""),
                "fallback_reason": metadata.get("fallback_reason", ""),
                "qubits": metadata.get("qubits", ""),
                "required_ones": metadata.get("required_ones", ""),
                "subspace_states": metadata.get(
                    "subspace_states", metadata.get("state_count", "")
                ),
                "depth_p": metadata.get("depth_p", ""),
                "shots": metadata.get("shots", ""),
                "optimizer_evaluations": metadata.get("optimizer_evaluations", ""),
                "unique_sampled_bitstrings": metadata.get(
                    "unique_sampled_bitstrings", len(search.counts)
                ),
                "cardinality_feasibility_rate": search.cardinality_feasibility_rate,
                "exact_expected_surrogate_energy": search.expected_surrogate_energy,
                "sampled_expected_surrogate_energy": metadata.get(
                    "sampled_expected_surrogate_energy", ""
                ),
                "best_sampled_energy": search.best_sampled_energy,
                "logical_two_qubit_gates": metadata.get("logical_two_qubit_gates", ""),
                "transpiled_two_qubit_gates": metadata.get(
                    "transpiled_two_qubit_gates", ""
                ),
                "transpiled_depth": metadata.get("transpiled_depth", ""),
                "preprocessing_seconds": metadata.get("preprocessing_seconds", 0.0),
                "angle_optimization_seconds": metadata.get(
                    "angle_optimization_seconds", 0.0
                ),
                "circuit_build_seconds": metadata.get("circuit_build_seconds", 0.0),
                "simulator_setup_seconds": metadata.get(
                    "simulator_setup_seconds", 0.0
                ),
                "transpile_seconds": metadata.get("transpile_seconds", 0.0),
                "sampling_seconds": metadata.get(
                    "simulation_seconds",
                    metadata.get("subspace_sampling_seconds", 0.0),
                ),
                "count_decode_seconds": metadata.get("count_decode_seconds", 0.0),
                "candidate_ranking_seconds": metadata.get(
                    "candidate_ranking_seconds", 0.0
                ),
                "allocation_oracle_seconds": metadata.get(
                    "allocation_oracle_seconds", 0.0
                ),
                "window_end_to_end_seconds": metadata.get(
                    "window_end_to_end_seconds", search.runtime
                ),
                "evaluated_supports": metadata.get("evaluated_supports", ""),
                "feasible_supports": metadata.get("feasible_supports", ""),
                "duplicate_supports": metadata.get("duplicate_supports", ""),
            }
        )
    return rows


def _save(fig: plt.Figure, path: Path) -> dict[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {path.stem: path, f"{path.stem}_pdf": pdf}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def plot_architecture(path: Path) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.axis("off")
    labels = [
        "Factor-QP\nrelaxation",
        "Valid sparse\nportfolio",
        "Adaptive\nchange window",
        "LNS  |  XY-QAOA",
        "Allocation oracle\n+ validator",
        "Gurobi bound\n+ Copilot",
    ]
    x = np.linspace(0.07, 0.93, len(labels))
    colors = ["#DCEBFA", "#DFF4EF", "#FFF0D7", "#EDE4F6", "#DFF4EF", "#DCEBFA"]
    for index, (position, label) in enumerate(zip(x, labels)):
        ax.text(
            position,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": colors[index],
                "edgecolor": "#51606F",
                "linewidth": 1.2,
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[index + 1] - 0.065, 0.52),
                xytext=(position + 0.065, 0.52),
                arrowprops={"arrowstyle": "->", "color": "#51606F", "lw": 1.4},
            )
    ax.text(
        0.5,
        0.08,
        "Every proposal returns to the exact financial model; only validated portfolios survive.",
        ha="center",
        fontsize=10,
        color="#34495E",
    )
    ax.set_title("Constraint-safe quantum-guided portfolio optimization", fontsize=15, pad=18)
    return _save(fig, path)


def plot_allocations(run: HybridRun, path: Path) -> dict[str, Path]:
    methods = _best_methods([run.initial, *run.results])
    series = [("Current", run.problem.w0)] + [
        (_label(result.method), result.weights) for result in methods
    ]
    importance = np.max(np.vstack([weights for _, weights in series]), axis=0)
    selected = np.argsort(importance)[::-1][: min(20, run.problem.n)]
    labels = [run.problem.asset_names[index] for index in selected]
    x = np.arange(len(selected))
    width = min(0.82 / len(series), 0.18)
    fig, ax = plt.subplots(figsize=(max(10, 0.52 * len(selected)), 5.8))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for index, ((name, weights), offset) in enumerate(zip(series, offsets)):
        color = "#AAB2BD" if index == 0 else COLORS[(index - 1) % len(COLORS)]
        ax.bar(x + offset, weights[selected], width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Allocation comparison on the most important assets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(series)))
    fig.tight_layout()
    return _save(fig, path)


def plot_objective_and_runtime(run: HybridRun, path: Path) -> dict[str, Path]:
    results = _best_methods(run.all_results())
    feasible_results = [
        result for result in results if result.model_type != "continuous_relaxation"
    ]
    feasible_labels = [_label(result.method) for result in feasible_results]
    best = min(result.objective for result in feasible_results)
    gaps = 1e6 * np.asarray([result.objective - best for result in feasible_results])
    runtime_labels = [_label(result.method) for result in results]
    runtimes = np.asarray([max(result.runtime, 1e-6) for result in results])
    quality_x = np.arange(len(feasible_results))
    runtime_x = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(results) * 1.7), 4.8))
    axes[0].bar(
        quality_x,
        gaps,
        color=[COLORS[index % len(COLORS)] for index in quality_x],
    )
    relaxation = next(
        (result for result in results if result.model_type == "continuous_relaxation"),
        None,
    )
    if relaxation is not None:
        bound_gap = 1e6 * (relaxation.objective - best)
        axes[0].text(
            0.02,
            0.96,
            f"Continuous-relaxation lower bound: {bound_gap:+.2f} ×10⁻⁶",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#34495E",
        )
    axes[0].set_ylabel("Feasible objective gap (×10⁻⁶; lower is better)")
    axes[0].set_title("Solution quality")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        runtime_x,
        runtimes,
        color=[COLORS[index % len(COLORS)] for index in runtime_x],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Component runtime in seconds (log scale)")
    axes[1].set_title("Runtime")
    axes[1].grid(axis="y", alpha=0.25, which="both")
    axes[0].set_xticks(quality_x)
    axes[0].set_xticklabels(feasible_labels, rotation=30, ha="right")
    axes[1].set_xticks(runtime_x)
    axes[1].set_xticklabels(runtime_labels, rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_risk_return(run: HybridRun, path: Path) -> dict[str, Path]:
    results = _best_methods(run.all_results())
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    for index, result in enumerate(results):
        marker = "D" if "qaoa" in result.method else ("s" if "gurobi" in result.method else "o")
        ax.scatter(
            result.metrics["volatility"],
            result.metrics["expected_return"],
            s=75,
            marker=marker,
            color=COLORS[index % len(COLORS)],
            edgecolor="white",
            linewidth=0.7,
            label=_label(result.method),
        )
    ax.set_xlabel("Expected annual volatility")
    ax.set_ylabel("Expected annual return")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Risk-return outcomes after exact allocation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_objective_timeline(run: HybridRun, path: Path) -> dict[str, Path]:
    rows = sorted(run.timeline, key=lambda row: float(row["elapsed_seconds"]))
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    objective = np.minimum.accumulate([float(row["objective"]) for row in rows])
    improvement = 1e6 * (objective[0] - objective)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.step(elapsed, improvement, where="post", color="#0B5CAD", linewidth=2.2)
    ax.scatter(elapsed, improvement, color="#F28E2B", s=45, zorder=3)
    for row, x, y in zip(rows, elapsed, improvement):
        ax.annotate(
            str(row["stage"]),
            (x, y),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("End-to-end elapsed seconds")
    ax.set_ylabel("Cumulative objective improvement (×10⁻⁶)")
    ax.set_title("Anytime behavior: best feasible portfolio versus time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_constraint_slacks(run: HybridRun, path: Path) -> dict[str, Path]:
    report = validate_weights(
        run.best.weights,
        run.problem,
        constraints=run.constraints,
        tol=VALIDATION_TOLERANCE,
    )
    active_assets = {
        run.problem.asset_names[index]
        for index in np.flatnonzero(run.best.weights > 1e-8)
    }
    candidates: list[tuple[Any, float, str]] = []
    for check in report.checks:
        family, _, asset = check.name.partition(":")
        if family in {"asset_lower", "eligible"}:
            continue
        if family in {"asset_upper", "implementation_upper"} and asset not in active_assets:
            continue
        if family == "asset_upper" and run.constraints.maximum_weights is not None:
            continue
        adjusted_slack = (
            0.0
            if check.slack < 0.0 and check.violation <= VALIDATION_TOLERANCE
            else check.slack
        )
        scale = max(abs(check.lhs), abs(check.rhs), 0.01)
        candidates.append((check, adjusted_slack / scale, family))

    family_limits = {
        "minimum_active_weight": 3,
        "implementation_upper": 3,
        "asset_upper": 3,
        "stress_floor": 5,
        "mandatory": 3,
    }
    selected: list[tuple[Any, float, str]] = []
    families = sorted({family for _, _, family in candidates})
    for family in families:
        members = sorted(
            (item for item in candidates if item[2] == family),
            key=lambda item: item[1],
        )
        selected.extend(members[: family_limits.get(family, len(members))])
    selected = sorted(selected, key=lambda item: item[1])[:30]
    checks = [item[0] for item in selected]
    slacks = np.asarray([item[1] for item in selected])
    labels = [check.name for check in checks]
    colors = [
        "#D1495B"
        if check.violation > VALIDATION_TOLERANCE
        else ("#F28E2B" if slack < 0.02 else "#00A6A6")
        for check, slack in zip(checks, slacks)
    ]
    fig, ax = plt.subplots(figsize=(10.0, max(5.0, len(checks) * 0.3)))
    y = np.arange(len(checks))
    ax.barh(y, slacks, color=colors)
    ax.scatter(slacks, y, color=colors, edgecolor="white", linewidth=0.5, s=24, zorder=3)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized headroom (0 = binding; negative = breach)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Independent validation: {report.breaches} hard-constraint breaches")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_group_exposure(run: HybridRun, path: Path) -> dict[str, Path]:
    current = run.problem.A @ run.problem.w0
    best = run.problem.A @ run.best.weights
    x = np.arange(run.problem.num_groups)
    fig, ax = plt.subplots(figsize=(max(8.0, run.problem.num_groups * 0.9), 4.8))
    ax.bar(x - 0.18, current, 0.36, label="Current", color="#AAB2BD")
    ax.bar(x + 0.18, best, 0.36, label="Recommended", color="#0B5CAD")
    ax.vlines(
        x,
        run.problem.group_lower,
        run.problem.group_upper,
        color="#D1495B",
        linewidth=3,
        label="Allowed band",
    )
    ax.scatter(x, run.problem.group_lower, color="#D1495B", s=28)
    ax.scatter(x, run.problem.group_upper, color="#D1495B", s=28)
    ax.set_xticks(x)
    ax.set_xticklabels(run.problem.group_names, rotation=25, ha="right")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Asset-group exposure and hard bounds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_factor_exposure(run: HybridRun, path: Path) -> dict[str, Path]:
    if not run.problem.has_factor_model:
        return {}
    current = run.problem.factor_loadings.T @ run.problem.w0
    best = run.problem.factor_loadings.T @ run.best.weights
    x = np.arange(run.problem.num_factors)
    fig, ax = plt.subplots(figsize=(max(8.0, run.problem.num_factors * 0.9), 4.8))
    ax.bar(x - 0.18, current, 0.36, label="Current", color="#AAB2BD")
    ax.bar(x + 0.18, best, 0.36, label="Recommended", color="#7B61A8")
    if run.constraints.factor_lower is not None:
        ax.vlines(
            x,
            run.constraints.factor_lower,
            run.constraints.factor_upper,
            color="#D1495B",
            linewidth=3,
            label="Allowed band",
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(run.problem.factor_names, rotation=25, ha="right")
    ax.set_ylabel("Factor exposure")
    ax.set_title("Factor-risk exposure")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_quantum_cardinality(run: HybridRun, path: Path) -> dict[str, Path]:
    if not run.quantum_searches:
        return {}
    distributions: list[tuple[str, CounterLike]] = []
    labels = _unique_method_labels(search.method for search in run.quantum_searches)
    for label, search in zip(labels, run.quantum_searches):
        histogram: dict[int, int] = {}
        for bits, count in search.counts.items():
            weight = sum(int(value) for value in bits)
            histogram[weight] = histogram.get(weight, 0) + int(count)
        distributions.append((label, histogram))
    weights = sorted({weight for _, histogram in distributions for weight in histogram})
    x = np.arange(len(weights))
    width = 0.8 / len(distributions)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for index, (name, histogram) in enumerate(distributions):
        total = max(sum(histogram.values()), 1)
        values = [histogram.get(weight, 0) / total for weight in weights]
        offset = x + (index - (len(distributions) - 1) / 2) * width
        ax.bar(
            offset,
            values,
            width,
            label=name,
            color=COLORS[index % len(COLORS)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(weights)
    ax.set_xlabel("Selected assets inside quantum window")
    ax.set_ylabel("Shot probability")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("XY mixing preserves cardinality; penalty QAOA may not")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_quantum_resources(run: HybridRun, path: Path) -> dict[str, Path]:
    if not run.quantum_searches:
        return {}
    labels = _unique_method_labels(search.method for search in run.quantum_searches)
    gates = [
        search.metadata.get(
            "transpiled_two_qubit_gates",
            search.metadata.get("logical_two_qubit_gates", 0),
        )
        for search in run.quantum_searches
    ]
    depths = [search.metadata.get("transpiled_depth", 0) for search in run.quantum_searches]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.5), 4.8))
    if any(float(value) > 0 for value in depths):
        ax.bar(x - 0.18, gates, 0.36, label="Two-qubit operations", color="#7B61A8")
        ax.bar(x + 0.18, depths, 0.36, label="Transpiled depth", color="#F28E2B")
        title = "Quantum circuit resources after compilation"
    else:
        ax.bar(x, gates, 0.55, label="Logical two-qubit operations", color="#7B61A8")
        title = "Logical quantum resources before hardware transpilation"
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_quantum_timing(run: HybridRun, path: Path) -> dict[str, Path]:
    """Show which XY-QAOA phases use the CPU, simulator backend, and oracle."""
    if not run.quantum_searches:
        return {}
    labels = _unique_method_labels(search.method for search in run.quantum_searches)
    rows = _quantum_execution_rows(run)
    phases = [
        (
            "CPU subspace setup",
            np.asarray([float(row["preprocessing_seconds"]) for row in rows]),
            "#5C677D",
        ),
        (
            "CPU angle optimization",
            np.asarray([float(row["angle_optimization_seconds"]) for row in rows]),
            "#0B5CAD",
        ),
        (
            "Circuit build + compile",
            np.asarray(
                [
                    float(row["circuit_build_seconds"])
                    + float(row["simulator_setup_seconds"])
                    + float(row["transpile_seconds"])
                    for row in rows
                ]
            ),
            "#7B61A8",
        ),
        (
            "Backend sampling",
            np.asarray(
                [
                    float(row["sampling_seconds"])
                    + float(row["count_decode_seconds"])
                    + float(row["candidate_ranking_seconds"])
                    for row in rows
                ]
            ),
            "#00A6A6",
        ),
        (
            "Classical allocation oracle",
            np.asarray([float(row["allocation_oracle_seconds"]) for row in rows]),
            "#F28E2B",
        ),
    ]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.7), 4.8))
    for name, values, color in phases:
        ax.bar(x, values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Quantum-search end-to-end phase timing")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation_communities(run: HybridRun, path: Path) -> dict[str, Path]:
    labels = market_communities(run.problem, seed=run.config.seed)
    display_limit = min(250, run.problem.n)
    buckets = {
        label: np.flatnonzero(labels == label).tolist()
        for label in sorted(set(labels.tolist()))
    }
    sampled: list[int] = []
    while len(sampled) < display_limit and any(buckets.values()):
        for label in buckets:
            if buckets[label] and len(sampled) < display_limit:
                sampled.append(int(buckets[label].pop(0)))
    displayed = np.asarray(sorted(sampled, key=lambda index: labels[index]), dtype=int)
    correlation = run.problem.corr[np.ix_(displayed, displayed)]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    sorted_labels = labels[displayed]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=0.7)
        ax.axvline(boundary, color="white", linewidth=0.7)
    ax.set_xlabel("Assets sorted by correlation community")
    ax.set_ylabel("Assets sorted by correlation community")
    suffix = "" if display_limit == run.problem.n else f" ({display_limit}-asset stratified sample)"
    ax.set_title(f"Market structure used for diverse candidate windows{suffix}")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return _save(fig, path)


def plot_backtest(
    run: HybridRun,
    realized_returns: np.ndarray,
    path: Path,
) -> dict[str, Path]:
    methods = _best_methods([run.initial, *run.results])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    for index, result in enumerate(methods):
        wealth = wealth_path(result.weights, realized_returns)
        peaks = np.maximum.accumulate(wealth)
        drawdown = 1.0 - wealth / np.maximum(peaks, 1e-15)
        color = COLORS[index % len(COLORS)]
        axes[0].plot(wealth, color=color, linewidth=1.8, label=_label(result.method))
        axes[1].plot(drawdown, color=color, linewidth=1.5)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Synthetic out-of-sample wealth paths")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Rebalancing periods")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Drawdown")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


# A small alias keeps the plot annotation readable without importing typing.Counter.
CounterLike = dict[int, int]


def write_hybrid_artifacts(
    run: HybridRun,
    output_directory: str | Path,
    *,
    realized_returns: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write a complete, auditable result package for the demo and presentation."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    artifacts: dict[str, Path] = {}

    artifacts["summary"] = _write_csv(output / "hybrid_summary.csv", run.summary_records())
    allocation_rows: list[dict[str, Any]] = []
    for result in run.all_results():
        for index, asset in enumerate(run.problem.asset_names):
            allocation_rows.append(
                {
                    "method": result.method,
                    "stage": result.metadata.get("stage", ""),
                    "iteration": result.metadata.get("iteration", ""),
                    "asset": asset,
                    "group": run.problem.group_names[run.problem.asset_group[index]],
                    "weight": result.weights[index],
                    "current_weight": run.problem.w0[index],
                    "change": result.weights[index] - run.problem.w0[index],
                    "selected": int(result.weights[index] > 1e-8),
                }
            )
    artifacts["allocations"] = _write_csv(output / "allocation_weights.csv", allocation_rows)

    report = validate_weights(
        run.best.weights,
        run.problem,
        constraints=run.constraints,
        tol=VALIDATION_TOLERANCE,
    )
    constraint_rows = [
        {
            "name": check.name,
            "sense": check.sense,
            "lhs": check.lhs,
            "rhs": check.rhs,
            "slack": check.slack,
            "violation": check.violation,
            "passed": check.violation <= VALIDATION_TOLERANCE,
        }
        for check in report.checks
    ]
    artifacts["constraints"] = _write_csv(output / "constraint_checks.csv", constraint_rows)
    artifacts["timeline"] = _write_csv(output / "objective_timeline.csv", run.timeline)

    window_rows: list[dict[str, Any]] = []
    communities = (
        market_communities(run.problem, seed=run.config.seed)
        if run.config.use_topology
        else None
    )
    for iteration, window in enumerate(run.windows):
        for index in window.indices:
            window_rows.append(
                {
                    "iteration": iteration,
                    "asset": run.problem.asset_names[index],
                    "asset_index": index,
                    "role": "weak_held" if index in window.weak_held else "promising_unheld",
                    "group": run.problem.group_names[run.problem.asset_group[index]],
                    "community": "" if communities is None else int(communities[index]),
                }
            )
    artifacts["windows"] = _write_csv(output / "change_windows.csv", window_rows)

    quantum_execution_rows = _quantum_execution_rows(run)
    if quantum_execution_rows:
        artifacts["quantum_execution"] = _write_csv(
            output / "quantum_execution.csv",
            quantum_execution_rows,
        )

    if realized_returns is not None:
        rows = []
        for result in _best_methods([run.initial, *run.results]):
            rows.append(
                {
                    "method": result.method,
                    **backtest_metrics(result.weights, realized_returns),
                }
            )
        artifacts["backtest"] = _write_csv(output / "backtest_summary.csv", rows)

    diagnostics = {
        "runtime_seconds": run.runtime,
        "oracle_calls": run.oracle_calls,
        "oracle_cache_hits": run.oracle_cache_hits,
        "skipped": run.skipped,
        "constraints": run.constraints.to_dict(),
        "preferences": run.preferences.to_dict(),
        "config": _jsonable(run.config.__dict__),
        "best_method": run.best.method,
        "best_objective": run.best.objective,
        "quantum": [
            {
                "method": search.method,
                "angles": search.angles,
                "runtime": search.runtime,
                "cardinality_feasibility_rate": search.cardinality_feasibility_rate,
                "expected_surrogate_energy": search.expected_surrogate_energy,
                "best_sampled_energy": search.best_sampled_energy,
                "metadata": search.metadata,
            }
            for search in run.quantum_searches
        ],
    }
    diagnostics_path = output / "hybrid_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_jsonable(diagnostics), indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts["diagnostics"] = diagnostics_path
    problem_path = output / "problem.json"
    problem_path.write_text(json.dumps(run.problem.to_dict(), indent=2) + "\n", encoding="utf-8")
    artifacts["problem"] = problem_path

    plot_functions = [
        (plot_architecture, "architecture.png"),
        (lambda path: plot_allocations(run, path), "allocation_comparison.png"),
        (lambda path: plot_objective_and_runtime(run, path), "objective_runtime.png"),
        (lambda path: plot_risk_return(run, path), "risk_return.png"),
        (lambda path: plot_objective_timeline(run, path), "objective_timeline.png"),
        (lambda path: plot_constraint_slacks(run, path), "constraint_slacks.png"),
        (lambda path: plot_group_exposure(run, path), "group_exposure.png"),
        (lambda path: plot_factor_exposure(run, path), "factor_exposure.png"),
        (lambda path: plot_quantum_cardinality(run, path), "quantum_cardinality.png"),
        (lambda path: plot_quantum_resources(run, path), "quantum_resources.png"),
        (lambda path: plot_quantum_timing(run, path), "quantum_timing.png"),
        (lambda path: plot_correlation_communities(run, path), "correlation_communities.png"),
    ]
    for function, filename in plot_functions:
        artifacts.update(function(plots / filename))
    if realized_returns is not None:
        artifacts.update(plot_backtest(run, realized_returns, plots / "out_of_sample_backtest.png"))

    rows = run.summary_records()
    best_row = next(row for row in rows if row["method"] == run.best.method)
    tied_methods = sorted(
        {
            _label(str(row["method"]))
            for row in rows
            if bool(row["feasible"])
            and str(row["model_type"]) != "continuous_relaxation"
            and abs(float(row["objective"]) - run.best.objective) <= 1e-10
        }
    )
    quantum_rows = [row for row in rows if "qaoa" in str(row["method"])]
    report_lines = [
        "# Hybrid portfolio optimization report",
        "",
        f"- Best valid incumbent: **{' = '.join(tied_methods)}**",
        f"- Objective: `{run.best.objective:.10g}`",
        f"- Expected return: `{float(best_row['expected_return']):.3%}`",
        f"- Volatility: `{float(best_row['volatility']):.3%}`",
        f"- L1 turnover: `{float(best_row['turnover']):.3%}` "
        f"(one-way convention: `{0.5 * float(best_row['turnover']):.3%}`)",
        f"- Selected assets: `{int(best_row['support_size'])}`",
        f"- Hard-constraint breaches: **{run.best.breaches}**",
        f"- Total runtime: `{run.runtime:.3f} s`",
        f"- Allocation-oracle calls/cache hits: `{run.oracle_calls}/{run.oracle_cache_hits}`",
        "",
        "## Interpretation",
        "",
        "The continuous relaxation supplies a lower bound and candidate scores. "
        "Classical LNS and XY-QAOA search the same fixed-cardinality windows. Every "
        "sampled support is reallocated with the complete continuous financial model "
        "and independently validated. Quantum output is therefore a proposal, never "
        "an unverified final portfolio.",
        "",
        "XY-QAOA angles are optimized by the exact fixed-Hamming-weight CPU subspace "
        "simulator. When selected, Aer GPU or IBM Runtime executes and samples the "
        "corresponding Qiskit circuit; the portable subspace backend samples on CPU. "
        "This split is intentional for small change windows and is reported explicitly "
        "in `quantum_execution.csv`.",
        "",
        "A hybrid result is globally heuristic even when its fixed-support allocation QP "
        "is solved optimally. Lower objective values are better.",
        "",
        "## Quantum results",
        "",
    ]
    if quantum_rows:
        for result_index, (row, search) in enumerate(
            zip(quantum_rows, run.quantum_searches),
        ):
            metadata = search.metadata
            window_index = int(metadata.get("iteration", result_index)) + 1
            device = metadata.get("execution_device", metadata.get("backend", "unknown"))
            verified = metadata.get("device_verification", "not recorded")
            report_lines.append(
                f"- Window {window_index}, {_label(str(row['method']))}: "
                f"objective `{float(row['objective']):.10g}`, "
                f"runtime `{float(row['runtime_seconds']):.3f} s`, breaches `{row['breaches']}`."
                f" Sampler device `{device}` ({verified}); cardinality-feasible shots "
                f"`{search.cardinality_feasibility_rate:.2%}`."
            )
    else:
        report_lines.append("- Quantum execution was disabled or unavailable for this run.")
    gurobi_rows = [row for row in rows if row["method"] == "gurobi_cardinality_miqp"]
    if gurobi_rows:
        row = gurobi_rows[-1]
        if row["best_bound"] != "" and row["reported_mip_gap"] != "":
            report_lines.extend(
                [
                    "",
                    "## Classical certification",
                    "",
                    f"- Gurobi incumbent: `{float(row['objective']):.10g}`",
                    f"- Certified lower bound: `{float(row['best_bound']):.10g}`",
                    f"- Reported MIP gap: `{float(row['reported_mip_gap']):.4%}`",
                    f"- Status: `{row['status']}` (optimal within the configured MIP tolerance).",
                ]
            )
    if run.skipped:
        report_lines.extend(["", "## Explicitly skipped components", ""])
        report_lines.extend(f"- `{name}`: {reason}" for name, reason in run.skipped.items())
    report_path = output / "hybrid_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifacts["report"] = report_path

    manifest = write_artifact_manifest(artifacts, output)
    artifacts["manifest"] = manifest
    return artifacts


__all__ = ["write_hybrid_artifacts"]
