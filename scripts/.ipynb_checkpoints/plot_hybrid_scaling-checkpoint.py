#!/usr/bin/env python3
"""Regenerate presentation-ready scaling figures from benchmark CSV evidence.

This script is intentionally read-only with respect to the benchmark. It does
not run solvers, alter configurations, or rewrite the evidence tables. Figures
are written to a separate presentation directory so the benchmark manifest
continues to describe the original artifacts.

Supported inputs
----------------
1. Current benchmark output: ``scaling_runs.csv`` (one row per trial).
2. Legacy benchmark output: ``hybrid_scaling.csv`` (one row per method/trial).

Examples
--------

    python scripts/plot_hybrid_scaling.py \
      --input results/hybrid_scaling \
      --output results/hybrid_scaling/presentation_plots

    python scripts/plot_hybrid_scaling.py \
      --input results/hybrid_scaling/hybrid_scaling.csv \
      --output results/hybrid_scaling/presentation_plots \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"),
)


VALIDATION_TOLERANCE = 1.0e-7
PRESENTATION_FILENAMES = (
    "scaling_runtime_presentation.png",
    "scaling_runtime_presentation.pdf",
    "scaling_quality_reliability_presentation.png",
    "scaling_quality_reliability_presentation.pdf",
    "scaling_memory_oracle_presentation.png",
    "scaling_memory_oracle_presentation.pdf",
    "scaling_quantum_presentation.png",
    "scaling_quantum_presentation.pdf",
)


@dataclass(frozen=True)
class Distribution:
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class SizeEvidence:
    n_assets: int
    total_runs: int
    successful_runs: int
    zero_breach_runs: int
    metrics: dict[str, Distribution]
    maximum_observed_violation: float | None

    @property
    def success_rate(self) -> float:
        return self.successful_runs / max(self.total_runs, 1)

    @property
    def zero_breach_rate(self) -> float:
        return self.zero_breach_runs / max(self.total_runs, 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, default: int = 0) -> int:
    number = _as_float(value)
    return default if number is None else int(number)


def _distribution(values: Iterable[float]) -> Distribution | None:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return Distribution(
        minimum=float(np.min(array)),
        q1=float(np.quantile(array, 0.25)),
        median=float(np.quantile(array, 0.50)),
        q3=float(np.quantile(array, 0.75)),
        maximum=float(np.max(array)),
        values=tuple(float(value) for value in array.tolist()),
    )


def _violation_from_error(message: str) -> float | None:
    match = re.search(
        r"max_violation\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        message or "",
    )
    return None if match is None else _as_float(match.group(1))


def _locate_input(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("scaling_runs.csv", "hybrid_scaling.csv"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{path} contains neither scaling_runs.csv nor hybrid_scaling.csv"
    )


def _current_trial_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for row in rows:
        trial = dict(row)
        trial["n_assets"] = _as_int(row.get("n_assets"))
        trial["repetition"] = _as_int(row.get("repetition"))
        trial["success"] = _as_bool(row.get("success"))
        trial["breaches"] = _as_int(row.get("breaches"), default=1)
        violation = _as_float(row.get("max_violation"))
        if violation is None:
            violation = _violation_from_error(str(row.get("error", "")))
        trial["observed_violation"] = violation
        trials.append(trial)
    return trials


def _legacy_trial_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(_as_int(row.get("n_assets")), _as_int(row.get("repetition")))].append(row)

    trials: list[dict[str, Any]] = []
    for (size, repetition), group in sorted(grouped.items()):
        failure = next((row for row in group if row.get("method") == "run_failure"), None)
        if failure is not None:
            trials.append(
                {
                    "n_assets": size,
                    "repetition": repetition,
                    "success": False,
                    "breaches": 1,
                    "observed_violation": _violation_from_error(
                        str(failure.get("error", ""))
                    ),
                    "error": failure.get("error", ""),
                }
            )
            continue

        best = next((row for row in group if row.get("method") == "best_valid"), None)
        reference = best or group[0]
        trial: dict[str, Any] = {
            "n_assets": size,
            "repetition": repetition,
            "success": _as_bool(reference.get("success")),
            "breaches": _as_int(reference.get("breaches"), default=1),
            "observed_violation": None,
            "time_to_first_valid_seconds": reference.get(
                "time_to_first_valid_seconds", ""
            ),
            "search_end_to_end_seconds": reference.get("runtime_seconds", ""),
            "full_end_to_end_seconds": reference.get("runtime_seconds", ""),
            "oracle_calls": reference.get("oracle_calls", ""),
            "oracle_cache_hits": reference.get("oracle_cache_hits", ""),
            "relative_gap_to_relaxation": reference.get(
                "objective_gap_to_relaxation", ""
            ),
        }
        method_to_field = {
            "osqp": "relaxation_seconds",
            "feasible_initial_portfolio": "initialization_seconds",
            "classical_tabu_lns": "classical_window_seconds",
            "classical_enumeration": "classical_window_seconds",
            "xy_qaoa_aer_gpu": "quantum_window_seconds",
            "xy_qaoa_aer_cpu": "quantum_window_seconds",
            "xy_qaoa_subspace": "quantum_window_seconds",
            "gurobi_cardinality_miqp": "gurobi_seconds",
        }
        for row in group:
            field = method_to_field.get(str(row.get("method", "")))
            if field is not None:
                trial[field] = row.get("runtime_seconds", "")
        trials.append(trial)
    return trials


def _load_trials(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    fields = set(rows[0])
    if "search_end_to_end_seconds" in fields:
        return _current_trial_rows(rows), "current run-level schema"
    if {"method", "runtime_seconds"}.issubset(fields):
        return _legacy_trial_rows(rows), "legacy method-level schema"
    raise ValueError(
        f"unsupported scaling CSV schema in {path}; columns={sorted(fields)}"
    )


def _aggregate(trials: Sequence[dict[str, Any]]) -> list[SizeEvidence]:
    metrics = (
        "search_end_to_end_seconds",
        "full_end_to_end_seconds",
        "time_to_first_valid_seconds",
        "data_generation_seconds",
        "relaxation_seconds",
        "initialization_seconds",
        "classical_window_seconds",
        "quantum_window_seconds",
        "window_overhead_seconds",
        "gurobi_seconds",
        "relative_gap_to_relaxation",
        "relative_hybrid_gap_to_gurobi",
        "peak_rss_gib",
        "dense_covariance_gib_avoided",
        "oracle_calls",
        "quantum_angle_seconds",
        "quantum_sampler_seconds",
        "quantum_allocation_seconds",
        "quantum_cardinality_rate",
    )
    result: list[SizeEvidence] = []
    sizes = sorted({int(row["n_assets"]) for row in trials})
    for size in sizes:
        group = [row for row in trials if int(row["n_assets"]) == size]
        successful = [row for row in group if row.get("success") is True]
        summaries: dict[str, Distribution] = {}
        for field in metrics:
            distribution = _distribution(
                value
                for row in successful
                if (value := _as_float(row.get(field))) is not None
            )
            if distribution is not None:
                summaries[field] = distribution
        violations = [
            value
            for row in group
            if (value := _as_float(row.get("observed_violation"))) is not None
        ]
        result.append(
            SizeEvidence(
                n_assets=size,
                total_runs=len(group),
                successful_runs=len(successful),
                zero_breach_runs=sum(
                    _as_int(row.get("breaches"), default=1) == 0
                    for row in successful
                ),
                metrics=summaries,
                maximum_observed_violation=max(violations) if violations else None,
            )
        )
    return result


def _save(fig: Any, output: Path, stem: str) -> list[Path]:
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        created.append(path)
    return created


def _available(evidence: Sequence[SizeEvidence], field: str) -> list[SizeEvidence]:
    return [row for row in evidence if field in row.metrics]


def _plot_distribution(
    ax: Any,
    evidence: Sequence[SizeEvidence],
    field: str,
    *,
    label: str,
    color: str,
    marker: str,
    raw: bool = True,
) -> None:
    rows = _available(evidence, field)
    if not rows:
        return
    x = np.asarray([row.n_assets for row in rows], dtype=float)
    distributions = [row.metrics[field] for row in rows]
    center = np.asarray([item.median for item in distributions])
    q1 = np.asarray([item.q1 for item in distributions])
    q3 = np.asarray([item.q3 for item in distributions])
    ax.plot(
        x,
        center,
        marker=marker,
        linewidth=2.2,
        markersize=6,
        color=color,
        label=label,
        zorder=4,
    )
    ax.fill_between(x, q1, q3, color=color, alpha=0.16, linewidth=0, zorder=2)
    if raw:
        for x_value, item in zip(x, distributions):
            count = len(item.values)
            offsets = np.linspace(-0.018, 0.018, count) if count > 1 else np.zeros(1)
            raw_x = x_value * np.power(10.0, offsets)
            ax.scatter(
                raw_x,
                item.values,
                s=20,
                color=color,
                alpha=0.45,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )


def _stage_matrix(
    evidence: Sequence[SizeEvidence],
) -> tuple[list[SizeEvidence], list[str], np.ndarray]:
    stage_fields = [
        "data_generation_seconds",
        "relaxation_seconds",
        "initialization_seconds",
        "classical_window_seconds",
        "quantum_window_seconds",
        "window_overhead_seconds",
    ]
    rows = [
        row
        for row in evidence
        if "search_end_to_end_seconds" in row.metrics
        and any(field in row.metrics for field in stage_fields)
    ]
    matrix = np.asarray(
        [
            [row.metrics[field].median if field in row.metrics else 0.0 for field in stage_fields]
            for row in rows
        ],
        dtype=float,
    )
    return rows, stage_fields, matrix


def _plots(evidence: Sequence[SizeEvidence], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    _plot_distribution(
        axes[0],
        evidence,
        "search_end_to_end_seconds",
        label="Complete hybrid search",
        color="#0B5CAD",
        marker="o",
    )
    _plot_distribution(
        axes[0],
        evidence,
        "time_to_first_valid_seconds",
        label="First independently valid portfolio",
        color="#00A6A6",
        marker="s",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Global asset universe size")
    axes[0].set_ylabel("Wall-clock seconds")
    axes[0].set_title("End-to-end scalability")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.02,
        "Line = median; band = IQR; dots = successful repetitions",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#51606F",
    )

    stage_rows, stage_fields, stage_values = _stage_matrix(evidence)
    stage_labels = {
        "data_generation_seconds": "Data/model construction",
        "relaxation_seconds": "Factor-QP relaxation",
        "initialization_seconds": "Support construction",
        "classical_window_seconds": "Classical LNS",
        "quantum_window_seconds": "XY-QAOA + allocation",
        "window_overhead_seconds": "Other orchestration",
    }
    stage_colors = {
        "data_generation_seconds": "#5C677D",
        "relaxation_seconds": "#0B5CAD",
        "initialization_seconds": "#9EC5E5",
        "classical_window_seconds": "#00A6A6",
        "quantum_window_seconds": "#F28E2B",
        "window_overhead_seconds": "#7B61A8",
    }
    if stage_rows:
        x = np.arange(len(stage_rows))
        bottom = np.zeros(len(stage_rows))
        for column, field in enumerate(stage_fields):
            values = stage_values[:, column]
            axes[1].bar(
                x,
                values,
                bottom=bottom,
                label=stage_labels[field],
                color=stage_colors[field],
                width=0.72,
            )
            bottom += values
        displayed_totals = bottom.copy()
        for position, total in zip(x, displayed_totals):
            axes[1].text(
                position,
                total,
                f"{total:.1f}s" if total >= 1 else f"{total:.2f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [f"{row.n_assets:,}" for row in stage_rows], rotation=25, ha="right"
        )
        axes[1].set_xlabel("Global asset universe size")
        axes[1].set_ylabel("Median wall-clock seconds")
        axes[1].set_title("Where the runtime is spent")
        axes[1].grid(axis="y", alpha=0.22)
        axes[1].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Hybrid solver scaling with a fixed-size quantum window")
    fig.tight_layout()
    created.extend(_save(fig, output, "scaling_runtime_presentation"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    _plot_distribution(
        axes[0],
        evidence,
        "relative_gap_to_relaxation",
        label="Gap to continuous relaxation",
        color="#00A6A6",
        marker="o",
    )
    _plot_distribution(
        axes[0],
        evidence,
        "relative_hybrid_gap_to_gurobi",
        label="Gap to Gurobi incumbent",
        color="#D1495B",
        marker="s",
    )
    axes[0].set_xscale("log")
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[0].set_xlabel("Global asset universe size")
    axes[0].set_ylabel("Relative objective gap")
    axes[0].set_title("Solution quality")
    axes[0].grid(alpha=0.22, which="both")
    if axes[0].lines:
        axes[0].legend(frameon=False)

    x = np.arange(len(evidence))
    success = np.asarray([row.success_rate for row in evidence])
    zero_breach = np.asarray([row.zero_breach_rate for row in evidence])
    width = 0.34
    axes[1].bar(
        x - width / 2,
        success,
        width,
        label="Completed runs",
        color="#0B5CAD",
    )
    axes[1].bar(
        x + width / 2,
        zero_breach,
        width,
        label="Zero-breach runs",
        color="#00A6A6",
    )
    for position, row, value in zip(x, evidence, success):
        axes[1].text(
            position - width / 2,
            min(value + 0.025, 1.04),
            f"{row.successful_runs}/{row.total_runs}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].axhline(1.0, color="#51606F", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{row.n_assets:,}" for row in evidence], rotation=25, ha="right"
    )
    axes[1].set_ylim(0.0, 1.10)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].set_xlabel("Global asset universe size")
    axes[1].set_ylabel("Run rate")
    axes[1].set_title("Reliability and independent validation")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    created.extend(_save(fig, output, "scaling_quality_reliability_presentation"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    has_memory = any("peak_rss_gib" in row.metrics for row in evidence)
    if has_memory:
        _plot_distribution(
            axes[0],
            evidence,
            "peak_rss_gib",
            label="Measured process peak RSS",
            color="#7B61A8",
            marker="o",
        )
        _plot_distribution(
            axes[0],
            evidence,
            "dense_covariance_gib_avoided",
            label="One dense covariance matrix avoided",
            color="#D1495B",
            marker="s",
            raw=False,
        )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Global asset universe size")
        axes[0].set_ylabel("GiB")
        axes[0].set_title("Factor-native memory scaling")
        axes[0].grid(alpha=0.22, which="both")
        if axes[0].lines:
            axes[0].legend(frameon=False)
    else:
        violation_rows = [
            row for row in evidence if row.maximum_observed_violation is not None
        ]
        if violation_rows:
            x_violation = np.asarray([row.n_assets for row in violation_rows], dtype=float)
            values = np.asarray(
                [float(row.maximum_observed_violation) for row in violation_rows]
            )
            axes[0].scatter(
                x_violation,
                values,
                s=55,
                color="#D1495B",
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            axes[0].plot(x_violation, values, color="#D1495B", linewidth=1.5)
            axes[0].axhline(
                VALIDATION_TOLERANCE,
                color="#51606F",
                linestyle="--",
                linewidth=1.2,
                label="Independent validation tolerance",
            )
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Global asset universe size")
            axes[0].set_ylabel("Maximum observed violation")
            axes[0].set_title("Numerical failures remain visible")
            axes[0].grid(alpha=0.22, which="both")
            axes[0].legend(frameon=False)
        else:
            axes[0].axis("off")
            axes[0].text(
                0.5,
                0.5,
                "Memory or validation-residual evidence\nwas not present in the input CSV.",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
                color="#51606F",
            )

    _plot_distribution(
        axes[1],
        evidence,
        "oracle_calls",
        label="Unique allocation-oracle calls",
        color="#7B61A8",
        marker="s",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Global asset universe size")
    axes[1].set_ylabel("Oracle calls per successful run")
    axes[1].set_title("Local search workload remains window-controlled")
    axes[1].grid(alpha=0.22, which="both")
    axes[1].text(
        0.02,
        0.02,
        "A flat trend supports the fixed-window decomposition claim",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#51606F",
    )
    fig.tight_layout()
    created.extend(_save(fig, output, "scaling_memory_oracle_presentation"))
    plt.close(fig)

    quantum_rows = [
        row
        for row in evidence
        if any(
            field in row.metrics
            for field in (
                "quantum_angle_seconds",
                "quantum_sampler_seconds",
                "quantum_allocation_seconds",
            )
        )
    ]
    if quantum_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
        for field, label, color, marker in (
            (
                "quantum_angle_seconds",
                "CPU fixed-weight angle optimization",
                "#7B61A8",
                "o",
            ),
            (
                "quantum_sampler_seconds",
                "Circuit sampling backend",
                "#F28E2B",
                "s",
            ),
            (
                "quantum_allocation_seconds",
                "Classical allocation oracle",
                "#00A6A6",
                "^",
            ),
        ):
            _plot_distribution(
                axes[0],
                quantum_rows,
                field,
                label=label,
                color=color,
                marker=marker,
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Global asset universe size")
        axes[0].set_ylabel("Seconds")
        axes[0].set_title("Fixed-window quantum phase timing")
        axes[0].grid(alpha=0.22, which="both")
        axes[0].legend(frameon=False)

        _plot_distribution(
            axes[1],
            quantum_rows,
            "quantum_cardinality_rate",
            label="Cardinality-valid shots",
            color="#0B5CAD",
            marker="o",
        )
        axes[1].set_xscale("log")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        axes[1].set_xlabel("Global asset universe size")
        axes[1].set_ylabel("Shot rate")
        axes[1].set_title("Cardinality preservation")
        axes[1].grid(alpha=0.22, which="both")
        axes[1].text(
            0.02,
            0.02,
            "Correctness diagnostic—not evidence of quantum advantage",
            transform=axes[1].transAxes,
            fontsize=8,
            color="#51606F",
        )
        fig.tight_layout()
        created.extend(_save(fig, output, "scaling_quantum_presentation"))
        plt.close(fig)

    return created


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/hybrid_scaling"),
        help="scaling CSV file or directory containing scaling_runs.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="figure directory; defaults to <input directory>/presentation_plots",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing presentation figures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = _locate_input(args.input)
    output = args.output or input_path.parent / "presentation_plots"
    existing = [output / name for name in PRESENTATION_FILENAMES if (output / name).is_file()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output} already contains presentation plots; pass --overwrite to replace them"
        )
    output.mkdir(parents=True, exist_ok=True)

    trials, schema = _load_trials(input_path)
    evidence = _aggregate(trials)
    if not evidence:
        raise ValueError(f"no scaling evidence could be aggregated from {input_path}")
    created = _plots(evidence, output)

    print(f"Input: {input_path}")
    print(f"Schema: {schema}")
    print(f"Asset sizes: {', '.join(f'{row.n_assets:,}' for row in evidence)}")
    print(f"Created {len(created)} presentation figures in {output}")
    failed = sum(row.total_runs - row.successful_runs for row in evidence)
    if failed:
        print(f"Failed benchmark trials shown in reliability plot: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
