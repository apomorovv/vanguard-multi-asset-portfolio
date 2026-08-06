#!/usr/bin/env python3
"""Run a reproducible scaling study for the hybrid portfolio optimizer.

Each case runs in a fresh process so peak memory and wall time are comparable.
The default study keeps portfolio cardinality and the quantum window fixed while
the asset universe grows from 250 to 20,000 assets. Gurobi certification is
attempted only up to a configurable universe size; larger cases measure the
factor-QP and neighborhood-search path without implying global certification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vanguard-mpl-cache"))

from vanguard_portfolio.data_generation import generate_factor_universe
from vanguard_portfolio.hybrid import HybridConfig, run_hybrid_optimizer
from vanguard_portfolio.quantum_solver import XYQAOAConfig
from vanguard_portfolio.schemas import PortfolioConstraints, Preferences

DEFAULT_SIZES = (250, 500, 1_000, 2_000, 5_000, 10_000, 20_000)
OUTPUT_FILENAMES = (
    "scaling_runs.csv",
    "scaling_methods.csv",
    "scaling_summary.csv",
    "scaling_environment.json",
    "scaling_config.json",
    "scaling_manifest.json",
    "scaling_evidence.png",
    "scaling_evidence.pdf",
)
LEGACY_PLOT_FILENAMES = (
    "scaling_runtime.png",
    "scaling_runtime.pdf",
    "scaling_quality_and_feasibility.png",
    "scaling_quality_and_feasibility.pdf",
    "scaling_memory_and_first_valid.png",
    "scaling_memory_and_first_valid.pdf",
    "scaling_quantum.png",
    "scaling_quantum.pdf",
)
RUN_COLUMNS = (
    "study_tier",
    "n_assets",
    "cardinality",
    "window_size",
    "iterations",
    "repetition",
    "seed",
    "success",
    "best_method",
    "best_objective",
    "hybrid_objective",
    "relaxation_objective",
    "relative_gap_to_relaxation",
    "gurobi_objective",
    "gurobi_best_bound",
    "gurobi_reported_gap",
    "relative_hybrid_gap_to_gurobi",
    "breaches",
    "max_violation",
    "time_to_first_valid_seconds",
    "data_generation_seconds",
    "relaxation_seconds",
    "relaxation_requested_tolerance",
    "relaxation_native_tolerance",
    "relaxation_iterations",
    "relaxation_primal_residual",
    "relaxation_dual_residual",
    "relaxation_rho_updates",
    "relaxation_status",
    "initialization_seconds",
    "classical_window_seconds",
    "quantum_window_seconds",
    "window_overhead_seconds",
    "gurobi_seconds",
    "search_end_to_end_seconds",
    "full_end_to_end_seconds",
    "oracle_calls",
    "oracle_cache_hits",
    "peak_rss_gib",
    "factor_risk_storage_mib",
    "dense_covariance_gib_avoided",
    "dense_covariance_and_correlation_gib_avoided",
    "quantum_backend_requested",
    "quantum_execution_device",
    "quantum_gpu_verified",
    "quantum_cardinality_rate",
    "quantum_angle_seconds",
    "quantum_sampler_seconds",
    "quantum_allocation_seconds",
    "certification_attempted",
    "certification_completed",
    "skipped_components",
    "error",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: Iterable[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a checkpoint while restoring simple scalar types."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw_row in raw:
        row: dict[str, Any] = {}
        for key, value in raw_row.items():
            if value in (None, ""):
                row[key] = ""
            elif value == "True":
                row[key] = True
            elif value == "False":
                row[key] = False
            else:
                try:
                    number = float(value)
                    row[key] = int(number) if number.is_integer() else number
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    byte_count = value if sys.platform == "darwin" else value * 1024.0
    return byte_count / 1024.0**3


def _relative_gap(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def _run_case(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = int(spec["n_assets"])
    cardinality = min(int(spec["cardinality"]), size - 1)
    groups = min(int(spec["groups"]), cardinality, size)
    seed = int(spec["seed"])
    data_start = time.perf_counter()
    problem = generate_factor_universe(
        n_assets=size,
        n_groups=groups,
        n_factors=min(int(spec["factors"]), size),
        seed=seed,
        current_cardinality=cardinality,
        materialize_covariance=bool(spec["materialize_covariance"]),
    )
    problem.max_turnover = float(spec["max_turnover"])
    data_seconds = time.perf_counter() - data_start

    minimum_weight = min(float(spec["minimum_active_weight"]), 0.5 / cardinality)
    maximum_weight = max(float(spec["maximum_weight"]), 1.5 / cardinality)
    constraints = PortfolioConstraints(
        exact_cardinality=cardinality,
        minimum_active_weight=minimum_weight,
        maximum_weights=np.full(size, maximum_weight),
    )
    certify = bool(spec["gurobi"]) and size <= int(spec["certification_max_assets"])
    quantum = XYQAOAConfig(
        depth=int(spec["quantum_depth"]),
        shots=int(spec["quantum_shots"]),
        optimizer_maxiter=int(spec["quantum_optimizer_maxiter"]),
        optimizer_starts=int(spec["quantum_optimizer_starts"]),
        seed=seed,
        initial_state="warm",
        mixer="ring",
        backend=str(spec["quantum_backend"]),
        maximum_subspace_states=int(spec["maximum_subspace_states"]),
        top_candidates=int(spec["quantum_top_candidates"]),
        transpile_optimization_level=int(spec["transpile_optimization_level"]),
    )
    config = HybridConfig(
        iterations=int(spec["iterations"]),
        window_size=min(int(spec["window_size"]), size),
        allocation_backend=str(spec["allocation_backend"]),
        allocation_options={
            "tol": float(spec["allocation_tolerance"]),
            "max_iter": int(spec["allocation_max_iter"]),
        },
        relaxation_options={
            "tol": float(spec["relaxation_tolerance"]),
            "max_iter": int(spec["relaxation_max_iter"]),
            "time_limit": float(spec["relaxation_time_limit"]),
        },
        initial_trials=int(spec["initial_trials"]),
        initial_milp_time_limit=float(spec["initial_milp_time_limit"]),
        classical_tabu_iterations=int(spec["classical_tabu_iterations"]),
        classical_tabu_tenure=int(spec["classical_tabu_tenure"]),
        classical_oracle_candidates=int(spec["classical_oracle_candidates"]),
        enumerate_windows_up_to=int(spec["enumerate_windows_up_to"]),
        run_quantum=bool(spec["quantum"]),
        quantum=quantum,
        run_penalty_qaoa=False,
        use_topology=True,
        maximum_quantum_edges=int(spec["maximum_quantum_edges"]),
        run_gurobi_reference=certify,
        gurobi_time_limit=float(spec["gurobi_time_limit"]),
        gurobi_mip_gap=float(spec["gurobi_mip_gap"]),
        seed=seed,
    )
    preferences = Preferences(
        lambda_return=1.0,
        lambda_risk=5.0,
        lambda_income=0.5,
        lambda_cost=1.0,
    )
    run = run_hybrid_optimizer(problem, preferences, constraints, config)
    gurobi_results = [
        result for result in run.results if result.method == "gurobi_cardinality_miqp"
    ]
    gurobi = gurobi_results[-1] if gurobi_results else None
    hybrid_candidates = [
        run.initial,
        *[
            result
            for result in run.results
            if result.method != "gurobi_cardinality_miqp" and result.success and result.feasible
        ],
    ]
    hybrid_best = min(hybrid_candidates, key=lambda result: result.objective)
    gurobi_seconds = 0.0 if gurobi is None else float(gurobi.runtime)
    search_solver_seconds = max(float(run.runtime) - gurobi_seconds, 0.0)
    relaxation_seconds = float(run.relaxation.runtime)
    relaxation_metadata = run.relaxation.metadata
    time_to_valid = float(run.timeline[0]["elapsed_seconds"])
    initialization_seconds = max(time_to_valid - relaxation_seconds, 0.0)
    classical_seconds = sum(
        float(result.runtime)
        for result in run.results
        if result.method in {"classical_enumeration", "classical_tabu_lns"}
    )
    quantum_searches = [
        search for search in run.quantum_searches if search.method.startswith("xy_qaoa_")
    ]
    quantum_rows = [search.metadata for search in quantum_searches]
    quantum_seconds = sum(
        float(row.get("window_end_to_end_seconds", search.runtime))
        for search, row in zip(quantum_searches, quantum_rows)
    )
    window_overhead = max(
        search_solver_seconds
        - relaxation_seconds
        - initialization_seconds
        - classical_seconds
        - quantum_seconds,
        0.0,
    )
    angle_seconds = sum(float(row.get("angle_optimization_seconds", 0.0)) for row in quantum_rows)
    sampler_seconds = sum(float(row.get("sampler_total_seconds", 0.0)) for row in quantum_rows)
    allocation_seconds = sum(
        float(row.get("allocation_oracle_seconds", 0.0)) for row in quantum_rows
    )
    cardinality_rate = (
        float(np.mean([float(search.cardinality_feasibility_rate) for search in quantum_searches]))
        if quantum_rows
        else np.nan
    )
    execution_devices = sorted({str(row.get("execution_device", "")) for row in quantum_rows})
    gpu_verified = bool(quantum_rows) and all(
        bool(row.get("gpu_accelerated", False)) for row in quantum_rows
    )
    factor_bytes = (
        problem.factor_loadings.nbytes
        + problem.factor_cov.nbytes
        + problem.idiosyncratic_var.nbytes
    )
    dense_covariance_gib = problem.dense_covariance_bytes() / 1024.0**3
    best_bound = "" if gurobi is None else gurobi.metadata.get("best_bound", "")
    reported_gap = "" if gurobi is None else gurobi.metadata.get("reported_mip_gap", "")
    record: dict[str, Any] = {
        "study_tier": "certified_reference" if certify else "scalable_hybrid_search",
        "n_assets": size,
        "cardinality": cardinality,
        "window_size": config.window_size,
        "iterations": config.iterations,
        "repetition": int(spec["repetition"]),
        "seed": seed,
        "success": bool(run.best.success and run.best.feasible),
        "best_method": run.best.method,
        "best_objective": run.best.objective,
        "hybrid_objective": hybrid_best.objective,
        "relaxation_objective": run.relaxation.objective,
        "relative_gap_to_relaxation": _relative_gap(
            hybrid_best.objective,
            run.relaxation.objective,
        ),
        "gurobi_objective": "" if gurobi is None else gurobi.objective,
        "gurobi_best_bound": best_bound,
        "gurobi_reported_gap": reported_gap,
        "relative_hybrid_gap_to_gurobi": ""
        if gurobi is None
        else _relative_gap(hybrid_best.objective, gurobi.objective),
        "breaches": int(run.best.breaches),
        "max_violation": float(run.best.max_violation),
        "time_to_first_valid_seconds": time_to_valid,
        "data_generation_seconds": data_seconds,
        "relaxation_seconds": relaxation_seconds,
        "relaxation_requested_tolerance": relaxation_metadata.get(
            "requested_tolerance", spec["relaxation_tolerance"]
        ),
        "relaxation_native_tolerance": relaxation_metadata.get(
            "native_tolerance", spec["relaxation_tolerance"]
        ),
        "relaxation_iterations": relaxation_metadata.get("iterations", ""),
        "relaxation_primal_residual": relaxation_metadata.get("primal_residual", ""),
        "relaxation_dual_residual": relaxation_metadata.get("dual_residual", ""),
        "relaxation_rho_updates": relaxation_metadata.get("rho_updates", ""),
        "relaxation_status": run.relaxation.status,
        "initialization_seconds": initialization_seconds,
        "classical_window_seconds": classical_seconds,
        "quantum_window_seconds": quantum_seconds,
        "window_overhead_seconds": window_overhead,
        "gurobi_seconds": gurobi_seconds,
        "search_end_to_end_seconds": data_seconds + search_solver_seconds,
        "full_end_to_end_seconds": data_seconds + float(run.runtime),
        "oracle_calls": int(run.oracle_calls),
        "oracle_cache_hits": int(run.oracle_cache_hits),
        "peak_rss_gib": _peak_rss_gib(),
        "factor_risk_storage_mib": factor_bytes / 1024.0**2,
        "dense_covariance_gib_avoided": 0.0
        if problem.has_dense_covariance
        else dense_covariance_gib,
        "dense_covariance_and_correlation_gib_avoided": 0.0
        if problem.has_dense_covariance
        else 2.0 * dense_covariance_gib,
        "quantum_backend_requested": spec["quantum_backend"] if spec["quantum"] else "disabled",
        "quantum_execution_device": ";".join(execution_devices),
        "quantum_gpu_verified": gpu_verified,
        "quantum_cardinality_rate": cardinality_rate,
        "quantum_angle_seconds": angle_seconds,
        "quantum_sampler_seconds": sampler_seconds,
        "quantum_allocation_seconds": allocation_seconds,
        "certification_attempted": certify,
        "certification_completed": gurobi is not None,
        "skipped_components": json.dumps(run.skipped, sort_keys=True),
        "error": "",
    }
    method_rows = []
    for row in run.summary_records():
        method_rows.append(
            {
                "n_assets": size,
                "repetition": int(spec["repetition"]),
                "seed": seed,
                **row,
            }
        )
    return record, method_rows


def _worker(spec_path: Path, output_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        record, methods = _run_case(spec)
        payload = {"record": record, "methods": methods}
    except Exception as exc:  # noqa: BLE001 - a failed case must remain auditable
        payload = {
            "record": {
                "study_tier": "run_failure",
                "n_assets": int(spec["n_assets"]),
                "cardinality": int(spec["cardinality"]),
                "window_size": int(spec["window_size"]),
                "iterations": int(spec["iterations"]),
                "repetition": int(spec["repetition"]),
                "seed": int(spec["seed"]),
                "success": False,
                "breaches": "",
                "peak_rss_gib": _peak_rss_gib(),
                "certification_attempted": bool(spec["gurobi"])
                and int(spec["n_assets"]) <= int(spec["certification_max_assets"]),
                "certification_completed": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "methods": [],
        }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.50)),
        float(np.quantile(array, 0.75)),
    )


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = (
        "search_end_to_end_seconds",
        "full_end_to_end_seconds",
        "time_to_first_valid_seconds",
        "data_generation_seconds",
        "relaxation_seconds",
        "initialization_seconds",
        "classical_window_seconds",
        "quantum_window_seconds",
        "gurobi_seconds",
        "relative_gap_to_relaxation",
        "relative_hybrid_gap_to_gurobi",
        "gurobi_reported_gap",
        "peak_rss_gib",
        "factor_risk_storage_mib",
        "dense_covariance_gib_avoided",
        "quantum_angle_seconds",
        "quantum_sampler_seconds",
        "quantum_allocation_seconds",
        "quantum_cardinality_rate",
    )
    rows: list[dict[str, Any]] = []
    for size in sorted({int(row["n_assets"]) for row in records}):
        group = [row for row in records if int(row["n_assets"]) == size]
        successful = [row for row in group if row.get("success") is True]
        summary: dict[str, Any] = {
            "n_assets": size,
            "runs": len(group),
            "successful_runs": len(successful),
            "success_rate": len(successful) / max(len(group), 1),
            "zero_breach_rate": sum(int(row.get("breaches", 1)) == 0 for row in successful)
            / max(len(successful), 1),
            "certified_runs": sum(bool(row.get("certification_completed")) for row in successful),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in successful
                if row.get(field) not in (None, "") and np.isfinite(float(row[field]))
            ]
            if values:
                q1, q2, q3 = _quantiles(values)
                summary[f"{field}_q1"] = q1
                summary[f"{field}_median"] = q2
                summary[f"{field}_q3"] = q3
        rows.append(summary)
    return rows


def _checkpoint_results(
    output: Path,
    records: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> None:
    """Persist recoverable raw and summarized evidence after every case."""

    _write_csv(output / "scaling_runs.csv", records, RUN_COLUMNS)
    _write_csv(output / "scaling_methods.csv", method_rows)
    _write_csv(output / "scaling_summary.csv", _summary_rows(records))


def _errorbar(ax: Any, rows: list[dict[str, Any]], field: str, **kwargs: Any) -> None:
    usable = [row for row in rows if f"{field}_median" in row]
    x = np.asarray([int(row["n_assets"]) for row in usable])
    center = np.asarray([float(row[f"{field}_median"]) for row in usable])
    lower = center - np.asarray([float(row[f"{field}_q1"]) for row in usable])
    upper = np.asarray([float(row[f"{field}_q3"]) for row in usable]) - center
    ax.errorbar(x, center, yerr=np.vstack([lower, upper]), capsize=3, **kwargs)


def _plots(summary: list[dict[str, Any]], output: Path) -> list[Path]:
    """Create one compact figure containing the scaling claims that matter."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    if not summary or not any(int(row.get("successful_runs", 0)) > 0 for row in summary):
        return []

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    _errorbar(
        axes[0, 0],
        summary,
        "search_end_to_end_seconds",
        marker="o",
        linewidth=2.0,
        color="#0B5CAD",
        label="Complete hybrid search",
    )
    _errorbar(
        axes[0, 0],
        summary,
        "relaxation_seconds",
        marker="s",
        linewidth=1.8,
        color="#7B61A8",
        label="Full-universe guide relaxation",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Global asset universe size")
    axes[0, 0].set_ylabel("Wall-clock seconds; median and IQR")
    axes[0, 0].set_title("End-to-end speed")
    axes[0, 0].grid(alpha=0.25, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8)

    _errorbar(
        axes[0, 1],
        summary,
        "relative_gap_to_relaxation",
        marker="o",
        linewidth=2.0,
        color="#00A6A6",
        label="Validated incumbent vs relaxation",
    )
    certified = [row for row in summary if "relative_hybrid_gap_to_gurobi_median" in row]
    if certified:
        _errorbar(
            axes[0, 1],
            certified,
            "relative_hybrid_gap_to_gurobi",
            marker="s",
            linewidth=1.8,
            color="#D1495B",
            label="Validated incumbent vs Gurobi",
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[0, 1].set_xlabel("Global asset universe size")
    axes[0, 1].set_ylabel("Relative objective gap; median and IQR")
    successful_runs = sum(int(row.get("successful_runs", 0)) for row in summary)
    zero_breach_runs = sum(
        round(float(row.get("zero_breach_rate", 0.0)) * int(row.get("successful_runs", 0)))
        for row in summary
    )
    axes[0, 1].set_title(
        f"Solution quality ({zero_breach_runs}/{successful_runs} valid runs had zero breaches)"
    )
    axes[0, 1].grid(alpha=0.25, which="both")
    axes[0, 1].legend(frameon=False, fontsize=8)

    memory_rows = [
        row
        for row in summary
        if "peak_rss_gib_median" in row
        and "factor_risk_storage_mib_median" in row
        and "dense_covariance_gib_avoided_median" in row
    ]
    if memory_rows:
        x = np.asarray([int(row["n_assets"]) for row in memory_rows], dtype=float)
        axes[1, 0].plot(
            x,
            [float(row["peak_rss_gib_median"]) for row in memory_rows],
            marker="o",
            linewidth=2.0,
            color="#7B61A8",
            label="Measured worker peak RSS",
        )
        axes[1, 0].plot(
            x,
            [
                float(row["factor_risk_storage_mib_median"]) / 1024.0
                for row in memory_rows
            ],
            marker="^",
            linewidth=1.8,
            color="#00A6A6",
            label=r"Factor arrays $B,\Omega,D$",
        )
        axes[1, 0].plot(
            x,
            [float(row["dense_covariance_gib_avoided_median"]) for row in memory_rows],
            marker="s",
            linewidth=1.8,
            color="#D1495B",
            label=r"Dense covariance $\Sigma$ (not allocated)",
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Global asset universe size")
    axes[1, 0].set_ylabel("GiB")
    axes[1, 0].set_title("Factor-native memory advantage")
    axes[1, 0].grid(alpha=0.25, which="both")
    axes[1, 0].legend(frameon=False, fontsize=8)

    _errorbar(
        axes[1, 1],
        summary,
        "classical_window_seconds",
        marker="o",
        linewidth=1.8,
        color="#00A6A6",
        label="Classical LNS + allocation",
    )
    _errorbar(
        axes[1, 1],
        summary,
        "quantum_window_seconds",
        marker="s",
        linewidth=1.8,
        color="#F28E2B",
        label="XY-QAOA proposals + allocation",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Global asset universe size")
    axes[1, 1].set_ylabel("Seconds; median and IQR")
    axes[1, 1].set_title("Fixed-window classical and quantum work")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Factor-native hybrid solver: validity, scalability, and quantum augmentation")
    fig.tight_layout()
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"scaling_evidence.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        created.append(path)
    plt.close(fig)
    return created


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "osqp", "qiskit", "qiskit-aer", "gurobipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    gpu_inventory: list[str] = []
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if query.returncode == 0:
            gpu_inventory = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": cpu_model,
        "cpu_count": os.cpu_count(),
        "gpu_inventory": gpu_inventory,
        "packages": packages,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _artifact_manifest(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted({Path(value) for value in paths}):
        payload = path.read_bytes()
        files[str(path.relative_to(root))] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {"algorithm": "sha256", "files": files}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cardinality", type=int, default=50)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--factors", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--backend", default="osqp")
    parser.add_argument(
        "--relaxation-tolerance",
        type=float,
        default=1e-10,
        help="explicit OSQP tolerance for the full-universe guide relaxation",
    )
    parser.add_argument("--relaxation-max-iter", type=int, default=250_000)
    parser.add_argument(
        "--relaxation-time-limit",
        type=float,
        default=300.0,
        help="native solver limit for the guide relaxation, in seconds",
    )
    parser.add_argument("--allocation-tolerance", type=float, default=1e-8)
    parser.add_argument("--allocation-max-iter", type=int, default=100_000)
    parser.add_argument("--minimum-active-weight", type=float, default=0.005)
    parser.add_argument("--maximum-weight", type=float, default=0.04)
    parser.add_argument("--max-turnover", type=float, default=0.40)
    parser.add_argument("--initial-trials", type=int, default=100)
    parser.add_argument("--initial-milp-time-limit", type=float, default=20.0)
    parser.add_argument("--classical-tabu-iterations", type=int, default=60)
    parser.add_argument("--classical-tabu-tenure", type=int, default=8)
    parser.add_argument("--classical-oracle-candidates", type=int, default=3)
    parser.add_argument("--enumerate-windows-up-to", type=int, default=5_000)
    parser.add_argument("--quantum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quantum-backend",
        choices=("subspace", "aer_cpu", "aer_gpu"),
        default="subspace",
    )
    parser.add_argument("--quantum-depth", type=int, default=1)
    parser.add_argument("--quantum-shots", type=int, default=4_096)
    parser.add_argument("--quantum-optimizer-maxiter", type=int, default=50)
    parser.add_argument("--quantum-optimizer-starts", type=int, default=2)
    parser.add_argument("--quantum-top-candidates", type=int, default=128)
    parser.add_argument("--maximum-subspace-states", type=int, default=400_000)
    parser.add_argument("--maximum-quantum-edges", type=int, default=60)
    parser.add_argument("--transpile-optimization-level", type=int, default=3)
    parser.add_argument("--gurobi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certification-max-assets", type=int, default=2_000)
    parser.add_argument("--gurobi-time-limit", type=float, default=60.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=1e-3)
    parser.add_argument("--materialize-covariance", action="store_true")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep completed size/repetition rows and run only missing cases",
    )
    parser.add_argument(
        "--case-time-limit",
        type=float,
        default=360.0,
        help="hard wall-clock limit for each isolated worker process",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/hybrid_scaling")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required with --worker-spec")
        return _worker(args.worker_spec, args.worker_output)
    if any(size < 2 for size in args.sizes):
        raise ValueError("every asset size must be at least two")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.case_time_limit <= 0.0:
        raise ValueError("case_time_limit must be positive")
    if args.relaxation_time_limit <= 0.0:
        raise ValueError("relaxation_time_limit must be positive")

    existing_outputs = [
        args.output / name
        for name in (*OUTPUT_FILENAMES, *LEGACY_PLOT_FILENAMES)
        if (args.output / name).is_file()
    ]
    if existing_outputs and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{args.output} contains a scaling result; pass --resume or --overwrite"
        )
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()
    args.output.mkdir(parents=True, exist_ok=True)
    records = _read_csv(args.output / "scaling_runs.csv") if args.resume else []
    method_rows = _read_csv(args.output / "scaling_methods.csv") if args.resume else []
    completed_keys = {
        (int(row["n_assets"]), int(row["repetition"]), int(row["seed"]))
        for row in records
        if row.get("n_assets") not in (None, "")
        and row.get("repetition") not in (None, "")
        and row.get("seed") not in (None, "")
    }
    with tempfile.TemporaryDirectory(prefix="vanguard-scaling-") as directory:
        work = Path(directory)
        case_index = 0
        total_cases = len(args.sizes) * args.repetitions
        for size in args.sizes:
            for repetition in range(args.repetitions):
                case_index += 1
                seed = int(args.seed + 10_000 * int(size) + repetition)
                case_key = (int(size), repetition, seed)
                if case_key in completed_keys:
                    print(
                        f"[{case_index}/{total_cases}] n={int(size):,} "
                        f"repetition={repetition + 1}: checkpoint already present",
                        flush=True,
                    )
                    continue
                spec = {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"worker_spec", "worker_output", "output", "sizes", "repetitions"}
                }
                spec.update(
                    {
                        "n_assets": int(size),
                        "repetition": repetition,
                        "seed": seed,
                        "allocation_backend": args.backend,
                    }
                )
                spec_path = work / f"case-{case_index}.json"
                result_path = work / f"result-{case_index}.json"
                spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[{case_index}/{total_cases}] n={int(size):,} repetition={repetition + 1}",
                    flush=True,
                )
                worker_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-spec",
                    str(spec_path),
                    "--worker-output",
                    str(result_path),
                ]
                try:
                    completed = subprocess.run(
                        worker_command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=float(args.case_time_limit),
                    )
                except subprocess.TimeoutExpired:
                    records.append(
                        {
                            "study_tier": "worker_timeout",
                            "n_assets": int(size),
                            "repetition": repetition,
                            "seed": seed,
                            "success": False,
                            "error": (
                                "worker exceeded case_time_limit="
                                f"{float(args.case_time_limit):g}s"
                            ),
                        }
                    )
                    _checkpoint_results(args.output, records, method_rows)
                    print("  timed out; checkpoint saved", flush=True)
                    continue
                if not result_path.is_file():
                    records.append(
                        {
                            "study_tier": "worker_failure",
                            "n_assets": int(size),
                            "repetition": repetition,
                            "seed": seed,
                            "success": False,
                            "error": completed.stderr.strip() or completed.stdout.strip(),
                        }
                    )
                    _checkpoint_results(args.output, records, method_rows)
                    continue
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                records.append(payload["record"])
                method_rows.extend(payload["methods"])
                record = payload["record"]
                if record.get("success"):
                    print(
                        "  completed: "
                        f"search={float(record['search_end_to_end_seconds']):.3f}s, "
                        f"breaches={record['breaches']}, "
                        f"peak={float(record['peak_rss_gib']):.2f} GiB",
                        flush=True,
                    )
                else:
                    print(f"  failed: {record.get('error', 'unknown error')}", flush=True)
                _checkpoint_results(args.output, records, method_rows)

    summary = _summary_rows(records)
    runs_path = args.output / "scaling_runs.csv"
    methods_path = args.output / "scaling_methods.csv"
    summary_path = args.output / "scaling_summary.csv"
    environment_path = args.output / "scaling_environment.json"
    config_path = args.output / "scaling_config.json"
    _checkpoint_results(args.output, records, method_rows)
    environment_path.write_text(
        json.dumps(_environment(), indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"worker_spec", "worker_output"}
    }
    config_path.write_text(json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8")
    plots = _plots(summary, args.output)
    manifest_path = args.output / "scaling_manifest.json"
    manifest_path.write_text(
        json.dumps(
            _artifact_manifest(
                [
                    runs_path,
                    methods_path,
                    summary_path,
                    environment_path,
                    config_path,
                    *plots,
                ],
                args.output,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failures = [row for row in records if row.get("success") is not True]
    print(
        f"Saved {len(records)} runs, {len(method_rows)} method rows, "
        f"{len(summary)} size summaries, and {len(plots)} plots to {args.output}"
    )
    if failures:
        print(f"Failed cases: {len(failures)}; inspect scaling_runs.csv")
        return 0 if args.allow_failures else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
