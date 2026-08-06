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
SCALING_SCHEMA_VERSION = 2
OUTPUT_FILENAMES = (
    "scaling_runs.csv",
    "scaling_methods.csv",
    "scaling_summary.csv",
    "scaling_environment.json",
    "scaling_config.json",
    "scaling_manifest.json",
    "scaling_evidence.png",
    "scaling_evidence.pdf",
    "scaling_runtime.png",
    "scaling_runtime.pdf",
    "scaling_runtime_presentation.png",
    "scaling_runtime_presentation.pdf",
    "scaling_quantum.png",
    "scaling_quantum.pdf",
)
LEGACY_PLOT_FILENAMES = (
    "scaling_quality_and_feasibility.png",
    "scaling_quality_and_feasibility.pdf",
    "scaling_memory_and_first_valid.png",
    "scaling_memory_and_first_valid.pdf",
)
RUN_COLUMNS = (
    "scaling_schema_version",
    "case_config_sha256",
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
    "relaxation_guide_objective",
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
    "relaxation_validation_breaches",
    "relaxation_validation_max_violation",
    "relaxation_accepted",
    "relaxation_bound_available",
    "relaxation_fallback_used",
    "relaxation_fallback_iterate_usable",
    "relaxation_guide_source",
    "initialization_seconds",
    "classical_window_seconds",
    "quantum_window_seconds",
    "window_overhead_seconds",
    "gurobi_seconds",
    "search_end_to_end_seconds",
    "full_end_to_end_seconds",
    "worker_wall_seconds",
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
    "quantum_other_seconds",
    "certification_attempted",
    "certification_completed",
    "certification_optimal",
    "skipped_components",
    "error",
)

CASE_CONFIG_FIELDS = (
    "cardinality",
    "groups",
    "factors",
    "window_size",
    "iterations",
    "backend",
    "relaxation_tolerance",
    "relaxation_max_iter",
    "relaxation_time_limit",
    "relaxation_fallback",
    "allocation_tolerance",
    "allocation_max_iter",
    "minimum_active_weight",
    "maximum_weight",
    "max_turnover",
    "initial_trials",
    "initial_milp_time_limit",
    "classical_tabu_iterations",
    "classical_tabu_tenure",
    "classical_oracle_candidates",
    "enumerate_windows_up_to",
    "quantum",
    "quantum_backend",
    "quantum_depth",
    "quantum_shots",
    "quantum_optimizer_maxiter",
    "quantum_optimizer_starts",
    "quantum_top_candidates",
    "maximum_subspace_states",
    "maximum_quantum_edges",
    "transpile_optimization_level",
    "gurobi",
    "certification_max_assets",
    "gurobi_time_limit",
    "gurobi_mip_gap",
    "materialize_covariance",
    "case_time_limit",
    "seed",
)
CHECKPOINT_CONFIG_MISMATCH_EXIT_CODE = 3


def _case_config_payload(values: argparse.Namespace | dict[str, Any]) -> dict[str, Any]:
    """Return the settings that must remain identical across resumed cases."""

    source = vars(values) if isinstance(values, argparse.Namespace) else values
    payload: dict[str, Any] = {"scaling_schema_version": SCALING_SCHEMA_VERSION}
    for field in CASE_CONFIG_FIELDS:
        if field not in source:
            raise KeyError(f"missing scaling configuration field {field!r}")
        value = source[field]
        payload[field] = str(value) if isinstance(value, Path) else value
    return payload


def _case_config_sha256(values: argparse.Namespace | dict[str, Any]) -> str:
    payload = json.dumps(
        _case_config_payload(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"worker_spec", "worker_output"}
    }
    resolved["scaling_schema_version"] = SCALING_SCHEMA_VERSION
    resolved["case_config_sha256"] = _case_config_sha256(args)
    return resolved


def _checkpoint_config_mismatch(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    """Describe settings that make an existing checkpoint unsafe to resume."""

    fields = ("scaling_schema_version", "case_config_sha256")
    return [
        f"{field}: existing={existing.get(field)!r}, requested={current.get(field)!r}"
        for field in fields
        if existing.get(field) != current.get(field)
    ]


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
                    if value.lstrip("+-").isdigit():
                        row[key] = int(value)
                        continue
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade plot-only legacy rows without pretending they are resumable."""

    row = dict(record)
    raw_version = row.get("scaling_schema_version", 1)
    version = 1 if raw_version in (None, "") else int(raw_version)
    if version < 2:
        first_valid = row.get("time_to_first_valid_seconds")
        construction = row.get("data_generation_seconds")
        if first_valid not in (None, "") and construction not in (None, ""):
            row["time_to_first_valid_seconds"] = float(first_valid) + float(construction)
    if row.get("quantum_other_seconds") in (None, ""):
        quantum_total = row.get("quantum_window_seconds")
        if quantum_total not in (None, ""):
            accounted = sum(
                float(row.get(field, 0.0) or 0.0)
                for field in (
                    "quantum_angle_seconds",
                    "quantum_sampler_seconds",
                    "quantum_allocation_seconds",
                )
            )
            row["quantum_other_seconds"] = max(float(quantum_total) - accounted, 0.0)
    return row


def _normalise_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalise_record(record) for record in records]


def _peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    byte_count = value if sys.platform == "darwin" else value * 1024.0
    return byte_count / 1024.0**3


def _relative_gap(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-12))


def _timing_residual(total: float, parts: Iterable[float], *, label: str) -> float:
    residual = float(total) - float(sum(parts))
    tolerance = max(1.0e-6, 1.0e-6 * max(abs(float(total)), 1.0))
    if residual < -tolerance:
        raise RuntimeError(
            f"{label} timing components exceed their measured total by "
            f"{-residual:.6g} seconds"
        )
    return max(residual, 0.0)


def _run_case(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size = int(spec["n_assets"])
    cardinality = int(spec["cardinality"])
    groups = int(spec["groups"])
    factors = int(spec["factors"])
    window_size = int(spec["window_size"])
    if not 1 <= cardinality < size:
        raise ValueError(
            f"cardinality must satisfy 1 <= K < n_assets; got K={cardinality}, "
            f"n_assets={size}"
        )
    if not 1 <= groups <= size:
        raise ValueError(f"groups must satisfy 1 <= groups <= n_assets; got {groups}")
    if not 1 <= factors <= size:
        raise ValueError(f"factors must satisfy 1 <= factors <= n_assets; got {factors}")
    if not 2 <= window_size <= size:
        raise ValueError(
            f"window_size must satisfy 2 <= window_size <= n_assets; got {window_size}"
        )
    seed = int(spec["seed"])
    data_start = time.perf_counter()
    problem = generate_factor_universe(
        n_assets=size,
        n_groups=groups,
        n_factors=factors,
        seed=seed,
        current_cardinality=cardinality,
        materialize_covariance=bool(spec["materialize_covariance"]),
    )
    problem.max_turnover = float(spec["max_turnover"])
    data_seconds = time.perf_counter() - data_start

    minimum_weight = float(spec["minimum_active_weight"])
    maximum_weight = float(spec["maximum_weight"])
    if not 0.0 <= minimum_weight <= maximum_weight:
        raise ValueError(
            "minimum_active_weight and maximum_weight must satisfy "
            f"0 <= minimum <= maximum; got {minimum_weight} and {maximum_weight}"
        )
    if cardinality * minimum_weight > problem.budget + 1.0e-12:
        raise ValueError("cardinality times minimum_active_weight exceeds the budget")
    if cardinality * maximum_weight < problem.budget - 1.0e-12:
        raise ValueError("cardinality times maximum_weight cannot fill the budget")
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
    relaxation_options: dict[str, Any] = {
        "tol": float(spec["relaxation_tolerance"]),
        "max_iter": int(spec["relaxation_max_iter"]),
        "time_limit": float(spec["relaxation_time_limit"]),
    }
    if str(spec["allocation_backend"]).strip().lower() == "osqp":
        relaxation_options["polish"] = False
    config = HybridConfig(
        iterations=int(spec["iterations"]),
        window_size=window_size,
        allocation_backend=str(spec["allocation_backend"]),
        allocation_options={
            "tol": float(spec["allocation_tolerance"]),
            "max_iter": int(spec["allocation_max_iter"]),
        },
        relaxation_options=relaxation_options,
        allow_relaxation_fallback=bool(spec["relaxation_fallback"]),
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
    relaxation_accepted = bool(run.relaxation.success)
    relaxation_bound_available = bool(run.relaxation.success and run.relaxation.optimal)
    relaxation_fallback_used = bool(relaxation_metadata.get("fallback_used", False))
    solver_time_to_valid = float(run.timeline[0]["elapsed_seconds"])
    time_to_valid = data_seconds + solver_time_to_valid
    initialization_seconds = max(solver_time_to_valid - relaxation_seconds, 0.0)
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
    window_overhead = _timing_residual(
        search_solver_seconds,
        (
            relaxation_seconds,
            initialization_seconds,
            classical_seconds,
            quantum_seconds,
        ),
        label="hybrid search",
    )
    angle_seconds = sum(float(row.get("angle_optimization_seconds", 0.0)) for row in quantum_rows)
    sampler_seconds = sum(float(row.get("sampler_total_seconds", 0.0)) for row in quantum_rows)
    allocation_seconds = sum(
        float(row.get("allocation_oracle_seconds", 0.0)) for row in quantum_rows
    )
    quantum_other_seconds = _timing_residual(
        quantum_seconds,
        (angle_seconds, sampler_seconds, allocation_seconds),
        label="XY-QAOA window",
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
    certification_completed = bool(gurobi is not None and gurobi.success and gurobi.feasible)
    certification_optimal = bool(certification_completed and gurobi.optimal)
    case_config_sha256 = str(
        spec.get("case_config_sha256") or _case_config_sha256(spec)
    )
    record: dict[str, Any] = {
        "scaling_schema_version": SCALING_SCHEMA_VERSION,
        "case_config_sha256": case_config_sha256,
        "study_tier": (
            "certified_reference"
            if certification_optimal
            else ("certification_attempted" if certify else "scalable_hybrid_search")
        ),
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
        "relaxation_objective": (
            run.relaxation.objective if relaxation_bound_available else ""
        ),
        "relaxation_guide_objective": relaxation_metadata.get(
            "guide_objective", run.relaxation.objective
        ),
        "relative_gap_to_relaxation": (
            _relative_gap(hybrid_best.objective, run.relaxation.objective)
            if relaxation_bound_available
            else ""
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
        "relaxation_validation_breaches": int(run.relaxation.breaches),
        "relaxation_validation_max_violation": float(run.relaxation.max_violation),
        "relaxation_accepted": relaxation_accepted,
        "relaxation_bound_available": relaxation_bound_available,
        "relaxation_fallback_used": relaxation_fallback_used,
        "relaxation_fallback_iterate_usable": relaxation_metadata.get(
            "fallback_iterate_usable", ""
        ),
        "relaxation_guide_source": relaxation_metadata.get("guide_source", ""),
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
        "quantum_other_seconds": quantum_other_seconds,
        "certification_attempted": certify,
        "certification_completed": certification_completed,
        "certification_optimal": certification_optimal,
        "skipped_components": json.dumps(run.skipped, sort_keys=True),
        "error": "",
    }
    method_rows = []
    for row in run.summary_records():
        method_rows.append(
            {
                "scaling_schema_version": SCALING_SCHEMA_VERSION,
                "case_config_sha256": case_config_sha256,
                "n_assets": size,
                "repetition": int(spec["repetition"]),
                "seed": seed,
                **row,
            }
        )
    return record, method_rows


def _worker(spec_path: Path, output_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    worker_start = time.perf_counter()
    try:
        record, methods = _run_case(spec)
        payload = {"record": record, "methods": methods}
    except Exception as exc:  # noqa: BLE001 - a failed case must remain auditable
        payload = {
            "record": {
                "scaling_schema_version": SCALING_SCHEMA_VERSION,
                "case_config_sha256": str(
                    spec.get("case_config_sha256") or _case_config_sha256(spec)
                ),
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
                "certification_optimal": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "methods": [],
        }
    payload["record"]["worker_wall_seconds"] = time.perf_counter() - worker_start
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
    records = _normalise_records(records)
    numeric_fields = (
        "search_end_to_end_seconds",
        "full_end_to_end_seconds",
        "worker_wall_seconds",
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
        "gurobi_reported_gap",
        "peak_rss_gib",
        "factor_risk_storage_mib",
        "dense_covariance_gib_avoided",
        "quantum_angle_seconds",
        "quantum_sampler_seconds",
        "quantum_allocation_seconds",
        "quantum_other_seconds",
        "quantum_cardinality_rate",
    )
    rows: list[dict[str, Any]] = []
    for size in sorted({int(row["n_assets"]) for row in records}):
        group = [row for row in records if int(row["n_assets"]) == size]
        successful = [row for row in group if row.get("success") is True]
        failed = [row for row in group if row.get("success") is not True]
        zero_breach_runs = sum(
            int(row.get("breaches", 1)) == 0 for row in successful
        )
        bound_runs = sum(
            bool(row.get("relaxation_bound_available")) for row in successful
        )
        summary: dict[str, Any] = {
            "n_assets": size,
            "runs": len(group),
            "successful_runs": len(successful),
            "success_rate": len(successful) / max(len(group), 1),
            "zero_breach_runs": zero_breach_runs,
            "zero_breach_rate": zero_breach_runs / max(len(successful), 1),
            "certified_runs": sum(
                bool(
                    row.get(
                        "certification_optimal",
                        row.get("certification_completed", False),
                    )
                )
                for row in successful
            ),
            "relaxation_acceptance_rate": sum(
                bool(row.get("relaxation_accepted")) for row in successful
            )
            / max(len(successful), 1),
            "relaxation_bound_runs": bound_runs,
            "relaxation_bound_rate": bound_runs / max(len(successful), 1),
            "relaxation_fallback_rate": sum(
                bool(row.get("relaxation_fallback_used")) for row in successful
            )
            / max(len(successful), 1),
        }
        measurement_tiers = sorted(
            {
                str(row.get("measurement_tier", "")).strip()
                for row in group
                if str(row.get("measurement_tier", "")).strip()
            }
        )
        if measurement_tiers:
            summary["measurement_tier"] = " | ".join(measurement_tiers)
        failed_seconds = [
            float(row["worker_wall_seconds"])
            for row in failed
            if row.get("worker_wall_seconds") not in (None, "")
            and np.isfinite(float(row["worker_wall_seconds"]))
        ]
        if failed_seconds:
            summary["failed_worker_seconds_median"] = float(np.median(failed_seconds))
        reasons = sorted({str(row.get("error", "")).strip() for row in failed if row.get("error")})
        if reasons:
            summary["failure_reasons"] = " | ".join(reasons)
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


def _median_band(
    ax: Any,
    rows: list[dict[str, Any]],
    field: str,
    *,
    records: list[dict[str, Any]],
    color: str,
    label: str,
    marker: str,
) -> None:
    usable = [row for row in rows if f"{field}_median" in row]
    if not usable:
        return
    x = np.asarray([int(row["n_assets"]) for row in usable], dtype=float)
    center = np.asarray([float(row[f"{field}_median"]) for row in usable])
    q1 = np.asarray([float(row[f"{field}_q1"]) for row in usable])
    q3 = np.asarray([float(row[f"{field}_q3"]) for row in usable])
    ax.plot(
        x,
        center,
        color=color,
        marker=marker,
        linewidth=2.0,
        label=label,
        zorder=3,
    )
    ax.fill_between(x, q1, q3, color=color, alpha=0.15, linewidth=0.0)
    for size in x.astype(int):
        values = [
            float(row[field])
            for row in records
            if row.get("success") is True
            and int(row.get("n_assets", -1)) == size
            and row.get(field) not in (None, "")
            and np.isfinite(float(row[field]))
        ]
        if not values:
            continue
        offsets = np.linspace(-0.012, 0.012, len(values))
        ax.scatter(
            size * np.power(10.0, offsets),
            values,
            s=18,
            facecolors="white",
            edgecolors=color,
            linewidths=0.8,
            alpha=0.85,
            zorder=4,
        )


def _representative_stage_rows(
    summary: list[dict[str, Any]],
    records: list[dict[str, Any]],
    stage_fields: list[str],
) -> list[dict[str, Any]]:
    """Select the actual repetition closest to each median total runtime."""

    representatives: list[dict[str, Any]] = []
    for aggregate in summary:
        size = int(aggregate["n_assets"])
        candidates = [
            row
            for row in records
            if row.get("success") is True
            and int(row.get("n_assets", -1)) == size
            and row.get("search_end_to_end_seconds") not in (None, "")
            and all(row.get(field) not in (None, "") for field in stage_fields)
        ]
        if candidates:
            target = float(aggregate["search_end_to_end_seconds_median"])
            selected = min(
                candidates,
                key=lambda row: (
                    abs(float(row["search_end_to_end_seconds"]) - target),
                    int(row.get("repetition", 0) or 0),
                ),
            )
            representative = {
                "n_assets": size,
                "total_seconds": float(selected["search_end_to_end_seconds"]),
                "fallback_used": bool(selected.get("relaxation_fallback_used")),
                **{field: float(selected[field]) for field in stage_fields},
            }
        else:
            values = [
                float(aggregate.get(f"{field}_median", 0.0) or 0.0)
                for field in stage_fields
            ]
            target = float(
                aggregate.get("search_end_to_end_seconds_median", sum(values))
            )
            raw_total = sum(values)
            scale = target / raw_total if raw_total > 0.0 else 0.0
            representative = {
                "n_assets": size,
                "total_seconds": target,
                "fallback_used": float(
                    aggregate.get("relaxation_fallback_rate", 0.0)
                )
                > 0.0,
                **{
                    field: value * scale
                    for field, value in zip(stage_fields, values)
                },
            }
        representatives.append(representative)
    return representatives


def _plots(
    summary: list[dict[str, Any]],
    output: Path,
    records: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Create the compact scaling, runtime-anatomy, and quantum figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    records = _normalise_records(records or [])
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
    failed_rows = [row for row in summary if "failed_worker_seconds_median" in row]
    if failed_rows:
        axes[0, 0].scatter(
            [int(row["n_assets"]) for row in failed_rows],
            [float(row["failed_worker_seconds_median"]) for row in failed_rows],
            marker="x",
            s=65,
            linewidths=2.0,
            color="#D1495B",
            label="Failed/time-limited case",
            zorder=4,
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Global asset universe size")
    axes[0, 0].set_ylabel("Wall-clock seconds; median and IQR")
    axes[0, 0].set_title("End-to-end speed")
    axes[0, 0].grid(alpha=0.25, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8)

    gap_rows = [row for row in summary if "relative_gap_to_relaxation_median" in row]

    def bound_count(row: dict[str, Any]) -> int:
        return int(
            row.get(
                "relaxation_bound_runs",
                round(
                    float(row.get("relaxation_bound_rate", 0.0))
                    * int(row.get("successful_runs", 0))
                ),
            )
        )

    complete_gap_rows = [
        row
        for row in gap_rows
        if bound_count(row) == int(row.get("successful_runs", 0))
    ]
    partial_gap_rows = [row for row in gap_rows if row not in complete_gap_rows]
    _errorbar(
        axes[0, 1],
        complete_gap_rows,
        "relative_gap_to_relaxation",
        marker="o",
        linewidth=2.0,
        color="#00A6A6",
        label="Validated incumbent vs solved relaxation",
    )
    if partial_gap_rows:
        _errorbar(
            axes[0, 1],
            partial_gap_rows,
            "relative_gap_to_relaxation",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            linewidth=1.8,
            color="#00A6A6",
            label="Partial solved-bound coverage",
        )
        for row in partial_gap_rows:
            axes[0, 1].annotate(
                (
                    f"{bound_count(row)}/"
                    f"{int(row.get('successful_runs', 0))} bounds"
                ),
                (
                    int(row["n_assets"]),
                    float(row["relative_gap_to_relaxation_median"]),
                ),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
                color="#51606F",
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
        int(
            row.get(
                "zero_breach_runs",
                round(
                    float(row.get("zero_breach_rate", 0.0))
                    * int(row.get("successful_runs", 0))
                ),
            )
        )
        for row in summary
    )
    axes[0, 1].set_title(
        f"Solution quality ({zero_breach_runs}/{successful_runs} valid runs had zero breaches)"
    )
    uncertified_guide_runs = sum(
        int(row.get("successful_runs", 0))
        - bound_count(row)
        for row in summary
    )
    if uncertified_guide_runs:
        axes[0, 1].text(
            0.98,
            0.04,
            f"{uncertified_guide_runs} valid run(s) had no solved relaxation bound",
            transform=axes[0, 1].transAxes,
            fontsize=8,
            color="#7B61A8",
            ha="right",
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

    stage_specs = (
        (
            "data_generation_seconds",
            "Data/model construction",
            "#5F6B7A",
            "o",
        ),
        (
            "relaxation_seconds",
            "Factor-QP relaxation",
            "#0B5CAD",
            "s",
        ),
        ("initialization_seconds", "Support construction", "#93B9D8", "D"),
        (
            "classical_window_seconds",
            "Classical LNS",
            "#00A6A6",
            "^",
        ),
        (
            "quantum_window_seconds",
            "XY-QAOA + allocation",
            "#F28E2B",
            "v",
        ),
        ("window_overhead_seconds", "Other orchestration", "#7B61A8", "P"),
    )
    active_stage_specs = [
        spec
        for spec in stage_specs
        if any(float(row.get(f"{spec[0]}_median", 0.0) or 0.0) > 0.0 for row in summary)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for field, label, color, marker in active_stage_specs:
        _errorbar(
            axes[0],
            summary,
            field,
            marker=marker,
            linewidth=1.8,
            color=color,
            label=label,
        )
    fallback_rows = [
        row
        for row in summary
        if float(row.get("relaxation_fallback_rate", 0.0)) > 0.0
        and "relaxation_seconds_median" in row
    ]
    if fallback_rows:
        axes[0].scatter(
            [int(row["n_assets"]) for row in fallback_rows],
            [float(row["relaxation_seconds_median"]) for row in fallback_rows],
            marker="o",
            s=95,
            facecolors="none",
            edgecolors="#D1495B",
            linewidths=1.8,
            label="Guide fallback used",
            zorder=5,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Global asset universe size")
    axes[0].set_ylabel("Seconds; median and IQR")
    axes[0].set_title("Absolute runtime by stage")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)

    stage_fields = [field for field, *_ in active_stage_specs]
    representative_rows = _representative_stage_rows(
        summary,
        records,
        stage_fields,
    )
    positions = np.arange(len(representative_rows))
    stage_values = np.asarray(
        [
            [float(row.get(field, 0.0) or 0.0) for row in representative_rows]
            for field, *_ in active_stage_specs
        ],
        dtype=float,
    )
    stage_totals = np.sum(stage_values, axis=0)
    safe_totals = np.where(stage_totals > 0.0, stage_totals, 1.0)
    bottom = np.zeros(len(representative_rows), dtype=float)
    for values, (_, label, color, _) in zip(stage_values, active_stage_specs):
        shares = 100.0 * values / safe_totals
        axes[1].bar(
            positions,
            shares,
            bottom=bottom,
            color=color,
            label=label,
        )
        bottom += shares
    tick_labels = []
    for row in representative_rows:
        label = f"{int(row['n_assets']):,}"
        if bool(row.get("fallback_used")):
            label += "*"
        tick_labels.append(label)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(tick_labels, rotation=35, ha="right")
    axes[1].set_ylim(0.0, 112.0)
    axes[1].set_xlabel("Global asset universe size")
    axes[1].set_ylabel("Share of representative median-total run")
    for position, row in zip(positions, representative_rows):
        total = row.get("total_seconds")
        if total not in (None, ""):
            axes[1].text(
                position,
                102.0,
                f"{float(total):.1f}s",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=35,
            )
    composition_title = "Representative median-run composition"
    if fallback_rows:
        composition_title += " (* = guide fallback)"
    axes[1].set_title(composition_title)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Hybrid runtime anatomy: global guide, classical search, and XY-QAOA")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"scaling_runtime.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        created.append(path)
    plt.close(fig)

    early_presentation_rows = [
        row for row in representative_rows if int(row["n_assets"]) <= 2_000
    ]
    presentation_rows = (
        early_presentation_rows
        if len(early_presentation_rows) >= 2
        else representative_rows
    )
    if len(presentation_rows) > 6:
        selected_indices = np.unique(
            np.rint(np.linspace(0, len(presentation_rows) - 1, 6)).astype(int)
        )
        presentation_rows = [presentation_rows[index] for index in selected_indices]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    _median_band(
        axes[0],
        summary,
        "search_end_to_end_seconds",
        records=records,
        color="#0B5CAD",
        label="Complete hybrid search",
        marker="o",
    )
    _median_band(
        axes[0],
        summary,
        "time_to_first_valid_seconds",
        records=records,
        color="#00A6A6",
        label="First independently valid portfolio",
        marker="s",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Global asset universe size")
    axes[0].set_ylabel("Wall-clock seconds")
    axes[0].set_title("End-to-end scalability")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[0].text(
        0.02,
        0.03,
        "Line = median; band = IQR; hollow dots = successful repetitions",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color="#51606F",
    )

    presentation_positions = np.arange(len(presentation_rows))
    presentation_bottom = np.zeros(len(presentation_rows), dtype=float)
    for field, label, color, _ in active_stage_specs:
        values = np.asarray(
            [float(row.get(field, 0.0) or 0.0) for row in presentation_rows]
        )
        axes[1].bar(
            presentation_positions,
            values,
            bottom=presentation_bottom,
            color=color,
            label=label,
        )
        presentation_bottom += values
    presentation_labels = [
        f"{int(row['n_assets']):,}{'*' if row.get('fallback_used') else ''}"
        for row in presentation_rows
    ]
    axes[1].set_xticks(presentation_positions)
    axes[1].set_xticklabels(presentation_labels, rotation=25, ha="right")
    axes[1].set_xlabel("Global asset universe size")
    axes[1].set_ylabel("Wall-clock seconds")
    runtime_title = "Where the runtime is spent"
    if any(bool(row.get("fallback_used")) for row in presentation_rows):
        runtime_title += " (* = guide fallback)"
    axes[1].set_title(runtime_title)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    maximum_total = max(
        (float(row["total_seconds"]) for row in presentation_rows),
        default=1.0,
    )
    axes[1].set_ylim(0.0, 1.18 * maximum_total)
    for position, row in zip(presentation_positions, presentation_rows):
        axes[1].text(
            position,
            float(row["total_seconds"]),
            f"{float(row['total_seconds']):.1f}s",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    fig.suptitle("Hybrid solver scaling with a fixed-size quantum window")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"scaling_runtime_presentation.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        created.append(path)
    plt.close(fig)

    quantum_fields = (
        (
            "quantum_angle_seconds",
            "CPU fixed-weight angle optimization",
            "#7B61A8",
            "o",
        ),
        ("quantum_sampler_seconds", "Circuit sampling backend", "#F28E2B", "s"),
        ("quantum_allocation_seconds", "Exact allocation oracle", "#00A6A6", "^"),
        (
            "quantum_other_seconds",
            "Circuit setup, ranking, and orchestration",
            "#9C9C9C",
            "P",
        ),
    )
    quantum_summary = [
        row
        for row in summary
        if any(float(row.get(f"{field}_median", 0.0) or 0.0) > 0.0 for field, *_ in quantum_fields)
    ]
    if quantum_summary:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
        for field, label, color, marker in quantum_fields:
            if not any(
                float(row.get(f"{field}_median", 0.0) or 0.0) > 0.0
                for row in quantum_summary
            ):
                continue
            _errorbar(
                axes[0],
                quantum_summary,
                field,
                marker=marker,
                linewidth=1.8,
                color=color,
                label=label,
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Global asset universe size")
        axes[0].set_ylabel("Seconds; median and IQR")
        axes[0].set_title("Fixed-window XY-QAOA phase timing")
        axes[0].grid(alpha=0.25, which="both")
        axes[0].legend(frameon=False, fontsize=8)

        cardinality_rows = [
            row for row in quantum_summary if "quantum_cardinality_rate_median" in row
        ]
        _errorbar(
            axes[1],
            cardinality_rows,
            "quantum_cardinality_rate",
            marker="o",
            linewidth=2.0,
            color="#0B5CAD",
        )
        rates = [
            float(row["quantum_cardinality_rate_median"]) for row in cardinality_rows
        ]
        axes[1].set_xscale("log")
        axes[1].set_ylim(0.9 if rates and min(rates) >= 0.9 else 0.0, 1.01)
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        axes[1].set_xlabel("Global asset universe size")
        axes[1].set_ylabel("Cardinality-valid sampled states")
        axes[1].set_title("XY mixer preserves the exact-cardinality subspace")
        axes[1].grid(alpha=0.25, which="both")
        axes[1].text(
            0.02,
            0.03,
            "Correctness diagnostic—not evidence of quantum advantage",
            transform=axes[1].transAxes,
            fontsize=8,
            color="#51606F",
        )
        fig.suptitle("Quantum-window anatomy and feasibility")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"scaling_quantum.{suffix}"
            fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
            created.append(path)
        plt.close(fig)
    return created


def create_scaling_plots(
    records: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    """Aggregate raw case records and write the evaluator-facing scaling figures."""

    output.mkdir(parents=True, exist_ok=True)
    normalised = _normalise_records(records)
    return _plots(_summary_rows(normalised), output, normalised)


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
        default=1e-8,
        help="explicit OSQP tolerance for the full-universe guide relaxation",
    )
    parser.add_argument("--relaxation-max-iter", type=int, default=250_000)
    parser.add_argument(
        "--relaxation-time-limit",
        type=float,
        default=30.0,
        help="native solver limit for the guide relaxation, in seconds",
    )
    parser.add_argument(
        "--relaxation-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "continue with a usable guide iterate or the current portfolio when "
            "the guide relaxation reaches its explicit limit"
        ),
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
        default=180.0,
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
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("asset sizes must be unique")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.case_time_limit <= 0.0:
        raise ValueError("case_time_limit must be positive")
    if args.relaxation_time_limit <= 0.0:
        raise ValueError("relaxation_time_limit must be positive")

    current_config = _resolved_config(args)
    config_path = args.output / "scaling_config.json"
    existing_outputs = [
        args.output / name
        for name in (*OUTPUT_FILENAMES, *LEGACY_PLOT_FILENAMES)
        if (args.output / name).is_file()
    ]
    if existing_outputs and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{args.output} contains a scaling result; pass --resume or --overwrite"
        )
    if args.resume and (args.output / "scaling_runs.csv").is_file():
        if not config_path.is_file():
            print(
                "Existing scaling checkpoints have no configuration fingerprint; "
                "use --overwrite once before resuming with this runner.",
                file=sys.stderr,
            )
            return CHECKPOINT_CONFIG_MISMATCH_EXIT_CODE
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"Could not validate existing scaling_config.json: {exc}. "
                "Use --overwrite to start a compatible checkpoint.",
                file=sys.stderr,
            )
            return CHECKPOINT_CONFIG_MISMATCH_EXIT_CODE
        mismatch = _checkpoint_config_mismatch(existing_config, current_config)
        if mismatch:
            print(
                "Existing scaling checkpoints were produced by an incompatible "
                "configuration or result schema:\n  - "
                + "\n  - ".join(mismatch)
                + "\nUse --overwrite or choose a new output directory; stale rows "
                "will not be mixed with this run.",
                file=sys.stderr,
            )
            return CHECKPOINT_CONFIG_MISMATCH_EXIT_CODE
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(current_config, indent=2) + "\n", encoding="utf-8")
    records = _read_csv(args.output / "scaling_runs.csv") if args.resume else []
    method_rows = _read_csv(args.output / "scaling_methods.csv") if args.resume else []
    completed_keys = {
        (int(row["n_assets"]), int(row["repetition"]), int(row["seed"]))
        for row in records
        if row.get("success") is True
        and row.get("n_assets") not in (None, "")
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
                records = [
                    row
                    for row in records
                    if (
                        int(row.get("n_assets", -1)),
                        int(row.get("repetition", -1)),
                        int(row.get("seed", -1)),
                    )
                    != case_key
                ]
                method_rows = [
                    row
                    for row in method_rows
                    if (
                        int(row.get("n_assets", -1)),
                        int(row.get("repetition", -1)),
                        int(row.get("seed", -1)),
                    )
                    != case_key
                ]
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
                        "scaling_schema_version": SCALING_SCHEMA_VERSION,
                        "case_config_sha256": current_config["case_config_sha256"],
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
                case_start = time.perf_counter()
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
                            "scaling_schema_version": SCALING_SCHEMA_VERSION,
                            "case_config_sha256": current_config["case_config_sha256"],
                            "study_tier": "worker_timeout",
                            "n_assets": int(size),
                            "repetition": repetition,
                            "seed": seed,
                            "success": False,
                            "certification_optimal": False,
                            "worker_wall_seconds": float(args.case_time_limit),
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
                            "scaling_schema_version": SCALING_SCHEMA_VERSION,
                            "case_config_sha256": current_config["case_config_sha256"],
                            "study_tier": "worker_failure",
                            "n_assets": int(size),
                            "repetition": repetition,
                            "seed": seed,
                            "success": False,
                            "certification_optimal": False,
                            "worker_wall_seconds": time.perf_counter() - case_start,
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
    _checkpoint_results(args.output, records, method_rows)
    environment_path.write_text(
        json.dumps(_environment(), indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(json.dumps(current_config, indent=2) + "\n", encoding="utf-8")
    plots = _plots(summary, args.output, records)
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
