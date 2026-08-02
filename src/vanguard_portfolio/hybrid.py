"""Production hybrid optimizer: factor relaxation, LNS/XY windows, exact audit."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from math import comb
from typing import Any

import numpy as np

from .allocation import (
    AllocationOracle,
    OracleEvaluation,
    find_feasible_initial_support,
    solve_relaxation,
)
from .classical_discrete import solve_cardinality_gurobi
from .quantum_solver import XYQAOAConfig, QuantumSearchResult, solve_penalty_qaoa, solve_xy_qaoa
from .qubo_builder import QUBOModel, build_window_qubo
from .schemas import (
    PortfolioConstraints,
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverUnavailableError,
)
from .topology import market_communities
from .window_search import (
    ChangeWindow,
    WindowSearchResult,
    construct_change_window,
    current_window_bits,
    enumerate_window,
    evaluate_bitstrings,
    tabu_window_search,
)


@dataclass(frozen=True)
class HybridConfig:
    iterations: int = 2
    window_size: int = 12
    held_fraction: float = 0.45
    allocation_backend: str = "scipy"
    allocation_options: dict[str, Any] = field(default_factory=dict)
    initial_trials: int = 250
    initial_milp_time_limit: float = 20.0
    classical_tabu_iterations: int = 50
    classical_tabu_tenure: int = 7
    classical_oracle_candidates: int = 4
    enumerate_windows_up_to: int = 10_000
    run_quantum: bool = True
    quantum: XYQAOAConfig = field(default_factory=XYQAOAConfig)
    run_penalty_qaoa: bool = False
    penalty_multiplier: float = 10.0
    use_topology: bool = True
    sparsify_threshold: float = 0.0
    maximum_quantum_edges: int | None = None
    run_gurobi_reference: bool = True
    gurobi_time_limit: float = 300.0
    gurobi_mip_gap: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.iterations) <= 0 or int(self.window_size) < 2:
            raise ValueError("iterations must be positive and window_size must exceed one")
        if not 0.0 < float(self.held_fraction) < 1.0:
            raise ValueError("held_fraction must be strictly between zero and one")
        if float(self.initial_milp_time_limit) <= 0.0:
            raise ValueError("initial_milp_time_limit must be positive")


@dataclass
class HybridRun:
    problem: PortfolioProblem
    preferences: Preferences
    constraints: PortfolioConstraints
    config: HybridConfig
    relaxation: SolveResult
    initial: SolveResult
    results: list[SolveResult]
    best: SolveResult
    windows: list[ChangeWindow]
    classical_searches: list[WindowSearchResult]
    quantum_searches: list[QuantumSearchResult]
    timeline: list[dict[str, float | int | str]]
    oracle_calls: int
    oracle_cache_hits: int
    runtime: float
    skipped: dict[str, str] = field(default_factory=dict)

    def all_results(self) -> list[SolveResult]:
        return [self.relaxation, self.initial, *self.results]

    def summary_records(self) -> list[dict[str, Any]]:
        reference = self.best.objective
        bound = None
        for result in self.results:
            value = result.metadata.get("best_bound")
            if value is not None and np.isfinite(value):
                bound = float(value) if bound is None else max(bound, float(value))
        rows: list[dict[str, Any]] = []
        for result in self.all_results():
            rows.append(
                {
                    "method": result.method,
                    "model_type": result.model_type,
                    "objective": result.objective,
                    "gap_to_best": result.objective - reference,
                    "gap_to_certified_bound": ""
                    if bound is None
                    else result.objective - bound,
                    "runtime_seconds": result.runtime,
                    "success": result.success,
                    "optimal": result.optimal,
                    "feasible": result.feasible,
                    "breaches": result.breaches,
                    "max_violation": result.max_violation,
                    "support_size": int(np.count_nonzero(result.weights > 1e-8)),
                    "expected_return": result.metrics.get("expected_return", np.nan),
                    "volatility": result.metrics.get("volatility", np.nan),
                    "income": result.metrics.get("income", np.nan),
                    "turnover": result.metrics.get("turnover", np.nan),
                    "transaction_cost": result.metrics.get("transaction_cost", np.nan),
                    "best_bound": result.metadata.get("best_bound", ""),
                    "reported_mip_gap": result.metadata.get("reported_mip_gap", ""),
                }
            )
        return rows


def _copy_as_method(
    evaluation: OracleEvaluation,
    method: str,
    runtime: float,
    metadata: dict[str, Any],
) -> SolveResult:
    if evaluation.result is None:
        raise ValueError(f"{method} has no allocation result")
    original = evaluation.result
    return replace(
        original,
        method=method,
        model_type="hybrid_sparse",
        weights=original.weights.copy(),
        runtime=float(runtime),
        metadata={**original.metadata, **metadata},
    )


def _penalty_strength(qubo: QUBOModel, multiplier: float) -> float:
    coefficient_scale = max(
        float(np.max(np.abs(qubo.linear), initial=0.0)),
        float(np.max(np.abs(qubo.quadratic), initial=0.0)),
        1e-4,
    )
    return float(multiplier) * coefficient_scale * max(1, qubo.n)


def run_hybrid_optimizer(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    config: HybridConfig | None = None,
) -> HybridRun:
    """Run the complete constraint-safe classical/quantum portfolio loop."""
    config = config or HybridConfig()
    constraints.validate_for(problem)
    if constraints.exact_cardinality is None:
        raise ValueError("the hybrid optimizer requires exact_cardinality")
    total_start = time.perf_counter()
    relaxation = solve_relaxation(
        problem,
        preferences,
        constraints,
        backend=config.allocation_backend,
        solver_options=dict(config.allocation_options),
    )
    if not relaxation.success:
        raise ValueError(f"continuous relaxation failed: {relaxation.status}")
    oracle = AllocationOracle(
        problem,
        preferences,
        constraints,
        backend=config.allocation_backend,
        solver_options=dict(config.allocation_options),
    )
    initial_evaluation = find_feasible_initial_support(
        oracle,
        relaxation.weights,
        max_trials=config.initial_trials,
        seed=config.seed,
        milp_time_limit=config.initial_milp_time_limit,
    )
    initial = _copy_as_method(
        initial_evaluation,
        "feasible_initial_portfolio",
        initial_evaluation.runtime,
        {"stage": "initialization"},
    )
    current = initial_evaluation
    results: list[SolveResult] = []
    windows: list[ChangeWindow] = []
    classical_searches: list[WindowSearchResult] = []
    quantum_searches: list[QuantumSearchResult] = []
    skipped: dict[str, str] = {}
    timeline: list[dict[str, float | int | str]] = [
        {
            "stage": "initial",
            "iteration": 0,
            "method": initial.method,
            "objective": initial.objective,
            "elapsed_seconds": time.perf_counter() - total_start,
        }
    ]
    communities = (
        market_communities(problem, seed=config.seed) if config.use_topology else None
    )

    for iteration in range(int(config.iterations)):
        try:
            window = construct_change_window(
                problem,
                preferences,
                constraints,
                current.weights,
                relaxation.weights,
                window_size=config.window_size,
                held_fraction=config.held_fraction,
                community_labels=communities,
            )
        except ValueError as exc:
            skipped[f"iteration_{iteration}:window"] = str(exc)
            break
        windows.append(window)
        qubo = build_window_qubo(
            problem,
            preferences,
            current.weights,
            window.indices,
            window.held_count,
            group_pressure=window.group_pressure,
            interaction_threshold=config.sparsify_threshold,
            maximum_edges=config.maximum_quantum_edges,
        )
        starting_bits = current_window_bits(window)
        search_space = comb(qubo.n, qubo.required_ones)
        if search_space <= int(config.enumerate_windows_up_to):
            classical = enumerate_window(
                qubo,
                oracle,
                window.frozen_support,
                max_candidates=config.enumerate_windows_up_to,
            )
        else:
            classical = tabu_window_search(
                qubo,
                oracle,
                window.frozen_support,
                starting_bits,
                max_iterations=config.classical_tabu_iterations,
                tabu_tenure=config.classical_tabu_tenure,
                oracle_candidates_per_iteration=config.classical_oracle_candidates,
                seed=config.seed + iteration,
            )
        classical_searches.append(classical)
        classical_result = _copy_as_method(
            classical.best,
            classical.method,
            classical.runtime,
            {
                **classical.metadata,
                "iteration": iteration,
                "window_size": qubo.n,
                "window_required_ones": qubo.required_ones,
                "evaluated_supports": classical.evaluated_supports,
                "feasible_supports": classical.feasible_supports,
                "duplicate_supports": classical.duplicate_supports,
            },
        )
        results.append(classical_result)
        candidates = [classical.best]

        if config.run_quantum:
            try:
                quantum = solve_xy_qaoa(qubo, starting_bits, config.quantum)
                quantum_searches.append(quantum)
                quantum_allocations = evaluate_bitstrings(
                    quantum.method,
                    qubo,
                    quantum.bitstrings,
                    oracle,
                    window.frozen_support,
                )
                quantum_result = _copy_as_method(
                    quantum_allocations.best,
                    quantum.method,
                    quantum.runtime + quantum_allocations.runtime,
                    {
                        **quantum.metadata,
                        "iteration": iteration,
                        "cardinality_feasibility_rate": quantum.cardinality_feasibility_rate,
                        "expected_surrogate_energy": quantum.expected_surrogate_energy,
                        "best_sampled_energy": quantum.best_sampled_energy,
                        "evaluated_supports": quantum_allocations.evaluated_supports,
                        "feasible_supports": quantum_allocations.feasible_supports,
                        "duplicate_supports": quantum_allocations.duplicate_supports,
                    },
                )
                results.append(quantum_result)
                candidates.append(quantum_allocations.best)
            except (ValueError, RuntimeError, SolverUnavailableError) as exc:
                skipped[f"iteration_{iteration}:xy_qaoa"] = str(exc)

        if config.run_penalty_qaoa and iteration == 0:
            penalty = _penalty_strength(qubo, config.penalty_multiplier)
            penalty_qubo = build_window_qubo(
                problem,
                preferences,
                current.weights,
                window.indices,
                window.held_count,
                group_pressure=window.group_pressure,
                cardinality_penalty=penalty,
                interaction_threshold=config.sparsify_threshold,
                maximum_edges=config.maximum_quantum_edges,
            )
            try:
                penalty_quantum = solve_penalty_qaoa(
                    penalty_qubo,
                    depth=config.quantum.depth,
                    shots=config.quantum.shots,
                    optimizer_maxiter=config.quantum.optimizer_maxiter,
                    seed=config.quantum.seed,
                    top_candidates=config.quantum.top_candidates,
                )
                quantum_searches.append(penalty_quantum)
                penalty_allocations = evaluate_bitstrings(
                    penalty_quantum.method,
                    penalty_qubo,
                    penalty_quantum.bitstrings,
                    oracle,
                    window.frozen_support,
                )
                penalty_result = _copy_as_method(
                    penalty_allocations.best,
                    penalty_quantum.method,
                    penalty_quantum.runtime + penalty_allocations.runtime,
                    {
                        **penalty_quantum.metadata,
                        "iteration": iteration,
                        "cardinality_penalty": penalty,
                        "cardinality_feasibility_rate": (
                            penalty_quantum.cardinality_feasibility_rate
                        ),
                    },
                )
                results.append(penalty_result)
                candidates.append(penalty_allocations.best)
            except (ValueError, RuntimeError) as exc:
                skipped["penalty_qaoa"] = str(exc)

        better = min(candidates, key=lambda item: item.objective)
        if better.objective < current.objective - 1e-12:
            current = better
        timeline.append(
            {
                "stage": "window",
                "iteration": iteration + 1,
                "method": "best_valid",
                "objective": current.objective,
                "elapsed_seconds": time.perf_counter() - total_start,
            }
        )

    if config.run_gurobi_reference:
        try:
            gurobi = solve_cardinality_gurobi(
                problem,
                preferences,
                constraints,
                warm_start=current.weights,
                time_limit=config.gurobi_time_limit,
                mip_gap=config.gurobi_mip_gap,
                seed=config.seed,
            )
            results.append(gurobi)
            if gurobi.success and gurobi.feasible and gurobi.objective < current.objective:
                current = OracleEvaluation(
                    support=tuple(np.flatnonzero(gurobi.weights > 1e-8).tolist()),
                    feasible=True,
                    objective=gurobi.objective,
                    weights=gurobi.weights.copy(),
                    runtime=gurobi.runtime,
                    reason=gurobi.status,
                    result=gurobi,
                )
            timeline.append(
                {
                    "stage": "gurobi",
                    "iteration": len(windows),
                    "method": gurobi.method,
                    "objective": gurobi.objective,
                    "elapsed_seconds": time.perf_counter() - total_start,
                }
            )
        except SolverUnavailableError as exc:
            skipped["gurobi_cardinality_miqp"] = str(exc)

    feasible_results = [
        initial,
        *(result for result in results if result.success and result.feasible),
    ]
    best = min(feasible_results, key=lambda result: result.objective)
    return HybridRun(
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        config=config,
        relaxation=relaxation,
        initial=initial,
        results=results,
        best=best,
        windows=windows,
        classical_searches=classical_searches,
        quantum_searches=quantum_searches,
        timeline=timeline,
        oracle_calls=oracle.calls,
        oracle_cache_hits=oracle.cache_hits,
        runtime=time.perf_counter() - total_start,
        skipped=skipped,
    )


__all__ = ["HybridConfig", "HybridRun", "run_hybrid_optimizer"]
