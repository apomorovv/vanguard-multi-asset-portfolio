"""Adaptive change-window construction and classical fixed-K search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Iterable

import numpy as np

from .allocation import AllocationOracle, OracleEvaluation
from .portfolio_model import risk_gradient
from .qubo_builder import QUBOModel
from .schemas import PortfolioConstraints, PortfolioProblem, Preferences


@dataclass(frozen=True)
class ChangeWindow:
    indices: tuple[int, ...]
    held_count: int
    frozen_support: tuple[int, ...]
    weak_held: tuple[int, ...]
    promising_unheld: tuple[int, ...]
    group_pressure: np.ndarray


@dataclass
class WindowSearchResult:
    method: str
    best: OracleEvaluation
    runtime: float
    evaluated_supports: int
    feasible_supports: int
    duplicate_supports: int
    trace: list[dict[str, float | int | str]] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


def binding_group_pressure(problem: PortfolioProblem, weights: np.ndarray) -> np.ndarray:
    """Return pressure only for genuinely near-binding group constraints.

    Positive values favor additions needed by a nonzero lower bound; negative
    values discourage additions near a finite upper bound.  A zero lower bound
    must not reward an almost-empty group, which was the source of repeated
    all-``Group_9`` candidate windows in the large benchmark.
    """
    exposure = problem.A @ np.asarray(weights, dtype=float)
    lower_slack = np.maximum(exposure - problem.group_lower, 0.0)
    upper_slack = np.maximum(problem.group_upper - exposure, 0.0)
    scale = np.maximum(problem.group_upper - problem.group_lower, 1e-6)
    activation = np.maximum(0.10 * scale, 1e-6)
    lower_signal = np.where(
        problem.group_lower > 1e-10,
        np.maximum(1.0 - lower_slack / activation, 0.0),
        0.0,
    )
    upper_signal = np.where(
        problem.group_upper < problem.budget - 1e-10,
        np.maximum(1.0 - upper_slack / activation, 0.0),
        0.0,
    )
    pressure = lower_signal - upper_signal
    maximum = np.max(np.abs(pressure))
    return pressure / maximum if maximum > 1e-12 else pressure


def _diverse_order(
    candidates: np.ndarray,
    score: np.ndarray,
    community_labels: np.ndarray | None,
    group_labels: np.ndarray | None = None,
) -> list[int]:
    ordered = candidates[np.argsort(score[candidates])[::-1]].tolist()
    if not ordered:
        return []
    if community_labels is None and group_labels is None:
        return [int(index) for index in ordered]

    communities = None if community_labels is None else np.asarray(community_labels)
    groups = None if group_labels is None else np.asarray(group_labels)

    def round_robin(values: list[int], labels: np.ndarray | None) -> list[int]:
        if labels is None:
            return values
        buckets: dict[int, list[int]] = {}
        for index in values:
            buckets.setdefault(int(labels[index]), []).append(int(index))
        label_order = sorted(
            buckets,
            key=lambda label: score[buckets[label][0]],
            reverse=True,
        )
        result: list[int] = []
        while any(buckets.values()):
            for label in label_order:
                if buckets[label]:
                    result.append(buckets[label].pop(0))
        return result

    if groups is None:
        return round_robin([int(index) for index in ordered], communities)

    group_buckets: dict[int, list[int]] = {}
    for index in ordered:
        group_buckets.setdefault(int(groups[index]), []).append(int(index))
    for group, values in group_buckets.items():
        group_buckets[group] = round_robin(values, communities)
    group_order = sorted(
        group_buckets,
        key=lambda group: score[group_buckets[group][0]],
        reverse=True,
    )
    result: list[int] = []
    while any(group_buckets.values()):
        for group in group_order:
            if group_buckets[group]:
                result.append(group_buckets[group].pop(0))
    return result


def construct_change_window(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    current_weights: np.ndarray,
    relaxation_weights: np.ndarray,
    *,
    window_size: int = 16,
    held_fraction: float = 0.45,
    community_labels: np.ndarray | None = None,
    excluded_unheld: Iterable[int] = (),
    support_tol: float = 1e-8,
) -> ChangeWindow:
    """Choose weak holdings and promising replacements without shrinking the universe."""
    constraints.validate_for(problem)
    current = np.asarray(current_weights, dtype=float).reshape(problem.n)
    relaxation = np.asarray(relaxation_weights, dtype=float).reshape(problem.n)
    support = set(np.flatnonzero(current > support_tol).tolist())
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(problem.lower > support_tol).tolist()
    )
    removable = np.asarray(sorted(support - mandatory), dtype=int)
    eligible = set(np.flatnonzero(constraints.eligible_mask(problem.n)).tolist())
    unheld = np.asarray(sorted(eligible - support), dtype=int)
    maximum_size = len(removable) + len(unheld)
    size = min(max(2, int(window_size)), maximum_size)
    if size < 2 or removable.size == 0 or unheld.size == 0:
        raise ValueError("no valid held/unheld swap window can be formed")
    held_count = min(
        removable.size,
        max(1, int(round(size * float(held_fraction)))),
        size - 1,
    )
    unheld_count = min(unheld.size, size - held_count)
    held_count = min(removable.size, size - unheld_count)

    marginal = (
        preferences.lambda_risk * risk_gradient(current, problem)
        - preferences.lambda_return * problem.mu
        - preferences.lambda_income * problem.y
        + preferences.lambda_cost * problem.c
    )
    # Large positive marginal and small current/relaxation weight indicate a weak holding.
    weakness = marginal - 0.25 * relaxation - 0.10 * current
    weak_order = removable[np.argsort(weakness[removable])[::-1]]
    weak = tuple(int(index) for index in weak_order[:held_count])

    pressure = binding_group_pressure(problem, current)
    group_bonus = pressure[np.asarray(problem.asset_group, dtype=int)]
    promise = (
        relaxation
        - marginal
        + 0.10 * group_bonus
        - 0.02 * problem.c / max(float(np.max(problem.c)), 1e-12)
    )
    excluded = set(int(index) for index in excluded_unheld)
    preferred = np.asarray([index for index in unheld if int(index) not in excluded], dtype=int)
    fallback = np.asarray([index for index in unheld if int(index) in excluded], dtype=int)
    group_labels = np.asarray(problem.asset_group, dtype=int)
    promising_order = _diverse_order(
        preferred,
        promise,
        community_labels,
        group_labels,
    ) + _diverse_order(
        fallback,
        promise,
        community_labels,
        group_labels,
    )
    promising = tuple(promising_order[:unheld_count])
    indices = weak + promising
    frozen = tuple(sorted(support - set(indices)))
    return ChangeWindow(
        indices=indices,
        held_count=len(weak),
        frozen_support=frozen,
        weak_held=weak,
        promising_unheld=promising,
        group_pressure=pressure,
    )


def current_window_bits(window: ChangeWindow) -> np.ndarray:
    bits = np.zeros(len(window.indices), dtype=int)
    bits[: len(window.weak_held)] = 1
    return bits


def evaluate_bitstrings(
    method: str,
    qubo: QUBOModel,
    bitstrings: Iterable[np.ndarray],
    oracle: AllocationOracle,
    frozen_support: Iterable[int],
    *,
    start_time: float | None = None,
) -> WindowSearchResult:
    beginning = time.perf_counter() if start_time is None else start_time
    frozen = set(int(index) for index in frozen_support)
    seen: set[tuple[int, ...]] = set()
    duplicates = 0
    feasible = 0
    best: OracleEvaluation | None = None
    trace: list[dict[str, float | int | str]] = []
    for rank, bits in enumerate(bitstrings):
        x = np.asarray(bits, dtype=int).reshape(-1)
        if x.shape != (qubo.n,) or not qubo.is_cardinality_feasible(x):
            continue
        support = tuple(sorted(frozen | set(qubo.support(x))))
        if support in seen:
            duplicates += 1
            continue
        seen.add(support)
        candidate = oracle.evaluate(support)
        if candidate.feasible:
            feasible += 1
            if best is None or candidate.objective < best.objective:
                best = candidate
        trace.append(
            {
                "candidate": rank,
                "surrogate_energy": qubo.energy(x),
                "objective": candidate.objective,
                "feasible": int(candidate.feasible),
                "best_objective": np.inf if best is None else best.objective,
                "elapsed_seconds": time.perf_counter() - beginning,
            }
        )
    if best is None:
        raise ValueError(f"{method} produced no support accepted by the allocation oracle")
    return WindowSearchResult(
        method=method,
        best=best,
        runtime=time.perf_counter() - beginning,
        evaluated_supports=len(seen),
        feasible_supports=feasible,
        duplicate_supports=duplicates,
        trace=trace,
    )


def enumerate_window(
    qubo: QUBOModel,
    oracle: AllocationOracle,
    frozen_support: Iterable[int],
    *,
    max_candidates: int = 200_000,
) -> WindowSearchResult:
    if qubo.required_ones is None:
        raise ValueError("window enumeration requires fixed cardinality")
    count = comb(qubo.n, qubo.required_ones)
    if count > int(max_candidates):
        raise ValueError(f"window enumeration would require {count:,} candidates")
    states: list[tuple[float, np.ndarray]] = []
    for selected in combinations(range(qubo.n), qubo.required_ones):
        bits = np.zeros(qubo.n, dtype=int)
        bits[list(selected)] = 1
        states.append((qubo.energy(bits), bits))
    states.sort(key=lambda pair: pair[0])
    result = evaluate_bitstrings(
        "classical_enumeration",
        qubo,
        (bits for _, bits in states),
        oracle,
        frozen_support,
    )
    result.metadata["search_space_size"] = count
    return result


def tabu_window_search(
    qubo: QUBOModel,
    oracle: AllocationOracle,
    frozen_support: Iterable[int],
    initial_bits: np.ndarray,
    *,
    max_iterations: int = 80,
    tabu_tenure: int = 7,
    oracle_candidates_per_iteration: int = 4,
    seed: int = 0,
) -> WindowSearchResult:
    """Surrogate-ranked tabu/LNS with exact allocation-oracle scoring."""
    start = time.perf_counter()
    current = np.asarray(initial_bits, dtype=int).reshape(qubo.n).copy()
    if not qubo.is_cardinality_feasible(current):
        raise ValueError("initial_bits violates fixed cardinality")
    rng = np.random.default_rng(seed)
    evaluated: list[np.ndarray] = [current.copy()]
    tabu_until: dict[tuple[int, int], int] = {}
    best_surrogate = qubo.energy(current)
    no_improvement = 0
    for iteration in range(int(max_iterations)):
        selected = np.flatnonzero(current)
        unselected = np.flatnonzero(1 - current)
        moves: list[tuple[float, int, int, np.ndarray]] = []
        for donor in selected:
            for receiver in unselected:
                trial = current.copy()
                trial[donor] = 0
                trial[receiver] = 1
                energy = qubo.energy(trial)
                move = (int(donor), int(receiver))
                if tabu_until.get(move, -1) > iteration and energy >= best_surrogate - 1e-12:
                    continue
                moves.append((energy, int(donor), int(receiver), trial))
        if not moves:
            break
        moves.sort(key=lambda item: item[0])
        shortlist = moves[: max(1, int(oracle_candidates_per_iteration))]
        evaluated.extend(item[3].copy() for item in shortlist)
        chosen = shortlist[0]
        current = chosen[3]
        tabu_until[(chosen[2], chosen[1])] = iteration + int(tabu_tenure)
        if chosen[0] < best_surrogate - 1e-12:
            best_surrogate = chosen[0]
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= max(10, tabu_tenure * 2):
            # Preserve Hamming weight while restarting from a random feasible state.
            current[:] = 0
            current[rng.choice(qubo.n, size=qubo.required_ones, replace=False)] = 1
            evaluated.append(current.copy())
            no_improvement = 0
    result = evaluate_bitstrings(
        "classical_tabu_lns",
        qubo,
        evaluated,
        oracle,
        frozen_support,
        start_time=start,
    )
    result.metadata.update(
        {
            "iterations": int(max_iterations),
            "tabu_tenure": int(tabu_tenure),
            "surrogate_best_energy": float(best_surrogate),
        }
    )
    return result


__all__ = [
    "ChangeWindow",
    "WindowSearchResult",
    "binding_group_pressure",
    "construct_change_window",
    "current_window_bits",
    "enumerate_window",
    "evaluate_bitstrings",
    "tabu_window_search",
]
