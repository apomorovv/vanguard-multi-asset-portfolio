"""Build the fixed-cardinality surrogate searched inside a change window.

The QUBO is deliberately a *candidate generator*.  Exact percentages and all
financial guardrails are evaluated by :class:`AllocationOracle`; therefore a
sparsified hardware surrogate can affect candidate quality but never final
feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Iterable

import numpy as np

from .portfolio_model import objective_value
from .schemas import PortfolioProblem, Preferences


@dataclass(frozen=True)
class IsingModel:
    constant: float
    fields: np.ndarray
    couplings: dict[tuple[int, int], float]


@dataclass
class QUBOModel:
    """Binary energy ``offset + linear @ x + x.T @ quadratic @ x``."""

    linear: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0
    variable_names: tuple[str, ...] = ()
    window_indices: tuple[int, ...] = ()
    required_ones: int | None = None
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float).reshape(-1)
        self.quadratic = np.asarray(self.quadratic, dtype=float)
        n = self.linear.size
        if self.quadratic.shape != (n, n):
            raise ValueError(f"quadratic must have shape ({n}, {n})")
        self.quadratic = 0.5 * (self.quadratic + self.quadratic.T)
        if not np.all(np.isfinite(self.linear)) or not np.all(np.isfinite(self.quadratic)):
            raise ValueError("QUBO coefficients must be finite")
        if not self.variable_names:
            self.variable_names = tuple(f"x_{index}" for index in range(n))
        if len(self.variable_names) != n:
            raise ValueError("variable_names must match the QUBO dimension")
        if self.window_indices and len(self.window_indices) != n:
            raise ValueError("window_indices must match the QUBO dimension")
        if self.required_ones is not None and not 0 <= self.required_ones <= n:
            raise ValueError("required_ones is outside the binary dimension")

    @property
    def n(self) -> int:
        return self.linear.size

    def energy(self, bits: np.ndarray | Iterable[int]) -> float:
        x = np.asarray(bits, dtype=float).reshape(-1)
        if x.shape != (self.n,) or np.any((x != 0.0) & (x != 1.0)):
            raise ValueError(f"bits must be a binary vector of shape ({self.n},)")
        return float(self.offset + self.linear @ x + x @ self.quadratic @ x)

    def is_cardinality_feasible(self, bits: np.ndarray | Iterable[int]) -> bool:
        return self.required_ones is None or int(np.sum(bits)) == self.required_ones

    def support(self, bits: np.ndarray | Iterable[int]) -> tuple[int, ...]:
        if not self.window_indices:
            raise ValueError("this QUBO has no asset window mapping")
        x = np.asarray(bits, dtype=int).reshape(-1)
        return tuple(
            self.window_indices[index] for index in np.flatnonzero(x)
        )

    def to_ising(self, *, drop_tolerance: float = 0.0) -> IsingModel:
        """Convert to ``constant + sum h_i Z_i + sum J_ij Z_i Z_j``."""
        row_sum = self.quadratic.sum(axis=1)
        fields = -0.5 * self.linear - 0.5 * row_sum
        constant = float(
            self.offset
            + 0.5 * self.linear.sum()
            + 0.25 * self.quadratic.sum()
            + 0.25 * np.trace(self.quadratic)
        )
        couplings: dict[tuple[int, int], float] = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                value = 0.5 * float(self.quadratic[i, j])
                if abs(value) > drop_tolerance:
                    couplings[(i, j)] = value
        return IsingModel(constant, fields, couplings)

    def exact_best(self, max_states: int = 2_000_000) -> tuple[np.ndarray, float]:
        """Enumerate a tiny fixed-cardinality window for certification."""
        required = self.required_ones
        states = 2**self.n if required is None else comb(self.n, required)
        if states > int(max_states):
            raise ValueError(f"exact QUBO enumeration would require {states:,} states")
        best_bits: np.ndarray | None = None
        best_energy = np.inf
        if required is None:
            iterator = (
                np.asarray([(state >> bit) & 1 for bit in range(self.n)], dtype=int)
                for state in range(2**self.n)
            )
        else:
            def fixed_weight_iterator():
                for selected in combinations(range(self.n), required):
                    bits = np.zeros(self.n, dtype=int)
                    bits[list(selected)] = 1
                    yield bits

            iterator = fixed_weight_iterator()
        for bits in iterator:
            value = self.energy(bits)
            if value < best_energy:
                best_bits = bits.copy()
                best_energy = value
        if best_bits is None:
            raise ValueError("QUBO has no states")
        return best_bits, float(best_energy)


def _sparsify_quadratic(
    quadratic: np.ndarray,
    *,
    interaction_threshold: float,
    maximum_edges: int | None,
) -> tuple[np.ndarray, int]:
    result = np.asarray(quadratic, dtype=float).copy()
    n = result.shape[0]
    pairs = [(i, j, abs(result[i, j])) for i in range(n) for j in range(i + 1, n)]
    keep: set[tuple[int, int]] | None = None
    if maximum_edges is not None and len(pairs) > int(maximum_edges):
        strongest = sorted(pairs, key=lambda item: item[2], reverse=True)[: int(maximum_edges)]
        keep = {(i, j) for i, j, _ in strongest}
    removed = 0
    for i, j, magnitude in pairs:
        if magnitude < float(interaction_threshold) or (
            keep is not None and (i, j) not in keep
        ):
            if result[i, j] != 0.0:
                removed += 1
            result[i, j] = 0.0
            result[j, i] = 0.0
    return result, removed


def build_window_qubo(
    problem: PortfolioProblem,
    preferences: Preferences,
    current_weights: np.ndarray,
    window_indices: Iterable[int],
    required_ones: int,
    *,
    group_pressure: np.ndarray | None = None,
    cardinality_penalty: float | None = None,
    interaction_threshold: float = 0.0,
    maximum_edges: int | None = None,
) -> QUBOModel:
    """Compile an equal-notional support surrogate for one adaptive window.

    With exactly ``required_ones`` selected assets, equal proxy notionals keep
    window capital constant.  Frozen holdings contribute exact cross-risk
    terms.  Transaction costs are binary-linear and therefore exact for this
    proxy; the final allocation oracle subsequently replaces the proxy weights
    with optimized continuous percentages.
    """
    current = np.asarray(current_weights, dtype=float).reshape(-1)
    if current.shape != (problem.n,):
        raise ValueError(f"current_weights must have shape ({problem.n},)")
    window = tuple(dict.fromkeys(int(index) for index in window_indices))
    if not window or any(index < 0 or index >= problem.n for index in window):
        raise ValueError("window_indices contains an invalid asset index")
    if not 0 < int(required_ones) <= len(window):
        raise ValueError("required_ones must be within the window size")
    r = int(required_ones)
    indices = np.asarray(window, dtype=int)
    frozen = current.copy()
    frozen[indices] = 0.0
    window_capital = float(current[indices].sum())
    if window_capital <= 1e-12:
        window_capital = problem.budget * r / max(problem.n, r)
    proxy_weight = window_capital / r

    cov_window = problem.covariance_submatrix(indices)
    cross = problem.covariance_matvec(frozen)[indices]
    quadratic = preferences.lambda_risk * proxy_weight**2 * cov_window
    linear = (
        2.0 * preferences.lambda_risk * proxy_weight * cross
        - preferences.lambda_return * proxy_weight * problem.mu[indices]
        - preferences.lambda_income * proxy_weight * problem.y[indices]
    )

    cost_zero = problem.c[indices] * np.abs(problem.w0[indices])
    cost_one = problem.c[indices] * np.abs(proxy_weight - problem.w0[indices])
    linear += preferences.lambda_cost * (cost_one - cost_zero)

    if group_pressure is not None:
        pressure = np.asarray(group_pressure, dtype=float).reshape(-1)
        if pressure.shape != (problem.num_groups,):
            raise ValueError("group_pressure must have one entry per group")
        linear -= proxy_weight * pressure[np.asarray(problem.asset_group)[indices]]

    constant = objective_value(frozen, problem, preferences)
    quadratic, removed_edges = _sparsify_quadratic(
        quadratic,
        interaction_threshold=interaction_threshold,
        maximum_edges=maximum_edges,
    )

    if cardinality_penalty is not None:
        penalty = float(cardinality_penalty)
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("cardinality_penalty must be finite and positive")
        quadratic += penalty * np.ones_like(quadratic)
        linear -= 2.0 * penalty * r
        constant += penalty * r**2

    active_edges = int(np.count_nonzero(np.triu(quadratic, 1)))
    return QUBOModel(
        linear=linear,
        quadratic=quadratic,
        offset=constant,
        variable_names=tuple(problem.asset_names[index] for index in window),
        window_indices=window,
        required_ones=r,
        metadata={
            "proxy_weight": proxy_weight,
            "window_capital": window_capital,
            "active_edges": active_edges,
            "removed_edges": removed_edges,
            "cardinality_mode": "penalty" if cardinality_penalty is not None else "preserved",
        },
    )


__all__ = ["IsingModel", "QUBOModel", "build_window_qubo"]
