"""Internal helper for constructing uniform solver results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import portfolio_metrics
from .portfolio_model import objective_value
from .schemas import PortfolioProblem, Preferences, SolveResult
from .validation import validate_weights


def make_result(
    *,
    method: str,
    model_type: str,
    weights: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
    runtime: float,
    status: str,
    success: bool,
    optimal: bool,
    units: int | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> SolveResult:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    report = validate_weights(weights, problem, units=units)
    valid = weights.shape == (problem.n,) and np.all(np.isfinite(weights))
    return SolveResult(
        method=method,
        model_type=model_type,
        weights=weights,
        objective=objective_value(weights, problem, preferences) if valid else np.inf,
        runtime=float(runtime),
        status=str(status),
        success=bool(success and valid and report.feasible),
        optimal=bool(optimal and valid and report.feasible),
        feasible=bool(valid and report.feasible),
        breaches=report.breaches,
        max_violation=report.max_violation,
        metrics=portfolio_metrics(weights, problem) if valid else {},
        metadata=dict(metadata or {}),
        units=units,
        seed=seed,
    )


