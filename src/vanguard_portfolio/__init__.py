"""Vanguard multi-asset portfolio baseline package."""

from .classical import (
    PRESETS,
    SolverSpec,
    benchmark_solvers,
    solve_continuous,
    solve_discrete,
    write_artifact_manifest,
    write_benchmark_artifacts,
)
from .data_generation import generate_factor_universe, generate_synthetic_universe
from .portfolio_model import objective_breakdown, objective_value
from .schemas import PortfolioProblem, Preferences, SolveResult
from .validation import validate_weights

__all__ = [
    "PRESETS",
    "PortfolioProblem",
    "Preferences",
    "SolveResult",
    "SolverSpec",
    "benchmark_solvers",
    "generate_factor_universe",
    "generate_synthetic_universe",
    "objective_breakdown",
    "objective_value",
    "solve_continuous",
    "solve_discrete",
    "validate_weights",
    "write_artifact_manifest",
    "write_benchmark_artifacts",
]

__version__ = "0.3.0"
