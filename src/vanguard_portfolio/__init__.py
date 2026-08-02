"""Constraint-safe hybrid multi-asset portfolio optimization package."""

from .classical import (
    PRESETS,
    SolverSpec,
    benchmark_solvers,
    solve_continuous,
    solve_discrete,
    write_artifact_manifest,
    write_benchmark_artifacts,
)
from .allocation import (
    AllocationOracle,
    find_feasible_initial_support,
    find_feasible_support_milp,
    solve_relaxation,
)
from .data_generation import (
    generate_backtest_returns,
    generate_factor_universe,
    generate_return_scenarios,
    generate_synthetic_universe,
)
from .hybrid import HybridConfig, HybridRun, run_hybrid_optimizer
from .portfolio_model import objective_breakdown, objective_value
from .quantum_solver import XYQAOAConfig, solve_penalty_qaoa, solve_xy_qaoa
from .qubo_builder import QUBOModel, build_window_qubo
from .schemas import PortfolioConstraints, PortfolioProblem, Preferences, SolveResult
from .validation import validate_weights

__all__ = [
    "PRESETS",
    "AllocationOracle",
    "HybridConfig",
    "HybridRun",
    "PortfolioConstraints",
    "PortfolioProblem",
    "Preferences",
    "SolveResult",
    "SolverSpec",
    "QUBOModel",
    "XYQAOAConfig",
    "benchmark_solvers",
    "generate_factor_universe",
    "generate_backtest_returns",
    "generate_return_scenarios",
    "generate_synthetic_universe",
    "objective_breakdown",
    "objective_value",
    "find_feasible_initial_support",
    "find_feasible_support_milp",
    "run_hybrid_optimizer",
    "solve_continuous",
    "solve_discrete",
    "solve_penalty_qaoa",
    "solve_relaxation",
    "solve_xy_qaoa",
    "build_window_qubo",
    "validate_weights",
    "write_artifact_manifest",
    "write_benchmark_artifacts",
]

__version__ = "1.0.0"
