from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.allocation import _linear_model, solve_relaxation
from vanguard_portfolio.data_generation import (
    generate_factor_universe,
    generate_return_scenarios,
)
from vanguard_portfolio.portfolio_model import empirical_cvar
from vanguard_portfolio.schemas import (
    PortfolioConstraints,
    Preferences,
    SolverUnavailableError,
)
from vanguard_portfolio.validation import validate_weights


def build_extended_case(
    *,
    scenario_count: int = 250,
) -> tuple:
    problem = generate_factor_universe(
        n_assets=60,
        n_groups=6,
        n_factors=6,
        seed=20260805,
        current_cardinality=12,
        materialize_covariance=False,
    )
    anchor = problem.w0.copy()
    problem.target_return = float(problem.mu @ anchor) - 1.0e-3
    problem.max_turnover = 0.40

    scenarios = generate_return_scenarios(
        problem,
        n_scenarios=scenario_count,
        seed=20260806,
    )
    factor_anchor = problem.factor_loadings.T @ anchor
    factor_band = np.maximum(0.005, 0.10 * np.maximum(np.abs(factor_anchor), 0.01))
    stress = scenarios[np.argsort(scenarios @ anchor)[:5]]
    alpha = 0.95
    anchor_cvar = empirical_cvar(anchor, scenarios, alpha)

    constraints = PortfolioConstraints(
        exact_cardinality=12,
        minimum_active_weight=0.01,
        minimum_income=float(problem.y @ anchor) - 1.0e-3,
        maximum_weights=np.full(problem.n, 0.15),
        factor_lower=factor_anchor - factor_band,
        factor_upper=factor_anchor + factor_band,
        stress_scenarios=stress,
        stress_floors=stress @ anchor - 0.0025,
        scenario_returns=scenarios,
        maximum_cvar=anchor_cvar + 0.005,
        cvar_alpha=alpha,
    )
    return problem, constraints


class ExtendedAllocationTests(unittest.TestCase):
    def test_cvar_matrix_is_sparse_and_linear_in_scenario_count(self) -> None:
        problem, constraints = build_extended_case(scenario_count=2_000)
        matrix, _, _, _ = _linear_model(problem, constraints)

        self.assertEqual(matrix.shape[1], 2 * problem.n + problem.num_factors + 1 + 2_000)
        self.assertLess(matrix.nnz, 2_000 * (problem.n + 5) + 20_000)

    def test_osqp_extended_relaxation_passes_independent_validation(self) -> None:
        problem, constraints = build_extended_case(scenario_count=250)
        try:
            result = solve_relaxation(
                problem,
                Preferences(lambda_income=0.5),
                constraints,
                backend="osqp",
                solver_options={"tol": 1.0e-8, "max_iter": 250_000},
            )
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))

        report = validate_weights(result.weights, problem, constraints=constraints)
        self.assertTrue(result.success, result.metadata)
        self.assertTrue(report.feasible, report.details)
        self.assertEqual(result.method, "osqp_extended_qp")
        self.assertEqual(result.metadata["cvar_scenarios"], 250)

    def test_clarabel_extended_relaxation_matches_osqp(self) -> None:
        problem, constraints = build_extended_case(scenario_count=100)
        preferences = Preferences(lambda_income=0.5)
        try:
            osqp_result = solve_relaxation(
                problem,
                preferences,
                constraints,
                backend="osqp",
                solver_options={"tol": 1.0e-8, "max_iter": 250_000},
            )
            clarabel_result = solve_relaxation(
                problem,
                preferences,
                constraints,
                backend="clarabel",
                solver_options={"tol": 1.0e-8, "max_iter": 10_000},
            )
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))

        self.assertTrue(osqp_result.success, osqp_result.metadata)
        self.assertTrue(clarabel_result.success, clarabel_result.metadata)
        self.assertLessEqual(abs(osqp_result.objective - clarabel_result.objective), 5.0e-6)


if __name__ == "__main__":
    unittest.main()
