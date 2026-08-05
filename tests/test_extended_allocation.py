from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from vanguard_portfolio.allocation import (
    AllocationOracle,
    _linear_model,
    solve_relaxation,
)
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

    def test_osqp_extended_relaxation_passes_relaxed_validation(self) -> None:
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

        # A continuous relaxation deliberately removes exact cardinality and
        # minimum-active-weight requirements. Validate it against the same
        # relaxed rule set rather than incorrectly requiring exactly 12 assets.
        relaxed_constraints = replace(
            constraints,
            exact_cardinality=None,
            minimum_active_weight=0.0,
            mandatory_assets=(),
        )
        report = validate_weights(
            result.weights,
            problem,
            constraints=relaxed_constraints,
        )

        self.assertTrue(result.success, result.metadata)
        self.assertTrue(report.feasible, report.details)
        self.assertEqual(result.method, "osqp_extended_qp")
        self.assertEqual(result.metadata["cvar_scenarios"], 250)

    def test_osqp_fixed_support_enforces_full_sparse_constraints(self) -> None:
        problem, constraints = build_extended_case(scenario_count=250)
        try:
            oracle = AllocationOracle(
                problem,
                Preferences(lambda_income=0.5),
                constraints,
                backend="osqp",
                solver_options={"tol": 1.0e-8, "max_iter": 250_000},
            )
            support = tuple(np.flatnonzero(problem.w0 > 1.0e-12).tolist())
            evaluated = oracle.evaluate(support)
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))

        self.assertTrue(evaluated.feasible, evaluated.reason)
        self.assertIsNotNone(evaluated.result)
        self.assertIsNotNone(evaluated.report)
        self.assertTrue(evaluated.report.feasible, evaluated.report.details)
        self.assertEqual(len(evaluated.support), constraints.exact_cardinality)
        self.assertEqual(evaluated.result.method, "osqp_extended_qp")
        self.assertEqual(evaluated.result.breaches, 0)

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
        self.assertEqual(clarabel_result.method, "clarabel_extended_qp")
        self.assertEqual(clarabel_result.metadata["cvar_scenarios"], 100)
        self.assertLessEqual(abs(osqp_result.objective - clarabel_result.objective), 5.0e-6)

    def test_gurobi_extended_relaxation_matches_osqp_when_available(self) -> None:
        # Keep this below the size limit of Gurobi's restricted test license.
        problem, constraints = build_extended_case(scenario_count=20)
        preferences = Preferences(lambda_income=0.5)
        try:
            osqp_result = solve_relaxation(
                problem,
                preferences,
                constraints,
                backend="osqp",
                solver_options={"tol": 1.0e-8, "max_iter": 250_000},
            )
            gurobi_result = solve_relaxation(
                problem,
                preferences,
                constraints,
                backend="gurobi",
                solver_options={"tol": 1.0e-8, "output": False},
            )
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))

        relaxed_constraints = replace(
            constraints,
            exact_cardinality=None,
            minimum_active_weight=0.0,
            mandatory_assets=(),
        )
        report = validate_weights(
            gurobi_result.weights,
            problem,
            constraints=relaxed_constraints,
        )
        self.assertTrue(gurobi_result.success, gurobi_result.metadata)
        self.assertTrue(report.feasible, report.details)
        self.assertEqual(gurobi_result.method, "gurobi_extended_qp")
        self.assertLessEqual(abs(osqp_result.objective - gurobi_result.objective), 5.0e-6)

    def test_empty_cvar_scenario_set_is_rejected(self) -> None:
        problem, constraints = build_extended_case(scenario_count=20)
        empty = replace(constraints, scenario_returns=np.empty((0, problem.n)))
        with self.assertRaisesRegex(ValueError, "at least one return scenario"):
            solve_relaxation(problem, Preferences(), empty, backend="scipy")

    def test_unknown_osqp_option_is_rejected(self) -> None:
        problem, constraints = build_extended_case(scenario_count=20)
        with self.assertRaisesRegex(ValueError, "unsupported OSQP solver option"):
            solve_relaxation(
                problem,
                Preferences(),
                constraints,
                backend="osqp",
                solver_options={"misspelled_tolerance": 1.0e-8},
            )

    def test_clarabel_alias_works_without_extended_rules(self) -> None:
        problem, _ = build_extended_case(scenario_count=20)
        basic_constraints = PortfolioConstraints(
            exact_cardinality=12,
            minimum_active_weight=0.01,
        )
        try:
            result = solve_relaxation(
                problem,
                Preferences(),
                basic_constraints,
                backend="clarabel",
            )
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))

        self.assertTrue(result.success, result.metadata)
        self.assertEqual(result.method, "clarabel_extended_qp")


if __name__ == "__main__":
    unittest.main()
