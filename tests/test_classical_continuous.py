from __future__ import annotations

import unittest
from dataclasses import replace

from vanguard_portfolio.classical_continuous import (
    solve_continuous_cvxpy,
    solve_continuous_gurobi,
    solve_continuous_osqp,
    solve_continuous_scipy,
)
from vanguard_portfolio.classical import PRESETS
from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.portfolio_model import objective_value
from vanguard_portfolio.schemas import Preferences, SolverUnavailableError


class ContinuousSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_synthetic_universe()

    def test_scipy_solution_is_feasible_and_improves_current(self) -> None:
        preferences = Preferences()
        result = solve_continuous_scipy(self.problem, preferences)
        self.assertTrue(result.success, result.status)
        self.assertTrue(result.feasible)
        self.assertEqual(result.breaches, 0)
        self.assertLessEqual(
            result.objective,
            objective_value(self.problem.w0, self.problem, preferences) + 1e-9,
        )
        self.assertIn("model_build_seconds", result.metadata)
        self.assertIn("feasible_start_seconds", result.metadata)
        self.assertIn("solve_seconds", result.metadata)
        self.assertGreaterEqual(result.runtime, result.metadata["solve_seconds"])

    def test_higher_risk_weight_reduces_variance(self) -> None:
        low = solve_continuous_scipy(self.problem, Preferences(lambda_risk=1.0))
        high = solve_continuous_scipy(self.problem, Preferences(lambda_risk=40.0))
        self.assertLessEqual(high.metrics["variance"], low.metrics["variance"] + 1e-9)

    def test_optional_target_return_is_hard(self) -> None:
        problem = replace(self.problem, target_return=0.055)
        result = solve_continuous_scipy(problem)
        self.assertTrue(result.success, result.status)
        self.assertGreaterEqual(result.metrics["expected_return"], 0.055 - 1e-7)

    def test_optional_turnover_limit_is_hard(self) -> None:
        problem = replace(self.problem, max_turnover=0.10)
        result = solve_continuous_scipy(problem)
        self.assertTrue(result.success, result.status)
        self.assertLessEqual(result.metrics["turnover"], 0.10 + 1e-7)

    def test_named_presets_change_the_intended_metrics(self) -> None:
        balanced = solve_continuous_scipy(self.problem, PRESETS["balanced"])
        income = solve_continuous_scipy(self.problem, PRESETS["income"])
        cost_sensitive = solve_continuous_scipy(self.problem, PRESETS["cost_sensitive"])
        self.assertGreaterEqual(income.metrics["income"], balanced.metrics["income"] - 1e-8)
        self.assertLessEqual(
            cost_sensitive.metrics["transaction_cost"],
            balanced.metrics["transaction_cost"] + 1e-8,
        )

    def test_osqp_matches_scipy_when_available(self) -> None:
        scipy_result = solve_continuous_scipy(self.problem)
        try:
            other = solve_continuous_osqp(self.problem)
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertTrue(other.success, other.status)
        self.assertAlmostEqual(other.objective, scipy_result.objective, places=6)

    def test_cvxpy_clarabel_matches_scipy_when_available(self) -> None:
        scipy_result = solve_continuous_scipy(self.problem)
        try:
            other = solve_continuous_cvxpy(self.problem, solver_name="CLARABEL")
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertTrue(other.success, other.status)
        self.assertAlmostEqual(other.objective, scipy_result.objective, places=6)

    def test_gurobi_matches_scipy_when_available(self) -> None:
        scipy_result = solve_continuous_scipy(self.problem)
        try:
            other = solve_continuous_gurobi(self.problem)
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertTrue(other.optimal, other.status)
        self.assertAlmostEqual(other.objective, scipy_result.objective, places=6)


if __name__ == "__main__":
    unittest.main()
