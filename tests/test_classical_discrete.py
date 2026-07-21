from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from vanguard_portfolio.classical_continuous import solve_continuous_scipy
from vanguard_portfolio.classical_discrete import (
    bounded_lot_allocation_count,
    solve_discrete_annealing,
    solve_discrete_cvxpy,
    solve_discrete_enumeration,
    solve_discrete_gurobi,
    solve_discrete_local_search,
)
from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.schemas import SolverUnavailableError
from vanguard_portfolio.schemas import SolverSkippedError


class DiscreteSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_synthetic_universe()

    def test_exact_enumeration_is_feasible_and_on_grid(self) -> None:
        result = solve_discrete_enumeration(self.problem, units=10)
        self.assertTrue(result.optimal)
        self.assertTrue(result.feasible)
        self.assertEqual(result.breaches, 0)
        np.testing.assert_allclose(result.weights * 10, np.rint(result.weights * 10), atol=1e-12)

    def test_continuous_relaxation_is_lower_bound(self) -> None:
        continuous = solve_continuous_scipy(self.problem)
        discrete = solve_discrete_enumeration(self.problem, units=10)
        self.assertLessEqual(continuous.objective, discrete.objective + 1e-8)

    def test_exact_cannot_be_beaten_by_heuristics(self) -> None:
        exact = solve_discrete_enumeration(self.problem, units=10)
        local = solve_discrete_local_search(self.problem, units=10)
        annealing = solve_discrete_annealing(
            self.problem, units=10, seed=3, n_iterations=1500
        )
        self.assertLessEqual(exact.objective, local.objective + 1e-12)
        self.assertLessEqual(exact.objective, annealing.objective + 1e-12)

    def test_annealing_is_reproducible_for_a_seed(self) -> None:
        first = solve_discrete_annealing(self.problem, units=10, seed=5, n_iterations=500)
        second = solve_discrete_annealing(self.problem, units=10, seed=5, n_iterations=500)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertAlmostEqual(first.objective, second.objective)

    def test_enumeration_guard_rejects_unsafe_search(self) -> None:
        self.assertGreater(bounded_lot_allocation_count(self.problem, units=10), 100)
        with self.assertRaises(SolverSkippedError):
            solve_discrete_enumeration(self.problem, units=10, max_candidates=100)

    def test_candidate_pool_local_search_remains_feasible(self) -> None:
        result = solve_discrete_local_search(
            self.problem,
            units=20,
            max_iterations=20,
            candidate_pool_size=3,
        )
        self.assertTrue(result.success, result.status)
        self.assertTrue(result.feasible)
        self.assertEqual(result.metadata["candidate_pool_size"], 3)

    def test_large_safe_feasible_start_honors_optional_guardrails(self) -> None:
        guarded = replace(self.problem, target_return=0.035, max_turnover=0.60)
        result = solve_discrete_local_search(guarded, units=20, max_iterations=25)
        self.assertTrue(result.feasible, result.status)
        self.assertGreaterEqual(result.metrics["expected_return"], 0.035 - 1e-7)
        self.assertLessEqual(result.metrics["turnover"], 0.60 + 1e-7)
        self.assertEqual(result.metadata["feasible_start_method"], "scipy_highs_milp")

    def test_gurobi_miqp_matches_enumeration_when_available(self) -> None:
        exact = solve_discrete_enumeration(self.problem, units=10)
        try:
            gurobi = solve_discrete_gurobi(self.problem, units=10)
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertTrue(gurobi.optimal, gurobi.status)
        self.assertAlmostEqual(gurobi.objective, exact.objective, places=8)

    def test_cvxpy_scip_matches_enumeration_when_available(self) -> None:
        exact = solve_discrete_enumeration(self.problem, units=10)
        try:
            scip = solve_discrete_cvxpy(self.problem, units=10, solver_name="SCIP")
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertTrue(scip.optimal, scip.status)
        self.assertAlmostEqual(scip.objective, exact.objective, places=8)


if __name__ == "__main__":
    unittest.main()
