from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from vanguard_portfolio.allocation import (
    AllocationOracle,
    find_feasible_initial_support,
    find_feasible_support_milp,
    solve_relaxation,
)
from vanguard_portfolio.classical_continuous import _repair_turnover_residual
from vanguard_portfolio.data_generation import (
    generate_factor_universe,
    generate_return_scenarios,
    generate_synthetic_universe,
)
from vanguard_portfolio.hybrid import HybridConfig, run_hybrid_optimizer
from vanguard_portfolio.presentation import write_hybrid_artifacts
from vanguard_portfolio.quantum_solver import XYQAOAConfig
from vanguard_portfolio.schemas import (
    PortfolioConstraints,
    PortfolioProblem,
    Preferences,
    SolverUnavailableError,
)
from vanguard_portfolio.validation import validate_weights
from vanguard_portfolio.window_search import (
    binding_group_pressure,
    construct_change_window,
)


class HybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_synthetic_universe()
        self.preferences = Preferences()
        self.constraints = PortfolioConstraints(
            exact_cardinality=4,
            minimum_active_weight=0.05,
        )

    def test_oracle_constructs_and_caches_valid_support(self) -> None:
        relaxation = solve_relaxation(
            self.problem, self.preferences, self.constraints
        )
        oracle = AllocationOracle(
            self.problem, self.preferences, self.constraints
        )
        initial = find_feasible_initial_support(oracle, relaxation.weights, max_trials=30)
        self.assertTrue(initial.feasible)
        report = validate_weights(
            initial.weights, self.problem, constraints=self.constraints
        )
        self.assertTrue(report.feasible, report.details)
        cached = oracle.evaluate(initial.support)
        self.assertTrue(cached.cached)
        self.assertEqual(oracle.cache_hits, 1)

    def test_allocation_oracle_accepts_factor_only_parent_problem(self) -> None:
        problem = generate_factor_universe(
            n_assets=40,
            n_groups=5,
            n_factors=6,
            seed=41,
            current_cardinality=10,
            materialize_covariance=False,
        )
        constraints = PortfolioConstraints(
            exact_cardinality=10,
            minimum_active_weight=0.02,
            maximum_weights=np.full(problem.n, 0.20),
        )
        oracle = AllocationOracle(problem, Preferences(), constraints)
        support = np.flatnonzero(problem.w0 > 1e-12)
        evaluated = oracle.evaluate(support)
        self.assertTrue(evaluated.feasible, evaluated.reason)
        self.assertEqual(evaluated.result.breaches, 0)

    def test_validator_detects_sparse_constraint_breaches(self) -> None:
        invalid = self.problem.w0.copy()
        report = validate_weights(
            invalid, self.problem, constraints=self.constraints
        )
        self.assertFalse(report.feasible)
        self.assertTrue(any(check.name == "exact_cardinality" for check in report.checks))

    def test_optional_income_stress_and_cvar_are_enforced_by_oracle(self) -> None:
        constraints = PortfolioConstraints(
            exact_cardinality=4,
            minimum_active_weight=0.05,
            minimum_income=0.015,
            stress_scenarios=np.full((1, self.problem.n), -0.02),
            stress_floors=np.array([-0.03]),
            scenario_returns=generate_return_scenarios(self.problem, 40, seed=11),
            maximum_cvar=0.50,
        )
        relaxation = solve_relaxation(self.problem, self.preferences, constraints)
        oracle = AllocationOracle(self.problem, self.preferences, constraints)
        initial = find_feasible_initial_support(oracle, relaxation.weights, max_trials=30)
        self.assertTrue(initial.feasible)
        names = {check.name for check in initial.report.checks}
        self.assertIn("minimum_income", names)
        self.assertIn("stress_floor:0", names)
        self.assertTrue(any(name.startswith("cvar_") for name in names))

    def test_feasibility_milp_handles_group_coverage_that_needs_multiple_assets(self) -> None:
        n = 6
        problem = PortfolioProblem.from_dict(
            {
                "asset_names": [f"Asset_{index}" for index in range(n)],
                "group_names": ["Group_0", "Group_1"],
                "asset_group": [0, 0, 0, 1, 1, 1],
                "mu": [0.20, 0.19, 0.18, 0.03, 0.02, 0.01],
                "sigma": [0.10] * n,
                "corr": np.eye(n).tolist(),
                "cov": (0.01 * np.eye(n)).tolist(),
                "y": [0.01] * n,
                "c": [0.001] * n,
                "w0": [1.0 / n] * n,
                "lower": [0.0] * n,
                "upper": [0.5] * n,
                "group_lower": [0.4, 0.4],
                "group_upper": [0.6, 0.6],
                "budget": 1.0,
            }
        )
        constraints = PortfolioConstraints(
            exact_cardinality=4,
            minimum_active_weight=0.1,
            maximum_weights=np.full(n, 0.3),
        )
        oracle = AllocationOracle(problem, Preferences(), constraints)
        result = find_feasible_support_milp(
            oracle,
            np.full(n, 1.0 / n),
            time_limit=5.0,
        )
        self.assertTrue(result.feasible)
        selected_groups = np.asarray(problem.asset_group)[list(result.support)]
        self.assertEqual(np.count_nonzero(selected_groups == 0), 2)
        self.assertEqual(np.count_nonzero(selected_groups == 1), 2)
        self.assertEqual(
            result.result.metadata["initialization"],
            "scipy_highs_feasibility_milp",
        )

    def test_large_osqp_relaxation_clears_independent_validation(self) -> None:
        problem = generate_factor_universe(
            n_assets=2_000,
            n_groups=10,
            n_factors=12,
            seed=20260802,
        )
        constraints = PortfolioConstraints(
            exact_cardinality=50,
            minimum_active_weight=0.005,
            maximum_weights=np.full(problem.n, 0.04),
        )
        try:
            result = solve_relaxation(
                problem,
                Preferences(lambda_income=0.5),
                constraints,
                backend="osqp",
                solver_options={"tol": 1e-10, "max_iter": 250_000},
            )
        except SolverUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertEqual(result.status.lower(), "solved")
        self.assertTrue(result.success, result.metadata)
        self.assertTrue(result.feasible)
        self.assertEqual(result.breaches, 0)
        self.assertLessEqual(result.max_violation, 1e-7)
        self.assertEqual(result.metadata["native_tolerance"], 1e-10)

    def test_complete_hybrid_run_has_zero_breaches(self) -> None:
        config = HybridConfig(
            iterations=1,
            window_size=5,
            enumerate_windows_up_to=1_000,
            run_gurobi_reference=False,
            quantum=XYQAOAConfig(
                shots=256,
                optimizer_maxiter=12,
                optimizer_starts=1,
                top_candidates=10,
                seed=2,
            ),
        )
        run = run_hybrid_optimizer(
            self.problem, self.preferences, self.constraints, config
        )
        self.assertTrue(run.best.feasible)
        self.assertEqual(run.best.breaches, 0)
        self.assertEqual(np.count_nonzero(run.best.weights > 1e-8), 4)
        self.assertTrue(any("xy_qaoa" in result.method for result in run.results))
        self.assertFalse(run.initial.optimal)
        self.assertTrue(
            all(
                not result.optimal
                for result in run.results
                if result.model_type == "hybrid_sparse"
            )
        )

        realized = np.random.default_rng(4).multivariate_normal(
            self.problem.mu / 12, self.problem.cov / 12, size=24
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_hybrid_artifacts(run, Path(directory), realized_returns=realized)
            self.assertGreaterEqual(len(artifacts), 25)
            self.assertTrue((Path(directory) / "hybrid_summary.csv").is_file())
            self.assertTrue((Path(directory) / "quantum_execution.csv").is_file())
            self.assertGreater(
                (Path(directory) / "plots/quantum_cardinality.png").stat().st_size,
                1_000,
            )
            self.assertGreater(
                (Path(directory) / "plots/quantum_timing.png").stat().st_size,
                1_000,
            )
            self.assertGreater(
                (Path(directory) / "plots/key_guardrails.png").stat().st_size,
                1_000,
            )
            with (Path(directory) / "quantum_execution.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                quantum_rows = list(csv.DictReader(handle))
            self.assertEqual(len(quantum_rows), 1)
            self.assertEqual(
                quantum_rows[0]["parameter_optimizer_backend"],
                "fixed_weight_subspace_cpu",
            )
            self.assertEqual(quantum_rows[0]["execution_device"], "CPU")
            with (Path(directory) / "constraint_checks.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                constraint_rows = list(csv.DictReader(handle))
            self.assertTrue(constraint_rows)
            self.assertTrue(all(row["passed"] == "True" for row in constraint_rows))

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_hybrid_artifacts(
                run,
                Path(directory),
                realized_returns=realized,
                profile="evaluation",
            )
            self.assertIn("summary", artifacts)
            self.assertIn("quantum_execution", artifacts)
            self.assertNotIn("allocations", artifacts)
            self.assertNotIn("problem", artifacts)
            self.assertTrue((Path(directory) / "plots/constraint_slacks.png").is_file())
            self.assertTrue(
                (Path(directory) / "plots/correlation_communities.png").is_file()
            )
            self.assertTrue((Path(directory) / "plots/key_guardrails.png").is_file())

    def test_hybrid_can_continue_after_an_explicit_guide_limit(self) -> None:
        accepted = solve_relaxation(self.problem, self.preferences, self.constraints)
        limited = replace(
            accepted,
            status="run time limit reached",
            success=False,
            optimal=False,
            feasible=False,
            breaches=1,
            max_violation=1.0e-5,
        )
        config = HybridConfig(
            iterations=1,
            window_size=5,
            enumerate_windows_up_to=1_000,
            run_quantum=False,
            run_gurobi_reference=False,
            allow_relaxation_fallback=True,
        )
        with mock.patch(
            "vanguard_portfolio.hybrid.solve_relaxation", return_value=limited
        ):
            run = run_hybrid_optimizer(
                self.problem, self.preferences, self.constraints, config
            )

        self.assertTrue(run.best.success)
        self.assertTrue(run.best.feasible)
        self.assertEqual(run.best.breaches, 0)
        self.assertFalse(run.relaxation.success)
        self.assertTrue(run.relaxation.metadata["fallback_used"])
        self.assertEqual(
            run.relaxation.metadata["guide_source"],
            "unaccepted_continuous_iterate",
        )
        self.assertIn("continuous_relaxation", run.skipped)
        relaxation_row = run.summary_records()[0]
        self.assertEqual(relaxation_row["certification"], "uncertified_guide_iterate")

    def test_turnover_residual_repair_is_a_convex_contraction(self) -> None:
        problem = generate_factor_universe(
            n_assets=20,
            n_groups=4,
            n_factors=4,
            seed=17,
            current_cardinality=5,
            materialize_covariance=False,
        )
        problem.max_turnover = 0.20
        candidate = problem.w0.copy()
        held = int(np.flatnonzero(problem.w0 > 0.0)[0])
        unheld = int(np.flatnonzero(problem.w0 == 0.0)[0])
        candidate[held] -= 0.1000001
        candidate[unheld] += 0.1000001

        repaired, metadata = _repair_turnover_residual(candidate, problem)

        self.assertTrue(metadata["turnover_repair_applied"])
        self.assertAlmostEqual(repaired.sum(), problem.budget)
        self.assertLessEqual(np.sum(np.abs(repaired - problem.w0)), 0.20)

    def test_change_windows_are_group_diverse_and_rotate_unheld_assets(self) -> None:
        problem = generate_factor_universe(
            n_assets=60,
            n_groups=6,
            n_factors=8,
            seed=29,
            current_cardinality=12,
        )
        constraints = PortfolioConstraints(
            exact_cardinality=12,
            minimum_active_weight=0.01,
            maximum_weights=np.full(problem.n, 0.15),
        )
        pressure = binding_group_pressure(problem, problem.w0)
        np.testing.assert_allclose(pressure, 0.0)
        communities = np.asarray(problem.asset_group, dtype=int)
        first = construct_change_window(
            problem,
            Preferences(),
            constraints,
            problem.w0,
            np.full(problem.n, 1.0 / problem.n),
            window_size=10,
            held_fraction=0.5,
            community_labels=communities,
        )
        first_groups = {
            problem.asset_group[index] for index in first.promising_unheld
        }
        self.assertEqual(len(first_groups), len(first.promising_unheld))
        second = construct_change_window(
            problem,
            Preferences(),
            constraints,
            problem.w0,
            np.full(problem.n, 1.0 / problem.n),
            window_size=10,
            held_fraction=0.5,
            community_labels=communities,
            excluded_unheld=first.promising_unheld,
        )
        self.assertTrue(set(first.promising_unheld).isdisjoint(second.promising_unheld))


if __name__ == "__main__":
    unittest.main()
