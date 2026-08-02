from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vanguard_portfolio.allocation import (
    AllocationOracle,
    find_feasible_initial_support,
    find_feasible_support_milp,
    solve_relaxation,
)
from vanguard_portfolio.data_generation import (
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
)
from vanguard_portfolio.validation import validate_weights


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

        realized = np.random.default_rng(4).multivariate_normal(
            self.problem.mu / 12, self.problem.cov / 12, size=24
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_hybrid_artifacts(run, Path(directory), realized_returns=realized)
            self.assertGreaterEqual(len(artifacts), 25)
            self.assertTrue((Path(directory) / "hybrid_summary.csv").is_file())
            self.assertGreater(
                (Path(directory) / "plots/quantum_cardinality.png").stat().st_size,
                1_000,
            )
            with (Path(directory) / "constraint_checks.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                constraint_rows = list(csv.DictReader(handle))
            self.assertTrue(constraint_rows)
            self.assertTrue(all(row["passed"] == "True" for row in constraint_rows))


if __name__ == "__main__":
    unittest.main()
