from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vanguard_portfolio.data_generation import (
    generate_factor_universe,
    generate_synthetic_universe,
    is_psd,
    load_problem,
    save_problem,
)
from vanguard_portfolio.validation import validate_weights


class DataGenerationTests(unittest.TestCase):
    def test_synthetic_covariance_is_psd_and_matches_definition(self) -> None:
        problem = generate_synthetic_universe()
        self.assertTrue(is_psd(problem.cov))
        np.testing.assert_allclose(
            problem.cov,
            problem.corr * np.outer(problem.sigma, problem.sigma),
            atol=1e-12,
        )

    def test_current_portfolio_is_hard_feasible(self) -> None:
        problem = generate_synthetic_universe()
        report = validate_weights(problem.w0, problem)
        self.assertTrue(report.feasible, report.details)

    def test_group_matrix_has_one_membership_per_asset(self) -> None:
        problem = generate_synthetic_universe()
        self.assertEqual(problem.A.shape, (problem.num_groups, problem.n))
        np.testing.assert_allclose(problem.A.sum(axis=0), 1.0)

    def test_json_round_trip(self) -> None:
        problem = generate_synthetic_universe()
        with tempfile.TemporaryDirectory() as directory:
            path = save_problem(problem, Path(directory))
            loaded = load_problem(path)
        self.assertEqual(problem.asset_names, loaded.asset_names)
        np.testing.assert_allclose(problem.cov, loaded.cov)
        np.testing.assert_allclose(problem.w0, loaded.w0)

    def test_factor_universe_has_requested_shape_and_psd_covariance(self) -> None:
        problem = generate_factor_universe(n_assets=30, n_groups=5, seed=17)
        self.assertEqual(problem.n, 30)
        self.assertEqual(problem.num_groups, 5)
        self.assertTrue(is_psd(problem.cov))
        self.assertTrue(validate_weights(problem.w0, problem).feasible)


if __name__ == "__main__":
    unittest.main()


