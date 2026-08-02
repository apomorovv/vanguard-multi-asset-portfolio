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
        self.assertTrue(problem.has_factor_model)
        reconstructed = (
            problem.factor_loadings @ problem.factor_cov @ problem.factor_loadings.T
            + np.diag(problem.idiosyncratic_var)
        )
        np.testing.assert_allclose(reconstructed, problem.cov, atol=1e-10)

        off_diagonal = problem.corr[np.triu_indices(problem.n, k=1)]
        self.assertGreater(float(off_diagonal.mean()), 0.20)
        self.assertTrue(np.all(problem.factor_loadings[:, 0] > 0.0))
        self.assertEqual(problem.factor_names[0], "Market")

    def test_factor_universe_can_create_a_sparse_balanced_incumbent(self) -> None:
        problem = generate_factor_universe(
            n_assets=120,
            n_groups=6,
            n_factors=8,
            seed=19,
            current_cardinality=24,
        )
        self.assertEqual(np.count_nonzero(problem.w0 > 1e-12), 24)
        self.assertTrue(validate_weights(problem.w0, problem).feasible)
        np.testing.assert_allclose(problem.A @ problem.w0, np.full(6, 1.0 / 6.0))

    def test_large_factor_universe_uses_scalable_validation_path(self) -> None:
        problem = generate_factor_universe(n_assets=400, n_groups=8, seed=23)
        self.assertEqual(problem.cov.shape, (400, 400))
        self.assertTrue(validate_weights(problem.w0, problem).feasible)


if __name__ == "__main__":
    unittest.main()
