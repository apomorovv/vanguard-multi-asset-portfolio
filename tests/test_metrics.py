from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.metrics import objective_gap, portfolio_metrics


class MetricsTests(unittest.TestCase):
    def test_metrics_match_direct_formulas(self) -> None:
        problem = generate_synthetic_universe()
        metrics = portfolio_metrics(problem.w0, problem)
        self.assertAlmostEqual(metrics["expected_return"], float(problem.mu @ problem.w0))
        self.assertAlmostEqual(metrics["variance"], float(problem.w0 @ problem.cov @ problem.w0))
        self.assertAlmostEqual(metrics["turnover"], 0.0)
        self.assertAlmostEqual(metrics["transaction_cost"], 0.0)
        self.assertAlmostEqual(metrics["effective_holdings"], 1.0 / np.sum(problem.w0**2))

    def test_minimization_gap_sign(self) -> None:
        absolute, relative = objective_gap(-0.020, -0.025)
        self.assertAlmostEqual(absolute, 0.005)
        self.assertAlmostEqual(relative, 0.2)


if __name__ == "__main__":
    unittest.main()

