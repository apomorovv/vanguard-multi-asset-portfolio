from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.validation import validate_weights


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_synthetic_universe()

    def test_valid_current_portfolio_has_zero_breaches(self) -> None:
        report = validate_weights(self.problem.w0, self.problem)
        self.assertTrue(report.feasible)
        self.assertEqual(report.breaches, 0)
        self.assertEqual(report.max_violation, 0.0)

    def test_budget_violation_is_detected(self) -> None:
        weights = self.problem.w0.copy()
        weights[0] += 0.03
        report = validate_weights(weights, self.problem)
        self.assertFalse(report.feasible)
        self.assertTrue(
            any(
                check.name == "budget" and check.violation > 0
                for check in report.checks
            )
        )

    def test_details_only_include_tolerance_breaches(self) -> None:
        weights = self.problem.w0.copy()
        weights[0] += 5.0e-8
        report = validate_weights(weights, self.problem, tol=1.0e-7)
        self.assertTrue(report.feasible)
        self.assertGreater(report.max_violation, 0.0)
        self.assertEqual(report.details, [])

    def test_negative_validation_tolerance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tol must be finite and nonnegative"):
            validate_weights(self.problem.w0, self.problem, tol=-1.0)

    def test_asset_and_group_violations_are_reported_separately(self) -> None:
        weights = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.2])
        report = validate_weights(weights, self.problem)
        names = {check.name for check in report.checks if check.violation > 1e-7}
        self.assertIn("asset_upper:US_Equity", names)
        self.assertIn("group_upper:Equity", names)
        self.assertIn("group_lower:FixedIncome", names)

    def test_lot_grid_is_checked(self) -> None:
        report = validate_weights(self.problem.w0, self.problem, units=20)
        self.assertTrue(report.feasible)
        off_grid = self.problem.w0.copy()
        off_grid[0] += 0.003
        off_grid[1] -= 0.003
        report = validate_weights(off_grid, self.problem, units=20)
        self.assertFalse(report.feasible)
        self.assertTrue(
            any(
                check.name.startswith("lot_grid:")
                for check in report.checks
                if check.violation > 1e-7
            )
        )


if __name__ == "__main__":
    unittest.main()

