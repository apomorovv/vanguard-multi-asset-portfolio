from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.portfolio_model import (
    build_continuous_qp,
    discrete_constraints_hold,
    lot_bounds,
    lots_to_weights,
    objective_breakdown,
    objective_value,
    swap_objective_delta,
)
from vanguard_portfolio.schemas import Preferences


class PortfolioModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = generate_synthetic_universe()
        self.preferences = Preferences()

    def test_objective_breakdown_sums_to_scalar(self) -> None:
        terms = objective_breakdown(self.problem.w0, self.problem, self.preferences)
        self.assertAlmostEqual(
            sum(terms.values()),
            objective_value(self.problem.w0, self.problem, self.preferences),
            places=14,
        )
        self.assertEqual(set(terms), {"risk_term", "return_term", "income_term", "cost_term"})

    def test_matrix_qp_matches_direct_objective(self) -> None:
        data = build_continuous_qp(self.problem, self.preferences)
        w = self.problem.w0.copy()
        w[0] -= 0.05
        w[2] += 0.05
        x = np.concatenate([w, np.abs(w - self.problem.w0)])
        upper = data.P.toarray()
        full = upper + np.triu(upper, 1).T
        qp_value = 0.5 * x @ full @ x + data.q @ x
        self.assertAlmostEqual(
            float(qp_value), objective_value(w, self.problem, self.preferences), places=12
        )

    def test_lot_bounds_round_in_feasibility_preserving_direction(self) -> None:
        problem = generate_synthetic_universe()
        problem.lower[0] = 0.11
        problem.upper[0] = 0.29
        low, high = lot_bounds(problem, units=10)
        self.assertEqual(low[0], 2)
        self.assertEqual(high[0], 2)

    def test_lots_convert_to_exact_budget(self) -> None:
        lots = np.array([2, 1, 2, 2, 0, 3])
        weights = lots_to_weights(lots, self.problem, units=10)
        self.assertAlmostEqual(float(weights.sum()), self.problem.budget)
        self.assertTrue(discrete_constraints_hold(lots, self.problem, units=10))

    def test_fast_swap_delta_matches_full_objective(self) -> None:
        lots = np.array([2, 1, 2, 2, 0, 3])
        weights = lots_to_weights(lots, self.problem, units=10)
        donor, receiver = 5, 0
        delta = swap_objective_delta(
            weights,
            self.problem.cov @ weights,
            donor,
            receiver,
            self.problem,
            self.preferences,
            units=10,
        )
        trial = lots.copy()
        trial[donor] -= 1
        trial[receiver] += 1
        exact_delta = (
            objective_value(
                lots_to_weights(trial, self.problem, units=10),
                self.problem,
                self.preferences,
            )
            - objective_value(weights, self.problem, self.preferences)
        )
        self.assertAlmostEqual(delta, exact_delta, places=14)


if __name__ == "__main__":
    unittest.main()
