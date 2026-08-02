from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.portfolio_model import objective_value
from vanguard_portfolio.quantum_solver import XYQAOAConfig, solve_xy_qaoa
from vanguard_portfolio.qubo_builder import QUBOModel, build_window_qubo
from vanguard_portfolio.schemas import Preferences


class QUBOAndQuantumTests(unittest.TestCase):
    def test_window_qubo_matches_equal_notional_proxy(self) -> None:
        problem = generate_synthetic_universe()
        preferences = Preferences()
        window = (0, 1, 2, 3)
        required = 2
        qubo = build_window_qubo(
            problem, preferences, problem.w0, window, required
        )
        bits = np.array([1, 0, 1, 0])
        proxy = problem.w0.copy()
        proxy[list(window)] = 0.0
        proxy[list(window)] = qubo.metadata["proxy_weight"] * bits
        self.assertAlmostEqual(
            qubo.energy(bits), objective_value(proxy, problem, preferences), places=12
        )

    def test_ising_conversion_preserves_every_energy(self) -> None:
        model = QUBOModel(
            linear=np.array([-0.2, 0.3, -0.1]),
            quadratic=np.array(
                [[0.1, -0.4, 0.2], [-0.4, 0.0, 0.5], [0.2, 0.5, -0.3]]
            ),
            offset=0.7,
        )
        ising = model.to_ising()
        for state in range(8):
            bits = np.array([(state >> index) & 1 for index in range(3)])
            spins = 1 - 2 * bits
            energy = ising.constant + ising.fields @ spins
            energy += sum(value * spins[i] * spins[j] for (i, j), value in ising.couplings.items())
            self.assertAlmostEqual(model.energy(bits), float(energy), places=13)

    def test_xy_qaoa_samples_only_required_cardinality(self) -> None:
        model = QUBOModel(
            linear=np.array([-2.0, -1.0, -0.5, 0.1, 0.2]),
            quadratic=np.zeros((5, 5)),
            required_ones=2,
            window_indices=tuple(range(5)),
        )
        config = XYQAOAConfig(
            depth=1,
            shots=512,
            optimizer_maxiter=20,
            optimizer_starts=1,
            seed=3,
        )
        result = solve_xy_qaoa(model, np.array([0, 0, 0, 1, 1]), config)
        self.assertEqual(result.cardinality_feasibility_rate, 1.0)
        self.assertTrue(all(int(bits.sum()) == 2 for bits in result.bitstrings))
        self.assertLessEqual(
            model.energy(result.bitstrings[0]), model.energy(np.array([0, 0, 0, 1, 1]))
        )


if __name__ == "__main__":
    unittest.main()
