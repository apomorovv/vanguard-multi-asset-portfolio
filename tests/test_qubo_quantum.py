from __future__ import annotations

import unittest

import numpy as np

from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.portfolio_model import objective_value
from vanguard_portfolio.quantum_solver import (
    XYQAOAConfig,
    _basis_energies,
    _fixed_weight_structure,
    _subspace_state,
    build_xy_qaoa_circuit,
    solve_xy_qaoa,
)
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

    def test_batched_sparse_energy_matches_scalar_qubo(self) -> None:
        model = QUBOModel(
            linear=np.array([-0.2, 0.3, -0.1, 0.4]),
            quadratic=np.array(
                [
                    [0.1, -0.4, 0.0, 0.2],
                    [-0.4, 0.0, 0.5, 0.0],
                    [0.0, 0.5, -0.3, 0.1],
                    [0.2, 0.0, 0.1, 0.2],
                ]
            ),
            offset=0.7,
        )
        bits = np.asarray(
            [[(state >> index) & 1 for index in range(model.n)] for state in range(16)]
        )
        np.testing.assert_allclose(
            _basis_energies(model, bits),
            np.asarray([model.energy(row) for row in bits]),
            atol=1e-13,
        )

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
        self.assertEqual(
            result.metadata["parameter_optimizer_backend"],
            "fixed_weight_subspace_cpu",
        )
        self.assertEqual(result.metadata["execution_device"], "CPU")
        self.assertFalse(result.metadata["gpu_accelerated"])
        self.assertGreaterEqual(result.metadata["angle_optimization_seconds"], 0.0)
        self.assertTrue(np.isfinite(result.metadata["sampled_expected_surrogate_energy"]))

    def test_qiskit_xy_circuit_matches_subspace_probabilities_when_available(self) -> None:
        try:
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            self.skipTest(str(exc))
        model = QUBOModel(
            linear=np.array([-0.7, -0.2, 0.1, 0.4, 0.6]),
            quadratic=np.array(
                [
                    [0.1, -0.2, 0.0, 0.0, 0.1],
                    [-0.2, 0.0, 0.3, 0.0, 0.0],
                    [0.0, 0.3, -0.1, -0.2, 0.0],
                    [0.0, 0.0, -0.2, 0.2, 0.25],
                    [0.1, 0.0, 0.0, 0.25, 0.0],
                ]
            ),
            required_ones=2,
            window_indices=tuple(range(5)),
        )
        initial_bits = np.array([1, 0, 0, 1, 0])
        angles = np.array([0.47, -0.31])
        integers, basis_bits, _, pairs = _fixed_weight_structure(5, 2, "ring")
        energies = _basis_energies(model, basis_bits)
        center = float(np.mean(energies))
        scale = max(float(np.std(energies)), float(np.ptp(energies)) / 10.0, 1e-12)
        initial = np.zeros(len(integers), dtype=complex)
        integer = np.uint64(
            sum(int(value) << index for index, value in enumerate(initial_bits))
        )
        initial[int(np.searchsorted(integers, integer))] = 1.0
        reference = _subspace_state(
            (energies - center) / scale,
            list(pairs),
            angles,
            initial,
        )
        circuit = build_xy_qaoa_circuit(
            model,
            initial_bits,
            angles,
            mixer="ring",
            cost_scale=scale,
        )
        qiskit_state = np.asarray(Statevector.from_instruction(circuit).data)
        np.testing.assert_allclose(
            np.abs(qiskit_state[integers]) ** 2,
            np.abs(reference) ** 2,
            atol=1e-12,
        )
        outside = np.ones(qiskit_state.size, dtype=bool)
        outside[integers] = False
        self.assertLess(float(np.sum(np.abs(qiskit_state[outside]) ** 2)), 1e-12)


if __name__ == "__main__":
    unittest.main()
