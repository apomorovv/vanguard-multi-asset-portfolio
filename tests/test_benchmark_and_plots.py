from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vanguard_portfolio.classical import benchmark_solvers, write_benchmark_artifacts
from vanguard_portfolio.data_generation import generate_synthetic_universe
from vanguard_portfolio.plotting import generate_benchmark_plots


class BenchmarkAndPlotTests(unittest.TestCase):
    def test_benchmark_writes_tables_and_nonempty_plots(self) -> None:
        problem = generate_synthetic_universe()
        report = benchmark_solvers(
            problem,
            units=10,
            continuous_backends=["scipy"],
            discrete_backends=["enumeration", "local_search", "annealing"],
            seeds=[0],
            annealing_iterations=300,
        )
        self.assertIsNotNone(report.reference_objective("continuous"))
        self.assertIsNotNone(report.reference_objective("discrete"))
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_benchmark_artifacts(report, directory)
            plots = generate_benchmark_plots(problem, report, directory)
            for path in [*artifacts.values(), *plots.values()]:
                self.assertTrue(Path(path).is_file(), path)
                self.assertGreater(Path(path).stat().st_size, 0, path)

    def test_missing_optional_solver_is_recorded_not_misreported(self) -> None:
        problem = generate_synthetic_universe()
        report = benchmark_solvers(
            problem,
            units=10,
            continuous_backends=["gurobi"],
            discrete_backends=[],
            missing_optional="skip",
        )
        # Depending on the machine, Gurobi either runs or is explicitly skipped.
        self.assertTrue(report.results or "continuous:gurobi" in report.skipped)


if __name__ == "__main__":
    unittest.main()


