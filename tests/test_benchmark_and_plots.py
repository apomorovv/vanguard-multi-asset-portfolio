from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vanguard_portfolio.classical import (
    benchmark_solvers,
    write_artifact_manifest,
    write_benchmark_artifacts,
)
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
        self.assertAlmostEqual(
            report.certified_lower_bound("continuous"),
            report.reference_objective("continuous"),
        )
        self.assertAlmostEqual(
            report.certified_lower_bound("discrete"),
            report.reference_objective("discrete"),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_benchmark_artifacts(
                report,
                directory,
                resolved_config={"problem": {"source": "synthetic"}},
            )
            plots = generate_benchmark_plots(problem, report, directory)
            for path in [*artifacts.values(), *plots.values()]:
                self.assertTrue(Path(path).is_file(), path)
                self.assertGreater(Path(path).stat().st_size, 0, path)
            manifest = write_artifact_manifest({**artifacts, **plots}, directory)
            self.assertGreater(manifest.stat().st_size, 0)
            self.assertIn("allocations", artifacts)
            self.assertIn("diagnostics", artifacts)
            self.assertIn("constraints", artifacts)
            allocation_text = artifacts["allocations"].read_text(encoding="utf-8")
            self.assertIn("optimized_weight", allocation_text)
            self.assertIn(problem.asset_names[0], allocation_text)

    def test_detailed_solver_specs_repeat_and_get_unique_ids(self) -> None:
        problem = generate_synthetic_universe()
        report = benchmark_solvers(
            problem,
            continuous_backends=[
                {
                    "name": "scipy",
                    "repetitions": 2,
                    "options": {"tol": 1e-8},
                }
            ],
            discrete_backends=[],
        )
        self.assertEqual(len(report.results), 2)
        self.assertEqual(len({result.run_id for result in report.results}), 2)
        self.assertEqual({result.repetition for result in report.results}, {0, 1})

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
