from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_scaling_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_hybrid_scaling.py"
    spec = importlib.util.spec_from_file_location("run_hybrid_scaling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScalingScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scaling = _load_scaling_module()

    def test_default_study_separates_scaling_from_certification(self) -> None:
        args = self.scaling._parser().parse_args([])
        self.assertEqual(args.sizes[-1], 20_000)
        self.assertEqual(args.certification_max_assets, 2_000)
        self.assertFalse(args.materialize_covariance)
        self.assertEqual(args.window_size, 16)
        self.assertEqual(args.relaxation_tolerance, 1e-8)
        self.assertEqual(args.allocation_tolerance, 1e-8)
        self.assertEqual(args.relaxation_time_limit, 30.0)
        self.assertEqual(args.case_time_limit, 180.0)
        self.assertTrue(args.relaxation_fallback)

    def test_summary_reports_medians_and_zero_breach_rate(self) -> None:
        records = [
            {
                "n_assets": 250,
                "success": True,
                "breaches": 0,
                "certification_completed": True,
                "search_end_to_end_seconds": 1.0,
                "full_end_to_end_seconds": 3.0,
                "relative_gap_to_relaxation": 0.02,
                "relaxation_accepted": True,
                "relaxation_bound_available": True,
                "relaxation_fallback_used": False,
            },
            {
                "n_assets": 250,
                "success": True,
                "breaches": 0,
                "certification_completed": True,
                "search_end_to_end_seconds": 2.0,
                "full_end_to_end_seconds": 4.0,
                "relative_gap_to_relaxation": 0.04,
                "relaxation_accepted": False,
                "relaxation_bound_available": False,
                "relaxation_fallback_used": True,
            },
        ]
        summary = self.scaling._summary_rows(records)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["zero_breach_rate"], 1.0)
        self.assertEqual(summary[0]["certified_runs"], 2)
        self.assertEqual(summary[0]["search_end_to_end_seconds_median"], 1.5)
        self.assertAlmostEqual(summary[0]["relative_gap_to_relaxation_median"], 0.03)
        self.assertEqual(summary[0]["relaxation_acceptance_rate"], 0.5)
        self.assertEqual(summary[0]["relaxation_bound_rate"], 0.5)
        self.assertEqual(summary[0]["relaxation_fallback_rate"], 0.5)

    def test_all_failed_checkpoint_does_not_attempt_log_scale_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            created = self.scaling._plots(
                [{"n_assets": 250, "runs": 1, "successful_runs": 0}],
                Path(directory),
            )
        self.assertEqual(created, [])

    def test_plots_include_runtime_and_quantum_anatomy(self) -> None:
        timed_fields = {
            "search_end_to_end_seconds": 5.0,
            "relaxation_seconds": 2.0,
            "data_generation_seconds": 0.2,
            "initialization_seconds": 0.3,
            "classical_window_seconds": 0.8,
            "quantum_window_seconds": 1.2,
            "window_overhead_seconds": 0.5,
            "relative_gap_to_relaxation": 0.01,
            "peak_rss_gib": 0.5,
            "factor_risk_storage_mib": 0.1,
            "dense_covariance_gib_avoided": 0.4,
            "quantum_angle_seconds": 0.4,
            "quantum_sampler_seconds": 0.3,
            "quantum_allocation_seconds": 0.5,
            "quantum_cardinality_rate": 1.0,
        }
        row = {
            "n_assets": 250,
            "runs": 3,
            "successful_runs": 3,
            "zero_breach_rate": 1.0,
            "relaxation_bound_rate": 1.0,
            "relaxation_fallback_rate": 0.0,
        }
        for field, value in timed_fields.items():
            row[f"{field}_q1"] = value
            row[f"{field}_median"] = value
            row[f"{field}_q3"] = value

        with tempfile.TemporaryDirectory() as directory:
            created = self.scaling._plots([row], Path(directory))
            names = {path.name for path in created}
            self.assertEqual(
                names,
                {
                    "scaling_evidence.png",
                    "scaling_evidence.pdf",
                    "scaling_runtime.png",
                    "scaling_runtime.pdf",
                    "scaling_quantum.png",
                    "scaling_quantum.pdf",
                },
            )
            self.assertTrue(all(path.is_file() for path in created))


if __name__ == "__main__":
    unittest.main()
