from __future__ import annotations

import importlib.util
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
            },
            {
                "n_assets": 250,
                "success": True,
                "breaches": 0,
                "certification_completed": True,
                "search_end_to_end_seconds": 2.0,
                "full_end_to_end_seconds": 4.0,
                "relative_gap_to_relaxation": 0.04,
            },
        ]
        summary = self.scaling._summary_rows(records)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["zero_breach_rate"], 1.0)
        self.assertEqual(summary[0]["certified_runs"], 2)
        self.assertEqual(summary[0]["search_end_to_end_seconds_median"], 1.5)
        self.assertAlmostEqual(summary[0]["relative_gap_to_relaxation_median"], 0.03)


if __name__ == "__main__":
    unittest.main()
