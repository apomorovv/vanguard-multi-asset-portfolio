from __future__ import annotations

import importlib.util
import json
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
            "time_to_first_valid_seconds": 2.5,
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
            "quantum_other_seconds": 0.0,
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
                    "scaling_runtime_presentation.png",
                    "scaling_runtime_presentation.pdf",
                    "scaling_quantum.png",
                    "scaling_quantum.pdf",
                },
            )
            self.assertTrue(all(path.is_file() for path in created))

    def test_legacy_plot_rows_get_consistent_first_valid_and_quantum_timing(self) -> None:
        row = self.scaling._normalise_record(
            {
                "n_assets": 250,
                "success": True,
                "data_generation_seconds": 0.25,
                "time_to_first_valid_seconds": 2.0,
                "quantum_window_seconds": 1.5,
                "quantum_angle_seconds": 0.2,
                "quantum_sampler_seconds": 0.5,
                "quantum_allocation_seconds": 0.6,
            }
        )
        self.assertEqual(row["time_to_first_valid_seconds"], 2.25)
        self.assertAlmostEqual(row["quantum_other_seconds"], 0.2)

    def test_runtime_composition_uses_one_actual_median_total_run(self) -> None:
        records = []
        for repetition, total, relaxation, quantum in (
            (0, 5.0, 1.0, 1.0),
            (1, 7.0, 2.0, 1.5),
            (2, 10.0, 4.0, 2.0),
        ):
            records.append(
                {
                    "scaling_schema_version": self.scaling.SCALING_SCHEMA_VERSION,
                    "n_assets": 250,
                    "repetition": repetition,
                    "success": True,
                    "search_end_to_end_seconds": total,
                    "data_generation_seconds": 0.5,
                    "relaxation_seconds": relaxation,
                    "initialization_seconds": 0.5,
                    "classical_window_seconds": 1.0,
                    "quantum_window_seconds": quantum,
                    "window_overhead_seconds": total
                    - 0.5
                    - relaxation
                    - 0.5
                    - 1.0
                    - quantum,
                }
            )
        summary = self.scaling._summary_rows(records)
        fields = [
            "data_generation_seconds",
            "relaxation_seconds",
            "initialization_seconds",
            "classical_window_seconds",
            "quantum_window_seconds",
            "window_overhead_seconds",
        ]
        selected = self.scaling._representative_stage_rows(
            summary,
            records,
            fields,
        )[0]
        self.assertEqual(selected["total_seconds"], 7.0)
        self.assertAlmostEqual(sum(selected[field] for field in fields), 7.0)

    def test_resume_fingerprint_ignores_only_size_and_repetition_grid(self) -> None:
        first = self.scaling._parser().parse_args(
            ["--sizes", "250", "500", "--repetitions", "1"]
        )
        extended = self.scaling._parser().parse_args(
            ["--sizes", "250", "500", "1000", "--repetitions", "3"]
        )
        changed = self.scaling._parser().parse_args(
            ["--relaxation-tolerance", "1e-9"]
        )
        self.assertEqual(
            self.scaling._case_config_sha256(first),
            self.scaling._case_config_sha256(extended),
        )
        self.assertNotEqual(
            self.scaling._case_config_sha256(first),
            self.scaling._case_config_sha256(changed),
        )

    def test_resume_rejects_an_unversioned_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "scaling_runs.csv").write_text(
                "n_assets,repetition,seed,success\n250,0,1,True\n",
                encoding="utf-8",
            )
            (output / "scaling_config.json").write_text(
                json.dumps({"relaxation_tolerance": 1.0e-9}) + "\n",
                encoding="utf-8",
            )
            return_code = self.scaling.main(
                ["--resume", "--output", str(output)]
            )
        self.assertEqual(
            return_code,
            self.scaling.CHECKPOINT_CONFIG_MISMATCH_EXIT_CODE,
        )


if __name__ == "__main__":
    unittest.main()
