from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Vanguard_Presentation_Benchmark_Suite.ipynb"
SCALING_SCRIPT = ROOT / "scripts" / "run_hybrid_scaling.py"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", ()))


def _joined_sources(notebook: dict[str, Any], cell_type: str | None = None) -> str:
    return "\n\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell_type is None or cell.get("cell_type") == cell_type
    )


def _heading_index(notebook: dict[str, Any], heading: str) -> int:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "markdown" and heading in _cell_source(cell):
            return index
    raise AssertionError(f"Notebook heading is missing: {heading}")


def _literal_assignments(notebook: dict[str, Any]) -> dict[str, list[object]]:
    assignments: dict[str, list[object]] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse(_cell_source(cell))
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            try:
                value = ast.literal_eval(statement.value)
            except (TypeError, ValueError):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(value)
    return assignments


def test_qpu_and_following_notebook_cells_compile() -> None:
    notebook = _load_notebook()
    frontier_index = _heading_index(
        notebook,
        "## 11A. IBM QPU hardware frontier and noise-management ablation",
    )
    audit_index = _heading_index(
        notebook,
        "## 11B. IBM quantum value-attribution and surrogate-alignment audit",
    )
    assert frontier_index < audit_index

    for index, cell in enumerate(notebook["cells"][frontier_index + 1 :], frontier_index + 1):
        if cell.get("cell_type") == "code":
            compile(_cell_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_all_notebook_cells_compile_and_outputs_are_clean() -> None:
    notebook = _load_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        compile(_cell_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_notebook_uses_safe_defaults_and_resumable_qpu_jobs() -> None:
    notebook = _load_notebook()
    sources = _joined_sources(notebook)
    code = _joined_sources(notebook, "code")
    assignments = _literal_assignments(notebook)

    expected_single_assignments = {
        "RUN_IBM_QPU": False,
        "QPU_RESUME": True,
        "QPU_FAIL_FAST": False,
        "QPU_STRESS_PRESET": "frontier",
        "QPU_STRESS_FRONTIER_BEST_MODE": "baseline",
        "QPU_STRESS_ENABLE_FRACTIONAL_GATES": False,
        "RUN_SCALING_STRETCH": False,
        "RUN_SCENARIO_PENALTY_SWEEP": True,
        "QP_TOLERANCE": 1.0e-9,
        "ALLOW_GUIDE_FALLBACK": True,
    }
    for name, expected in expected_single_assignments.items():
        assert assignments.get(name) == [expected], (
            f"{name} must be assigned exactly once to {expected!r}; "
            f"found {assignments.get(name)!r}"
        )

    execution_versions = assignments.get("QPU_EXECUTION_SCHEMA_VERSION", [])
    analysis_versions = assignments.get("QPU_ANALYSIS_SCHEMA_VERSION", [])
    assert len(execution_versions) == len(analysis_versions) == 1
    assert isinstance(execution_versions[0], int)
    assert isinstance(analysis_versions[0], int)
    assert analysis_versions[0] > execution_versions[0]

    assert 'python -m pip install -e ".[full]"' in sources
    assert "qiskit-aer-gpu-cu11" in sources

    # Completed result tables are loaded without contacting IBM. If only the
    # submission manifest exists, the exact submitted Runtime job IDs are
    # retrieved instead of submitting a replacement campaign.
    assert "if QPU_RESUME and QPU_STRESS_RESULTS.is_file():" in code
    assert "completed hardware rows; no QPU job submitted." in code
    assert "if QPU_RESUME and QPU_STRESS_MANIFEST.is_file():" in code
    assert "Retrieving previously submitted Runtime jobs:" in code
    assert "_write_json(QPU_STRESS_MANIFEST, manifest)" in code


def test_notebook_scaling_contract_matches_the_runner() -> None:
    notebook = _load_notebook()
    code = _joined_sources(notebook, "code")
    scaling_script = SCALING_SCRIPT.read_text(encoding="utf-8")

    assert 'command.append("--resume")' in code
    assert ".isin(GLOBAL_CORE_SIZES)" in code
    assert ".isin(GLOBAL_STRETCH_SIZES)" in code
    assert "missing_stretch_sizes" in code
    assert "completed_legacy_sizes" in code
    assert "create_scaling_plots" in code
    assert "Checkpoint settings or schema changed" in code
    assert "case_config_sha256" in code
    assert 'profile="evaluation"' in code

    for artifact in (
        "scaling_runtime.png",
        "scaling_runtime_presentation.png",
        "scaling_quantum.png",
    ):
        assert artifact in scaling_script

    assert "scaling_20k" not in code
    assert "Reopen under a separate IBM Runtime kernel" not in code


def test_offline_quantum_audit_is_safe_without_saved_hardware_results() -> None:
    notebook = _load_notebook()
    audit_heading = _heading_index(
        notebook,
        "## 11B. IBM quantum value-attribution and surrogate-alignment audit",
    )
    audit_code = next(
        _cell_source(cell)
        for cell in notebook["cells"][audit_heading + 1 :]
        if cell.get("cell_type") == "code"
    )

    assert 'removeprefix("06_ibm_qpu_stress_")' in audit_code
    assert "Final quantum audit skipped: no saved frontier hardware results" in audit_code
    assert "RUN_FINAL_QUANTUM_AUDIT = False" in audit_code
    assert "raise FileNotFoundError" not in audit_code


def test_scenario_penalty_experiment_is_controlled_and_validated() -> None:
    notebook = _load_notebook()
    gauntlet_index = _heading_index(
        notebook,
        "## 6. Experiment C — all-constraints validity and sparse-CVaR scaling",
    )
    penalty_index = _heading_index(
        notebook,
        "## 6A. Experiment D - scenario-based CVaR penalty frontier",
    )
    ablation_index = _heading_index(
        notebook,
        "### Optional constraint ablation (off by default)",
    )
    assert gauntlet_index < penalty_index < ablation_index

    markdown = _cell_source(notebook["cells"][penalty_index])
    code = next(
        _cell_source(cell)
        for cell in notebook["cells"][penalty_index + 1 :]
        if cell.get("cell_type") == "code"
    )
    assignments = _literal_assignments(notebook)

    assert assignments["SCENARIO_PENALTY_WEIGHTS"] == [
        (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
    ]
    assert assignments["SCENARIO_PENALTY_ALPHA"] == [0.95]
    assert assignments["SCENARIO_PENALTY_TRAIN_SCENARIOS"] == [2_000]
    assert assignments["SCENARIO_PENALTY_TEST_SCENARIOS"] == [20_000]
    assert assignments["SCENARIO_PENALTY_REPETITIONS"] == [5]

    assert "Rockafellar and Uryasev (2000)" in markdown
    assert "Krokhmal, Palmquist, and Uryasev (2002)" in markdown
    assert "penalty_multiplier" in markdown
    assert "financial downside preference" in markdown
    assert "exact-$K=20$ support" in markdown

    assert "penalty_weight * training_cvar_expression" in code
    assert "heldout_scenarios = generate_return_scenarios" in code
    assert "validate_weights(" in code
    assert "if not report.feasible:" in code
    assert "Scenario-penalty control support does not match exact cardinality" in code
    assert '"qpu_jobs_submitted": 0' in code
    assert '"qubo_penalty_multiplier_used": False' in code
    assert "run_hybrid_optimizer(" not in code
    assert "RUN_IBM_QPU" not in code

    for artifact in (
        "scenario_penalty_runs.csv",
        "scenario_penalty_summary.csv",
        "scenario_penalty_constraint_checks.csv",
        "scenario_penalty_frontier.{suffix}",
        "scenario_penalty_allocation_shift.{suffix}",
    ):
        assert artifact in code
