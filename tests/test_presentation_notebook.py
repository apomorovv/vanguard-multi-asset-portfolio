from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "Vanguard_Presentation_Benchmark_Suite.ipynb"
)


def test_qpu_and_following_notebook_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    heading_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "## 11. IBM QPU hardware-validation experiment"
        in "".join(cell.get("source", ()))
    )
    for index, cell in enumerate(notebook["cells"][heading_index + 1 :], heading_index + 1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_all_notebook_cells_compile_and_outputs_are_clean() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        compile(
            "".join(cell.get("source", ())),
            f"{NOTEBOOK.name}:cell-{index}",
            "exec",
        )


def test_notebook_uses_unified_environment_and_resumable_qpu_jobs() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert 'python -m pip install -e \\".[full]\\"' in text
    assert "qiskit-aer-gpu-cu11" in text
    assert "QPU_RESUME = True" in text
    assert "QPU_FAIL_FAST = False" in text
    assert "RUN_IBM_QPU = False" in text
    assert "Skipping completed QPU job" in text
    assert "QP_TOLERANCE = 1.0e-8" in text
    assert "ALLOW_GUIDE_FALLBACK = True" in text
    assert "RUN_SCALING_STRETCH = True" in text
    assert 'command.append(\\\"--resume\\\")' in text
    assert '.isin(GLOBAL_CORE_SIZES)' in text
    assert '.isin(GLOBAL_STRETCH_SIZES)' in text
    assert "missing_stretch_sizes" in text
    assert "completed_legacy_sizes" in text
    assert "create_scaling_plots" in text
    assert "scaling_runtime.png" in text
    assert "scaling_quantum.png" in text
    assert "profile=\\\"evaluation\\\"" in text
    assert "scaling_20k" not in text
    assert "Reopen under a separate IBM Runtime kernel" not in text
