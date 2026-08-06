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


def test_notebook_uses_unified_environment_and_resumable_qpu_jobs() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert 'python -m pip install -e \\".[full]\\"' in text
    assert "qiskit-aer-gpu-cu11" in text
    assert "QPU_RESUME = True" in text
    assert "QPU_FAIL_FAST = False" in text
    assert "Skipping completed QPU job" in text
    assert "scaling_20k" not in text
    assert "Reopen under a separate IBM Runtime kernel" not in text
