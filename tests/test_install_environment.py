from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_environment.py"
SPEC = importlib.util.spec_from_file_location("install_environment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_linux_x86_64_selects_runtime_compatible_gpu_aer() -> None:
    assert MODULE.supports_gpu_wheel(system="Linux", machine="x86_64")
    assert (
        MODULE.expected_aer_distribution(system="Linux", machine="x86_64")
        == "qiskit-aer-gpu-cu11"
    )
    assert MODULE.PINNED_VERSIONS == {
        "qiskit": "2.5.1",
        "qiskit-aer": "0.17.2",
        "qiskit-aer-gpu-cu11": "0.17.2",
        "qiskit-ibm-runtime": "0.48.0",
    }


def test_non_linux_or_non_x86_uses_cpu_aer() -> None:
    assert not MODULE.supports_gpu_wheel(system="Darwin", machine="arm64")
    assert not MODULE.supports_gpu_wheel(system="Linux", machine="aarch64")
    assert (
        MODULE.expected_aer_distribution(system="Darwin", machine="arm64")
        == "qiskit-aer"
    )


def test_repair_removes_every_overlapping_distribution() -> None:
    removed = MODULE.cleanup_distributions()
    for name in (
        "qiskit",
        "qiskit-terra",
        "qiskit-aer",
        "qiskit-aer-gpu",
        "qiskit-aer-gpu-cu11",
        "qiskit-ibm-runtime",
        "samplomatic",
        "ibm-quantum-schemas",
    ):
        assert name in removed


def test_full_extra_contains_one_platform_selected_aer_and_runtime() -> None:
    configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    full = configuration.split("full = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"qiskit==2.5.1"' in full
    assert '"qiskit-ibm-runtime==0.48.0"' in full
    assert "qiskit-aer-gpu-cu11==0.17.2" in full
    assert "qiskit-aer==0.17.2" in full
    assert "qiskit-aer-gpu==" not in full
    assert "jupyterlab" in full
    assert "ipykernel" in full


def test_installer_dry_run_uses_only_full_extra(capsys) -> None:
    assert MODULE.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "[full]" in output
    assert "full-gpu" not in output
    assert "qiskit-aer-gpu==0.15.1" not in output
