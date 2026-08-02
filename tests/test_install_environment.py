from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_environment.py"
SPEC = importlib.util.spec_from_file_location("install_environment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_cuda_major() -> None:
    output = "NVIDIA-SMI 580.159.04  Driver Version: 580.159.04  CUDA Version: 13.0"
    assert MODULE.parse_cuda_major(output) == 13
    assert MODULE.parse_cuda_major("no CUDA field") is None


def test_cuda_13_selects_cuda_12_gpu_wheel() -> None:
    choice = MODULE.choose_accelerator(
        system="Linux",
        machine="x86_64",
        nvidia_smi_output="CUDA Version: 13.0",
    )
    assert choice.expect_gpu
    assert choice.cuda_major == 13
    assert choice.extra == "quantum-gpu"
    assert choice.aer_distribution.startswith("qiskit-aer-gpu")
    assert "cu11" not in choice.aer_distribution
    assert MODULE.project_extra("full", choice) == "full-gpu"
    removed = MODULE.cleanup_distributions(choice)
    assert "qiskit-ibm-runtime" in removed
    assert "samplomatic" in removed
    assert "ibm-quantum-schemas" in removed


def test_cuda_11_selects_cu11_wheel() -> None:
    choice = MODULE.choose_accelerator(
        system="Linux",
        machine="x86_64",
        nvidia_smi_output="CUDA Version: 11.8",
    )
    assert choice.expect_gpu
    assert choice.extra == "quantum-gpu-cu11"
    assert choice.aer_distribution.startswith("qiskit-aer-gpu-cu11")
    assert MODULE.project_extra("full", choice) == "full-gpu-cu11"
    assert "qiskit-ibm-runtime" in MODULE.cleanup_distributions(choice)


def test_missing_gpu_uses_cpu_wheel() -> None:
    choice = MODULE.choose_accelerator(
        system="Linux",
        machine="x86_64",
        nvidia_smi_output=None,
    )
    assert not choice.expect_gpu
    assert choice.extra == "quantum-cpu"
    assert choice.aer_distribution.startswith("qiskit-aer")
    assert MODULE.project_extra("full", choice) == "full"
    assert "qiskit-ibm-runtime" not in MODULE.cleanup_distributions(choice)


def test_non_linux_uses_cpu_wheel() -> None:
    choice = MODULE.choose_accelerator(
        system="Darwin",
        machine="arm64",
        nvidia_smi_output="CUDA Version: 13.0",
    )
    assert not choice.expect_gpu
    assert choice.extra == "quantum-cpu"


def test_force_cpu_overrides_gpu() -> None:
    choice = MODULE.choose_accelerator(
        system="Linux",
        machine="x86_64",
        nvidia_smi_output="CUDA Version: 13.0",
        force_cpu=True,
    )
    assert not choice.expect_gpu
    assert choice.extra == "quantum-cpu"
