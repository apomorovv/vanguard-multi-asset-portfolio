#!/usr/bin/env python3
"""Install the project with a Qiskit Aer build compatible with this machine.

The CPU, CUDA 12, and CUDA 11 Aer distributions are separate packages. The
CUDA 12 wheel currently remains at Aer 0.15.1 and must be paired with Qiskit
1.4.x because Qiskit 2.0 removed an API imported by that Aer Python layer.
This script detects the machine, installs a known-compatible package set, and
executes a real Aer job to verify the selected backend.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


AER_DISTRIBUTIONS = (
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qiskit-aer-gpu-cu11",
)
QISKIT_DISTRIBUTIONS = ("qiskit", "qiskit-terra")


@dataclass(frozen=True)
class AcceleratorChoice:
    extra: str
    aer_distribution: str
    qiskit_requirement: str
    expect_gpu: bool
    reason: str
    cuda_major: int | None = None


def parse_cuda_major(nvidia_smi_output: str) -> int | None:
    """Extract the maximum CUDA version reported by ``nvidia-smi``."""
    match = re.search(r"CUDA Version:\s*(\d+)(?:\.\d+)?", nvidia_smi_output)
    return None if match is None else int(match.group(1))


def choose_accelerator(
    *,
    system: str,
    machine: str,
    nvidia_smi_output: str | None,
    force_cpu: bool = False,
) -> AcceleratorChoice:
    """Choose a compatible project extra and Qiskit/Aer pair."""
    if force_cpu:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer==0.17.2",
            qiskit_requirement="qiskit>=2.0,<3",
            expect_gpu=False,
            reason="CPU installation was explicitly requested",
        )
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer==0.17.2",
            qiskit_requirement="qiskit>=2.0,<3",
            expect_gpu=False,
            reason="prebuilt Aer GPU wheels require x86_64 Linux",
        )
    if not nvidia_smi_output:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer==0.17.2",
            qiskit_requirement="qiskit>=2.0,<3",
            expect_gpu=False,
            reason="no working NVIDIA GPU was detected with nvidia-smi",
        )

    cuda_major = parse_cuda_major(nvidia_smi_output)
    if cuda_major is None:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer==0.17.2",
            qiskit_requirement="qiskit>=2.0,<3",
            expect_gpu=False,
            reason="nvidia-smi did not report a CUDA compatibility version",
        )
    if cuda_major >= 12:
        return AcceleratorChoice(
            extra="quantum-gpu",
            aer_distribution="qiskit-aer-gpu==0.15.1",
            qiskit_requirement="qiskit==1.4.6",
            expect_gpu=True,
            reason=(
                f"NVIDIA driver reports CUDA {cuda_major}; selecting Aer's CUDA 12 "
                "wheel and its required Qiskit 1.4 compatibility line"
            ),
            cuda_major=cuda_major,
        )
    if cuda_major == 11:
        return AcceleratorChoice(
            extra="quantum-gpu-cu11",
            aer_distribution="qiskit-aer-gpu-cu11==0.17.2",
            qiskit_requirement="qiskit>=2.0,<3",
            expect_gpu=True,
            reason="NVIDIA driver reports CUDA 11; selecting current CUDA 11 Aer",
            cuda_major=cuda_major,
        )
    return AcceleratorChoice(
        extra="quantum-cpu",
        aer_distribution="qiskit-aer==0.17.2",
        qiskit_requirement="qiskit>=2.0,<3",
        expect_gpu=False,
        reason=f"CUDA {cuda_major} is older than the supported prebuilt GPU wheels",
        cuda_major=cuda_major,
    )


def read_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def project_extra(profile: str, choice: AcceleratorChoice) -> str:
    if profile == "quantum":
        return choice.extra
    return {
        "quantum-cpu": "full",
        "quantum-gpu": "full-gpu",
        "quantum-gpu-cu11": "full-gpu-cu11",
    }[choice.extra]


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def verify_aer(*, expect_gpu: bool, dry_run: bool) -> None:
    """Verify import, device discovery, and one real simulator execution."""
    if dry_run:
        return
    code = r'''
import os
from importlib import metadata

names = (
    "qiskit",
    "qiskit-terra",
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qiskit-aer-gpu-cu11",
    "qiskit-ibm-runtime",
)
installed = {}
for name in names:
    try:
        installed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print("Installed Qiskit distributions:", installed)

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except Exception as exc:
    raise SystemExit(
        "Qiskit Aer import failed after installation. This usually means an "
        f"incompatible or stale Qiskit/Aer pair remains: {exc}"
    ) from exc

devices = tuple(AerSimulator().available_devices())
print("Aer available devices:", devices)
expect_gpu = os.environ.get("VANGUARD_EXPECT_AER_GPU") == "1"
if expect_gpu and "GPU" not in devices:
    raise SystemExit(
        "A GPU Aer package was selected, but Aer reports no GPU device. "
        "Check CUDA runtime libraries and LD_LIBRARY_PATH."
    )

device = "GPU" if expect_gpu else "CPU"
simulator = AerSimulator(method="statevector", device=device)
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
compiled = transpile(circuit, simulator, optimization_level=0)
result = simulator.run(compiled, shots=128, seed_simulator=7).result()
if not result.success:
    raise SystemExit(f"Aer {device} verification job failed: {result.status}")
counts = result.get_counts()
if sum(counts.values()) != 128:
    raise SystemExit(f"Aer verification returned an unexpected shot count: {counts}")
print(f"Aer {device} verification counts:", counts)
'''
    env = os.environ.copy()
    env["VANGUARD_EXPECT_AER_GPU"] = "1" if expect_gpu else "0"
    print("+ verifying Qiskit/Aer import and simulator execution", flush=True)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("quantum", "full"),
        default="full",
        help="install only quantum dependencies or the complete project stack",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="install CPU Aer even when an NVIDIA GPU is present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print package choices and commands without changing the environment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("repository root containing pyproject.toml was not found")

    choice = choose_accelerator(
        system=platform.system(),
        machine=platform.machine().lower(),
        nvidia_smi_output=read_nvidia_smi(),
        force_cpu=bool(args.force_cpu),
    )
    extra = project_extra(args.profile, choice)

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Qiskit selection: {choice.qiskit_requirement}")
    print(f"Aer selection: {choice.aer_distribution}")
    print(f"Reason: {choice.reason}")
    print(f"Project extra: {extra}")

    # Aer variants install the same qiskit_aer import tree. Qiskit 1.x and 2.x
    # also replace the same package tree, so remove both before reinstalling.
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            *AER_DISTRIBUTIONS,
            *QISKIT_DISTRIBUTIONS,
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "-e",
            f"{root}[{extra}]",
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [sys.executable, "-m", "pip", "check"],
        dry_run=args.dry_run,
    )
    verify_aer(expect_gpu=choice.expect_gpu, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
