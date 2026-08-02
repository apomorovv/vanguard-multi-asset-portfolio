#!/usr/bin/env python3
"""Install the project with the correct Qiskit Aer build for this machine.

The standard ``qiskit-aer`` distribution is CPU-only. Qiskit publishes
separate GPU distributions for CUDA 12+ and CUDA 11 on x86_64 Linux. Static
Python extras cannot inspect the host GPU, so this script performs that
selection before invoking pip and verifies that Aer can actually see the GPU.
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


@dataclass(frozen=True)
class AcceleratorChoice:
    extra: str
    aer_distribution: str
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
    """Choose the compatible project extra and Aer distribution."""
    if force_cpu:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer>=0.17",
            expect_gpu=False,
            reason="CPU installation was explicitly requested",
        )
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer>=0.17",
            expect_gpu=False,
            reason="prebuilt Aer GPU wheels require x86_64 Linux",
        )
    if not nvidia_smi_output:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer>=0.17",
            expect_gpu=False,
            reason="no working NVIDIA GPU was detected with nvidia-smi",
        )

    cuda_major = parse_cuda_major(nvidia_smi_output)
    if cuda_major is None:
        return AcceleratorChoice(
            extra="quantum-cpu",
            aer_distribution="qiskit-aer>=0.17",
            expect_gpu=False,
            reason="nvidia-smi did not report a CUDA compatibility version",
        )
    if cuda_major >= 12:
        return AcceleratorChoice(
            extra="quantum-gpu",
            aer_distribution="qiskit-aer-gpu>=0.15.1",
            expect_gpu=True,
            reason=(
                f"NVIDIA driver reports CUDA {cuda_major}; selecting Aer's CUDA 12 "
                "GPU wheel because newer NVIDIA drivers are backward compatible"
            ),
            cuda_major=cuda_major,
        )
    if cuda_major == 11:
        return AcceleratorChoice(
            extra="quantum-gpu-cu11",
            aer_distribution="qiskit-aer-gpu-cu11>=0.17",
            expect_gpu=True,
            reason="NVIDIA driver reports CUDA 11; selecting Aer's CUDA 11 GPU wheel",
            cuda_major=cuda_major,
        )
    return AcceleratorChoice(
        extra="quantum-cpu",
        aer_distribution="qiskit-aer>=0.17",
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
    if dry_run:
        return
    code = r'''
import os
from importlib import metadata
from qiskit_aer import AerSimulator

names = ("qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11")
installed = {}
for name in names:
    try:
        installed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass

devices = tuple(AerSimulator().available_devices())
print("Installed Aer distributions:", installed)
print("Aer available devices:", devices)
if os.environ.get("VANGUARD_EXPECT_AER_GPU") == "1" and "GPU" not in devices:
    raise SystemExit(
        "A GPU Aer package was selected, but Aer still reports no GPU device. "
        "Check CUDA runtime libraries, then rerun. Use --force-cpu only when "
        "CPU execution is intentional."
    )
'''
    env = os.environ.copy()
    env["VANGUARD_EXPECT_AER_GPU"] = "1" if expect_gpu else "0"
    print("+ verifying qiskit_aer device discovery", flush=True)
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
        help="install the CPU Aer wheel even when an NVIDIA GPU is present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected packages and commands without changing the environment",
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
    print(f"Selection: {choice.aer_distribution}")
    print(f"Reason: {choice.reason}")
    print(f"Project extra: {extra}")

    # CPU and GPU Aer wheels install the same qiskit_aer import package. Remove
    # all variants first so stale files or metadata cannot mask the selection.
    run_command(
        [sys.executable, "-m", "pip", "uninstall", "-y", *AER_DISTRIBUTIONS],
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
    verify_aer(expect_gpu=choice.expect_gpu, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
