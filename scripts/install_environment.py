#!/usr/bin/env python3
"""Install and verify the unified CPU, NVIDIA Aer, and IBM Runtime stack.

The CUDA-11 build of Aer 0.17.2 is intentionally used on Linux x86_64 even
when ``nvidia-smi`` reports CUDA 12 or 13.  A new NVIDIA driver can execute
older CUDA runtime binaries, and this Aer release shares Qiskit 2.x with the
current IBM Runtime client.  The GPU Aer distribution also contains the CPU
simulator, so exactly one Aer distribution is installed in the environment.

Use this script to repair an existing environment because pip does not know
that the three Aer distribution names overwrite the same ``qiskit_aer`` files.
On a clean environment, ``python -m pip install -e \".[full]\"`` installs the
same package set directly.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

AER_DISTRIBUTIONS = (
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qiskit-aer-gpu-cu11",
)
QISKIT_DISTRIBUTIONS = ("qiskit", "qiskit-terra")
IBM_RUNTIME_DISTRIBUTIONS = (
    "qiskit-ibm-runtime",
    "samplomatic",
    "ibm-quantum-schemas",
)
PINNED_VERSIONS = {
    "qiskit": "2.5.1",
    "qiskit-aer": "0.17.2",
    "qiskit-aer-gpu-cu11": "0.17.2",
    "qiskit-ibm-runtime": "0.48.0",
}


def supports_gpu_wheel(*, system: str, machine: str) -> bool:
    """Return whether the prebuilt unified Aer GPU wheel supports the platform."""
    return system == "Linux" and machine.lower() in {"x86_64", "amd64"}


def read_nvidia_smi() -> str | None:
    """Return ``nvidia-smi`` output only when a visible NVIDIA device responds."""
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


def expected_aer_distribution(*, system: str, machine: str) -> str:
    """Return the sole Aer distribution selected by the ``full`` extra."""
    if supports_gpu_wheel(system=system, machine=machine):
        return "qiskit-aer-gpu-cu11"
    return "qiskit-aer"


def cleanup_distributions() -> tuple[str, ...]:
    """Return distributions removed before a deterministic reinstall."""
    return (*AER_DISTRIBUTIONS, *QISKIT_DISTRIBUTIONS, *IBM_RUNTIME_DISTRIBUTIONS)


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def verify_environment(*, expect_gpu: bool, dry_run: bool) -> None:
    """Run imports plus real CPU/GPU simulator jobs in the installed interpreter."""
    if dry_run:
        return
    code = r'''
import os
import platform
from importlib import metadata

names = (
    "qiskit",
    "qiskit-terra",
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qiskit-aer-gpu-cu11",
    "qiskit-ibm-runtime",
    "samplomatic",
    "ibm-quantum-schemas",
)
installed = {}
for name in names:
    try:
        installed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print("Installed Qiskit distributions:", installed)

aer_names = {
    name for name in ("qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11")
    if name in installed
}
if len(aer_names) != 1:
    raise SystemExit(
        "Exactly one Aer distribution must be installed; found "
        f"{sorted(aer_names)}. Run scripts/install_environment.py to repair it."
    )

expected_aer = os.environ["VANGUARD_EXPECT_AER_DISTRIBUTION"]
if aer_names != {expected_aer}:
    raise SystemExit(
        f"Expected {expected_aer!r} on {platform.system()} {platform.machine()}, "
        f"but found {sorted(aer_names)}"
    )

expected_versions = {
    "qiskit": "2.5.1",
    expected_aer: "0.17.2",
    "qiskit-ibm-runtime": "0.48.0",
}
wrong = {
    name: {"expected": version, "installed": installed.get(name)}
    for name, version in expected_versions.items()
    if installed.get(name) != version
}
if wrong:
    raise SystemExit(f"The pinned unified stack was not installed: {wrong}")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
except Exception as exc:
    raise SystemExit(f"Unified Qiskit/Aer/Runtime import failed: {exc}") from exc

# Referencing the classes verifies the Runtime API without requiring or reading
# an IBM credential during installation.
assert QiskitRuntimeService is not None and SamplerV2 is not None
devices = tuple(str(device) for device in AerSimulator().available_devices())
print("Aer available devices:", devices)

def execute(device: str) -> None:
    simulator = AerSimulator(method="statevector", device=device)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=128, seed_simulator=7).result()
    if not bool(getattr(result, "success", True)):
        raise SystemExit(f"Aer {device} verification job failed: {result.status}")
    counts = result.get_counts()
    if sum(counts.values()) != 128:
        raise SystemExit(f"Aer {device} returned an unexpected shot count: {counts}")
    experiment_metadata = dict(result.results[0].metadata or {})
    reported = str(experiment_metadata.get("device", device)).upper()
    if reported != device:
        raise SystemExit(
            f"Aer {device} verification metadata reports device={reported!r}"
        )
    print(f"Aer {device} verification counts:", counts)

execute("CPU")
expect_gpu = os.environ.get("VANGUARD_EXPECT_AER_GPU") == "1"
if expect_gpu and "GPU" not in devices:
    raise SystemExit(
        "nvidia-smi sees a GPU, but Aer does not advertise GPU execution. "
        "The package versions are compatible; check that the NVIDIA device is "
        "exposed to this process/container and inspect LD_LIBRARY_PATH."
    )
if expect_gpu:
    execute("GPU")
else:
    print("Aer GPU execution: not requested (no visible nvidia-smi device)")
print("IBM Runtime import: OK (account/network access is checked by the notebook)")
'''
    env = os.environ.copy()
    env["VANGUARD_EXPECT_AER_GPU"] = "1" if expect_gpu else "0"
    env["VANGUARD_EXPECT_AER_DISTRIBUTION"] = expected_aer_distribution(
        system=platform.system(),
        machine=platform.machine(),
    )
    print("+ verifying unified Qiskit/Aer/IBM Runtime environment", flush=True)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not install; validate the packages already in this interpreter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print repair/install commands without changing the environment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("repository root containing pyproject.toml was not found")

    system = platform.system()
    machine = platform.machine()
    nvidia_smi_output = read_nvidia_smi()
    expect_gpu = bool(
        nvidia_smi_output
        and supports_gpu_wheel(system=system, machine=machine)
    )
    print(f"Platform: {system} {machine}")
    print(
        "Aer distribution: "
        f"{expected_aer_distribution(system=system, machine=machine)}==0.17.2"
    )
    print("Qiskit: 2.5.1; IBM Runtime: 0.48.0")
    print("Visible NVIDIA GPU:", "yes" if expect_gpu else "no")

    if not args.verify_only:
        # Aer variants own the same import tree, and pip does not model that
        # conflict.  Remove every variant before installing the one selected by
        # the full extra's platform markers.
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                *cleanup_distributions(),
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
                f"{root}[full]",
            ],
            dry_run=args.dry_run,
        )
        run_command(
            [sys.executable, "-m", "pip", "check"],
            dry_run=args.dry_run,
        )
    verify_environment(expect_gpu=expect_gpu, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
