# CPU, GPU, and IBM QPU Installation

## 1. Supported Single-Environment Stack

CPU simulation, NVIDIA GPU simulation, and IBM Runtime hardware access can use
one Python environment. The project pins:

| Component | Version | Purpose |
|---|---:|---|
| Qiskit | `2.5.1` | Circuits and transpilation |
| Qiskit Aer | `0.17.2` | CPU or NVIDIA simulation |
| IBM Runtime | `0.48.0` | IBM QPU submission |
| Python | `3.10`–`3.13` | Available Aer wheel range |

On Linux x86_64, the `full` extra installs
`qiskit-aer-gpu-cu11==0.17.2`. Despite its name, that distribution includes
both the CPU and GPU simulator implementations. A CUDA-12/13-capable NVIDIA
driver can run the older CUDA-11 runtime binaries bundled through pip, so this
is also the supported choice for the RTX A6000 system.

On macOS, Windows, Linux ARM, and other unsupported GPU-wheel platforms,
`full` installs `qiskit-aer==0.17.2` instead.

## 2. What Each Device Accelerates

| Workload | Device used |
|---|---|
| Full-universe factor QP and sparse CVaR | CPU through OSQP/Clarabel/Gurobi |
| Fixed-support allocation and classical LNS | CPU |
| XY-QAOA parameter optimization | CPU fixed-weight subspace simulator |
| Qiskit Aer circuit sampling | GPU when a real CUDA probe succeeds; CPU otherwise |
| IBM hardware sampling | Selected IBM QPU through Runtime |

The GPU wheel does not make the large-universe OSQP scaling experiment a GPU
solve. The notebook records the requested and actual Aer device and falls back
cleanly when CUDA is not visible. Do not infer GPU acceleration from
`available_devices()` alone; the project verifier executes a real circuit.

## 3. Why the Previous Split Failed

The old installer selected `qiskit-aer-gpu==0.15.1` whenever
`nvidia-smi` advertised CUDA 12 or newer. That stale CUDA-12 wheel required
Qiskit 1.4.x. Current IBM Runtime requires Qiskit 2.3 or newer, so pip could
not create a consistent CPU/GPU/QPU environment.

The fix is not to install CPU Aer and GPU Aer together. All Aer distributions
write the same `qiskit_aer` package tree. The fix is to install exactly one
current GPU-capable Aer distribution that also supplies CPU execution.

## 4. Clean Installation

Create and activate one environment, then run from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
python -m pip check
python scripts/install_environment.py --verify-only
```

After installation, restart the Jupyter kernel and confirm that
`sys.executable` points to this environment.

The `full` extra also installs pandas, JupyterLab, and ipykernel because the
presentation notebook uses them directly.

## 5. Repair an Existing Environment

Plain pip cannot infer that `qiskit-aer`, `qiskit-aer-gpu`, and
`qiskit-aer-gpu-cu11` overwrite the same import files. If this environment
has ever used another Aer/Qiskit profile, run:

```bash
python scripts/install_environment.py
```

The repair script:

1. removes all Aer distributions, Qiskit/Terra, and Runtime schema helpers;
2. installs the pinned `full` extra;
3. runs `pip check`;
4. verifies that exactly one Aer distribution exists;
5. imports Qiskit, Aer, and IBM Runtime together;
6. executes a real Aer CPU job;
7. executes a real Aer GPU job when a GPU is visible through `nvidia-smi`.

Preview the repair without changing the environment:

```bash
python scripts/install_environment.py --dry-run
```

Verify an already installed environment without changing it:

```bash
python scripts/install_environment.py --verify-only
```

## 6. Verify From Python

```python
from importlib import metadata

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

for name in (
    "qiskit",
    "qiskit-aer",
    "qiskit-aer-gpu",
    "qiskit-aer-gpu-cu11",
    "qiskit-ibm-runtime",
):
    try:
        print(f"{name}: {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass

devices = tuple(AerSimulator().available_devices())
print("Aer devices:", devices)

simulator = AerSimulator(method="statevector", device="CPU")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
compiled = transpile(circuit, simulator)
result = simulator.run(compiled, shots=128).result()
print("CPU success:", result.success, result.get_counts())
print("IBM Runtime imports:", QiskitRuntimeService, SamplerV2)
```

A GPU-capable wheel can advertise `"GPU"` even when a container has no CUDA device. The repository verifier therefore submits a real GPU job when `nvidia-smi` sees a device; package names and `available_devices()` alone are not accepted as proof.

## 7. IBM Account Setup

Package installation does not authenticate an IBM account. Save the account
once outside committed files:

```python
from getpass import getpass
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=getpass("IBM Quantum API token: "),
    overwrite=True,
    set_as_default=True,
)
```

Never place the token in YAML, notebooks, shell scripts, logs, or Git history.

## 8. Notebook Execution

Run the presentation notebook from this same kernel. Section 11 checks:

- the pinned Qiskit and Runtime versions;
- that exactly one Aer distribution is installed;
- IBM account availability;
- backend access, status, and qubit capacity.

The hardware experiment checkpoints
`results/presentation_benchmark_suite/ibm_qpu_hardware_validation.csv` after
every result. With `QPU_RESUME=True`, reruns skip successful jobs and retry
only missing or failed jobs.

## 9. Troubleshooting

### More Than One Aer Distribution Is Installed

```bash
python -m pip list | grep -E 'qiskit|aer'
python scripts/install_environment.py
```

Do not install `qiskit-aer` alongside either GPU Aer distribution.

### GPU Is Visible to `nvidia-smi` but Not Aer

```bash
nvidia-smi
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
```

Confirm that the NVIDIA device is exposed to the Python process or container.
Then inspect `LD_LIBRARY_PATH` only if the repair script reports a loader
failure. The solver automatically falls back to Aer CPU for normal local runs,
but the installation verifier deliberately fails when `nvidia-smi` sees a
GPU and Aer cannot use it.

### IBM Runtime Imports but Has No Account

Run the one-time `save_account` block above. Do not pass the token through a
notebook configuration or command-line argument.

### Jupyter Still Imports Old Packages

Restart the kernel and print:

```python
import sys
print(sys.executable)
```

Install with that exact interpreter if it differs from the terminal:

```bash
/path/from/sys/executable -m pip install -e ".[full]"
```
