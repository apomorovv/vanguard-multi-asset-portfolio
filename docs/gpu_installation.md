# Qiskit Aer CPU and GPU Installation

## 1. Why Separate Environments Are Necessary

Qiskit Aer is distributed as different packages for:

- CPU execution;
- NVIDIA CUDA 12 or newer;
- NVIDIA CUDA 11.

All three distributions expose the same Python import, `qiskit_aer`. Installing
multiple Aer variants in one environment can leave conflicting files.

This branch also pins different Qiskit versions for different simulator stacks:

| Use case | Qiskit | Aer package |
|---|---|---|
| CPU Aer | `qiskit>=2.0,<3` | `qiskit-aer==0.17.2` |
| CUDA 12+ Aer | `qiskit==1.4.6` | `qiskit-aer-gpu==0.15.1` |
| CUDA 11 Aer | `qiskit>=2.0,<3` | `qiskit-aer-gpu-cu11==0.17.2` |
| IBM Runtime | Current Qiskit 2.x line | `qiskit-ibm-runtime>=0.40` |

The CUDA 12 simulator environment is intentionally separate from IBM Runtime
because the pinned Aer wheel uses the Qiskit 1.4 compatibility line.

## 2. Recommended Automatic Installation

From the repository root:

```bash
python scripts/install_environment.py --profile full
```

The installer:

1. checks the operating system and machine architecture;
2. runs `nvidia-smi` when available;
3. reads the CUDA compatibility version reported by the driver;
4. selects the matching project extra;
5. removes conflicting Qiskit/Aer distributions;
6. removes IBM Runtime packages from GPU simulator environments;
7. installs the editable project;
8. runs `pip check`;
9. imports Aer;
10. executes a real simulator job;
11. verifies that GPU execution is actually available when a GPU package was
    selected.

The installer falls back to CPU on unsupported operating systems,
architectures, old CUDA drivers, or machines without a working NVIDIA setup.

## 3. Preview Without Changing the Environment

```bash
python scripts/install_environment.py \
  --profile full \
  --dry-run
```

## 4. Install Only Quantum Dependencies

```bash
python scripts/install_environment.py --profile quantum
```

## 5. Force CPU Aer

```bash
python scripts/install_environment.py \
  --profile full \
  --force-cpu
```

## 6. Recommended Environments

### 6.1 Portable CPU Aer and IBM Runtime

```bash
python -m venv .venv-vanguard-cpu
source .venv-vanguard-cpu/bin/activate
python -m pip install --upgrade pip
python scripts/install_environment.py --profile full --force-cpu
python -m pip check
```

This environment can contain CPU Aer and IBM Runtime together.

### 6.2 CUDA 12+ Aer GPU

```bash
conda create -n vanguard-aer-gpu python=3.11 -y
conda activate vanguard-aer-gpu
python scripts/install_environment.py --profile full
python -m pip check
```

Do not install current IBM Runtime into this environment.

### 6.3 IBM Runtime Hardware

```bash
conda create -n vanguard-ibm-runtime python=3.11 -y
conda activate vanguard-ibm-runtime
python -m pip install --upgrade pip
python -m pip install -e ".[qp,ibm-runtime,test]"
python -m pip check
```

The simulator and hardware environments may use the same repository checkout
and YAML configuration files.

## 7. Explicit Extras

The repository exposes:

```bash
python -m pip install -e ".[quantum-cpu]"
python -m pip install -e ".[quantum-gpu]"
python -m pip install -e ".[quantum-gpu-cu11]"
```

Complete-stack variants are:

```bash
python -m pip install -e ".[full]"
python -m pip install -e ".[full-gpu]"
python -m pip install -e ".[full-gpu-cu11]"
```

`full-gpu` and `full-gpu-cu11` intentionally exclude IBM Runtime.

## 8. Verify the Installed Aer Distribution

```bash
python - <<'PY'
from importlib import metadata
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

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

devices = AerSimulator().available_devices()
print("Available Aer devices:", devices)

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
result = simulator.run(compiled, shots=128).result()
print("Execution success:", result.success)
print("Counts:", result.get_counts())
PY
```

For a correctly configured NVIDIA environment, the available device tuple must
include `GPU`.

## 9. Run the Hybrid Model With Aer GPU

The YAML file may set:

```yaml
hybrid:
  quantum:
    backend: aer_gpu
```

or the command line may override it:

```bash
python scripts/run_hybrid.py \
  --config configs/large_hybrid.yaml \
  --quantum-backend aer_gpu \
  --overwrite
```

Inspect `quantum_execution.csv` and `hybrid_diagnostics.json` to confirm the
actual device. A requested GPU backend is not proof that GPU execution occurred.

## 10. What the GPU Accelerates

In the final hybrid path:

1. the exact fixed-cardinality subspace simulator optimizes QAOA angles on the
   CPU;
2. Aer CPU or GPU samples the corresponding physical circuit after the angles
   are selected.

For the default 16-qubit, 7-excitation window, the compact subspace has only

$$
\binom{16}{7}=11{,}440
$$

states. Repeatedly launching small sequential COBYLA evaluations on a GPU may be
slower than the compact CPU calculation.

GPU Aer is most useful for:

- physical-circuit sampling;
- larger circuit widths;
- higher shot counts;
- explicit CPU-versus-GPU backend comparisons.

It does not accelerate the full-universe factor QP, support allocation, or
Gurobi model unless those components use their own accelerator-enabled
software.

## 11. Common Problems

### Aer Imports but No GPU Is Available

Check:

```bash
nvidia-smi
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
```

A CPU-only Aer package may still import successfully.

### Conflicting Aer Packages

Remove all variants and reinstall one:

```bash
python -m pip uninstall -y \
  qiskit-aer \
  qiskit-aer-gpu \
  qiskit-aer-gpu-cu11
```

Then rerun the repository installer.

### Qiskit Version Conflict

Do not upgrade the CUDA 12 environment to Qiskit 2.x while retaining
`qiskit-aer-gpu==0.15.1`.

### IBM Runtime Breaks the CUDA 12 Environment

Create a separate IBM Runtime environment rather than trying to satisfy both
stacks with one set of packages.

## 12. Security

IBM credentials must be stored through the supported Runtime account
configuration. Never commit tokens to:

- YAML files;
- notebooks;
- shell scripts;
- logs;
- Git history.
