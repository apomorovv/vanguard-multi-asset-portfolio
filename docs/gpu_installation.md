# Qiskit Aer CPU/GPU installation

Qiskit Aer uses different Python distributions for CPU, CUDA 12+, and CUDA 11.
The distributions install the same `qiskit_aer` import package, so installing
more than one variant can leave conflicting files in an environment.

Use the repository installer rather than installing the `quantum` or `full`
extra directly when GPU auto-detection is desired:

```bash
python scripts/install_environment.py --profile full
```

The installer:

1. checks that the host is x86_64 Linux;
2. calls `nvidia-smi`;
3. reads the reported CUDA compatibility version;
4. selects `qiskit-aer-gpu` for CUDA 12 or newer;
5. selects `qiskit-aer-gpu-cu11` for CUDA 11;
6. otherwise selects CPU `qiskit-aer`;
7. removes all existing Aer/Qiskit variants before installation;
8. for either GPU profile, also removes `qiskit-ibm-runtime`, `samplomatic`,
   and `ibm-quantum-schemas`, keeping simulator and hardware stacks isolated;
9. installs the selected editable project extra;
10. runs `pip check`; and
11. executes a real Aer job and records the reported execution device.

A driver reporting CUDA 13 selects the CUDA 12 Aer package. NVIDIA drivers are
backward compatible with applications built against an older CUDA major
version supported by the driver.

To preview the selection without changing the environment:

```bash
python scripts/install_environment.py --profile full --dry-run
```

To install only Qiskit and Aer dependencies:

```bash
python scripts/install_environment.py --profile quantum
```

To deliberately force CPU Aer:

```bash
python scripts/install_environment.py --profile full --force-cpu
```

After installation, verify manually:

```bash
python - <<'PY'
from importlib import metadata
from qiskit_aer import AerSimulator

for name in ("qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11"):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        pass

print("Devices:", AerSimulator().available_devices())
PY
```

On a correctly configured NVIDIA host the device tuple must include `GPU`.
The installer exits with an error rather than silently accepting CPU-only Aer
when it selected a GPU package.

## CUDA 12 and IBM Runtime require separate environments

The CUDA 12 simulator wheel available to this project is
`qiskit-aer-gpu==0.15.1`, paired with `qiskit==1.4.6`. Current IBM Runtime uses
Qiskit 2.x. Installing both stacks in one environment can pass an import in one
order and then fail after the next package operation. Do not combine them.

Use the main project environment for the RTX 6000 simulator:

```bash
conda create -n vanguard-aer-gpu python=3.11 -y
conda activate vanguard-aer-gpu
python scripts/install_environment.py --profile full
python -m pip check
```

Use a separate environment for IBM hardware submissions:

```bash
conda create -n vanguard-ibm-runtime python=3.11 -y
conda activate vanguard-ibm-runtime
python -m pip install -e ".[ibm-runtime]"
python -m pip check
```

The two environments can use the same repository checkout and YAML files.
Credentials remain in IBM Runtime's account configuration, never in Git.

## Explicit extras

The project also exposes explicit extras:

```bash
python -m pip install -e ".[quantum-cpu]"
python -m pip install -e ".[quantum-gpu]"       # CUDA 12+
python -m pip install -e ".[quantum-gpu-cu11]" # CUDA 11
```

Complete-stack variants are `full`, `full-gpu`, and `full-gpu-cu11`.
The legacy `quantum` and `full` extras remain portable CPU defaults.

Both `full-gpu` variants intentionally exclude IBM Runtime. CUDA 11 is package
compatible with current Runtime, but keeping all GPU simulators separate makes
the install reproducible and prevents later upgrades from changing the Aer
environment.

## What the GPU accelerates

The production XY-QAOA path optimizes angles with the exact fixed-cardinality
CPU subspace simulator, then sends the final Qiskit circuit to Aer GPU for
sampling. For the default 16-qubit window, the subspace has only
`C(16, 7) = 11,440` amplitudes; repeated GPU launches during COBYLA are normally
slower than this CPU calculation. Every run writes `quantum_execution.csv` and
`hybrid_diagnostics.json`, including the actual Aer device and phase timings.
