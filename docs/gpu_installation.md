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
7. removes all existing Aer variants before installation;
8. installs the selected editable project extra; and
9. verifies `AerSimulator().available_devices()`.

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

## Explicit extras

The project also exposes explicit extras:

```bash
python -m pip install -e ".[quantum-cpu]"
python -m pip install -e ".[quantum-gpu]"       # CUDA 12+
python -m pip install -e ".[quantum-gpu-cu11]" # CUDA 11
```

Complete-stack variants are `full`, `full-gpu`, and `full-gpu-cu11`.
The legacy `quantum` and `full` extras remain portable CPU defaults.
