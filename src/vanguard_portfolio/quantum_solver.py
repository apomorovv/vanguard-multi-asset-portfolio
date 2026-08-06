"""Constraint-preserving XY-QAOA for adaptive portfolio change windows.

The dependency-free subspace simulator is the deterministic reference.  It
stores only bitstrings with the required Hamming weight and applies XY swap
gates directly.  Optional Qiskit paths reuse the optimized angles for Aer GPU
or IBM Runtime sampling and return measured support candidates to the same
classical allocation oracle used by every baseline.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize

from .qubo_builder import QUBOModel
from .schemas import SolverUnavailableError


@dataclass(frozen=True)
class XYQAOAConfig:
    depth: int = 1
    shots: int = 4_096
    optimizer_maxiter: int = 60
    optimizer_starts: int = 3
    seed: int = 0
    initial_state: str = "warm"
    mixer: str = "ring"
    backend: str = "subspace"
    maximum_subspace_states: int = 400_000
    top_candidates: int = 128
    transpile_optimization_level: int = 3
    ibm_backend: str | None = None

    enable_dynamical_decoupling: bool = True
    dynamical_decoupling_sequence: str = "XY4"

    enable_pauli_twirling: bool = False
    pauli_twirling_strategy: str = "active"
    pauli_twirling_num_randomizations: int = 32

    def __post_init__(self) -> None:
        if int(self.depth) <= 0 or int(self.shots) <= 0:
            raise ValueError("depth and shots must be positive")
        if int(self.optimizer_maxiter) <= 0 or int(self.optimizer_starts) <= 0:
            raise ValueError("optimizer budgets must be positive")
        if self.initial_state not in {"warm", "dicke"}:
            raise ValueError("initial_state must be 'warm' or 'dicke'")
        if self.mixer not in {"ring", "complete"}:
            raise ValueError("mixer must be 'ring' or 'complete'")
        if self.backend not in {"subspace", "aer_cpu", "aer_gpu", "ibm_runtime"}:
            raise ValueError("unknown XY-QAOA backend")


@dataclass
class QuantumSearchResult:
    method: str
    bitstrings: list[np.ndarray]
    counts: dict[str, int]
    angles: np.ndarray
    expected_surrogate_energy: float
    best_sampled_energy: float
    cardinality_feasibility_rate: float
    runtime: float
    metadata: dict[str, Any] = field(default_factory=dict)


def mixer_edges(n_qubits: int, topology: str = "ring") -> tuple[tuple[int, int], ...]:
    if topology == "complete":
        return tuple(combinations(range(n_qubits), 2))
    if n_qubits == 2:
        return ((0, 1),)
    return tuple((index, (index + 1) % n_qubits) for index in range(n_qubits))


def _fixed_weight_basis(n: int, weight: int) -> tuple[np.ndarray, np.ndarray]:
    integers = np.fromiter(
        (sum(1 << index for index in selected) for selected in combinations(range(n), weight)),
        dtype=np.uint64,
        count=comb(n, weight),
    )
    integers.sort()
    bits = ((integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)
    return integers, bits


def _basis_energies(qubo: QUBOModel, bits: np.ndarray) -> np.ndarray:
    """Evaluate a QUBO on a batch without multiplying by its zero entries."""
    diagonal_linear = qubo.linear + np.diag(qubo.quadratic)
    energies = qubo.offset + bits @ diagonal_linear
    left, right = np.nonzero(np.triu(qubo.quadratic, k=1))
    if left.size:
        interactions = bits[:, left] * bits[:, right]
        energies = energies + 2.0 * (interactions @ qubo.quadratic[left, right])
    return np.asarray(energies, dtype=float)


def _mixer_pairs(
    integers: np.ndarray,
    bits: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for left, right in edges:
        first = np.flatnonzero((bits[:, left] == 1) & (bits[:, right] == 0))
        partners = integers[first] ^ np.uint64((1 << left) | (1 << right))
        second = np.searchsorted(integers, partners)
        if np.any(second >= integers.size) or not np.array_equal(integers[second], partners):
            raise RuntimeError("fixed-weight mixer basis is internally inconsistent")
        pairs.append((first, second))
    return pairs


@lru_cache(maxsize=8)
def _fixed_weight_structure(
    n: int,
    weight: int,
    topology: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, int], ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    """Cache window-size-dependent basis and mixer indices across iterations."""
    integers, bits = _fixed_weight_basis(n, weight)
    edges = mixer_edges(n, topology)
    pairs = tuple(_mixer_pairs(integers, bits, edges))
    integers.setflags(write=False)
    bits.setflags(write=False)
    for first, second in pairs:
        first.setflags(write=False)
        second.setflags(write=False)
    return integers, bits, edges, pairs


def _apply_xy_layer(
    state: np.ndarray,
    beta: float,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    cosine = np.cos(beta)
    sine = -1j * np.sin(beta)
    for left, right in pairs:
        left_values = state[left].copy()
        right_values = state[right].copy()
        state[left] = cosine * left_values + sine * right_values
        state[right] = cosine * right_values + sine * left_values


def _subspace_state(
    normalized_energies: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    angles: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    depth = angles.size // 2
    gammas = angles[:depth]
    betas = angles[depth:]
    state = initial.astype(complex, copy=True)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * normalized_energies)
        _apply_xy_layer(state, float(beta), pairs)
    return state


def _optimize_subspace_angles(
    normalized_energies: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    initial: np.ndarray,
    config: XYQAOAConfig,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    rng = np.random.default_rng(config.seed)
    depth = int(config.depth)
    history: list[float] = []

    def loss(values: np.ndarray) -> float:
        state = _subspace_state(normalized_energies, pairs, values, initial)
        value = float(np.dot(np.square(np.abs(state)), normalized_energies))
        history.append(value)
        return value

    best_angles: np.ndarray | None = None
    best_value = np.inf
    for start in range(int(config.optimizer_starts)):
        if start == 0:
            guess = np.concatenate(
                [np.linspace(0.8, 0.4, depth), np.linspace(0.35, 0.15, depth)]
            )
        else:
            guess = np.concatenate(
                [rng.uniform(-np.pi, np.pi, depth), rng.uniform(-np.pi / 2, np.pi / 2, depth)]
            )
        optimized = minimize(
            loss,
            guess,
            method="COBYLA",
            options={
                "maxiter": int(config.optimizer_maxiter),
                "rhobeg": 0.5,
                "tol": 1e-4,
            },
        )
        if float(optimized.fun) < best_value:
            best_value = float(optimized.fun)
            best_angles = np.asarray(optimized.x, dtype=float)
    if best_angles is None:
        raise RuntimeError("QAOA parameter optimization produced no result")
    state = _subspace_state(normalized_energies, pairs, best_angles, initial)
    return best_angles, state, best_value, history


def _bit_key(bits: np.ndarray) -> str:
    return "".join(str(int(value)) for value in bits)


def _decode_qiskit_counts(raw: dict[str, int], n: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, count in raw.items():
        compact = key.replace(" ", "")
        logical = compact[::-1][:n]
        result[logical] = result.get(logical, 0) + int(count)
    return result


def build_xy_qaoa_circuit(
    qubo: QUBOModel,
    initial_bits: np.ndarray,
    angles: np.ndarray,
    *,
    mixer: str = "ring",
    cost_scale: float = 1.0,
    dicke: bool = False,
):
    """Return an unmeasured Qiskit circuit; Qiskit is imported lazily."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
    except ImportError as exc:
        raise SolverUnavailableError(
            "Qiskit is not installed; install the 'quantum' extra"
        ) from exc
    bits = np.asarray(initial_bits, dtype=int).reshape(qubo.n)
    depth = angles.size // 2
    circuit = QuantumCircuit(qubo.n)
    if dicke:
        amplitudes = np.zeros(2**qubo.n, dtype=complex)
        for selected in combinations(range(qubo.n), int(bits.sum())):
            index = sum(1 << qubit for qubit in selected)
            amplitudes[index] = 1.0
        amplitudes /= np.linalg.norm(amplitudes)
        circuit.append(StatePreparation(amplitudes), range(qubo.n))
    else:
        for index in np.flatnonzero(bits):
            circuit.x(int(index))

    ising = qubo.to_ising()
    fields = ising.fields / float(cost_scale)
    couplings = {pair: value / float(cost_scale) for pair, value in ising.couplings.items()}
    for gamma, beta in zip(angles[:depth], angles[depth:]):
        for index, field_value in enumerate(fields):
            if field_value:
                circuit.rz(2.0 * float(gamma) * float(field_value), index)
        for (left, right), coupling in couplings.items():
            circuit.rzz(2.0 * float(gamma) * float(coupling), left, right)
        for left, right in mixer_edges(qubo.n, mixer):
            circuit.rxx(float(beta), left, right)
            circuit.ryy(float(beta), left, right)
    return circuit


def _sample_aer(
    circuit,
    *,
    shots: int,
    seed: int,
    use_gpu: bool,
    optimization_level: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SolverUnavailableError(
            "Qiskit Aer is not installed; install the 'quantum' extra"
        ) from exc

    def execute(gpu: bool) -> tuple[dict[str, int], dict[str, Any]]:
        sampler_start = time.perf_counter()
        available_devices = tuple(str(device) for device in AerSimulator().available_devices())
        if gpu and "GPU" not in available_devices:
            raise RuntimeError(
                f'Aer does not advertise a GPU device; available={available_devices}'
            )
        setup_start = time.perf_counter()
        kwargs: dict[str, Any] = {"method": "statevector"}
        if gpu:
            kwargs["device"] = "GPU"
        simulator = AerSimulator(**kwargs)
        measured = circuit.copy()
        measured.measure_all()
        setup_seconds = time.perf_counter() - setup_start
        transpile_start = time.perf_counter()
        compiled = transpile(
            measured,
            simulator,
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed),
        )
        transpile_seconds = time.perf_counter() - transpile_start
        # Some CPU-only Aer builds accept device="GPU" at construction and
        # fail only when the job executes.  Keep the whole execution inside the
        # fallback boundary rather than checking only simulator construction.
        simulation_start = time.perf_counter()
        result = simulator.run(compiled, shots=int(shots), seed_simulator=int(seed)).result()
        simulation_seconds = time.perf_counter() - simulation_start
        if not bool(getattr(result, "success", True)):
            raise RuntimeError(f"Aer simulation failed: {getattr(result, 'status', 'unknown')}")
        decode_start = time.perf_counter()
        counts = _decode_qiskit_counts(result.get_counts(), circuit.num_qubits)
        decode_seconds = time.perf_counter() - decode_start
        operations = {str(name): int(value) for name, value in compiled.count_ops().items()}
        two_qubit = sum(
            count
            for name, count in operations.items()
            if name.lower() in {"cx", "cz", "ecr", "rxx", "ryy", "rzz", "swap"}
        )
        experiment_metadata = {}
        if getattr(result, "results", None):
            experiment_metadata = dict(getattr(result.results[0], "metadata", {}) or {})
        reported_device = str(experiment_metadata.get("device", "")).upper()
        if gpu and reported_device and reported_device != "GPU":
            raise RuntimeError(
                f"Aer accepted device='GPU' but result metadata reports {reported_device!r}"
            )
        execution_device = reported_device or ("GPU" if gpu else "CPU")
        verification = (
            "result_metadata"
            if reported_device
            else "available_devices_and_successful_device_execution"
        )
        return counts, {
            "backend": "aer_gpu" if gpu else "aer_cpu",
            "requested_backend": "aer_gpu" if use_gpu else "aer_cpu",
            "available_devices": list(available_devices),
            "execution_device": execution_device,
            "gpu_accelerated": bool(gpu and execution_device == "GPU"),
            "device_verification": verification,
            "simulator_method": str(experiment_metadata.get("method", "statevector")),
            "transpiled_depth": int(compiled.depth()),
            "transpiled_size": int(compiled.size()),
            "transpiled_two_qubit_gates": int(two_qubit),
            "transpiled_operations": operations,
            "simulator_setup_seconds": setup_seconds,
            "transpile_seconds": transpile_seconds,
            "simulation_seconds": simulation_seconds,
            "count_decode_seconds": decode_seconds,
            "sampler_total_seconds": time.perf_counter() - sampler_start,
        }

    try:
        return execute(use_gpu)
    except Exception as first_error:
        if not use_gpu:
            raise SolverUnavailableError(
                f"Aer CPU simulation failed: {first_error}"
            ) from first_error
        try:
            counts, metadata = execute(False)
        except Exception as fallback_error:
            raise SolverUnavailableError(
                "Aer GPU simulation failed and the automatic CPU fallback also failed: "
                f"GPU={first_error}; CPU={fallback_error}"
            ) from fallback_error
        metadata.update(
            {
                "requested_backend": "aer_gpu",
                "fallback_reason": str(first_error),
                "gpu_accelerated": False,
            }
        )
        return counts, metadata


def _sample_ibm_runtime(
    circuit,
    *,
    backend_name: str,
    shots: int,
    optimization_level: int,
    config: XYQAOAConfig,
) -> tuple[dict[str, int], dict[str, Any]]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise SolverUnavailableError(
            "IBM Runtime packages are not installed; install the 'ibm-runtime' extra"
        ) from exc
    sampler_start = time.perf_counter()
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    backend_version = str(getattr(backend, "backend_version", ""))
    calibration_timestamp = ""
    try:
        properties = backend.properties()
        last_update = getattr(properties, "last_update_date", None)
        if last_update is not None:
            calibration_timestamp = (
                last_update.isoformat()
                if hasattr(last_update, "isoformat")
                else str(last_update)
            )
    except Exception:
        pass
    try:
        backend_status = backend.status()
        pending_jobs = int(getattr(backend_status, "pending_jobs", -1))
        operational = bool(getattr(backend_status, "operational", True))
    except Exception:
        pending_jobs = -1
        operational = True
    measured = circuit.copy()
    measured.measure_all()
    transpile_start = time.perf_counter()
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(optimization_level),
    )
    isa_circuit = pass_manager.run(measured)
    transpile_seconds = time.perf_counter() - transpile_start
    sampler = SamplerV2(mode=backend)

    sampler.options.dynamical_decoupling.enable = config.enable_dynamical_decoupling
    if config.enable_dynamical_decoupling:
        sampler.options.dynamical_decoupling.sequence_type = config.dynamical_decoupling_sequence

    sampler.options.twirling.enable_gates = config.enable_pauli_twirling

    if config.enable_pauli_twirling:
        sampler.options.twirling.strategy = config.pauli_twirling_strategy
        sampler.options.twirling.num_randomizations = (
            config.pauli_twirling_num_randomizations
        )

    execution_start = time.perf_counter()
    job = sampler.run([isa_circuit], shots=int(shots))
    publication = job.result()[0]
    execution_seconds = time.perf_counter() - execution_start
    qpu_seconds = None
    try:
        metrics = job.metrics()
        usage = metrics.get("usage", metrics.get("usage_estimation", {}))
        if isinstance(usage, dict) and usage.get("quantum_seconds") is not None:
            qpu_seconds = float(usage["quantum_seconds"])
    except Exception:
        pass
    decode_start = time.perf_counter()
    raw = publication.data.meas.get_counts()
    counts = _decode_qiskit_counts(raw, circuit.num_qubits)
    decode_seconds = time.perf_counter() - decode_start
    operations = {str(name): int(value) for name, value in isa_circuit.count_ops().items()}
    two_qubit = sum(
        count
        for name, count in operations.items()
        if name.lower() in {"cx", "cz", "ecr", "rxx", "ryy", "rzz", "swap"}
    )
    return counts, {
        "backend": backend_name,
        "requested_backend": "ibm_runtime",
        "execution_device": backend_name,
        "gpu_accelerated": False,
        "device_verification": "ibm_runtime_job",
        "runtime_mode": "job",
        "job_id": job.job_id(),
        "backend_version": backend_version,
        "backend_calibration_timestamp": calibration_timestamp,
        "backend_num_qubits": int(getattr(backend, "num_qubits", circuit.num_qubits)),
        "backend_operational_at_submission": operational,
        "backend_pending_jobs_at_submission": pending_jobs,
        "transpiled_depth": int(isa_circuit.depth()),
        "transpiled_size": int(isa_circuit.size()),
        "transpiled_two_qubit_gates": int(two_qubit),
        "transpiled_operations": operations,
        "transpile_seconds": transpile_seconds,
        "simulation_seconds": execution_seconds,
        "qpu_wall_seconds": execution_seconds,
        "qpu_usage_seconds": qpu_seconds,
        "count_decode_seconds": decode_seconds,
        "sampler_total_seconds": time.perf_counter() - sampler_start,
    }


def solve_xy_qaoa(
    qubo: QUBOModel,
    initial_bits: np.ndarray,
    config: XYQAOAConfig | None = None,
) -> QuantumSearchResult:
    """Optimize in the exact-Hamming-weight subspace, then sample candidates."""
    config = config or XYQAOAConfig()
    start = time.perf_counter()
    bits0 = np.asarray(initial_bits, dtype=int).reshape(-1)
    if bits0.shape != (qubo.n,) or np.any((bits0 != 0) & (bits0 != 1)):
        raise ValueError("initial_bits must be a binary vector matching the QUBO")
    required = qubo.required_ones if qubo.required_ones is not None else int(bits0.sum())
    if int(bits0.sum()) != required:
        raise ValueError("initial_bits must have the required Hamming weight")
    state_count = comb(qubo.n, required)
    if state_count > int(config.maximum_subspace_states):
        raise ValueError(
            f"fixed-weight simulator requires {state_count:,} states; reduce the window "
            "or raise maximum_subspace_states intentionally"
        )
    preprocessing_start = time.perf_counter()
    integers, basis_bits, edges, pairs = _fixed_weight_structure(
        qubo.n,
        required,
        config.mixer,
    )
    energies = _basis_energies(qubo, basis_bits)
    center = float(np.mean(energies))
    scale = max(float(np.std(energies)), float(np.ptp(energies)) / 10.0, 1e-12)
    normalized = (energies - center) / scale
    initial = np.zeros(state_count, dtype=complex)
    if config.initial_state == "dicke":
        initial[:] = 1.0 / np.sqrt(state_count)
    else:
        integer = np.uint64(sum(int(value) << index for index, value in enumerate(bits0)))
        location = int(np.searchsorted(integers, integer))
        if location >= state_count or integers[location] != integer:
            raise RuntimeError("warm-start bitstring is absent from fixed-weight basis")
        initial[location] = 1.0
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    optimization_start = time.perf_counter()
    angles, optimized_state, _, history = _optimize_subspace_angles(
        normalized, list(pairs), initial, config
    )
    angle_optimization_seconds = time.perf_counter() - optimization_start
    probabilities = np.square(np.abs(optimized_state))
    probabilities /= probabilities.sum()
    exact_expected = float(np.dot(probabilities, energies))
    circuit_build_seconds = 0.0
    subspace_sampling_seconds = 0.0
    if config.backend == "subspace":
        sampling_start = time.perf_counter()
        rng = np.random.default_rng(config.seed)
        sampled_locations = rng.choice(
            state_count, size=int(config.shots), replace=True, p=probabilities
        )
        subspace_counts = Counter(int(value) for value in sampled_locations)
        counts = {
            _bit_key(basis_bits[location]): int(count)
            for location, count in subspace_counts.items()
        }
        subspace_sampling_seconds = time.perf_counter() - sampling_start
        backend_metadata: dict[str, Any] = {
            "backend": "subspace",
            "requested_backend": "subspace",
            "execution_device": "CPU",
            "gpu_accelerated": False,
            "device_verification": "not_applicable",
            "sampler_total_seconds": subspace_sampling_seconds,
        }
    else:
        circuit_start = time.perf_counter()
        circuit = build_xy_qaoa_circuit(
            qubo,
            bits0,
            angles,
            mixer=config.mixer,
            cost_scale=scale,
            dicke=config.initial_state == "dicke",
        )
        circuit_build_seconds = time.perf_counter() - circuit_start
        if config.backend in {"aer_cpu", "aer_gpu"}:
            counts, backend_metadata = _sample_aer(
                circuit,
                shots=config.shots,
                seed=config.seed,
                use_gpu=config.backend == "aer_gpu",
                optimization_level=config.transpile_optimization_level,
            )
        else:
            if not config.ibm_backend:
                raise ValueError("ibm_backend is required for IBM Runtime sampling")
            counts, backend_metadata = _sample_ibm_runtime(
                circuit,
                backend_name=config.ibm_backend,
                shots=config.shots,
                optimization_level=config.transpile_optimization_level,
                config=config
            )

    ranking_start = time.perf_counter()
    ranked: list[tuple[float, int, np.ndarray]] = []
    valid_shots = 0
    sampled_energy_sum = 0.0
    for key, count in counts.items():
        bits = np.fromiter((int(value) for value in key), dtype=int, count=qubo.n)
        energy = qubo.energy(bits)
        sampled_energy_sum += int(count) * energy
        if int(bits.sum()) == required:
            valid_shots += int(count)
            ranked.append((energy, -int(count), bits))
    ranked.sort(key=lambda item: (item[0], item[1]))
    candidates = [item[2] for item in ranked[: int(config.top_candidates)]]
    if not candidates:
        raise RuntimeError("XY-QAOA sampling returned no fixed-cardinality candidate")
    total_shots = max(sum(counts.values()), 1)
    sampled_expected = sampled_energy_sum / total_shots
    candidate_ranking_seconds = time.perf_counter() - ranking_start
    logical_two_qubit = int(config.depth) * (
        len(qubo.to_ising().couplings) + 2 * len(edges)
    )
    metadata = {
        "qubits": qubo.n,
        "required_ones": required,
        "subspace_states": state_count,
        "depth_p": int(config.depth),
        "shots": int(config.shots),
        "mixer_edges": len(edges),
        "cost_edges": len(qubo.to_ising().couplings),
        "logical_two_qubit_gates": logical_two_qubit,
        "optimizer_evaluations": len(history),
        "parameter_optimizer_backend": "fixed_weight_subspace_cpu",
        "cost_center": center,
        "cost_scale": scale,
        "exact_expected_surrogate_energy": exact_expected,
        "sampled_expected_surrogate_energy": sampled_expected,
        "unique_sampled_bitstrings": len(counts),
        "preprocessing_seconds": preprocessing_seconds,
        "angle_optimization_seconds": angle_optimization_seconds,
        "subspace_sampling_seconds": subspace_sampling_seconds,
        "circuit_build_seconds": circuit_build_seconds,
        "candidate_ranking_seconds": candidate_ranking_seconds,
        **backend_metadata,
    }
    runtime = time.perf_counter() - start
    metadata["search_runtime_seconds"] = runtime
    return QuantumSearchResult(
        method=f"xy_qaoa_{metadata['backend']}",
        bitstrings=candidates,
        counts=counts,
        angles=angles,
        expected_surrogate_energy=exact_expected,
        best_sampled_energy=float(ranked[0][0]),
        cardinality_feasibility_rate=valid_shots / total_shots,
        runtime=runtime,
        metadata=metadata,
    )


def solve_penalty_qaoa(
    qubo: QUBOModel,
    *,
    depth: int = 1,
    shots: int = 4_096,
    optimizer_maxiter: int = 60,
    seed: int = 0,
    top_candidates: int = 128,
    maximum_qubits: int = 20,
) -> QuantumSearchResult:
    """Standard X-mixer penalty-QAOA baseline on the full binary space."""
    if qubo.n > int(maximum_qubits):
        raise ValueError("penalty-QAOA full-state simulation exceeds its safety limit")
    start = time.perf_counter()
    preprocessing_start = time.perf_counter()
    count = 2**qubo.n
    integers = np.arange(count, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(qubo.n, dtype=np.uint64)) & 1).astype(np.uint8)
    energies = _basis_energies(qubo, bits)
    center = float(np.mean(energies))
    scale = max(float(np.std(energies)), 1e-12)
    normalized = (energies - center) / scale
    initial = np.full(count, 1.0 / np.sqrt(count), dtype=complex)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    def state_for(angles: np.ndarray) -> np.ndarray:
        state = initial.copy()
        for gamma, beta in zip(angles[:depth], angles[depth:]):
            state *= np.exp(-1j * gamma * normalized)
            cosine = np.cos(beta)
            sine = -1j * np.sin(beta)
            for qubit in range(qubo.n):
                left = np.flatnonzero(bits[:, qubit] == 0)
                right = left | (1 << qubit)
                left_values = state[left].copy()
                right_values = state[right].copy()
                state[left] = cosine * left_values + sine * right_values
                state[right] = cosine * right_values + sine * left_values
        return state

    history: list[float] = []

    def loss(angles: np.ndarray) -> float:
        state = state_for(angles)
        value = float(np.dot(np.square(np.abs(state)), normalized))
        history.append(value)
        return value

    guess = np.concatenate([np.full(depth, 0.5), np.full(depth, 0.35)])
    optimization_start = time.perf_counter()
    optimized = minimize(
        loss,
        guess,
        method="COBYLA",
        options={"maxiter": int(optimizer_maxiter), "rhobeg": 0.5, "tol": 1e-4},
    )
    state = state_for(np.asarray(optimized.x))
    angle_optimization_seconds = time.perf_counter() - optimization_start
    probabilities = np.square(np.abs(state))
    sampling_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    samples = rng.choice(count, size=int(shots), p=probabilities)
    sampled_counts = Counter(int(value) for value in samples)
    counts = {_bit_key(bits[index]): value for index, value in sampled_counts.items()}
    sampling_seconds = time.perf_counter() - sampling_start
    ranking_start = time.perf_counter()
    ranked = sorted(
        (
            (energies[index], -value, bits[index].astype(int))
            for index, value in sampled_counts.items()
        ),
        key=lambda item: (item[0], item[1]),
    )
    candidates = [item[2] for item in ranked[: int(top_candidates)]]
    valid_shots = sum(
        value
        for index, value in sampled_counts.items()
        if qubo.required_ones is None or int(bits[index].sum()) == qubo.required_ones
    )
    logical_two_qubit = int(depth) * len(qubo.to_ising().couplings)
    sampled_expected = sum(
        int(value) * float(energies[index])
        for index, value in sampled_counts.items()
    ) / max(int(shots), 1)
    candidate_ranking_seconds = time.perf_counter() - ranking_start
    runtime = time.perf_counter() - start
    return QuantumSearchResult(
        method="penalty_qaoa_statevector",
        bitstrings=candidates,
        counts=counts,
        angles=np.asarray(optimized.x, dtype=float),
        expected_surrogate_energy=float(np.dot(probabilities, energies)),
        best_sampled_energy=float(ranked[0][0]),
        cardinality_feasibility_rate=valid_shots / max(int(shots), 1),
        runtime=runtime,
        metadata={
            "backend": "numpy_statevector",
            "requested_backend": "numpy_statevector",
            "execution_device": "CPU",
            "gpu_accelerated": False,
            "device_verification": "not_applicable",
            "parameter_optimizer_backend": "full_state_numpy_cpu",
            "qubits": qubo.n,
            "required_ones": qubo.required_ones,
            "state_count": count,
            "depth_p": int(depth),
            "shots": int(shots),
            "optimizer_evaluations": len(history),
            "logical_two_qubit_gates": logical_two_qubit,
            "exact_expected_surrogate_energy": float(np.dot(probabilities, energies)),
            "sampled_expected_surrogate_energy": sampled_expected,
            "unique_sampled_bitstrings": len(counts),
            "preprocessing_seconds": preprocessing_seconds,
            "angle_optimization_seconds": angle_optimization_seconds,
            "subspace_sampling_seconds": sampling_seconds,
            "circuit_build_seconds": 0.0,
            "candidate_ranking_seconds": candidate_ranking_seconds,
            "sampler_total_seconds": sampling_seconds,
            "search_runtime_seconds": runtime,
        },
    )


__all__ = [
    "QuantumSearchResult",
    "XYQAOAConfig",
    "build_xy_qaoa_circuit",
    "mixer_edges",
    "solve_penalty_qaoa",
    "solve_xy_qaoa",
]
