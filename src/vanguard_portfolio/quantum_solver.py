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
    return (
        qubo.offset
        + bits @ qubo.linear
        + np.einsum("bi,ij,bj->b", bits, qubo.quadratic, bits, optimize=True)
    )


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
    kwargs: dict[str, Any] = {"method": "statevector"}
    if use_gpu:
        kwargs["device"] = "GPU"
    try:
        simulator = AerSimulator(**kwargs)
    except Exception as exc:
        if not use_gpu:
            raise SolverUnavailableError(f"AerSimulator could not initialize: {exc}") from exc
        simulator = AerSimulator(method="statevector")
        use_gpu = False
    measured = circuit.copy()
    measured.measure_all()
    compiled = transpile(
        measured,
        simulator,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )
    result = simulator.run(compiled, shots=int(shots), seed_simulator=int(seed)).result()
    counts = _decode_qiskit_counts(result.get_counts(), circuit.num_qubits)
    operations = {str(name): int(value) for name, value in compiled.count_ops().items()}
    two_qubit = sum(
        count
        for name, count in operations.items()
        if name.lower() in {"cx", "cz", "ecr", "rxx", "ryy", "rzz", "swap"}
    )
    return counts, {
        "backend": "aer_gpu" if use_gpu else "aer_cpu",
        "transpiled_depth": int(compiled.depth()),
        "transpiled_size": int(compiled.size()),
        "transpiled_two_qubit_gates": int(two_qubit),
        "transpiled_operations": operations,
    }


def _sample_ibm_runtime(
    circuit,
    *,
    backend_name: str,
    shots: int,
    optimization_level: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise SolverUnavailableError(
            "IBM Runtime packages are not installed; install the 'ibm-runtime' extra"
        ) from exc
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    measured = circuit.copy()
    measured.measure_all()
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(optimization_level),
    )
    isa_circuit = pass_manager.run(measured)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=int(shots))
    publication = job.result()[0]
    raw = publication.data.meas.get_counts()
    counts = _decode_qiskit_counts(raw, circuit.num_qubits)
    operations = {str(name): int(value) for name, value in isa_circuit.count_ops().items()}
    two_qubit = sum(
        count
        for name, count in operations.items()
        if name.lower() in {"cx", "cz", "ecr", "rxx", "ryy", "rzz", "swap"}
    )
    return counts, {
        "backend": backend_name,
        "job_id": job.job_id(),
        "transpiled_depth": int(isa_circuit.depth()),
        "transpiled_size": int(isa_circuit.size()),
        "transpiled_two_qubit_gates": int(two_qubit),
        "transpiled_operations": operations,
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
    integers, basis_bits = _fixed_weight_basis(qubo.n, required)
    energies = _basis_energies(qubo, basis_bits)
    center = float(np.mean(energies))
    scale = max(float(np.std(energies)), float(np.ptp(energies)) / 10.0, 1e-12)
    normalized = (energies - center) / scale
    edges = mixer_edges(qubo.n, config.mixer)
    pairs = _mixer_pairs(integers, basis_bits, edges)

    initial = np.zeros(state_count, dtype=complex)
    if config.initial_state == "dicke":
        initial[:] = 1.0 / np.sqrt(state_count)
    else:
        integer = np.uint64(sum(int(value) << index for index, value in enumerate(bits0)))
        location = int(np.searchsorted(integers, integer))
        if location >= state_count or integers[location] != integer:
            raise RuntimeError("warm-start bitstring is absent from fixed-weight basis")
        initial[location] = 1.0

    angles, optimized_state, _, history = _optimize_subspace_angles(
        normalized, pairs, initial, config
    )
    probabilities = np.square(np.abs(optimized_state))
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(config.seed)
    sampled_locations = rng.choice(
        state_count, size=int(config.shots), replace=True, p=probabilities
    )
    subspace_counts = Counter(int(value) for value in sampled_locations)
    counts = {
        _bit_key(basis_bits[location]): int(count)
        for location, count in subspace_counts.items()
    }
    backend_metadata: dict[str, Any] = {"backend": "subspace"}
    if config.backend != "subspace":
        circuit = build_xy_qaoa_circuit(
            qubo,
            bits0,
            angles,
            mixer=config.mixer,
            cost_scale=scale,
            dicke=config.initial_state == "dicke",
        )
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
            )

    ranked: list[tuple[float, int, np.ndarray]] = []
    valid_shots = 0
    for key, count in counts.items():
        bits = np.fromiter((int(value) for value in key), dtype=int, count=qubo.n)
        if int(bits.sum()) == required:
            valid_shots += int(count)
            ranked.append((qubo.energy(bits), -int(count), bits))
    ranked.sort(key=lambda item: (item[0], item[1]))
    candidates = [item[2] for item in ranked[: int(config.top_candidates)]]
    if not candidates:
        raise RuntimeError("XY-QAOA sampling returned no fixed-cardinality candidate")
    expected = float(np.dot(probabilities, energies))
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
        "cost_center": center,
        "cost_scale": scale,
        **backend_metadata,
    }
    return QuantumSearchResult(
        method=f"xy_qaoa_{metadata['backend']}",
        bitstrings=candidates,
        counts=counts,
        angles=angles,
        expected_surrogate_energy=expected,
        best_sampled_energy=float(ranked[0][0]),
        cardinality_feasibility_rate=valid_shots / max(sum(counts.values()), 1),
        runtime=time.perf_counter() - start,
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
    count = 2**qubo.n
    integers = np.arange(count, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(qubo.n, dtype=np.uint64)) & 1).astype(np.uint8)
    energies = _basis_energies(qubo, bits)
    center = float(np.mean(energies))
    scale = max(float(np.std(energies)), 1e-12)
    normalized = (energies - center) / scale
    initial = np.full(count, 1.0 / np.sqrt(count), dtype=complex)

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
    optimized = minimize(
        loss,
        guess,
        method="COBYLA",
        options={"maxiter": int(optimizer_maxiter), "rhobeg": 0.5, "tol": 1e-4},
    )
    state = state_for(np.asarray(optimized.x))
    probabilities = np.square(np.abs(state))
    rng = np.random.default_rng(seed)
    samples = rng.choice(count, size=int(shots), p=probabilities)
    sampled_counts = Counter(int(value) for value in samples)
    counts = {_bit_key(bits[index]): value for index, value in sampled_counts.items()}
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
    return QuantumSearchResult(
        method="penalty_qaoa_statevector",
        bitstrings=candidates,
        counts=counts,
        angles=np.asarray(optimized.x, dtype=float),
        expected_surrogate_energy=float(np.dot(probabilities, energies)),
        best_sampled_energy=float(ranked[0][0]),
        cardinality_feasibility_rate=valid_shots / max(int(shots), 1),
        runtime=time.perf_counter() - start,
        metadata={
            "backend": "numpy_statevector",
            "qubits": qubo.n,
            "depth_p": int(depth),
            "shots": int(shots),
            "optimizer_evaluations": len(history),
            "logical_two_qubit_gates": logical_two_qubit,
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
