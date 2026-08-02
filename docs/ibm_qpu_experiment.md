# IBM QPU experiment protocol

The IBM experiment is a hardware demonstration of one adaptive XY-QAOA change
window. It is not a full-universe QPU solve: the factor QP, window construction,
angle optimization, allocation oracle, and validation remain classical.

## Experimental question

For a fixed window and transferred QAOA angles, can current IBM hardware
sample useful supports while approximately preserving the required window
cardinality?

The primary outputs are therefore:

- raw and postselected cardinality-feasibility rate;
- number of unique sampled supports;
- best valid objective after exact reallocation;
- improvement relative to the warm-start support;
- transpiled depth and two-qubit-operation count;
- shots, QPU usage time, queue-inclusive wall time, and complete end-to-end
  time.

The experiment does not establish quantum advantage unless it is compared with
classical LNS under an equal total time budget, including queueing and all
classical work.

## Environment

Use a separate environment from CUDA 12 Aer:

```bash
python -m venv .venv-ibm-runtime
source .venv-ibm-runtime/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[qp,ibm-runtime,test]"
```

Save IBM credentials through the supported `qiskit-ibm-runtime` account
configuration. Do not place tokens in YAML, shell scripts, notebooks, logs, or
Git history.

`SamplerV2` is used because it returns measured bitstrings. The implementation
submits one Runtime job per selected window and records the Runtime job ID and
available usage metrics. IBM documents the sampler interface and execution
modes in the following primary sources:

- [SamplerV2 API](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/sampler-v2)
- [Batch execution](https://quantum.cloud.ibm.com/docs/guides/run-jobs-batch)
- [Session execution](https://quantum.cloud.ibm.com/docs/guides/run-jobs-session)
- [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)

Job mode is the most portable default. Batch mode is appropriate when the
8-, 12-, and 16-qubit circuits are independent and ready together. Session mode
is useful for iterative hardware refinement only when the account plan and
workload support it.

## Backend selection

Do not hardcode a processor in committed configuration. On the experiment
date, select a backend that has:

1. enough operational qubits for the transpiled window;
2. low two-qubit error on a connected subgraph;
3. acceptable readout error;
4. an acceptable queue;
5. no maintenance or calibration warning.

Record the backend name, calibration timestamp, target version, and chosen
transpilation seed with the result. A backend that is best today need not be
best on a later run.

## Recommended sequence

Run the same window first with the exact subspace sampler and Aer, then use the
QPU:

1. 8 qubits, depth \(p=1\), 4,096 shots;
2. 12 qubits, depth \(p=1\), 4,096-8,192 shots;
3. 16 qubits, depth \(p=1\), 8,192 shots if transpiled resources remain
   reasonable;
4. depth \(p=2\) only after the \(p=1\) comparison is stable.

Twenty qubits is an optional extension, not a submission requirement. Proceed
only when the transpiled two-qubit depth and calibration data justify the
additional width.

Example command:

```bash
python scripts/run_hybrid.py \
  --config configs/large_hybrid.yaml \
  --quantum-backend ibm_runtime \
  --ibm-backend <backend-name> \
  --window-size 12 \
  --iterations 1 \
  --quantum-shots 4096 \
  --no-gurobi \
  --output results/qpu_12 \
  --overwrite
```

The classical angle optimizer runs in the exact fixed-weight subspace. This is
intentional: at the default 16-qubit, seven-excitation window it operates on
only \(\binom{16}{7}=11{,}440\) states. The QPU samples the transferred physical
circuit; it is not used for dozens of sequential COBYLA evaluations.

## Acceptance and reporting

Before using a QPU result in the presentation, confirm:

- the actual backend is `ibm_runtime` and a job ID is present;
- no fallback reason is recorded;
- the transpiled circuit width matches the intended window;
- raw counts sum to the requested shots;
- every presented support was reallocated and independently validated;
- invalid-cardinality samples are reported, not silently treated as valid;
- QPU usage time and complete wall time are both disclosed;
- the classical comparison uses the same window, objective, shots/candidate
  budget, and end-to-end timing convention.

Read `quantum_execution.csv`, `hybrid_diagnostics.json`, and
`constraint_checks.csv` together. A high cardinality rate is a circuit
correctness result; it is not a portfolio-quality or speed result by itself.
