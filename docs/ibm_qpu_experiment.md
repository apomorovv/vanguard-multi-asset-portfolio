# IBM QPU Experiment Protocol

## 1. Scope

The IBM experiment demonstrates one or more adaptive XY-QAOA change windows on
real hardware.

It is not a full-universe QPU portfolio solve. The following stages remain
classical:

- factor-risk data representation;
- full-universe convex relaxation;
- valid exact-cardinality initialization;
- change-window construction;
- QAOA angle optimization;
- fixed-support continuous allocation;
- independent validation;
- optional Gurobi certification.

The QPU samples candidate support bitstrings for a small window.

## 2. Experimental Question

For a fixed support window and transferred QAOA angles:

> Can current IBM hardware sample useful candidate supports while maintaining a
> meaningful fixed-cardinality rate and producing competitive validated
> portfolios after exact reallocation?

## 3. Primary Outputs

Report:

- requested and actual backend;
- Runtime job identifier;
- experiment date and calibration timestamp when available;
- window size $F$;
- required Hamming weight $r$;
- QAOA depth $p$;
- shots;
- raw cardinality-feasibility rate;
- postselected cardinality-feasibility rate, if used;
- number of unique raw bitstrings;
- number of unique fixed-weight supports;
- number of supports sent to the allocation oracle;
- number of financially feasible supports;
- best validated portfolio objective;
- improvement relative to the warm-start support;
- transpiled width;
- transpiled circuit depth;
- two-qubit operation count;
- transpilation seed and optimization level;
- QPU usage time when reported;
- queue-inclusive Runtime wall time;
- complete end-to-end time.

A high cardinality rate is a circuit-correctness result. It is not by itself a
portfolio-quality or speed result.

## 4. Environment

Use a separate environment from the CUDA 12 Aer GPU stack:

```bash
python -m venv .venv-ibm-runtime
source .venv-ibm-runtime/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[qp,ibm-runtime,test]"
python -m pip check
```

Configure credentials with the supported `qiskit-ibm-runtime` account
mechanism. Never place an API token in repository files.

## 5. Backend Selection

Do not hardcode one processor in the committed default configuration. Hardware
availability and calibration change.

On the experiment date, choose an operational backend with:

1. enough qubits for the transpiled circuit;
2. a connected subgraph compatible with the mixer;
3. low two-qubit error on the selected edges;
4. acceptable readout error;
5. acceptable queue length;
6. no maintenance or calibration warning.

Record the selected backend rather than describing it as permanently best.

## 6. Reference Sequence

For each selected window, run:

1. exact fixed-weight subspace simulator;
2. Aer CPU or GPU physical-circuit sampling;
3. IBM QPU sampling.

Use the same:

- QUBO;
- initial bitstring;
- optimized angles;
- mixer;
- depth;
- shots where practical;
- candidate-ranking rule;
- allocation oracle;
- validation tolerance.

This separates algorithmic behavior from physical execution noise.

## 7. Recommended Hardware Progression

Start conservatively:

1. 8 qubits, $p=1$, 4,096 shots;
2. 12 qubits, $p=1$, 4,096 to 8,192 shots;
3. 16 qubits, $p=1$, 8,192 shots if transpiled resources remain reasonable;
4. $p=2$ only after the $p=1$ baseline is stable.

A 20-qubit case is an extension, not a requirement. Proceed only when
transpiled two-qubit depth and calibration quality justify it.

## 8. Example Command

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

`--ibm-backend` is required when the selected quantum backend is
`ibm_runtime`.

## 9. Why Angles Are Optimized Classically

The current implementation optimizes angles in the exact fixed-weight CPU
subspace, then transfers those angles to the physical circuit.

For a 16-qubit, 7-excitation window, the optimizer works with

$$
\binom{16}{7}=11{,}440
$$

states.

Using the QPU for dozens of sequential COBYLA evaluations would add queue and
latency costs and would make the experiment difficult to reproduce. Angle
transfer isolates the hardware-sampling question.

## 10. Cardinality on Hardware

In the ideal model, the XY mixer preserves Hamming weight.

On hardware, invalid-cardinality samples may appear because of:

- state-preparation error;
- gate error;
- routing and added operations;
- decoherence;
- readout error.

Report the raw rate:

$$
\text{raw fixed-weight rate}
=
\frac{\text{shots with Hamming weight }r}
{\text{total shots}}.
$$

If postselection is used, also report how many shots and unique supports remain.
Do not report only the postselected rate.

## 11. Allocation and Financial Validation

For every selected fixed-weight support:

1. combine it with the frozen holdings;
2. solve the fixed-support continuous allocation;
3. reconstruct the full-universe weight vector;
4. validate budget, bounds, groups, turnover, exact cardinality, active-weight
   rules, eligibility, mandatory holdings, and every enabled optional guardrail;
5. reject any support with a breach above tolerance.

The best measured bitstring is not automatically the best validated portfolio.
The QUBO is a surrogate and the exact allocation changes the weights.

## 12. Fair Classical Comparison

A quantum-advantage claim requires more than observing an improving QPU sample.

Compare against classical LNS using:

- the same window;
- the same starting support;
- the same QUBO;
- the same allocation oracle;
- the same validation rule;
- an equal end-to-end time budget.

The end-to-end QPU time includes:

- classical preprocessing;
- angle optimization;
- circuit construction;
- transpilation;
- queue time;
- QPU execution;
- result retrieval;
- decoding;
- allocation;
- validation.

Report QPU usage time separately, but do not substitute it for total wall time.

## 13. Acceptance Checklist

Before presenting a hardware result, confirm:

- `actual_backend` is IBM Runtime;
- a Runtime job ID is present;
- no fallback reason is recorded;
- the intended window size is reflected in the transpiled circuit;
- counts sum to the requested shots;
- raw invalid-cardinality samples are disclosed;
- every reported support was reallocated and validated;
- complete wall time and QPU usage time are both shown;
- the classical comparison uses the same window and timing convention;
- the archived result package contains all raw tables and diagnostics.

Read together:

- `quantum_execution.csv`;
- `hybrid_diagnostics.json`;
- `change_windows.csv`;
- `constraint_checks.csv`;
- `hybrid_summary.csv`.

## 14. Claims the Experiment Can Support

Appropriate claims include:

- the circuit executed on a named IBM backend;
- ideal and hardware cardinality rates were measured;
- the QPU generated one or more financially feasible supports after allocation;
- the best QPU-guided support improved or did not improve the warm start;
- hardware depth and error reduced candidate quality relative to ideal
  simulation.

Do not claim:

- a full 2,000-asset quantum optimization;
- global optimality from QPU sampling;
- quantum speedup from QPU usage time alone;
- quantum advantage without an equal-budget classical comparison.
