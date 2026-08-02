# Constraint-Safe Hybrid Portfolio Optimization

This repository studies sparse, long-only multi-asset portfolio optimization
with classical and quantum-guided neighborhood search. The full financial
model remains classical: a factor quadratic program assigns continuous
weights, an allocation oracle enforces hard constraints, and an independent
validator checks every accepted portfolio. XY-QAOA proposes support changes
inside a small fixed-cardinality window.

The implementation is designed to answer three separate questions without
conflating them:

1. How quickly can a valid, high-quality sparse portfolio be found?
2. How close is that portfolio to a continuous lower bound or a Gurobi bound?
3. Does the quantum window generator preserve cardinality and produce useful
   candidate supports?

No quantum-speedup claim is made. CPU, GPU, QPU, and solver time are recorded
separately so the evidence can be interpreted directly.

## Architecture

```mermaid
flowchart TD
    A["Factor-model universe"] --> B["Full-universe factor QP"]
    B --> C["Valid exact-K portfolio"]
    C --> D["Adaptive change window"]
    D --> E["Classical LNS or XY-QAOA"]
    E --> F["Fixed-support allocation oracle"]
    F --> G["Independent validator"]
    G -->|"better and valid"| C
    C --> H["Optional Gurobi bound"]
```

The quantum circuit never produces final portfolio weights. It samples asset
supports; every support returns to the same continuous allocation model and
validator used by the classical search.

## Optimization model

For weights \(w\), support variables \(z\), current weights \(w^0\), and
transaction-cost auxiliaries \(t\), the canonical objective is

\[
\min\; \lambda_r w^T\Sigma w
-\lambda_g\mu^T w
-\lambda_y y^T w
+\lambda_c c^Tt.
\]

The benchmark normally enforces

\[
\mathbf 1^T w=1,\qquad
m_i z_i\le w_i\le u_i z_i,\qquad
\sum_i z_i=K,
\]

together with group bounds and a turnover cap. Eligibility, mandatory
holdings, income, factor exposure, stress-return, and empirical-CVaR limits are
available when the data support them.

Generated large instances use

\[
\Sigma=B\Omega B^T+D.
\]

The solver evaluates risk from \(B\), \(\Omega\), and diagonal \(D\); it does
not need to materialize an \(n\times n\) covariance matrix. This changes
storage from quadratic in the number of assets to \(O(nk)\), where \(k\) is
the number of factors.

Inside an \(F\)-asset change window, XY-QAOA solves a surrogate support problem

\[
\min_{x\in\{0,1\}^F} x^TQx+h^Tx,
\qquad \sum_i x_i=r.
\]

The XY mixer exchanges `10` and `01`. Starting from an \(r\)-asset bitstring
therefore preserves the number of selected window assets in ideal execution.

## What is implemented

- Factor-native continuous optimization with SciPy, OSQP, CVXPY, or Gurobi.
- Deterministic exact-\(K\) initialization with a HiGHS feasibility-MILP
  fallback.
- Cached fixed-support allocation oracle.
- Exact window enumeration for small neighborhoods and tabu/LNS for larger
  neighborhoods.
- Explicit QUBO/Ising construction and constraint-preserving XY-QAOA.
- Fixed-Hamming-weight CPU simulator for parameter optimization.
- Optional Qiskit Aer CPU/GPU sampling and IBM Runtime QPU sampling.
- Direct Gurobi exact-cardinality MIQP with incumbent, bound, and MIP-gap
  reporting.
- Independent validation without clipping or post-solve renormalization.
- CSV/JSON tables, checksums, and matched PNG/PDF presentation graphics.
- A matrix-free scaling study from 250 to 20,000 assets.

## Installation

Python 3.10 or newer is required.

Portable tests and core solver:

```bash
python -m pip install -e ".[test]"
```

Classical QP and MIQP stack:

```bash
python -m pip install -e ".[all-solvers,test]"
```

Portable CPU Aer and IBM Runtime:

```bash
python -m pip install -e ".[full]"
```

On an NVIDIA system, use the compatibility-aware installer:

```bash
python scripts/install_environment.py --profile full
```

The CUDA 12 Aer wheel used by the project requires a Qiskit 1.4 environment,
whereas current IBM Runtime uses Qiskit 2.x. Keep the Aer GPU and IBM Runtime
profiles in separate environments. See [GPU installation](docs/gpu_installation.md)
and [IBM QPU protocol](docs/ibm_qpu_experiment.md).

Gurobi requires a valid license. IBM credentials are managed by
`qiskit-ibm-runtime` and must not be committed to the repository.

## Reproducible runs

Run the tests first:

```bash
python -m pytest -q
```

Tiny certification example:

```bash
python scripts/run_hybrid.py \
  --config configs/tiny_hybrid.yaml \
  --overwrite
```

Two-thousand-asset reference run:

```bash
python scripts/run_hybrid.py \
  --config configs/large_hybrid.yaml \
  --overwrite
```

The large configuration uses matrix-free factor risk, exactly 50 holdings, a
40% L1 turnover cap (20% one-way turnover), three 16-asset change windows, and
optional Gurobi certification.

### Scaling study

The default study runs seven sizes, three seeded repetitions per size, and
writes runtime, quality, feasibility, memory, and quantum timing plots:

```bash
python scripts/run_hybrid_scaling.py \
  --quantum-backend aer_gpu \
  --output results/hybrid_scaling \
  --overwrite
```

Default sizes are 250, 500, 1,000, 2,000, 5,000, 10,000, and 20,000 assets.
Gurobi is attempted only through 2,000 assets; larger cases measure the
scalable hybrid-search path and are not described as globally certified.
Each case runs in a fresh process, and the output records peak resident memory,
time to first valid portfolio, component runtimes, objective gaps, and all
skipped components.

A quick portable check is:

```bash
python scripts/run_hybrid_scaling.py \
  --sizes 250 500 1000 2000 \
  --repetitions 1 \
  --quantum-backend subspace \
  --no-gurobi \
  --output results/hybrid_scaling_quick \
  --overwrite
```

### IBM QPU demonstration

Use a separate IBM Runtime environment and run one selected window at 8, 12,
or 16 qubits. An explicit backend is required:

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

The hardware run is a component demonstration, not a replacement for the
20,000-asset classical factor solve. The asset-universe size and QPU width are
decoupled: the QPU sees only the adaptive change window.

## Evidence package

Every hybrid run writes:

- `hybrid_summary.csv` for objective, runtime, feasibility, and solver status;
- `allocation_weights.csv` for weights and trades;
- `constraint_checks.csv` for all independently recomputed guardrails;
- `change_windows.csv` for the searched asset neighborhoods;
- `quantum_execution.csv` for requested and actual devices, circuit resources,
  cardinality rate, and phase timing;
- `objective_timeline.csv` and `backtest_summary.csv`;
- `hybrid_diagnostics.json`, `problem.json`, and a SHA-256 artifact manifest;
- matched PNG/PDF figures, including a six-guardrail presentation view and the
  full constraint-slack appendix.

The scaling study writes raw run rows, method rows, per-size summaries, the
resolved study configuration, CPU/GPU and package metadata, SHA-256 checksums,
and four matched PNG/PDF figure pairs. A presentation claim should be traceable
to these tables rather than inferred from a plot.

## Repository map

| Path | Purpose |
|---|---|
| `src/vanguard_portfolio/schemas.py` | Problem, constraints, results, and factor-native matrix operations |
| `src/vanguard_portfolio/portfolio_model.py` | Objective, risk, QP construction, CVaR |
| `src/vanguard_portfolio/allocation.py` | Relaxation, initialization, allocation oracle |
| `src/vanguard_portfolio/window_search.py` | Change windows, enumeration, tabu/LNS |
| `src/vanguard_portfolio/quantum_solver.py` | XY-QAOA, Aer, IBM Runtime, device evidence |
| `src/vanguard_portfolio/classical_discrete.py` | Direct Gurobi MIQP and classical references |
| `src/vanguard_portfolio/validation.py` | Independent hard-constraint checks |
| `src/vanguard_portfolio/presentation.py` | Tables, reports, figures, checksums |
| `scripts/run_hybrid.py` | One configured hybrid experiment |
| `scripts/run_hybrid_scaling.py` | Multi-size, repeated scaling experiment |
| `docs/portfolio_optimization_report.md` | Research-style problem and method report |

## Interpretation limits

- The model is single-period, long-only, and based on estimated inputs.
- Synthetic backtests illustrate behavior; they are not forecasts or financial
  advice.
- Fixed-support QP optimality does not make LNS or QAOA globally optimal.
- A time-limited Gurobi result is reported with its bound and gap, not as an
  unconditional proof.
- Cardinality-preserving quantum samples establish circuit correctness, not
  quantum advantage.
- QPU queue time and all classical preprocessing must be included in an
  end-to-end hardware comparison.

The study methodology, two-thousand-asset reference result, scaling protocol,
and limitations are documented in
[the research report](docs/portfolio_optimization_report.md), with a
[typeset PDF](docs/portfolio_optimization_report.pdf) included for review.
