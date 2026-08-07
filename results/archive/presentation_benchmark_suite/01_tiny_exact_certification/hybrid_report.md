# Hybrid portfolio optimization report

- Best valid incumbent: **Classical exact window = Gurobi MIQP = Penalty QAOA = Valid initial**
- Objective: `-0.03681145`
- Expected return: `4.350%`
- Volatility: `6.540%`
- L1 turnover: `70.000%` (one-way convention: `35.000%`)
- Selected assets: `4`
- Hard-constraint breaches: **0**
- Total runtime: `0.133 s`
- Allocation-oracle calls/cache hits: `15/17`

## Interpretation

The continuous relaxation supplies a lower bound and candidate scores. Classical LNS and XY-QAOA search the same fixed-cardinality windows. Every sampled support is reallocated with the complete continuous financial model and independently validated. Quantum output is therefore a proposal, never an unverified final portfolio.

XY-QAOA angles are optimized by the exact fixed-Hamming-weight CPU subspace simulator. When selected, Aer GPU or IBM Runtime executes and samples the corresponding Qiskit circuit; the portable subspace backend samples on CPU. This split is intentional for small change windows and is reported explicitly in `quantum_execution.csv`.

A hybrid result is globally heuristic even when its fixed-support allocation QP is solved optimally. Lower objective values are better.

## Quantum results

- Window 1, Penalty QAOA: objective `-0.03681145`, runtime `0.005 s`, breaches `0`. Sampler device `CPU` (not_applicable); cardinality-feasible shots `100.00%`.

## Classical certification

- Gurobi incumbent: `-0.03681145`
- Certified lower bound: `-0.03681145`
- Reported MIP gap: `0.0000%`
- Status: `optimal` (optimal within the configured MIP tolerance).

## Explicitly skipped components

- `iteration_0:xy_qaoa`: xy_qaoa_subspace produced no support accepted by the allocation oracle
