# Hybrid portfolio optimization report

- Best valid incumbent: **Classical tabu/LNS**
- Objective: `-0.03762016669`
- Expected return: `6.791%`
- Volatility: `9.090%`
- L1 turnover: `40.000%` (one-way convention: `20.000%`)
- Selected assets: `25`
- Hard-constraint breaches: **0**
- Total runtime: `41.097 s`
- Allocation-oracle calls/cache hits: `10/1`

## Interpretation

The continuous relaxation supplies a lower bound and candidate scores. Classical LNS and XY-QAOA search the same fixed-cardinality windows. Every sampled support is reallocated with the complete continuous financial model and independently validated. Quantum output is therefore a proposal, never an unverified final portfolio.

XY-QAOA angles are optimized by the exact fixed-Hamming-weight CPU subspace simulator. When selected, Aer GPU or IBM Runtime executes and samples the corresponding Qiskit circuit; the portable subspace backend samples on CPU. This split is intentional for small change windows and is reported explicitly in `quantum_execution.csv`.

A hybrid result is globally heuristic even when its fixed-support allocation QP is solved optimally. Lower objective values are better.

## Quantum results

- Window 1, XY-QAOA (Aer GPU sample): objective `-0.03628184002`, runtime `0.720 s`, breaches `0`. Sampler device `GPU` (result_metadata); cardinality-feasible shots `100.00%`.
