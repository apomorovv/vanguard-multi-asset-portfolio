# Classical baseline benchmark

All objective values use the same minimization convention. A zero breach count is required.
Wall-clock runtime includes Python model construction and solver execution.

| model | method | runs | feasible | best objective | reference/bound gap | median runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| continuous | cvxpy_clarabel | 3 | 100% | -0.09852504 | 0.000e+00 | 0.048098 |
| continuous | gurobi_qp | 3 | 100% | -0.09852503 | 1.007e-09 | 0.040367 |
| continuous | osqp | 3 | 100% | -0.09852503 | 9.799e-10 | 0.033732 |
| continuous | scipy_slsqp | 1 | 100% | -0.09852502 | 1.594e-08 | 16.967903 |
| discrete | cvxpy_scip_miqp | 1 | 100% | -0.09852490 | 4.732e-08 | 0.698560 |
| discrete | gurobi_miqp | 1 | 100% | -0.09852495 | 0.000e+00 | 0.055679 |
| discrete | simulated_annealing_swap | 10 | 100% | -0.09852466 | 2.878e-07 | 3.994635 |
| discrete | swap_local_search | 1 | 100% | -0.09852466 | 2.878e-07 | 3.982114 |

## Interpretation

- The continuous optimum is a relaxation and can be better than the discrete optimum.
- Enumeration and an optimal MIQP solve should agree at the same lot resolution.
- Heuristic gaps are measured against an exact/optimal reference, never against another heuristic.
- A missing commercial license is recorded as skipped; it is never presented as a failed model.

## Auditable files

- `allocation_weights.csv` contains every numeric asset weight and discrete lot count.
- `constraint_checks.csv` contains every independently recomputed hard-constraint check.
- `solver_diagnostics.json` preserves native iterations, bounds, nodes, gaps, and timing phases.
- `problem.json` and `resolved_config.yaml` reconstruct the exact run inputs.
