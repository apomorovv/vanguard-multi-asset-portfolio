# Result packages

Generated result folders are intentionally excluded from version control. This
file documents how to preserve and interpret a reproducible experiment.

## Single hybrid run

`scripts/run_hybrid.py` writes one directory such as
`results/large_hybrid/`. Keep the directory intact because every figure is
backed by tables and metadata.

| File | Contents |
|---|---|
| `hybrid_summary.csv` | One row per relaxation, initial, classical, quantum, and Gurobi result |
| `allocation_weights.csv` | Exact weights, trades, groups, and selected assets |
| `constraint_checks.csv` | Every independently recomputed hard constraint |
| `change_windows.csv` | Weak holdings and proposed replacements in each window |
| `quantum_execution.csv` | Requested/actual backend, device evidence, circuit resources, phase timing |
| `objective_timeline.csv` | Best valid objective versus elapsed time |
| `backtest_summary.csv` | Synthetic return, volatility, CVaR, drawdown, and wealth statistics |
| `hybrid_diagnostics.json` | Configuration, solver metadata, skips, and quantum diagnostics |
| `problem.json` | Complete reproducible problem instance; dense matrices may be omitted for factor-native runs |
| `hybrid_report.md` | Short interpretation of the run |
| `artifact_manifest.json` | File sizes and SHA-256 checksums |
| `plots/` | Matched PNG and PDF figures |

Acceptance requires all of the following:

- `success = True`;
- `feasible = True`;
- `breaches = 0`;
- exact support size equal to the configured cardinality;
- no silently skipped component represented as having run.

The six-row `key_guardrails` figure is intended for the main presentation.
`constraint_slacks` is the complete appendix view. The latter can be dense for
large portfolios and should not replace the underlying CSV.

## Scaling study

`scripts/run_hybrid_scaling.py` writes:

| File | Contents |
|---|---|
| `scaling_runs.csv` | One auditable row per size, repetition, and seed |
| `scaling_methods.csv` | Method-level result rows from every completed case |
| `scaling_summary.csv` | Per-size medians and interquartile ranges |
| `scaling_config.json` | Resolved command-line study configuration |
| `scaling_environment.json` | Python, platform, CPU count, package versions, timestamp |
| `scaling_manifest.json` | SHA-256 checksums and byte counts |
| `scaling_runtime.*` | End-to-end runtime and component composition |
| `scaling_quality_and_feasibility.*` | Objective gaps and zero-breach rate |
| `scaling_memory_and_first_valid.*` | Peak memory, dense storage avoided, time to validity |
| `scaling_quantum.*` | Fixed-window phase timing and cardinality rate |

The default protocol uses fresh worker processes and three seeded repetitions.
Gurobi certification is attempted only for configured sizes at or below 2,000
assets. Rows above that threshold are scalable-search measurements and must not
be called globally certified.

## Interpreting the numbers

Lower objective values are better. The objective combines risk, return,
income, and trading cost and is not itself a percentage return.

The continuous factor-QP objective is a lower bound for the corresponding
sparse minimization problem. Gurobi contributes an incumbent and a best bound;
its reported MIP gap is the appropriate certification measure. A classical or
quantum-guided support remains heuristic even when its conditional allocation
QP is solved optimally.

XY-QAOA output is a set of candidate supports, not portfolio weights. A quantum
row in `hybrid_summary.csv` has already passed through the continuous
allocation oracle and validator. The raw shot-cardinality rate and actual
execution device are recorded separately in `quantum_execution.csv`.

Backtests use generated out-of-sample paths. They are useful for checking
coherent risk behavior but do not establish future performance. Report the
seed, number of paths, horizon, and uncertainty whenever making a statistical
claim.

## Preserving a presentation result

Before rerunning an experiment, copy the entire result folder to a uniquely
named archive outside the overwrite target. Preserve together:

1. the Git commit identifier;
2. the exact YAML configuration and command;
3. all raw CSV/JSON files;
4. the environment manifest;
5. plots cited in the presentation or report;
6. the artifact manifest and checksums.

Never keep a plot without its source tables and run metadata.
