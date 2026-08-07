# Results guide

The repository keeps two kinds of result material:

1. generated experiment directories created by the command-line scripts; and
2. the curated, versioned final evidence package in
   [`final_submission/`](final_submission/).

Start with [`final_submission/README.md`](final_submission/README.md), then use
[`final_submission/claim_evidence_map.csv`](final_submission/claim_evidence_map.csv)
to trace a report or presentation claim to its raw table and qualification.

## Standard hybrid package

`scripts/run_hybrid.py` writes files such as:

| File | Meaning |
|---|---|
| `hybrid_summary.csv` | One row per guide, initialization, classical, quantum, and exact method |
| `allocation_weights.csv` | Exact weights, trades, groups, and selected assets |
| `constraint_checks.csv` | Independently recomputed rule checks |
| `quantum_execution.csv` | Actual device, circuit resources, shots, cardinality, and phase times |
| `objective_timeline.csv` | Best valid objective versus elapsed time |
| `backtest_summary.csv` | Synthetic held-out robustness metrics |
| `hybrid_diagnostics.json` | Configuration, status, fallback, and detailed diagnostics |
| `artifact_manifest.json` | File sizes and SHA-256 checksums |

Accept a reported portfolio only when it is successful, feasible, has the
configured support size, has zero breaches, and does not describe a skipped
component as having run.

## Scaling package

`scripts/run_hybrid_scaling.py` writes per-run, per-method, and per-size tables,
plus resolved configuration, environment, manifest, and presentation figures.
The important fields include time to first validity, full time, peak memory,
factor storage, dense storage avoided, guide status, fallback status, bound
availability, breach count, execution device, and quantum cardinality rate.

Blank gap fields are meaningful. They indicate that a guide or exact solve did
not supply a valid bound; do not fill or interpolate them.

## Evidence levels

- **Certified:** matching exact incumbent and bound, or exhaustive enumeration.
- **Bounded:** valid sparse result with a solved continuous lower bound.
- **Heuristic:** valid result without a solved quality bound.
- **Hardware observation:** measured QPU behavior; not proof of advantage.

Lower objective values are better, but the objective is not a percentage
return. Backtests use synthetic paths and are not forecasts.
