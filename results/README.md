# Generated results

Run:

```bash
python scripts/run_classical.py --config configs/baseline.yaml
```

The runner creates:

- `benchmark_runs.csv`: one row per solver/seed.
- `benchmark_summary.csv`: median runtime, feasibility, and reference gaps.
- `benchmark_metadata.json`: objective settings and skipped optional solvers.
- `test_report.txt`: the most recent full unit-test log.
- `classical_baseline_report.md`: human-readable benchmark summary.
- `allocation_comparison.png`: current versus solver allocations.
- `risk_return.png`: expected return versus volatility.
- `runtime_comparison.png`: wall-clock comparison with stochastic IQR bars.
- `objective_gap.png`: gap to an optimal/exact reference within each model class.
- `correlation_heatmap.png`: synthetic-universe correlation matrix.
- `constraint_slacks.png`: distance to each hard-constraint boundary.
- `risk_aversion_sweep.png`: continuous mean-variance trade-off.

Generated result files are ignored by Git because timing is machine-dependent.
Commit only selected figures or tables that are cited in a report.

