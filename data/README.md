# Data policy

- `data/synthetic/` contains deterministic, non-confidential test data.
- `data/raw/` is for immutable source data and is ignored by Git.
- `data/processed/` is for reproducible derived data and is ignored by Git.

All solver comparisons must read the same serialized `PortfolioProblem` instance.
Do not commit proprietary or personally identifiable portfolio data.

Every benchmark output directory now contains its own immutable-by-convention
`problem.json` plus a SHA-256 problem fingerprint in
`benchmark_metadata.json`. The standard runner only refreshes
`data/synthetic/synthetic_universe.json` for `source: synthetic`; loading a
custom or factor-generated problem no longer overwrites that demonstration
file.

Large reproducible test universes use `problem.source: factor` with explicit
`n_assets`, `n_groups`, `n_factors`, and `seed` fields. Factor covariance is PSD
by construction and does not require a cubic nearest-correlation projection.
