# Data policy

- `data/synthetic/` contains deterministic, non-confidential test data.
- `data/raw/` is for immutable source data and is ignored by Git.
- `data/processed/` is for reproducible derived data and is ignored by Git.

All solver comparisons must read the same serialized `PortfolioProblem` instance.
Do not commit proprietary or personally identifiable portfolio data.


