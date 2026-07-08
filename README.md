# Vanguard Multi-Asset Portfolio Optimization

This project compares classical and quantum approaches to constrained
multi-asset portfolio construction.

## Package structure

- `data_generation.py`: Generate portfolio problem instances.
- `portfolio_model.py`: Evaluate return, risk, turnover, cost and constraints.
- `classical_continuous.py`: Continuous mean-variance baseline.
- `classical_discrete.py`: Exact discrete baseline.
- `qubo_builder.py`: Convert the discrete model to QUBO.
- `quantum_solver.py`: Solve the QUBO.
- `validation.py`: Independently validate every returned portfolio.
- `metrics.py`: Compare solutions.
- `copilot_app.py`: Interactive user interface.

## Installation

```bash
conda create -n vanguard-portfolio python=3.11 -y
conda activate vanguard-portfolio
pip install -e ".[dev,app]"
