# Validation Protocol

## Gate 1: input integrity

Before optimization:

- all arrays have consistent dimensions and finite values;
- asset and group names are unique;
- each asset belongs to exactly one declared group;
- correlation is symmetric with unit diagonal;
- covariance equals `corr * outer(sigma, sigma)`;
- covariance is PSD;
- transaction costs are nonnegative;
- lower bounds do not exceed upper bounds;
- asset bounds can satisfy the budget;
- the synthetic current portfolio is feasible.

Invalid input raises an exception instead of being silently repaired. Synthetic
generation may explicitly construct/project a valid PSD correlation before it
creates a `PortfolioProblem`.

## Gate 2: algebraic equivalence

For a feasible test vector \(w\), verify:

\[
F_{\mathrm{direct}}(w)
=\tfrac12x^TPx+q^Tx,
\qquad x=[w,|w-w^{(0)}|].
\]

This catches the most common factor-of-two and sign errors.

## Gate 3: independent feasibility

Every decoded candidate is checked without clipping or renormalization. The
validator reports one check per constraint, signed slack, breach count, and the
largest raw violation.

The default feasibility tolerance is \(10^{-7}\). A tolerance excuses numerical
roundoff; it does not authorize post-solve repair.

Discrete candidates additionally satisfy:

\[
\frac{w_i}{B/M}\in\mathbb Z\quad\forall i.
\]

## Gate 4: optimality relationships

For each preference configuration:

1. Continuous optimum \(\le\) exact discrete optimum.
2. Exact discrete optimum \(\le\) every feasible heuristic objective.
3. Enumeration = optimal MIQP at identical \(M\), within tolerance.
4. Independent continuous backends agree within numerical tolerance.

## Gate 5: reproducibility

- synthetic inputs are deterministic;
- random universes and heuristics store their seeds;
- the configuration file is saved with the experiment;
- all stochastic raw runs are retained;
- runtime comparisons identify the machine/environment separately when used in
  a paper or presentation.

## Gate 6: graphics

Each benchmark run must produce and visually inspect:

- allocation comparison;
- risk-return scatter;
- runtime comparison;
- optimality-gap comparison;
- correlation heatmap;
- hard-constraint slack plot;
- risk-aversion sweep when enabled.

Plots are explanatory outputs. CSV/JSON results remain the auditable source.

## Test commands

```bash
python -m pytest -q
PYTHONPATH=src python -m unittest discover -s tests -v
```

Optional solver tests may skip only for a clearly reported missing package or
license. A backend that runs and produces the wrong objective must fail.


