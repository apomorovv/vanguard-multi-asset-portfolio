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

Hybrid sparse candidates additionally check exact cardinality, minimum active
weight, eligibility, mandatory holdings, implementation-specific weight caps,
income/factor/stress rules, and empirical CVaR when configured. The validator
receives the full-universe weight vector even though the allocation QP is
solved in support-reduced coordinates.

Raw quantum bitstrings are not portfolios. They become reportable only after:

1. fixed-Hamming-weight decoding;
2. full-support reconstruction with frozen holdings;
3. exact continuous allocation;
4. independent full-universe validation.

## Gate 4: optimality relationships

For each preference configuration:

1. Continuous optimum \(\le\) exact discrete optimum.
2. Exact discrete optimum \(\le\) every feasible heuristic objective.
3. Enumeration = optimal MIQP at identical \(M\), within tolerance.
4. Independent continuous backends agree within numerical tolerance.
5. The continuous relaxation is no worse than any exact-`K` feasible result.
6. XY-QAOA ideal samples have the required window Hamming weight.
7. A heuristic or time-limited result is never labeled globally optimal.

## Gate 5: reproducibility

- synthetic inputs are deterministic;
- random universes and heuristics store their seeds;
- the configuration file is saved with the experiment;
- all stochastic raw runs are retained;
- runtime comparisons identify the machine/environment separately when used in
  a paper or presentation.
- each run has a unique `run_id` shared by run, allocation, constraint, and
  diagnostic artifacts;
- exact solver options and repetitions are retained in `resolved_config.yaml`;
- exact problem bytes are represented by a SHA-256 fingerprint;
- every generated artifact is listed with its size and SHA-256 checksum.

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

For large universes, allocation plots show the 30 largest assets plus an
aggregate remainder, correlation tick labels are hidden above 60 assets, and
constraint plots show the 50 most binding/violated limits. No information is
discarded from `allocation_weights.csv` or `constraint_checks.csv`.

## Gate 7: large-instance safety

- enumeration must remain below its configured candidate-count guard;
- large heuristic starts must be found and validated without recursive
  enumeration;
- finite candidate-pool local search must not be described as a complete
  one-swap optimum;
- time-limited MIQP runs must preserve incumbent, best bound, and gap when
  available;
- all accepted large-instance results must pass the same unmodified-weight
  validation used for tiny tests.

## Test commands

```bash
python -m pytest -q
PYTHONPATH=src python -m unittest discover -s tests -v
```

Optional solver tests may skip only for a clearly reported missing package or
license. A backend that runs and produces the wrong objective must fail.
