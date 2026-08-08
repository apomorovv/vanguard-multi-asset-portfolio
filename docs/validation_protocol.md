# Validation Protocol

## 1. Purpose

A solver status is not sufficient evidence that a portfolio is valid. Every
candidate is checked by code that is separate from the candidate-generation
logic.

The validator receives the unmodified full-universe weight vector. It does not:

- clip small negative values;
- renormalize the portfolio;
- round weights to satisfy cardinality;
- repair group exposure;
- replace invalid quantum samples with nearby valid samples.

The default hard-feasibility tolerance is $10^{-7}$. The support threshold used
to decide whether an asset is active is separately configurable and defaults to
$10^{-8}$.

A tolerance covers floating-point residuals. It does not authorize post-solve
repair.

## 2. Gate 1: Input Integrity

Before optimization, verify:

- all arrays have consistent dimensions;
- all values are finite;
- asset names are unique;
- group names are unique;
- every asset belongs to an existing group;
- returns, yields, costs, bounds, and current weights use consistent decimal
  units;
- costs and volatilities are nonnegative;
- lower bounds do not exceed upper bounds;
- aggregate asset bounds can satisfy the budget;
- group lower bounds do not exceed group upper bounds;
- correlation is symmetric with a unit diagonal;
- covariance is symmetric;
- covariance equals `corr * outer(sigma, sigma)`;
- covariance is positive semidefinite;
- factor covariance is positive semidefinite;
- idiosyncratic variances are nonnegative;
- the factor model diagonal equals `sigma**2`;
- a supplied factor model reconstructs dense covariance when both are present;
- optional constraint arrays have the required shapes;
- mandatory assets are eligible;
- exact cardinality is compatible with eligible and mandatory sets;
- minimum active weights can fit inside the budget.

Synthetic data generation may deliberately project or construct valid
correlation data before creating `PortfolioProblem`. User-supplied invalid data
raise an exception.

## 3. Gate 2: Objective Equivalence

For a test weight vector $w$, compare the direct objective

$$
F_{\mathrm{direct}}(w) = \lambda_{\mathrm{risk}}w^\top\Sigma w - \lambda_{\mathrm{return}}\mu^\top w -
\lambda_{\mathrm{income}}y^\top w + \lambda_{\mathrm{cost}}c^\top|w-w^0|
$$

with the solver representation.

For OSQP form,

$$
F_{\mathrm{qp}}(x) = \frac12x^\top Px+q^\top x,
$$

where $x=[w,t]$ or $x=[w,t,f]$ and $t=|w-w^0|$.

Verify

$$
F_{\mathrm{direct}}(w)=F_{\mathrm{qp}}(x)
$$

within numerical tolerance.

This test catches:

- a missing factor of two in the quadratic block;
- a wrong return or income sign;
- a missing cost coefficient;
- an incorrect factor-link equation;
- inconsistent dense and factor risk.

## 4. Gate 3: Base Constraint Validation

For every candidate $w$, check:

### 4.1 Budget

$$
\sum_i w_i=B_{\mathrm{budget}}.
$$

### 4.2 Asset Bounds

$$
\ell_i\le w_i\le u_i.
$$

### 4.3 Group Bounds

$$
L_g\le\sum_i A_{gi}w_i\le U_g.
$$

### 4.4 Target Return

When configured,

$$
\mu^\top w\ge R_{\min}.
$$

### 4.5 Turnover

When configured,

$$
\sum_i|w_i-w_i^0|\le T_{\max}.
$$

Each check records:

- constraint name;
- sense;
- left-hand side;
- right-hand side;
- signed slack;
- raw violation;
- pass or fail at the configured tolerance.

## 5. Gate 4: Sparse-Support Validation

When hybrid constraints are present, also check:

### 5.1 Eligibility

Every ineligible asset has zero weight.

### 5.2 Mandatory Holdings

Every mandatory asset has at least the required active weight.

### 5.3 Exact Cardinality

$$
\left| \left[ i : w_{i} \gt \tau_{\mathrm{support}} \right] \right| = K
$$

### 5.4 Minimum Active Weight

For every active asset,

$$
w_i\ge m.
$$

### 5.5 Implementation Maximum Weight

$$
w_i\le\bar u_i.
$$

The base upper bound and implementation cap are both checked.

## 6. Gate 5: Optional Financial Constraints

### 6.1 Minimum Income

$$
y^\top w\ge Y_{\min}.
$$

### 6.2 Factor Bands

$$
f=B^\top w,
$$

$$
f_{\min}\le f\le f_{\max}.
$$

### 6.3 Stress Floors

$$
r_s^\top w\ge q_s.
$$

### 6.4 Empirical CVaR

Compute losses

$$
L_s=-R_s^\top w.
$$

The validator independently evaluates empirical CVaR at level $\alpha$ and
requires

$$
\mathrm{C}\mathrm{V}\mathrm{a}\mathrm{R}_{\alpha}(w) \le C_{\max}
$$

It does not reuse the optimizer's $\eta$ and $u_s$ values.

## 7. Gate 6: Equal-Lot Validation

For a discrete model with $M$ lots,

$$
\delta=\frac{B_{\mathrm{budget}}}{M}.
$$

Each weight must satisfy

$$
\frac{w_i}{\delta}\in\mathbb Z
$$

within numerical tolerance.

The validator also checks the ordinary budget, asset, group, return, and
turnover constraints on the decoded weights.

## 8. Gate 7: Quantum Candidate Validation

A raw bitstring is not a portfolio.

A quantum sample becomes reportable only after:

1. decoding the logical window bits;
2. checking the required Hamming weight;
3. combining selected window assets with the frozen support;
4. solving the exact fixed-support continuous allocation;
5. reconstructing the full-universe weight vector;
6. running every base and optional constraint check.

For ideal subspace XY-QAOA, the cardinality rate should be 100%. For Aer or IBM
hardware, the raw rate may be lower and must be reported.

Postselection may be analyzed, but raw and postselected rates must not be
confused.

## 9. Gate 8: Optimality Relationships

On tractable instances, verify:

1. the continuous relaxation objective is no worse than any exact-$K$ feasible
   objective;
2. the exact equal-lot optimum is no worse than any equal-lot heuristic;
3. exact enumeration equals optimal equal-lot MIQP at the same $M$;
4. independent continuous backends agree within tolerance;
5. exact window enumeration is no worse in QUBO energy than heuristic window
   search;
6. every accepted classical or quantum support passes allocation and validation;
7. a time-limited or heuristic result is never called globally optimal;
8. a fixed-support QP optimum is not promoted to a global support certificate.

For minimization, "no worse" means "less than or equal to."

## 10. Gate 9: Solver Status Interpretation

| Situation | Allowed claim |
|---|---|
| Continuous solver converged and candidate validates | Valid continuous solution; global optimum is supported by convexity and backend evidence. |
| Exact enumeration completed | Globally optimal for that finite enumerated model. |
| Gurobi returns optimal status | Globally optimal within configured tolerances; report numeric MIP gap. |
| Gurobi reaches time limit with incumbent | Valid incumbent; report best bound and gap. |
| LNS or QAOA support passes oracle | Valid heuristic incumbent. |
| Fixed-support QP is optimal | Optimal only for that support. |
| Quantum sample has correct Hamming weight | Valid support encoding only; financial feasibility still unknown. |

## 11. Gate 10: Reproducibility

Every experiment should preserve:

- Git commit identifier;
- branch name;
- exact command;
- resolved YAML configuration;
- all random seeds;
- complete problem data or deterministic generation parameters;
- solver versions;
- Python version;
- operating system and hardware summary;
- requested and actual quantum backend;
- all raw stochastic repetitions;
- unique run identifiers;
- solver diagnostics;
- validation checks;
- artifact checksums.

A reported table should be reproducible without reading values from a figure.

## 12. Gate 11: Timing Integrity

Record end-to-end wall time and, where available:

- data/model preparation;
- continuous relaxation;
- feasible initialization;
- window construction;
- QUBO construction;
- angle optimization;
- circuit setup and transpilation;
- sampling;
- allocation-oracle evaluation;
- validation;
- Gurobi build and solve;
- IBM queue-inclusive wall time;
- IBM reported QPU usage time.

Do not compare a native solver timer against another method's end-to-end time.

## 13. Gate 12: Scaling Safety

- reject exact enumeration before execution when the candidate count exceeds the
  configured guard;
- use a MILP feasible start rather than recursive enumeration for large
  equal-lot instances;
- do not call a finite-pool local search a complete one-swap optimum;
- preserve Gurobi incumbent, best bound, node count, and MIP gap;
- run large sizes in fresh processes when measuring peak memory;
- distinguish globally certified sizes from heuristic scaling sizes;
- use the same validator at every universe size.

## 14. Gate 13: Graphics

Every figure must have:

- a descriptive title;
- labeled axes;
- units;
- readable legends;
- a source table;
- no claim stronger than the underlying solver evidence.

For large universes, plots may aggregate or display the largest positions, but
the complete allocation must remain in the CSV output.

## 15. Required Commands

Run the full test suite:

```bash
python -m pytest -q
```

Run a tiny hybrid correctness case:

```bash
python scripts/run_hybrid.py \
  --config configs/tiny_hybrid.yaml \
  --overwrite
```

Run a portable large case without optional quantum or Gurobi components:

```bash
python scripts/run_hybrid.py \
  --config configs/large_hybrid.yaml \
  --no-quantum \
  --no-gurobi \
  --overwrite
```

Optional solver tests may skip only when a dependency or license is clearly
reported as unavailable. A backend that runs and returns an incorrect objective
or invalid portfolio must fail.
