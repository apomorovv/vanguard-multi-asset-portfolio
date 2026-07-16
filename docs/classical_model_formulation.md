# Classical Portfolio Solution — Implementation Guide

This document explains the *classical* portion of the multi-asset portfolio
project. It is the reference implementation that the QUBO and quantum solvers
will later be validated against, and it directly implements Sections 5–8 and 11
of [mathematical_model.md](mathematical_model.md).

---

## 1. What was implemented

| Component | File | Model reference |
|-----------|------|-----------------|
| Synthetic asset universe | [src/data.py](../src/data.py) | Sections 2–3 |
| Portfolio metrics + constraint checks | [src/metrics.py](../src/metrics.py) | Sections 5, 10, 11 |
| Continuous mean-variance optimizer | [src/classical.py](../src/classical.py) | Sections 6–7 |
| Exact discrete optimizer | [src/classical.py](../src/classical.py) | Section 8 |
| Results / demo script | [scripts/run_classical.py](../scripts/run_classical.py) | Section 12 |
| Tests | [tests/test_metrics.py](../tests/test_metrics.py), [tests/test_classical.py](../tests/test_classical.py) | — |

This closes the gap identified earlier: the mathematical model was fully
specified, but no classical optimizer existed in src/.

---

## 2. The problem being solved

Given expected returns $\mu$, a covariance matrix $\Sigma$, income yields $y$,
transaction-cost coefficients $c$, and the current allocation $w^{(0)}$, choose
target weights $w$ that minimise

$$
\lambda_{\text{risk}}\, w^{\mathsf T}\Sigma w
-\lambda_{\text{return}}\, \mu^{\mathsf T} w
-\lambda_{\text{income}}\, y^{\mathsf T} w
+\lambda_{\text{cost}}\, \sum_i c_i\,|w_i - w_i^{(0)}|
$$

subject to the *hard constraints*:

- *Full investment:* $\sum_i w_i = 1$
- *Long-only / per-asset bounds:* $l_i \le w_i \le u_i$
- *Group exposure limits:* $L_g \le \sum_i a_{gi} w_i \le U_g$

A solution that violates any hard constraint is infeasible regardless of its
objective value (Section 10). This matches the challenge rule that the winning
solution must have *zero hard-constraint breaches*.

---

## 3. Synthetic data (src/data.py)

generate_synthetic_universe() builds a deterministic six-asset universe (US
equity, international equity, government bonds, corporate bonds, commodities,
cash) grouped into equity, fixed income, alternatives and cash. All values are
illustrative and contain no real or confidential data, satisfying the
challenge's data-privacy rule.

Key points:

- The covariance matrix is built as $\Sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$
  and checked for positive semidefiniteness; if needed, _nearest_psd clips
  negative eigenvalues. A PSD covariance keeps the continuous problem convex.
- PortfolioProblem is a dataclass holding every parameter. The group
  membership matrix $A$ (shape $G\times n$) is derived from asset_group.
- save_problem / load_problem serialise the universe to
  data/synthetic_universe.json so every solver reads identical inputs.

---

## 4. Metrics and constraint checking (src/metrics.py)

All solvers are scored with the *same* functions so comparisons are fair:

- expected_return, variance, volatility, income, turnover,
  transaction_cost — the quantities from Sections 5 and 11.
- constraint_report(w, problem) returns a ConstraintReport with:
  - feasible — whether every hard constraint holds,
  - breaches — number of violated constraints,
  - max_violation — the largest single violation,
  - details — human-readable descriptions (used by the co-pilot rationale).

portfolio_metrics(w, problem) bundles all six metrics into a dictionary.

---

## 5. Continuous optimizer (src/classical.py)

solve_continuous(problem, prefs) solves the convex quadratic program of
Section 6 using SciPy's SLSQP method. To keep the transaction-cost term linear,
the decision vector is augmented as $z = [w, t]$ where $t_i$ models
$|w_i - w_i^{(0)}|$ through the two inequalities
$t_i \ge w_i - w_i^{(0)}$ and $t_i \ge w_i^{(0)} - w_i$ (Section 6). The budget
is an equality constraint; bounds and group limits are inequality constraints.

The returned SolveResult carries the weights, objective value, runtime,
feasibility flag, breach count, maximum violation, and the full metric set. This
continuous optimum is the *ground-truth reference* for the whole project and,
being a relaxation, is a lower bound on the discrete objective.

### Tunable investor goals

Preferences holds the four $\lambda$ coefficients. PRESETS exposes five
goals (Deliverable 5):

| Preset | Emphasis |
|--------|----------|
| balanced | Neutral risk/return trade-off |
| growth | Higher return weight, lower risk aversion |
| income | Adds income-yield preference |
| drawdown_control | Strong risk aversion (lower volatility) |
| cost_sensitive | Heavy penalty on turnover / transaction cost |

---

## 6. Discrete optimizer (src/classical.py)

solve_discrete(problem, prefs, units=M) implements Section 8: weights are
restricted to multiples of $1/M$, i.e. $w_i = q_i/M$ with integer $q_i$ and
$\sum_i q_i = M$. It enumerates *every* feasible integer allocation
(_integer_allocations prunes on per-asset bounds and remaining budget), rejects
those breaching group limits, and returns the one with the lowest objective.

Because it is exhaustive, its result is the *exact discrete optimum* — the
reference the QUBO/quantum solvers must reproduce. At $M=10$ with six assets the
search space is only a few thousand allocations, so it runs instantly.

---

## 7. How to run and see results

Install dependencies (note the file is named requiremnts.txt):

powershell
python -m pip install -r requiremnts.txt


Run the demo:

powershell
python scripts/run_classical.py


The script:

1. Builds and saves the synthetic universe to data/synthetic_universe.json.
2. Prints the recommended allocation for the continuous and discrete solvers
   (with a text bar chart) plus the L1 distance between them.
3. Prints a *solver-comparison table* across all five presets showing
   objective, expected return, volatility, income, turnover, cost, breaches and
   runtime.
4. Saves a bar chart of the balanced allocation vs. the current portfolio to
   results/allocation_balanced.png.

---

## 8. Tests

Run the suite from the repository root:

powershell
python -m pytest -q


Coverage highlights:

- *Data:* covariance is PSD and matches $\rho\sigma\sigma$; group matrix is
  well-formed; the current portfolio is feasible; save/load round-trips.
- *Metrics:* each formula matches its definition; turnover/cost are zero at the
  current allocation; the constraint report flags budget, bound and group
  breaches.
- *Optimizers:* continuous and discrete solutions are feasible with zero
  breaches; discrete weights are multiples of $1/M$; the discrete objective is
  never better than the continuous relaxation; higher risk aversion lowers
  volatility; the income preset raises income; the cost-sensitive preset lowers
  turnover; every preset stays feasible.

---

## 9. Where this fits in the roadmap

This classical layer delivers Deliverables 3–7 (synthetic data, baseline
mean-variance optimizer with constraints and scenario weights, tunable goals,
comparison metrics, and classical validation). It provides the exact discrete
optimum and metric harness that the QUBO encoding and quantum solver will be
checked against next.