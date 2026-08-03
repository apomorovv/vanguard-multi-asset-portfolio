# Classical Models and Solver Baselines

## 1. Purpose

The repository uses several classical models for different roles:

1. solve the continuous full-universe relaxation;
2. allocate exact weights on a fixed support;
3. provide tiny exact equal-lot references;
4. provide scalable heuristic controls;
5. provide an exact-cardinality Gurobi incumbent, bound, and possible global
   certificate.

These models must use the same canonical objective and independent validator.

## 2. Continuous Convex Model

The continuous model minimizes

$$
F(w) = \lambda_{\mathrm{risk}}w^\top\Sigma w - \lambda_{\mathrm{return}}\mu^\top w - \lambda_{\mathrm{income}}y^\top w + \lambda_{\mathrm{cost}}c^\top t
$$

subject to budget, base asset bounds, group limits, turnover epigraph
constraints, and optional return limits.

For the base model, define

$$
x=[w,t].
$$

For factor-native risk, define

$$
x=[w,t,f],
\qquad
f=B^\top w.
$$

## 3. OSQP Matrix Form

OSQP solves

$$
\min_x \frac{1}{2}x^\top Px+q^\top x
$$

subject to

$$
\ell_{\mathrm{qp}}\le Ax\le u_{\mathrm{qp}}.
$$

Because the financial objective has no hidden factor of $1/2$, the risk block is

$$
P_{ww}=2\lambda_{\mathrm{risk}}\Sigma
$$

for a dense covariance model.

For the factor-native formulation,

$$
P
=
\operatorname{blockdiag}
\left(
2\lambda_{\mathrm{risk}}D,
0,
2\lambda_{\mathrm{risk}}\Omega
\right).
$$

The linear vector is

$$
q
=
\begin{bmatrix}
-\lambda_{\mathrm{return}}\mu-\lambda_{\mathrm{income}}y\\
\lambda_{\mathrm{cost}}c\\
0
\end{bmatrix}.
$$

The constraint matrix contains rows for:

- asset bounds;
- nonnegative turnover variables;
- budget equality;
- group bounds;
- absolute-value epigraph;
- target return;
- turnover cap;
- factor definition.

The direct objective evaluator remains the source of truth. The matrix form is
tested against it to catch sign and factor-of-two errors.

## 4. Continuous Backends

| Backend | Role | Evidence |
|---|---|---|
| SciPy SLSQP | Always-available nonlinear constrained solver and fixed-support fallback. | Native convergence plus independent validation. |
| OSQP | Specialized sparse convex QP solver. | Solver status, residuals, and independent validation. |
| CVXPY with CLARABEL or another installed solver | Independent modeling stack. | Backend status and independent validation. |
| Gurobi QP | Commercial continuous cross-check. | Optimal status and independent validation. |

A convex mathematical model has a global optimum, but a generic numerical
solver status is not accepted without checking the returned weights.

## 5. Continuous Relaxation Versus Fixed-Support Allocation

### 5.1 Full-Universe Relaxation

The relaxation removes exact cardinality and minimum-active-weight rules. It
provides a lower bound on the sparse objective.

### 5.2 Fixed-Support Allocation

Given support $\mathcal H$, set

$$
w_i=0
\qquad
\forall i\notin\mathcal H.
$$

For selected assets, use

$$
w_i\ge\max(\ell_i,m)
$$

and the effective upper cap.

The optimizer solves the reduced continuous problem and reconstructs a full
$n$-asset vector for validation.

A fixed-support optimum is conditional on $\mathcal H$ and is not a certificate
over all supports.

## 6. Equal-Lot Discrete Baseline

The legacy discrete baseline divides the budget into $M$ equal lots:

$$
\delta=\frac{B_{\mathrm{budget}}}{M},
\qquad
w_i=\delta q_i,
\qquad
q_i\in\mathbb Z_{\ge0},
$$

$$
\sum_iq_i=M.
$$

The integer bounds are

$$
q_i^{\min}
=
\left\lceil\frac{\ell_i}{\delta}\right\rceil,
\qquad
q_i^{\max}
=
\left\lfloor\frac{u_i}{\delta}\right\rfloor.
$$

Lower bounds must round upward and upper bounds downward. Reversing these
directions would admit portfolios that violate the original continuous bounds.

Group, target-return, and turnover rules are applied to the decoded weights
$w=\delta q$.

## 7. Exact Enumeration

Without bounds, the number of nonnegative lot allocations is

$$
\binom{M+n-1}{n-1}.
$$

The implementation first uses an $O(nM)$ dynamic program to count allocations
that satisfy asset lot bounds. If the count exceeds `max_candidates`, exact
enumeration is rejected before recursion starts.

Enumeration is a truth oracle for tiny cases only. It proves the optimum of the
equal-lot model at that particular $M$.

## 8. Feasible Starts for Large Equal-Lot Searches

Large heuristics do not recursively enumerate until they find a starting point.
A SciPy/HiGHS feasibility MILP searches for integer lots satisfying:

- total lot budget;
- integer asset bounds;
- group bounds;
- turnover epigraph;
- target return;
- turnover cap.

The returned start is independently checked before use.

## 9. Swap Local Search and Simulated Annealing

A one-lot move transfers one lot from a donor asset to a receiver asset. The
total budget remains invariant.

The implementation caches

$$
\Sigma w
$$

so the exact objective change for a proposed swap can be evaluated in constant
time after reading the relevant cached entries and covariance pair. An accepted
move updates the cache in $O(n)$ time.

A finite candidate pool is a heuristic speed control. It must not be described
as a complete one-swap optimality certificate.

Simulated annealing uses seeded stochastic acceptance to escape local minima,
then applies a deterministic swap polish. All seeds and raw repetitions must be
reported.

## 10. Direct Exact-Cardinality Gurobi MIQP

The final sparse reference is not the equal-lot model. It uses continuous
weights and binary support variables:

$$
m_i z_i\le w_i\le \hat u_i z_i,
$$

$$
\sum_i z_i=K.
$$

The model also includes all configured financial guardrails and factorized risk.

Gurobi may return:

- a globally optimal solution;
- a feasible time-limited incumbent;
- a best bound;
- a numerical MIP gap;
- no incumbent;
- an infeasibility certificate.

Only the first case is described as a global optimum. A small reported gap is
strong evidence, but the actual status, bound, tolerance, and gap must still be
shown.

## 11. Backend Comparison Matrix

| Method | Variables | Main use | Global certificate |
|---|---|---|---|
| SciPy SLSQP | Continuous | Portable continuous solve | No formal dual bound |
| OSQP | Continuous QP | Sparse factor-QP | Solver status and residuals |
| CVXPY backend | Continuous or MIQP | Independent formulation | Depends on backend |
| Exact enumeration | Integer equal lots | Tiny truth oracle | Yes, for enumerated model |
| Swap local search | Integer equal lots | Deterministic heuristic | No |
| Simulated annealing | Integer equal lots | Stochastic heuristic | No |
| Window enumeration | Binary fixed-weight window | Tiny local QUBO truth | Only for that window surrogate |
| Tabu/LNS | Binary support changes | Scalable hybrid control | No |
| Gurobi exact-cardinality MIQP | Continuous weights plus binary support | Final classical reference | Yes only with optimal status |

## 12. Fair Comparison Rules

Every method being compared must receive:

- the same serialized problem data;
- the same preference coefficients;
- the same enabled hard constraints;
- the same numerical validation tolerance;
- the same seed policy;
- the same timing boundary.

For equal-lot comparisons, methods must also use the same $M$.

Continuous and discrete gaps require different references. Comparing an
equal-lot heuristic directly with the continuous optimum mixes search error
with unavoidable discretization error.

For hybrid windows, classical and quantum methods must receive the same window,
required Hamming weight, QUBO, allocation oracle, and candidate budget.

## 13. Timing

Common wall-clock timing begins before model construction and ends after solver
execution and required postprocessing.

When available, record separately:

- model-build time;
- feasible-start time;
- solver-setup time;
- solve or optimization time;
- sampling time;
- allocation-oracle time;
- validation time;
- native solver runtime.

Do not replace end-to-end time with a favorable internal solver timer.

## 14. Required Cross-Checks

On tractable instances:

1. exact enumeration objective equals optimal equal-lot MIQP objective;
2. no feasible heuristic beats the exact optimum in a minimization problem;
3. the continuous optimum is no worse than the equal-lot optimum;
4. independent continuous backends agree within tolerance;
5. every reported portfolio has zero independent breaches;
6. direct objective evaluation equals solver-matrix evaluation;
7. a time-limited incumbent is not labeled optimal.

A failure should stop the quantum comparison until the mismatch is resolved.

## 15. Auditable Output

A complete benchmark should preserve:

- exact weights and lots;
- objective terms and financial metrics;
- constraint checks;
- solver diagnostics;
- problem data or problem fingerprint;
- resolved configuration;
- package and platform versions;
- random seeds;
- file sizes and SHA-256 hashes.

A plot is never a substitute for the underlying numerical output.
