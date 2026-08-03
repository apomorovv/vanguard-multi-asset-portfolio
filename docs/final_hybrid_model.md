# Hybrid Portfolio Model and Algorithm

This document defines the mathematical model and optimization algorithm implemented by the hybrid benchmark.

Legacy equal-lot solvers remain available as small-instance classical baselines. They are not part of the continuous-weight hybrid optimization path described here.

## 1. Decision Variables

For each asset \(i\):

- \(z_i \in \{0,1\}\) indicates whether asset \(i\) is held.
- \(w_i \ge 0\) is the exact portfolio weight assigned to asset \(i\).
- \(t_i \ge |w_i-w_i^0|\) represents the turnover associated with asset \(i\).

The main benchmark fixes the number of selected holdings:

$$
\sum_i z_i = K.
$$

Continuous portfolio weights are linked to the binary support variables through

$$
m_i z_i \le w_i \le u_i z_i.
$$

Here, \(m_i\) and \(u_i\) are the minimum and maximum permitted weights for asset \(i\).

This is not an equal-weight portfolio model. Two selected assets may receive substantially different portfolio weights.

## 2. Objective Function and Factor Risk

Every final candidate minimizes the canonical portfolio objective

$$
F(w)
=
\lambda_r w^\top \Sigma w
-
\lambda_g \mu^\top w
-
\lambda_y y^\top w
+
\lambda_c c^\top t.
$$

The terms represent:

- portfolio risk, \(w^\top \Sigma w\);
- expected return, \(\mu^\top w\);
- income or yield, \(y^\top w\);
- transaction costs and turnover, \(c^\top t\).

When factor-model data are available, the covariance matrix is represented as

$$
\Sigma = B\Omega B^\top + D,
$$

where:

- \(B\) is the asset-factor exposure matrix;
- \(\Omega\) is the factor covariance matrix;
- \(D\) is the diagonal idiosyncratic-risk matrix.

The portfolio risk can then be evaluated as

$$
w^\top \Sigma w
=
(B^\top w)^\top \Omega (B^\top w)
+
\sum_i D_{ii}w_i^2.
$$

`PortfolioProblem.cov` may be retained for audits on small instances. Large runs can omit both dense covariance and correlation matrices. All required matrix-vector products and support submatrices are reconstructed from the factor representation.

## 3. Hard Constraints

The always-active base constraints include:

- full investment of the available budget;
- long-only portfolio weights;
- asset-level weight bounds;
- group-level weight bounds;
- asset eligibility;
- mandatory holdings;
- binary-to-continuous support linkage;
- independent post-solution validation.

The benchmark normally also enables:

- exact cardinality;
- minimum active weight;
- maximum position size;
- turnover awareness;
- transaction-cost awareness.

Data-dependent optional constraints include the following.

### 3.1 Minimum Expected Return

$$
\mu^\top w \ge R_{\min}.
$$

### 3.2 Minimum Income or Yield

$$
y^\top w \ge Y_{\min}.
$$

### 3.3 Factor-Exposure Bounds

$$
f_{\min} \le B^\top w \le f_{\max}.
$$

### 3.4 Stress-Scenario Requirements

For stress scenario \(s\),

$$
r_s^\top w \ge s_s.
$$

### 3.5 Empirical CVaR Limit

The empirical Conditional Value at Risk constraint is represented by

$$
\eta
+
\frac{1}{(1-\alpha)S}
\sum_{s=1}^{S} u_s
\le C_{\max},
$$

subject to

$$
u_s \ge -r_s^\top w-\eta,
$$

and

$$
u_s \ge 0.
$$

The fixed-support SciPy oracle and the Gurobi MIQP model use this linear epigraph formulation. The validator independently recomputes empirical CVaR from the returned portfolio weights.

## 4. Full-Universe Relaxation

The first optimization step temporarily removes:

- the exact-cardinality constraint;
- the minimum-active-weight constraints.

All remaining convex financial constraints are retained.

The relaxation returns:

- provisional portfolio weights;
- a lower bound on the sparse optimization objective;
- asset-attractiveness information;
- marginal-risk signals;
- binding group constraints;
- binding factor constraints.

The factor-QP variables are

$$
[w,t,f],
$$

with the linear factor-exposure definition

$$
f = B^\top w.
$$

The quadratic objective contains only:

- diagonal idiosyncratic risk from \(D\);
- the relatively small factor covariance block \(\Omega\).

This avoids constructing or factorizing a dense full-universe covariance matrix.

## 5. Guaranteed Valid Initial Portfolio

The initializer protects:

- all mandatory assets;
- at least one strong candidate from every group with a positive minimum-weight requirement.

The remaining portfolio slots are ranked using:

- the continuous relaxation;
- the smooth marginal objective;
- asset-level attractiveness signals.

When this initial support is not valid, a SciPy/HiGHS feasibility MILP jointly determines continuous weights and binary support decisions under:

- exact cardinality;
- support linkage;
- group constraints;
- turnover constraints;
- income constraints;
- factor-exposure constraints;
- stress constraints;
- CVaR constraints.

Each support returned by the feasibility procedure is reoptimized by the continuous allocation oracle.

Randomized weighted trials and safe enumeration of tiny instances remain available as time-limit fallbacks.

The search stops only after a fully valid portfolio containing exactly \(K\) assets has been found. If the feasibility MILP proves that the model is infeasible, the pipeline terminates with an explicit error.

No quantum optimization routine runs before a valid initial portfolio exists.

## 6. Adaptive Change Window

At hybrid iteration \(j\), the current portfolio is divided into:

- selected assets frozen outside the change window;
- weak removable holdings inside the window;
- promising eligible unheld assets inside the window.

Mandatory assets and assets with positive required lower bounds are never presented as removable candidates.

Group slack and optional market-community information influence the ranking of window candidates, but they do not remove assets from the global investment universe.

Suppose \(r\) assets inside the window are currently selected. Every candidate window solution must satisfy

$$
\sum_{i\in W} x_i = r.
$$

The frozen support contains \(K-r\) assets. Therefore, preserving \(r\) selected assets inside the window also preserves the full-portfolio cardinality:

$$
(K-r)+r=K.
$$

## 7. Window Surrogate

Let \(a\) denote the equal proxy notional assigned to each selected window asset, and let \(w_F\) denote the frozen allocation outside the window.

The surrogate allocation is

$$
\widetilde{w}(x)=w_F+aS_Wx,
$$

where \(S_W\) maps the binary window vector \(x\) into the full asset universe.

Expanding the canonical objective produces a quadratic binary model:

$$
x^\top Qx+h^\top x+\text{constant}.
$$

The matrix \(Q\) contains covariance interactions between window assets.

The linear vector \(h\) contains:

- covariance interactions with frozen holdings;
- expected-return contributions;
- income or yield contributions;
- binary-exact proxy transaction-cost changes;
- group-pressure signals.

Strong-interaction sparsification is optional when targeting quantum hardware. Sparsification affects only proposal quality because every proposed support is subsequently evaluated using the complete covariance model and the full continuous allocation oracle.

## 8. Classical Window Search

Tiny fixed-cardinality windows are solved by exact enumeration.

Larger windows use tabu search and large-neighborhood search:

1. Generate \(1\rightarrow0\) and \(0\rightarrow1\) swap neighbors.
2. Rank the neighbors using the QUBO surrogate energy.
3. Send only the most promising supports to the continuous allocation oracle.
4. Cache duplicate supports.
5. Retain the best valid support according to the exact financial objective.
6. Use tabu tenure and fixed-weight randomized restarts to escape local minima.

The surrogate identifies promising combinations, while the allocation oracle determines their exact continuous portfolio weights.

## 9. XY-QAOA

The cost Hamiltonian is obtained by converting the window QUBO into an Ising Hamiltonian.

The XY mixer is

$$
H_M
=
\frac{1}{2}
\sum_{(i,j)\in E}
\left(
X_iX_j+Y_iY_j
\right).
$$

An XY interaction exchanges the computational-basis states \(10\) and \(01\). It therefore preserves physical Hamming weight.

Starting from the current \(r\)-asset window bitstring ensures that every ideal sample continues to contain exactly \(r\) selected window assets.

The parameterized quantum state is

$$
\lvert\psi(\gamma,\beta)\rangle
=
\prod_{l=1}^{p}
e^{-i\beta_lH_M}
e^{-i\gamma_lH_C}
\lvert\psi_0\rangle.
$$

The default configuration uses:

- a shallow circuit;
- a warm start from the current support;
- a ring mixer.

Dicke-state initialization and a complete mixer are retained as explicit ablation configurations.

Cost coefficients are normalized before variational-angle optimization. Multiple COBYLA starts are independently seeded and recorded.

### 9.1 Fixed-Weight Subspace Simulator

The portable subspace simulator stores only

$$
\binom{F}{r}
$$

basis states, where \(F\) is the window size and \(r\) is the required Hamming weight.

This exact fixed-weight CPU simulator is used to optimize the variational angles.

### 9.2 Aer and IBM Runtime Execution

Aer compiles and samples the same logical circuit on either CPU or GPU.

Repeatedly sending the default 16-qubit COBYLA objective evaluations to a GPU would generally introduce more kernel-launch and transfer overhead than useful computation. The GPU backend is therefore intended primarily for larger simulation workloads or explicit backend comparisons.

IBM Runtime sampling requires an explicitly selected backend.

Every execution outside the subspace simulator records:

- requested device;
- actual device;
- phase timings;
- cardinality feasibility;
- circuit depth;
- gate counts;
- additional circuit-resource information.

These values are written to:

- `quantum_execution.csv`;
- `hybrid_diagnostics.json`.

## 10. Allocation Oracle and Validation

For every support proposed by a classical or quantum method, the allocation oracle performs the following steps:

1. Set all upper bounds outside the proposed support to zero.
2. Apply active-position lower bounds to selected assets.
3. Solve the continuous convex financial model.
4. Independently recompute every base and optional constraint.
5. Reject the support when any violation exceeds the configured tolerance.
6. Cache the support and its result.

Portfolio weights are never clipped or renormalized after optimization. Such modifications could invalidate constraints or destroy optimality for the fixed support.

## 11. Exact Reference Model

The final direct Gurobi model includes:

- continuous portfolio weights \(w\);
- binary support variables \(z\);
- turnover epigraph variables \(t\);
- factorized portfolio risk;
- exact cardinality;
- all configured hard constraints;
- the best hybrid portfolio as a MIP start.

A time-limited Gurobi run returns:

- incumbent objective value;
- best objective bound;
- reported optimality gap;
- explored node count;
- model-build time;
- solve time.

Only a solver status proving optimality is described as global certification. Any such claim is reported together with:

- the configured solver tolerance;
- the numerical MIP gap.

Fixed-support QP optimality is recorded separately. It does not upgrade an LNS or QAOA result to global optimality over all possible supports.

## 12. Required Comparisons

The benchmark includes the following comparisons:

- continuous factor-QP lower bound;
- valid initialization;
- exact classical enumeration on tiny windows;
- classical tabu search and large-neighborhood search;
- XY-QAOA using the same windows and allocation oracle;
- standard X-mixer penalty-QAOA;
- Gurobi exact-cardinality MIQP;
- optional equal-lot classical baselines for small instances.

The primary performance comparison assigns equal end-to-end runtime to each method.

A secondary warm-start comparison may give every method the same amount of final Gurobi refinement time.

## 13. Interpretation

The quantum circuit does not:

- calculate final portfolio percentages;
- validate financial constraints;
- replace the continuous allocation solver;
- replace Gurobi as the exact reference method.

Its role is to propose asset combinations within a constraint-preserving neighborhood.

The defensible contribution of the hybrid approach is a constraint-preserving quantum neighborhood search embedded within a scalable, auditable classical optimization and validation system.
