# Final hybrid portfolio model and algorithm

This document is normative for the final production path. The earlier
equal-lot, QAOA/PCE, VQE, and VQE/PCE documents describe retained research
baselines.

## 1. Decisions

For each asset `i`:

- `z_i in {0,1}` says whether the asset is held;
- `w_i >= 0` is its exact portfolio percentage;
- `t_i >= |w_i-w_i^0|` represents trading turnover.

The main benchmark fixes the number of holdings:

\[
\sum_i z_i=K.
\]

The continuous percentages are linked to support with

\[
m_i z_i\le w_i\le u_i z_i.
\]

This is not an equal-weight model: two selected assets can receive very
different percentages.

## 2. Objective and factor risk

All final candidates minimize

\[
F(w)=\lambda_r w^T\Sigma w-\lambda_g\mu^Tw
-\lambda_y y^Tw+\lambda_c c^Tt.
\]

When factor data are available,

\[
\Sigma=B\Omega B^T+D
\]

and

\[
w^T\Sigma w=(B^Tw)^T\Omega(B^Tw)+\sum_iD_{ii}w_i^2.
\]

`PortfolioProblem.cov` is still retained for audit and small calculations, but
the OSQP and Gurobi formulations can use the lower-rank factor expression.

## 3. Hard constraints

Always-on base rules are budget, long-only domains, asset bounds, group weight
bounds, eligibility, mandatory holdings, support linkage, and independent
validation. The benchmark normally also enables exact cardinality, minimum
active weight, maximum position, and turnover/cost awareness.

Data-dependent optional rules are:

\[
\mu^Tw\ge R_{min},\qquad y^Tw\ge Y_{min},
\]

\[
f_{min}\le B^Tw\le f_{max},
\]

\[
r_s^Tw\ge s_s,
\]

and empirical scenario CVaR:

\[
\eta+\frac{1}{(1-\alpha)S}\sum_su_s\le C_{max},
\]

\[
u_s\ge-r_s^Tw-\eta,\qquad u_s\ge0.
\]

The fixed-support SciPy oracle and Gurobi MIQP use this linear epigraph. The
validator independently recomputes empirical CVaR from the returned weights.

## 4. Full-universe relaxation

The first solve temporarily removes exact cardinality and minimum-active-weight
rules while retaining the remaining convex financial constraints. It returns:

- provisional percentages;
- a lower bound on the sparse minimization objective;
- asset attractiveness information;
- marginal risk signals;
- binding group/factor information.

The factor-QP variables are `[w,t,f]`, with the linear definition

\[
f=B^Tw.
\]

The quadratic block contains only diagonal idiosyncratic risk and the small
`factor_cov` block.

## 5. Guaranteed valid initial portfolio

The initializer protects mandatory assets and at least one strong candidate
from every group having a positive weight floor. Remaining slots are ranked by
the relaxation and smooth marginal objective. If that first support is not
valid, a SciPy/HiGHS feasibility MILP jointly solves continuous weights and
binary support decisions under cardinality, linkage, group, turnover, income,
factor, stress, and CVaR rules. Each returned support is reoptimized by the
allocation oracle. Randomized weighted trials and safe tiny enumeration remain
time-limit fallbacks. Search stops only after a fully valid exact-`K` portfolio
exists; a proven-infeasible MILP stops the pipeline with an explicit error.

No quantum routine runs before this point.

## 6. Adaptive change window

At iteration `j`, the current portfolio is split into:

- frozen selected assets outside the window;
- weak removable holdings inside the window;
- promising eligible unheld assets inside the window.

Mandatory or positive-lower-bound assets are never offered as removable.
Group slack and optional market communities influence ranking, but do not
remove assets from the global universe.

If `r` window assets are currently selected, every candidate must satisfy

\[
\sum_{i\in W}x_i=r.
\]

This preserves full-portfolio cardinality because frozen support size is
`K-r`.

## 7. Window surrogate

Let `a` be the equal proxy notional assigned to a selected window asset and
`w_F` the frozen outside-window allocation. The surrogate allocation is

\[
\widetilde w(x)=w_F+aS_Wx.
\]

Expanding the canonical objective gives

\[
x^TQx+h^Tx+\text{constant}.
\]

`Q` contains window covariance interactions. `h` contains covariance with
frozen holdings, return, income, binary-exact proxy transaction-cost changes,
and group-pressure signals. Strong-interaction sparsification is optional for
hardware; it changes proposal quality only because final candidates return to
the complete covariance model.

## 8. Classical window search

Tiny fixed-cardinality windows are enumerated exactly. Larger windows use
tabu/LNS:

1. Generate `1 -> 0` / `0 -> 1` swap neighbors.
2. Rank neighbors by QUBO energy.
3. Send only the most promising supports to the allocation oracle.
4. Cache duplicate supports.
5. Retain the best valid exact financial objective.
6. Use tabu tenure and fixed-weight random restarts to escape local minima.

## 9. XY-QAOA

The cost Hamiltonian is the Ising conversion of the window QUBO. The mixer is

\[
H_M=\frac12\sum_{(i,j)\in E}(X_iX_j+Y_iY_j).
\]

An XY gate exchanges `10` and `01` and therefore commutes with physical Hamming
weight. Starting from the current `r`-asset window bitstring means every ideal
sample still has `r` selected assets.

The state is

\[
|\psi(\gamma,\beta)\rangle=
\prod_{l=1}^p e^{-i\beta_lH_M}e^{-i\gamma_lH_C}|\psi_0\rangle.
\]

The default is a shallow warm-started ring mixer. Dicke initialization and a
complete mixer are explicit ablations. Cost coefficients are normalized before
angle optimization. Multiple COBYLA starts are seeded and recorded.

The portable subspace simulator stores only

\[
\binom{F}{r}
\]

states. Aer sampling compiles the same logical circuit for CPU or GPU. IBM
Runtime sampling requires an explicitly chosen backend and records job/circuit
metadata.

## 10. Allocation oracle and validation

For every classical or quantum support:

1. Set all outside-support upper bounds to zero.
2. Apply active position lower bounds to selected assets.
3. Solve the continuous convex financial model.
4. Recompute every base and optional guardrail independently.
5. Reject the support if any breach exceeds tolerance.
6. Cache the support result.

Weights are never clipped or renormalized after solving.

## 11. Exact reference

The final direct Gurobi model uses continuous `w`, binary `z`, turnover
epigraph `t`, factorized risk, exact cardinality, all configured hard rules,
and the best hybrid portfolio as a MIP start. A time-limited run returns the
incumbent, best bound, reported gap, nodes, build time, and solve time. Only an
optimal status is described as global certification, and always together with
the configured tolerance and numeric MIP gap. Fixed-support QP optimality is
recorded separately and never upgrades LNS or QAOA to global optimality.

## 12. Required comparisons

- Continuous factor-QP lower bound.
- Valid initialization.
- Classical enumeration on tiny windows.
- Classical tabu/LNS.
- XY-QAOA with the same windows and allocation oracle.
- Standard X-mixer penalty-QAOA.
- Gurobi exact-cardinality MIQP.
- Optional PCE/VQE legacy ablations.

The fair performance comparison uses equal end-to-end time. A second warm-start
comparison can give each method the same final Gurobi time.

## 13. Interpretation

The quantum circuit does not calculate percentages, validate constraints, or
replace Gurobi. It proposes combinations. The defensible contribution is a
constraint-preserving quantum neighborhood embedded in a scalable, auditable
classical optimization and validation system.
