# Constraint-Safe Quantum-Guided Large-Neighborhood Search for Sparse Multi-Asset Portfolios

**Final technical report and WISER Quantum Challenge evidence paper**  
**August 2026**

---

## Abstract

Real portfolio construction is not only a search for an attractive risk-return
trade-off. It must also respect a budget, hold an exact or limited number of
assets, cap position sizes and turnover, maintain group and factor exposures,
meet return and income targets, and control stress and tail losses. The
continuous portfolio weights and the discrete choice of which assets to hold
therefore form a mixed-integer quadratic optimization problem. This paper
presents a constraint-safe hybrid solver that separates those two decisions.
A factor-based quadratic program first solves a continuous full-universe guide.
Classical large-neighborhood search (LNS) and a fixed-cardinality quantum
alternating-operator circuit (XY-QAOA) then propose changes inside a small,
adaptive asset window. Every proposed asset set is passed to the same classical
allocation oracle, which assigns continuous weights, and to an independent
validator, which recomputes every hard constraint. Exact mixed-integer
optimization supplies global certificates where tractable.

The strongest correctness results are exact: four continuous solvers agree to
an objective spread of $1.41\times10^{-9}$; tiny exhaustive enumeration and
Gurobi return the same sparse optimum; the 100-asset case is solved to a 0.0%
reported mixed-integer gap; and a separate 60-asset gauntlet passes 244 checks
across 17 constraint families at a globally optimal objective of
$-0.0320099372$. The broader study remains feasible as the workload grows. A
250-asset, 10,000-scenario case passes 858 independent checks. All 21 repeated
full-hybrid runs from 250 to 20,000 assets and all 27 stretch runs from 1,000 to
300,000 assets return valid portfolios with zero reported hard-constraint
breaches. At 300,000 assets, median time to the first valid 50-asset portfolio
is 34.69 seconds, median complete search time is 112.60 seconds, and the factor
risk arrays occupy 29.76 MiB instead of the 670.55 GiB required by one dense
double-precision covariance matrix. The stretch runs are heuristic scaling
measurements, not global optimality certificates.

On IBM hardware, one 8,192-shot Runtime job evaluates ten width-depth cases
from 8 to 28 qubits with three repeated observations per case. The median raw
fixed-cardinality survival rate is 29.14%; all 30 post-allocated observations
still pass the financial validator. However, the QPU proposal beats a
matched-random proposal in only 6 of 30 strict comparisons, and the QUBO
surrogate has only 0.237-0.276 Spearman rank correlation with the final
allocated objective in exact 8-, 12-, and 16-variable enumerations. The study
therefore makes no quantum-advantage claim. Its contribution is a scalable,
auditable interface in which quantum hardware can propose local supports
without being allowed to violate portfolio rules or assign the final capital.

**Keywords:** portfolio optimization; cardinality constraint; mixed-integer
quadratic programming; large-neighborhood search; factor model; conditional
value at risk; QAOA; XY mixer; quantum hardware; independent validation.

## 1. Introduction

### 1.1 Problem and challenge objective

Portfolio optimization decides how much capital to place in each available
asset. In the classical mean-variance model of Markowitz [1], expected return
is the reward and covariance—the tendency of asset returns to move together—is
the source of portfolio variance, a common risk measure. A practical portfolio,
however, is constrained by implementation and policy. It may need exactly
$K$ active holdings; lower and upper position limits; eligibility and
mandatory-holding rules; asset-class or sector bands; a turnover budget;
minimum expected return and income; factor-exposure bands; scenario loss
floors; and a conditional value-at-risk (CVaR) limit. CVaR is the average loss
in the worst fraction of scenarios, rather than the loss at only one percentile
[2].

The Vanguard WISER Quantum Challenge asks for a mathematically explicit,
quantum-compatible, explainable portfolio optimizer; comparison with a
classical baseline; synthetic or anonymized data; risk, return, turnover, and
constraint-breach reporting; classical validation; and a presentation-quality
prototype [22]. The competition goal is therefore broader than obtaining a low
number from a quantum circuit. A useful result must be an actual allocation,
must explain its trade-offs, and must remain within the investor's guardrails.

This work addresses that goal with one central design rule:

> A classical or quantum search method may propose which assets to hold, but
> only the continuous allocation model and an independent validator may accept
> a portfolio and assign its final percentages.

This rule turns hardware noise, imperfect surrogates, and heuristic search into
recoverable proposal failures rather than financial-rule failures.

### 1.2 Literature overview

Markowitz's mean-variance framework established the quadratic relationship
between portfolio weights and covariance [1]. Adding a binary variable for
whether an asset is held produces a cardinality-constrained mixed-integer
quadratic program (MIQP): “mixed-integer” means that continuous weights and
integer selection decisions appear in the same model. Bienstock's early
computational study [3] and later work on fixed costs and minimum lots [4]
showed why realistic portfolio rules create hard combinatorial structure.
These studies also motivate hybrid methods that solve smaller mixed-integer
subproblems rather than enumerate every possible asset set. In the present
solver, exact enumeration, branch-and-bound MIQP, swap search, tabu search, and
LNS are all retained as classical references. The continuous subproblems are
quadratic programs (QPs); OSQP's operator-splitting method is particularly
useful for large sparse QPs and warm starts [5].

Tail loss is modeled using Rockafellar and Uryasev's convex CVaR formulation
[2]. It introduces one threshold variable and one nonnegative excess-loss
variable per scenario. This is larger than a mean-variance model but remains a
convex continuous problem for a fixed support. The experiments therefore study
both the number of assets and the number of loss scenarios.

Quantum Approximate Optimization Algorithm (QAOA) alternates a cost evolution
with a mixing evolution [6]. The quantum alternating-operator extension allows
mixers that remain inside a feasible subspace [7]. In particular, an XY mixer
exchanges the bit patterns `10` and `01` and therefore preserves Hamming weight,
the number of selected binary variables [8]. Hodson et al. applied hard
constraint mixers to an eight-stock, discrete-lot portfolio-rebalancing
example [9]. These ideas are directly relevant to exact-cardinality support
selection: a window that must keep $r$ assets can begin with $r$ one-bits
and, ideally, never leave that subspace.

The project also investigated several neighboring quantum formulations.
Variational quantum eigensolvers (VQE), introduced by Peruzzo et al. [10], can
minimize an Ising Hamiltonian obtained from a quadratic unconstrained binary
optimization (QUBO) model. Buonaiuto et al. studied encoding, penalty, ansatz,
optimizer, and hardware choices for portfolios of up to four assets on several
IBM devices [11]. Scursulim et al. later used multiple Dicke-state ansatzes—a
Dicke state is a superposition of bitstrings with the same Hamming weight—to
enforce multiclass allocation counts without a cardinality penalty [12]. This
is an elegant feasible-manifold construction, but its discrete multiclass
weights do not replace the continuous allocation and broad guardrail set needed
here. CVaR can also be used as a variational sample-aggregation objective [13];
in this project CVaR instead enters the financial allocation model, where its
meaning and units remain explicit.

Other portfolio research explores different depth and encoding trade-offs.
Digitized counterdiabatic algorithms add problem-inspired operators to improve
short-depth evolution [14], but the additional circuit structure can raise
routing and noise costs. Dynamic multi-period portfolios have been studied on
quantum processors and with tensor-network simulation [15]. Quantum annealing
has been evaluated with control benchmarks [16] and reverse annealing seeded by
classical local search [17]. Reverse annealing is conceptually close to the
warm-started local search used here, but it depends on annealing hardware and a
different experimental access path. Pauli Correlation Encoding (PCE) assigns
multiple binary variables to a qubit and iteratively partitions a market graph,
allowing a gate-based study with more than 250 variables [18]. PCE compresses
the binary representation, whereas the present architecture decouples global
universe size from qubit count by selecting a small adaptive change window.
The latter keeps continuous allocation and full-universe validation exact after
each proposal.

Broad reviews of quantum finance [19, 20] stress that practical claims must
distinguish ideal simulation, noisy hardware, preprocessing, and total
end-to-end cost. This paper follows that standard. It reports classical exact
results, heuristic scaling, ideal fixed-weight simulation, Aer GPU circuit
sampling, and IBM QPU observations as separate evidence tiers. It also includes
matched candidate budgets and random baselines, because a quantum method is not
demonstrably useful merely because it produces a valid bitstring.

### 1.3 Contributions

The project makes six contributions.

1. It formulates a sparse, multi-objective portfolio with continuous weights,
   binary support, turnover, group, eligibility, mandatory, return, income,
   factor, stress, CVaR, and implementation constraints.
2. It separates support generation from capital allocation through a cached
   fixed-support allocation oracle, allowing classical and quantum proposals
   to be evaluated by exactly the same financial model.
3. It uses an XY mixer to preserve window cardinality in the ideal circuit and
   a validator to protect the full portfolio when hardware noise does not.
4. It avoids dense covariance construction through a matrix-free factor risk
   representation and demonstrates repeated end-to-end runs up to 300,000
   candidate assets.
5. It triangulates correctness with four continuous backends, exhaustive tiny
   enumeration, exact Gurobi MIQP certificates, 17-family validation
   certificates, repeated seeds, scenario sweeps, and synthetic out-of-sample
   paths.
6. It performs a fair IBM hardware audit through 28 qubits and reports the
   unfavorable comparisons—weak QUBO/allocation alignment and no consistent
   advantage over matched random sampling—alongside the successful feasibility
   results.

## 2. Mathematical formulation

### 2.1 Data and variables

Let $n$ be the number of candidate assets. The model uses the following
inputs:

- $\mu_i$: estimated expected return of asset $i$;
- $y_i$: income yield, such as a dividend or coupon yield;
- $c_i$: linear cost per unit of turnover;
- $w_i^0$: current portfolio weight;
- $\Sigma$: return covariance matrix;
- $m_i,u_i$: minimum and maximum weight when asset $i$ is selected;
- $K$: required number of selected assets;
- $B$: asset-by-factor loading matrix; and
- $R_{si}$: return of asset $i$ in scenario $s$.

A **weight** is a fraction of total capital. The decision variable
$w_i\in\mathbb{R}$ is the new weight of asset $i$. The binary variable
$z_i\in\{0,1\}$ equals one if the asset is selected. The auxiliary variable
$t_i\ge0$ represents absolute turnover and satisfies
$t_i\ge w_i-w_i^0$ and $t_i\ge w_i^0-w_i$. The **support** of a portfolio
is the set $S=\{i:w_i>0\}$; its size is the portfolio's **cardinality**.

### 2.2 Objective

The canonical model minimizes

$$ 
\lambda_r w^T\Sigma w -\lambda_g\mu^T w -\lambda_y y^T w +\lambda_c c^Tt +\lambda_s\Phi(w). \quad (1) 
$$

The terms are, in order, variance risk, expected growth, income, trading cost,
and an optional scenario penalty $\Phi$. The nonnegative coefficients
$\lambda_r,\lambda_g,\lambda_y,\lambda_c,\lambda_s$ express investor
preferences. Risk and costs have positive signs because they are minimized;
return and income have negative signs because larger values should improve a
minimization objective. **Lower objective values are better, but the objective
is a composite ranking score—not a percentage return.** Results therefore
report return, volatility, income, turnover, tail loss, and breaches separately.

### 2.3 Hard guardrails

The core constraints are

$$
\begin{aligned}
&\mathbf{1}^T w = 1, &&\text{full investment},\\
&w_i\ge0, &&\text{long-only positions},\\
&m_i z_i\le w_i\le u_i z_i, &&\text{selection and position limits},\\
&\sum_{i=1}^{n}z_i=K, &&\text{exact cardinality},\\
&L_g\le\sum_{i\in g}w_i\le U_g, &&\text{group exposure bands},\\
&\sum_i t_i\le T_{\max}, &&\text{turnover cap}.
\end{aligned}
\tag{2}
$$

Here $\mathbf{1}$ is a vector of ones, so the first line states that weights
sum to 100%. A group $g$ may be an asset class, region, sector, or any policy
bucket, and $L_g,U_g$ are its minimum and maximum allocation. The turnover
definition is the two-way $L_1$ change; a 40% $L_1$ cap corresponds to 20%
one-way buys in a fully invested rebalance.

When enabled, the model also enforces

$$
\begin{aligned}
&z_i=0 &&\text{for ineligible assets},\\
&z_i=1 &&\text{for mandatory assets},\\
&\mu^T w\ge R_{\min} &&\text{minimum expected return},\\
&y^T w\ge Y_{\min} &&\text{minimum income},\\
&f_{\min}\le B^T w\le f_{\max} &&\text{factor-exposure bands},\\
&R_s^T w\ge q_s &&\text{stress-scenario floor},\\
&\operatorname{CVaR}_{\alpha}(w)\le C_{\max} &&\text{tail-loss limit},\\
&w_i\le \bar u_i^{\mathrm{impl}} &&\text{implementation cap}.
\end{aligned}
\tag{3}
$$

An implementation cap is a potentially tighter tradability limit derived from
liquidity or operational rules. Factor exposure $B^Tw$ measures the
portfolio's sensitivity to common drivers. A stress floor limits loss under a
named scenario. For scenario losses $\ell_s(w)=-R_s^Tw$, empirical CVaR at
confidence $\alpha$ is represented by

$$
\operatorname{CVaR}_{\alpha}(w)=
\eta+\frac{1}{(1-\alpha)N_s}\sum_{s=1}^{N_s}\xi_s,
\quad
\xi_s\ge \ell_s(w)-\eta,
\quad \xi_s\ge0,
\tag{4}
$$

where $\eta$ is a loss threshold and $\xi_s$ is scenario $s$'s loss
above that threshold [2].

### 2.4 Factor-native risk

For large universes the covariance is represented as

$$
\Sigma=B\Omega B^T+D,
\tag{5}
$$

where $B\in\mathbb{R}^{n\times k}$ holds $k$ factor loadings,
$\Omega\in\mathbb{R}^{k\times k}$ is the factor covariance, and $D$ is a
diagonal matrix of asset-specific variances. Portfolio variance becomes

$$
w^T\Sigma w=(B^Tw)^T\Omega(B^Tw)+\sum_i D_{ii}w_i^2.
\tag{6}
$$

Equation (6) can be evaluated without forming an $n\times n$ matrix. With a
fixed factor count, the dominant stored arrays grow linearly with $n$, while
a dense covariance grows quadratically.

## 3. Algorithm

### 3.1 End-to-end pipeline

The solver follows seven stages.

1. **Continuous guide.** Remove exact cardinality and minimum-active-weight
   rules, then solve the remaining convex factor QP over the full universe.
2. **Valid exact-$K$ initialization.** Construct a support containing the
   mandatory assets and sufficient representation for positive group floors;
   solve its allocation and, if needed, invoke a feasibility MILP.
3. **Adaptive window.** Select weak held assets and promising unheld assets to
   form a small change window.
4. **Candidate generation.** Use classical LNS, exact window enumeration,
   random fixed-weight sampling, penalty QAOA, or XY-QAOA to propose supports.
5. **Allocation oracle.** For each distinct support, solve the complete
   continuous model with all other weights fixed to zero.
6. **Independent validation.** Recompute every constraint directly from the
   returned weights. Accept only a better portfolio with zero breaches.
7. **Optional certification.** Solve the direct MIQP with Gurobi, optionally
   warm-started by the best valid hybrid portfolio.

The first valid portfolio is retained throughout the search. A failed guide,
infeasible proposal, incorrect hardware cardinality, solver timeout, or worse
candidate therefore cannot erase feasibility.

### 3.2 Continuous guide and initialization

The continuous guide supplies two things: a lower bound when it is solved to
the required status, and a ranking signal for promising assets. It does not
solve the sparse problem because it may use more than $K$ nonzero weights.
A deterministic initialization then assembles an exact-$K$ support and calls
the allocation oracle. Mandatory holdings are protected, group floors are
covered, and ranked alternatives are tried before a feasibility MILP is used.
The search begins only after this portfolio passes independent validation.

At large sizes, the guide is time-limited. If OSQP returns a usable iterate or
the current portfolio is already feasible, the search may continue, but the
unsolved guide is not reported as an optimality bound. This distinction is
especially important above 20,000 assets in the stretch experiment.

### 3.3 Fixed-support allocation oracle

For a proposed support $S$, the oracle imposes $w_i=0$ for every
$i\notin S$ and solves the resulting convex QP. Because the binary choice is
fixed, the oracle can assign continuous percentages while enforcing every
linear and convex constraint. Its result is cached by support to avoid
re-solving duplicate classical and quantum candidates.

The oracle is called an **oracle** because candidate generators query it for a
definitive score and feasibility decision; the word does not imply quantum
oracle access. The model never repairs a solution by clipping a negative
weight or renormalizing after the solve, because such post-processing could
silently violate a bound.

### 3.4 Classical large-neighborhood search

Large-neighborhood search changes several discrete decisions while freezing
the rest. In a window $W$ of $F$ assets, suppose $r$ are currently held.
A candidate bitstring $x\in\{0,1\}^F$ must satisfy

$$
\sum_{i\in W}x_i=r.
\tag{7}
$$

The window includes removable current holdings and attractive unheld assets,
using guide weights, marginal risk, group pressure, and correlation-community
coverage. Exact enumeration is possible for small $\binom{F}{r}$. At larger
sizes a tabu mechanism temporarily forbids recently reversed swaps, encouraging
the search to explore new supports. Only distinct, high-ranking proposals are
sent to the more expensive allocation oracle.

### 3.5 QUBO surrogate and XY-QAOA

Inside the same window, the support ranking is approximated by

$$
E(x)=x^TQx+h^Tx,
\qquad x\in\{0,1\}^F,
\tag{8}
$$

where $Q$ contains pairwise risk interactions and $h$ contains linearized
return, income, trading cost, frozen-portfolio covariance, and group-pressure
effects. A QUBO is “unconstrained” in its algebraic form, but this implementation
uses a mixer that restricts evolution to the fixed-weight subspace in (7). The
standard binary-to-Ising substitution maps (8) to Pauli $Z$ operators [21].

The XY mixer is

$$
H_M=\frac{1}{2}\sum_{(i,j)\in E}(X_iX_j+Y_iY_j),
\tag{9}
$$

where $X_i,Y_i$ are Pauli operators on qubit $i$, and $E$ is a ring or
hardware-aware edge set. Each term exchanges one selected and one unselected
bit. Starting from the current window support therefore preserves exactly
$r$ selections in ideal execution. At QAOA depth $p$, cost and mixer
evolutions are alternated $p$ times and their angles are tuned classically.

The default window uses $F=16$ rather than one qubit per global asset. For
example, with seven required selections, the exact feasible subspace has

$$
\binom{16}{7}=11{,}440
\tag{10}
$$

states. A CPU subspace simulator tunes angles efficiently; Aer GPU executes
the corresponding physical circuit and verifies the actual device. IBM
Runtime samples selected hardware cases. Hardware bitstrings with the wrong
Hamming weight are recorded rather than disguised. Valid distinct bitstrings
are passed to the same allocation oracle as classical proposals.

### 3.6 Certification and claim tiers

Gurobi's branch-and-bound MIQP reports an incumbent, a best bound, and a mixed-
integer programming gap. A 0.0% reported gap means that the incumbent and bound
coincide to the solver's reporting tolerance; it is the appropriate global
certificate for the stated model. By contrast, an optimally solved
fixed-support QP proves only that no better weights exist on that one support.

This paper uses four evidence labels:

- **Exact/certified:** exhaustive enumeration or an MIQP incumbent with a
  matching bound.
- **Bounded:** a valid sparse portfolio compared with a solved continuous lower
  bound, without a mixed-integer certificate.
- **Heuristic:** a valid portfolio with no solved global or relaxation bound.
- **Hardware observation:** measured circuit and post-allocation behavior on a
  named QPU; not evidence of asymptotic quantum advantage.

## 4. Experimental design

### 4.1 Data, privacy, and reproducibility

All portfolio instances and out-of-sample paths in the submitted benchmark
packages are synthetic or anonymized. No client holdings, account identifiers,
or confidential market data are used. Each result package records configuration,
random seed, software environment, solver status, raw tables, checksums, and
plots. The original supplied archives and challenge brief are fingerprinted in
`results/final_submission/archive_manifest.csv`.

The benchmark suite contains 224 relevant files before duplicate and temporary
notebook artifacts are removed: 73 CSV tables, 24 JSON records, 64 PNG figures,
52 PDF figures or documents, six Markdown reports, four text files, and one YAML
configuration. The curated repository package preserves the source tables,
provenance, and presentation figures needed to audit every headline claim;
very large duplicate diagnostics, raw shot-count payloads, and machine-specific
environment inventories remain represented by archive checksums rather than
duplicated in Git. The resolved numerical and solver settings are published;
one absolute output path is normalized to its repository-relative equivalent.

### 4.2 Experimental cases

| Case | Purpose | Scale | Evidence tier |
|---|---|---:|---|
| Continuous cross-check | Verify the convex implementation across backends | 4 solvers | Exact numerical agreement |
| Tiny sparse case | Compare enumeration, QAOA, and Gurobi | $K=4$ | Exact/certified |
| Main sparse case | Compare initialization, LNS, two QAOA forms, and Gurobi | 100 assets, $K=20$ | Exact/certified |
| Full-constraint gauntlet | Exercise every guardrail family | 60 assets, $K=12$ | Exact/certified |
| Scenario-rich gauntlet | Combine all guardrails with empirical CVaR | 250 assets, 10,000 scenarios, $K=25$ | Heuristic sparse result |
| Scenario count scaling | Measure the CVaR continuous solve | 500-100,000 scenarios | Continuous-solver benchmark |
| Preference and tail sweeps | Show controllable investor trade-offs | 30 tail runs plus repeated profiles | Repeated heuristic/continuous |
| Full hybrid scaling | Repeated end-to-end hybrid protocol | 250-20,000 assets, 21 runs | Bounded where guide solved |
| Stretch scaling | Test matrix-free engineering scale | 1,000-300,000 assets, 27 runs | Heuristic above solved guides |
| Frozen 16-variable window | Match candidate methods and oracle budgets | 6 methods | Controlled local benchmark |
| Width-depth sweep | Measure subspace growth and circuit cost | 8-24 variables | Ideal simulation |
| IBM campaign | Audit cardinality survival and proposal quality | 8-28 qubits, 30 observations | Hardware observation |
| Equal-lot classical baseline | Compare QP, MIQP, SCIP, annealing | 250 assets, 1,000 units | Exact/near-exact classical |

### 4.3 Metrics

Every accepted portfolio reports:

- expected return and income yield;
- volatility, the square root of modeled variance;
- turnover and estimated transaction cost;
- maximum drawdown and 95% CVaR for synthetic paths where applicable;
- objective value, always with the reminder that lower is better;
- support size, breach count, and maximum numerical violation;
- runtime to first valid output and complete runtime;
- solver status, bound, and gap when available; and
- for quantum runs, execution device, circuit depth, two-qubit gates, shots,
  Hamming-weight survival, distinct candidates, and allocation outcome.

Synthetic backtests use 30 generated paths in the main robustness experiment.
They test whether risk behavior is coherent under held-out draws; they are not
forecasts, live track records, or financial advice.

## 5. Results

### 5.1 Continuous and discrete correctness

SciPy SLSQP, OSQP, Clarabel, and Gurobi all solve the same continuous case with
zero breaches. Their objective spread is only $1.4063\times10^{-9}$.

| Backend | Objective | Runtime (s) | Breaches |
|---|---:|---:|---:|
| OSQP | -0.049140752870 | 0.0041 | 0 |
| Clarabel | -0.049140751526 | 0.0087 | 0 |
| Gurobi QP | -0.049140752932 | 0.0190 | 0 |
| SciPy SLSQP | -0.049140752606 | 1.4517 | 0 |

In the tiny sparse case, exhaustive enumeration and Gurobi return the same
objective, $-0.03681145$, at $K=4$; Gurobi reports a 0.0% gap. The penalty-
QAOA proposal reaches the same support-level result after allocation. The
continuous relaxation attains a lower objective because it is not restricted
to the same sparse feasible set and must not be called the sparse winner.

The independent 250-asset equal-lot benchmark supplies a second classical
check. Clarabel, OSQP, and Gurobi agree on a continuous objective near
$-0.09852504$. Gurobi's 1,000-unit MIQP reaches $-0.09852495$ in 0.0557
seconds; SCIP reaches a very similar solution in 0.6986 seconds; and swap
annealing is within $2.92\times10^{-6}$ relative gap in about 3.99 seconds.
All are feasible. This result establishes a strong classical baseline and
shows that the project did not adopt a quantum method before validating the
classical model.

### 5.2 Main 100-asset result

The main case requires exactly 20 holdings. Gurobi certifies the global sparse
objective $-0.0384147146$ with a 0.0% reported gap and zero breaches. The
valid initial portfolio scores $-0.0346876359$. Classical LNS, XY-QAOA on
Aer GPU, and penalty QAOA all eventually reach the same best heuristic support,
with allocated objective $-0.0380448157$. That support closes 90.08% of the
initial-to-certified objective difference.

| Method | Final objective | Status | Breaches |
|---|---:|---|---:|
| Valid initialization | -0.0346876359 | Feasible | 0 |
| Classical tabu/LNS | -0.0380448157 | Heuristic | 0 |
| XY-QAOA, Aer GPU | -0.0380448157 | Heuristic; GPU verified | 0 |
| Penalty QAOA, statevector | -0.0380448157 | Heuristic | 0 |
| Gurobi cardinality MIQP | **-0.0384147146** | Global optimum; 0.0% gap | 0 |

All three Gurobi starts reach the same optimum at one explored node. Median-like
single-run total times are 0.03574 seconds from cold, 0.02721 seconds from the
valid initialization, and 0.02698 seconds from the hybrid incumbent. On this
easy certified case the hybrid start reduces total time by 24.5%, but the
sample is too small to claim a general warm-start speedup.

Across 30 synthetic out-of-sample paths, the LNS/XY/penalty support has median
terminal wealth 1.7909, annualized volatility 8.22%, return-to-volatility ratio
0.7316, maximum drawdown 13.53%, and period CVaR 4.23%. The initial portfolio's
corresponding medians are 1.7358, 8.40%, 0.7083, 14.37%, and 4.32%. Gurobi's
globally best in-sample objective does not dominate every held-out statistic:
its median terminal wealth is 1.7692 and return-to-volatility ratio is 0.7106.
This is a useful warning against equating one estimated objective with certain
future performance.

### 5.3 All-constraint certification

The full gauntlet selects exactly 12 of 60 assets and exercises 17 constraint
families. Gurobi returns $-0.0320099372$ with a 0.0% reported gap. All 244
independently recomputed checks pass, including 60 lower and 60 upper position
checks, eligibility, mandatory holdings, group bands, turnover, return, income,
factor bands, five stress floors, empirical 95% CVaR, and implementation caps.

![Certified guardrail checks](results/final_submission/figures/all_constraints_guardrails.png)

The validator tolerates only configured numerical residuals; for example, the
turnover residual is $5.55\times10^{-17}$, effectively floating-point zero.
The certificate demonstrates that zero breaches is not merely a summary flag:
it is backed by row-level left-hand sides, limits, slacks, and pass/fail results.

The larger scenario-rich case selects 25 of 250 assets while processing 10,000
scenarios. Classical LNS returns the strongest sparse objective,
$-0.0376201667$, and zero breaches; the 16-qubit Aer-GPU XY proposal returns
$-0.0362818400$, 100% ideal cardinality, transpiled depth 43, and 61 two-qubit
gates. Seventeen constraint families and 858 independent checks pass. This is a
validated heuristic result; no global MIQP certificate was completed.

### 5.4 Tail-risk and preference controls

The scenario-penalty experiment evaluates six penalty weights with five
repetitions each. All 30 allocations have zero breaches. Moving from no
scenario penalty to $\lambda_s=1$ reduces median held-out 95% CVaR from
11.1051% to 10.5619%, an absolute improvement of 54.31 basis points. A basis
point is one hundredth of a percentage point. The trade-off is 41.55 basis
points of expected return. The result is a continuous frontier rather than a
claim that one setting is universally best.

![Scenario penalty frontier](results/final_submission/figures/scenario_penalty_frontier.png)

One-at-a-time sensitivity sweeps also preserve feasibility. Raising risk
aversion from 0.5 to 25 moves expected return from 7.95% to 5.46% and volatility
from 12.03% to 7.81%. Raising the income weight from 0 to 4 increases income
yield from 2.18% to 2.50%. Raising the cost weight from 1 to 25 reduces turnover
from 40.00% to 12.36% and estimated transaction cost from 5.81 to 0.71 basis
points. In five repeated risk sweeps, the normalized frontier knee occurs at
$\lambda_r=3$ four times and $\lambda_r=5$ once.

The named presets make these controls accessible to a non-specialist:

| Preset | Expected return | Volatility | Income | Turnover | Held-out CVaR | Breaches |
|---|---:|---:|---:|---:|---:|---:|
| Growth | 7.90% | 11.81% | 2.06% | 40.00% | 16.11% | 0 |
| Balanced | 5.96% | 8.18% | 2.18% | 40.00% | 10.65% | 0 |
| Income | 5.83% | 8.35% | 2.46% | 40.00% | 11.06% | 0 |
| Drawdown control | 5.46% | 7.81% | 2.23% | 40.00% | 10.37% | 0 |
| Cost sensitive | 6.18% | 8.66% | 2.21% | 30.00% | 11.56% | 0 |

![Preference sensitivity](results/final_submission/figures/preference_sensitivity.png)

### 5.5 Scenario-count scaling

The 250-asset CVaR continuous model is repeated with 500, 2,000, 10,000,
50,000, and 100,000 scenarios. Clarabel's median time grows from 0.0186 to
13.9239 seconds; OSQP grows from 0.0198 to 13.6907 seconds. Every run succeeds
with zero breaches. The 100,000-scenario return array occupies 190.74 MiB. This
experiment isolates scenario growth from the discrete search and shows that
tail-risk controls remain computationally manageable at the studied scale.

### 5.6 Full-hybrid and stretch scaling

The repeated full-hybrid study uses exactly 50 holdings, 12 factors, three
adaptive 16-variable windows, and three seeds at each universe size from 250 to
20,000 assets. All 21 runs succeed and have zero breaches; ideal quantum
cardinality is 100% throughout. Median complete times are 1.49, 2.21, 2.30,
3.67, 8.99, 20.06, and 54.92 seconds at 250, 500, 1,000, 2,000, 5,000, 10,000,
and 20,000 assets. Median time to first validity rises from 0.033 to 30.354
seconds. Where a solved continuous guide is available, median relative sparse
gaps remain approximately 0.37%-0.76%; only those solved rows support a bound
comparison.

The separate stretch protocol uses one adaptive iteration and a fixed 16-qubit
Aer-GPU window at 1,000, 10,000, 20,000, 50,000, 80,000, 100,000, 150,000,
200,000, and 300,000 assets, with three seeds each. All 27 runs return exactly
50 holdings with zero breaches. At 300,000 assets, median time to first valid
output is 34.6929 seconds, complete hybrid time is 112.5998 seconds, and peak
resident memory is 10.657 GiB.

![Scaling runtime](results/final_submission/figures/scaling_runtime.png)

Above 20,000 assets, most 30-second guide relaxations reach the time limit and
the solver continues from a valid fallback. Their relaxation-gap cells are
deliberately blank. These rows demonstrate safe engineering scale, not global
quality or optimality.

At 300,000 assets, the 12-factor representation occupies 0.02906 GiB (29.76
MiB). One dense covariance would occupy 670.55 GiB, a storage ratio of about
23,076 to 1. This dense figure is an analytical storage calculation; the dense
matrix was intentionally not allocated.

![Factor versus dense storage](results/final_submission/figures/factor_vs_dense_memory.png)

### 5.7 Controlled quantum candidate benchmark

The frozen 16-variable window compares candidate generators before the global
window changes. Random fixed-weight sampling finds the largest objective
improvement, 0.0022968, after 128 allocation-oracle calls. Exact QUBO top states
and classical LNS each improve by 0.0021369, but LNS requires only 11 oracle
calls and therefore has the best improvement per call. XY-QAOA subspace, Aer
CPU, and Aer GPU each improve by 0.0006971 with five or six evaluated supports.
Aer GPU is verified, but at this small workload it is slower than Aer CPU and
the specialized subspace simulator.

![Frozen-window candidate efficiency](results/final_submission/figures/frozen_window_candidate_efficiency.png)

The width-depth sweep explains the fixed 16-qubit design. The feasible
subspace $\binom{F}{r}$ grows from 70 states at $F=8,r=4$ to 2,496,144 at
$F=24,r=11$. Depth-one subspace runtime grows from 0.059 seconds at width 8
to 36.57 seconds at width 24. A second QAOA layer often improves QUBO energy
but increases optimizer evaluations, logical gates, and runtime. Classical LNS
produces the larger final objective improvement in every matched row of this
sweep.

### 5.8 IBM QPU audit

The hardware campaign uses `ibm_kingston` version 1.0.0 with calibration
timestamp 2026-08-07 00:37:20-04:00. Runtime job
`d9qmec1dsedc73afg2d0` packs ten cases with 8,192 shots each and three repeated
observations per case. Widths range from 8 to 28 qubits; depths range from one
to three where tested. The circuits contain 101-483 two-qubit gates and
transpiled depths of 171-923.

The median raw fixed-cardinality survival across all 30 observations is 29.14%,
with a minimum of 7.48% and maximum of 67.44%. Median survival falls from
67.13% at $F=8,p=1$ to 7.75% at $F=28,p=1$; deeper 12- and 20-qubit
circuits also reduce survival. Nevertheless, postselection, exact allocation,
and validation produce zero hard-constraint breaches in all 30 observations.
The result validates the safety boundary, not noise-free cardinality.

![IBM QPU frontier](results/final_submission/figures/ibm_qpu_frontier.png)

Proposal quality is less favorable. The QPU objective improvement exceeds the
matched-random result in only 6 of 30 strict comparisons. Median QPU improvement
is 0.002063 versus 0.002205 for random sampling. Exact enumeration of all
fixed-weight supports at widths 8, 12, and 16 finds only 0.237-0.276 Spearman
rank correlation between QUBO energy and final allocated objective. Spearman
correlation measures whether two rankings agree; one means identical ordering,
zero means no monotonic ordering. Many top-QUBO supports therefore miss the
best support after continuous allocation.

![QUBO and allocation alignment](results/final_submission/figures/qubo_alignment.png)

The correct conclusion is deliberately narrow: IBM hardware generated usable
fixed-cardinality candidates through 28 qubits, and the classical safety layer
converted every reported observation into a valid portfolio. The experiment
does **not** demonstrate a speed, quality, or scaling advantage over the best
classical candidate generator.

## 6. Output and explainability

### 6.1 Portfolio output

The final user-facing output is not a bitstring. It is a table with, for every
asset, the current weight, recommended weight, buy or sell change, selection
status, group, expected return, income contribution, risk contribution, and
estimated transaction cost. Summary fields explain:

- why the selected preset or preference weights were used;
- expected return, volatility, income, turnover, transaction cost, CVaR, and
  stress results;
- which constraints are active or close to active;
- whether the result is certified, bounded, or heuristic; and
- which classical, simulator, GPU, or QPU components actually ran.

An **active constraint** has zero or nearly zero slack, meaning the portfolio is
at that rule's limit. Reporting active constraints explains why the optimizer
cannot improve one goal without changing another rule or preference.

### 6.2 Copilot interaction

The Streamlit Copilot exposes investor-friendly controls for growth, income,
drawdown control, implementation cost, turnover, return, factor, stress, and
CVaR settings. Each change triggers a fresh solve and independent validation.
If the requested combination is impossible, the interface reports
infeasibility instead of silently relaxing a rule. The command is:

```bash
streamlit run src/vanguard_portfolio/copilot_app.py
```

### 6.3 Evidence outputs

The curated evidence index is
`results/final_submission/README.md`. The most important machine-readable files
are `evidence_summary.csv`, `claim_evidence_map.csv`, the row-level constraint
certificates, the scaling run tables, and the IBM provenance and frontier
tables. Figures are explanatory views of those tables, not standalone proof.

## 7. Discussion

### 7.1 What is strongest

The strongest challenge result is the combination of safety and scale. Exact
small and medium cases validate the mathematics; a 17-family certificate shows
that “zero breaches” covers the full rule set; preference and scenario sweeps
show that the output responds coherently to investor goals; repeated scaling
shows that the same acceptance boundary survives through 300,000 candidate
assets; and hardware tests show that noisy quantum samples cannot bypass it.

The architecture is also modular. A better classical heuristic, a future
fault-tolerant algorithm, PCE, a Dicke-state ansatz, reverse annealing, or a
learned candidate generator can replace the proposal engine without changing
the allocation oracle or validator. This makes quantum experimentation useful
without making portfolio safety depend on a quantum claim.

### 7.2 Why the quantum result is still informative

The negative quantum comparison diagnoses the interface rather than invalidating
the entire approach. Exact QUBO/allocation rank correlations below 0.28 show
that improving the surrogate is more urgent than increasing QAOA depth. The
width-depth results show that deeper circuits raise both classical tuning cost
and hardware burden. The hardware frontier shows rapidly falling cardinality
survival. These results recommend three next steps: learn or derive a
better allocation-aware surrogate; concentrate hardware trials on shallow,
well-calibrated windows; and continue matching oracle-call and time budgets
against strong classical and random baselines.

### 7.3 Limitations

The model is single-period and long-only. Expected returns, covariances, factor
loadings, and scenarios are estimates and can be wrong. Linear transaction cost
does not model nonlinear market impact. Taxes, tax lots, leverage, shorting,
liquidity dynamics, and multi-period recourse are outside the present scope.
Synthetic paths test internal consistency but cannot establish investable
performance. Results at 300,000 assets use a generated factor structure and a
fixed search budget; they do not prove that every real universe of that size is
easy.

The quantum circuit solves a local surrogate, not the entire constrained
portfolio. Hamming-weight preservation is exact only for ideal execution;
hardware noise can break it. Postselection reduces usable shots. The IBM
campaign is one backend/calibration campaign, and its queue-inclusive wall time
is not compared directly with local kernels. No result in this paper establishes
quantum advantage.

## 8. Reproducibility, tools, and contributions

Every headline value in this paper maps to a repository table through
`results/final_submission/claim_evidence_map.csv`. The supplied result archives
were integrity-tested before analysis. The curated package omits large raw
diagnostic duplicates from Git but records their source archive SHA-256
fingerprints. Reproduction should use the published numerical and solver
settings, pin the required dependencies, preserve seeds, record the new
execution environment, and report timeouts or fallbacks rather than
interpolating missing bounds.

The project uses Python; NumPy, SciPy, CVXPY, OSQP, Clarabel, HiGHS, Gurobi,
Qiskit, Qiskit Aer, Qiskit IBM Runtime, Matplotlib, pandas, and Streamlit as
applicable to a run. Gurobi requires a license; IBM hardware requires an
authorized Runtime account. AI-assisted coding and writing tools were used for
review, test scaffolding, analysis organization, and documentation drafting.
All reported numerical values come from the preserved experiment outputs and
were checked against their source tables and independent validators. The human
team remains responsible for model choices, execution, interpretation, and the
submitted work.

**Author-contribution statement.** The project team jointly covered
conceptualization, methodology and software, experiment execution, validation
and analysis, visualization and presentation, and writing and review. The
report does not assign those roles to named individuals because no verified
team roster accompanied the result archives.

## 9. Conclusion

This project delivers a practical hybrid portfolio optimizer whose most
important invariant is simple: no proposal becomes a portfolio until exact
continuous allocation and independent validation say that it can. Classical
backends agree numerically, exact sparse cases are certified, all 17 guardrail
families pass in the full gauntlet, investor and tail-risk controls behave
coherently, and repeated zero-breach outputs extend to a 300,000-asset
factor-model universe. The IBM campaign shows that quantum hardware can be
inserted into this pipeline without compromising the financial rules, even
when raw cardinality survival is low.

The same evidence also prevents overclaiming. Classical LNS is the most
efficient tested local proposal method, matched random sampling often equals
or beats the QPU, and QUBO energy is only weakly aligned with the final
allocated objective. The winning case for the solver is therefore not an
unsupported promise of quantum advantage. It is a rigorously tested,
explainable, production-oriented system that is useful with today's classical
tools and provides a fair, safe experimental slot for better quantum methods
as hardware and encodings improve.

## References

1. H. Markowitz, “Portfolio Selection,” *The Journal of Finance*, 7(1),
   77-91, 1952. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
2. R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,”
   *The Journal of Risk*, 2(3), 21-41, 2000.
   https://doi.org/10.21314/JOR.2000.038
3. D. Bienstock, “Computational study of a family of mixed-integer quadratic
   programming problems,” *Mathematical Programming*, 74, 121-140, 1996.
   https://doi.org/10.1007/BF02592208
4. H. Kellerer, R. Mansini, and M. G. Speranza, “Selecting Portfolios with Fixed
   Costs and Minimum Transaction Lots,” *Annals of Operations Research*, 99,
   287-304, 2000. https://doi.org/10.1023/A:1019279918596
5. B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, “OSQP: an
   operator splitting solver for quadratic programs,” *Mathematical Programming
   Computation*, 12, 637-672, 2020.
   https://doi.org/10.1007/s12532-020-00179-2
6. E. Farhi, J. Goldstone, and S. Gutmann, “A Quantum Approximate Optimization
   Algorithm,” arXiv:1411.4028, 2014. https://arxiv.org/abs/1411.4028
7. S. Hadfield, Z. Wang, B. O'Gorman, E. Rieffel, D. Venturelli, and R. Biswas,
   “From the Quantum Approximate Optimization Algorithm to a Quantum Alternating
   Operator Ansatz,” *Algorithms*, 12(2), 34, 2019.
   https://doi.org/10.3390/a12020034
8. Z. Wang, N. C. Rubin, J. M. Dominy, and E. G. Rieffel, “XY mixers: Analytical
   and numerical results for the quantum alternating operator ansatz,”
   *Physical Review A*, 101, 012320, 2020.
   https://doi.org/10.1103/PhysRevA.101.012320
9. M. Hodson, B. Ruck, H. Ong, D. Garvin, and S. Dulman, “Portfolio rebalancing
   experiments using the Quantum Alternating Operator Ansatz,” arXiv:1911.05296,
   2019. https://arxiv.org/abs/1911.05296
10. A. Peruzzo et al., “A variational eigenvalue solver on a photonic quantum
    processor,” *Nature Communications*, 5, 4213, 2014.
    https://doi.org/10.1038/ncomms5213
11. G. Buonaiuto, F. Gargiulo, G. De Pietro, M. Esposito, and M. Pota, “Best
    practices for portfolio optimization by quantum computing, experimented on
    real quantum devices,” *Scientific Reports*, 13, 19434, 2023.
    https://doi.org/10.1038/s41598-023-45392-w
12. J. V. S. Scursulim et al., “Multiclass portfolio optimization via
    variational quantum Eigensolver with Dicke state ansatz,” *Scientific
    Reports*, 16, 6208, 2026.
    https://doi.org/10.1038/s41598-026-36333-4
13. P. K. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, and S. Woerner,
    “Improving Variational Quantum Optimization using CVaR,” *Quantum*, 4, 256,
    2020. https://doi.org/10.22331/q-2020-04-20-256
14. N. N. Hegade et al., “Portfolio optimization with digitized
    counterdiabatic quantum algorithms,” *Physical Review Research*, 4, 043204,
    2022. https://doi.org/10.1103/PhysRevResearch.4.043204
15. S. Mugel et al., “Dynamic portfolio optimization with real datasets using
    quantum processors and quantum-inspired tensor networks,” *Physical Review
    Research*, 4, 013006, 2022.
    https://doi.org/10.1103/PhysRevResearch.4.013006
16. E. Grant, T. S. Humble, and B. Stump, “Benchmarking Quantum Annealing
    Controls with Portfolio Optimization,” *Physical Review Applied*, 15,
    014012, 2021. https://doi.org/10.1103/PhysRevApplied.15.014012
17. D. Venturelli and A. Kondratyev, “Reverse quantum annealing approach to
    portfolio optimization problems,” *Quantum Machine Intelligence*, 1,
    17-30, 2019. https://doi.org/10.1007/s42484-019-00001-w
18. V. P. Soloviev and M. Krompiec, “Large-scale portfolio optimization using
    Pauli correlation encoding,” *Scientific Reports*, 16, 2026.
    https://doi.org/10.1038/s41598-026-54244-2
19. R. Orús, S. Mugel, and E. Lizaso, “Quantum computing for finance: Overview
    and prospects,” *Reviews in Physics*, 4, 100028, 2019.
    https://doi.org/10.1016/j.revip.2019.100028
20. D. J. Egger et al., “Quantum Computing for Finance: State-of-the-Art and
    Future Prospects,” *IEEE Transactions on Quantum Engineering*, 1, 2020.
    https://doi.org/10.1109/TQE.2020.3030314
21. A. Lucas, “Ising formulations of many NP problems,” *Frontiers in Physics*,
    2, 5, 2014. https://doi.org/10.3389/fphy.2014.00005
22. Vanguard and WISER, “Quantum Challenge: Multi-Asset Portfolio Optimization,”
    challenge brief supplied with the project, 2026.
