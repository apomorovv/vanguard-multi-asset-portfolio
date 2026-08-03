# Multi-Asset Portfolio Construction Model

> **Final implementation note:** this file remains the original baseline model.
> The exact-cardinality continuous-weight hybrid model, factor-QP decomposition,
> optional CVaR/stress constraints, allocation oracle, and XY-QAOA window are
> specified in [`final_hybrid_model.md`](final_hybrid_model.md).

## 1. Purpose

The model constructs a single-period, long-only portfolio from a universe of
eligible assets. It balances four economic goals:

1. lower portfolio risk;
2. higher expected return;
3. higher income yield;
4. lower implementation cost.

The model also enforces practical rules such as full investment, position
limits, group exposure limits, turnover limits, exact portfolio cardinality,
mandatory holdings, factor bands, stress tests, and empirical CVaR limits.

All returns, volatilities, yields, costs, and portfolio weights use decimal
units. For example, `0.07` means 7%.

## 2. Sets and Indices

Let:

- $\mathcal A = \{1,\ldots,n\}$ be the asset set;
- $\mathcal G = \{1,\ldots,G\}$ be the group set;
- $\mathcal F = \{1,\ldots,k\}$ be the factor set;
- $\mathcal S = \{1,\ldots,S\}$ be the scenario set.

An asset belongs to exactly one declared group in the current data schema.

## 3. Input Data

| Symbol | Shape | Meaning |
|---|---:|---|
| $\mu$ | $n$ | Annual expected asset returns. |
| $\sigma$ | $n$ | Annual asset volatilities. |
| $\rho$ | $n\times n$ | Asset correlation matrix. |
| $\Sigma$ | $n\times n$ | Asset covariance matrix. |
| $y$ | $n$ | Annual income or yield estimates. |
| $c$ | $n$ | Proportional transaction-cost coefficients. |
| $w^0$ | $n$ | Current portfolio weights. |
| $\ell$ | $n$ | Base lower bounds on asset weights. |
| $u$ | $n$ | Base upper bounds on asset weights. |
| $A$ | $G\times n$ | Group-membership matrix. |
| $L$ | $G$ | Group lower bounds. |
| $U$ | $G$ | Group upper bounds. |
| $B_{\mathrm{budget}}$ | scalar | Total portfolio budget, normally 1. |
| $K$ | scalar | Required number of selected assets. |
| $m$ | scalar | Minimum positive weight for a selected asset. |
| $\bar u$ | $n$ | Optional implementation-specific maximum weights. |

When dense covariance data are present,

$$
\Sigma_{ij} = \rho_{ij}\sigma_i\sigma_j.
$$

The covariance matrix must be symmetric and positive semidefinite:

$$
v^\top\Sigma v \ge 0
\qquad
\text{for every } v\in\mathbb R^n.
$$

Positive semidefiniteness is financially necessary because a variance cannot be
negative. It also makes the continuous risk term convex.

## 4. Factor-Risk Representation

Large instances may omit dense covariance and correlation matrices and instead store:

$$
\Sigma = B\Omega B^\top + D
$$

where:
- $B\in\mathbb{R}^{n\times k}$ contains asset-factor loadings;
- $\Omega\in\mathbb{R}^{k\times k}$ is the factor covariance matrix;
- $D = \mathrm{diag}(d_1, \ldots, d_n)$ contains idiosyncratic variances.

For portfolio weights $w$,

$$
w^\top\Sigma w = (B^\top w)^\top\Omega(B^\top w) + \sum_{i=1}^n d_iw_i^2
$$

This representation avoids storing an $n \times n$ dense matrix. Storage for the main risk arrays scales as $O(nk+k^2)$ instead of $O(n^2)$.

The implementation can still recover:

- $\Sigma w$ through a factor matrix-vector product;
- a small covariance submatrix for a selected support or change window;
- a small correlation submatrix when required for diagnostics.

## 5. Decision Variables

The final sparse portfolio uses:

| Variable | Domain | Meaning |
|---|---|---|
| $w_i$ | continuous, $w_i\ge0$ | Final weight of asset $i$. |
| $z_i$ | binary | 1 when asset $i$ is selected, otherwise 0. |
| $t_i$ | continuous, $t_i\ge0$ | Epigraph variable for $\lvert w_i-w_i^0 \rvert$. |
| $f_j$ | continuous | Portfolio exposure to factor $j$. |
| $\eta$ | continuous | VaR threshold used in the CVaR epigraph. |
| $u_s$ | continuous, $u_s\ge0$ | Scenario loss above $\eta$. |

The variables $f$, $\eta$, and $u$ are present only when their associated
constraints are enabled.

## 6. Portfolio Quantities

### 6.1 Expected Return

$$
R(w)=\mu^\top w.
$$

### 6.2 Variance and Volatility

$$
V(w)=w^\top\Sigma w,
\qquad
\sigma_p(w)=\sqrt{V(w)}.
$$

Variance is the quantity in the optimization objective. Volatility is reported
for interpretation.

### 6.3 Income

$$
Y(w)=y^\top w.
$$

### 6.4 Turnover

The implementation uses total L1 turnover:

$$
T(w)=\sum_{i=1}^n |w_i-w_i^0|.
$$

When the current and target portfolios have the same total budget, the common
one-way turnover convention is $T(w)/2$. Therefore, an L1 cap of `0.40`
corresponds to 20% one-way turnover.

### 6.5 Transaction Cost

$$
C(w)=\sum_{i=1}^n c_i|w_i-w_i^0|.
$$

Turnover and transaction cost are different. Turnover measures how much capital
is traded; transaction cost weights each trade by its asset-specific cost.

## 7. Canonical Objective

Every accepted portfolio is evaluated with the same minimization objective:

$$
F(w) = \lambda_{\mathrm{risk}}\,w^\top\Sigma w - \lambda_{\mathrm{return}}\,\mu^\top w - \lambda_{\mathrm{income}}\,y^\top w + \lambda_{\mathrm{cost}}\,c^\top t
$$

The preference coefficients are nonnegative:

$$
\lambda_{\mathrm{risk}}, \lambda_{\mathrm{return}}, \lambda_{\mathrm{income}}, \lambda_{\mathrm{cost}} \ge 0
$$


Interpretation:

- the risk term increases the objective, so lower variance is preferred;
- return and income have negative signs, so larger values are preferred;
- transaction cost increases the objective, so unnecessary trading is
  discouraged.

The objective value is a ranking score. It is not itself a return percentage.

The turnover epigraph is enforced by

$$
t_i \ge w_i-w_i^0,
\qquad
t_i \ge w_i^0-w_i,
\qquad
t_i\ge0.
$$

When $c_i\lambda_{\mathrm{cost}}>0$, minimizing the objective drives
$t_i$ to $|w_i-w_i^0|$. The validator nevertheless recomputes turnover and
cost directly from $w$.

## 8. Base Hard Constraints

### 8.1 Full Investment

$$
\sum_{i=1}^n w_i = B_{\mathrm{budget}}.
$$

The benchmark normally uses $B_{\mathrm{budget}}=1$.

### 8.2 Long-Only Asset Bounds

$$
\ell_i \le w_i \le u_i.
$$

The current implementation requires nonnegative lower bounds.

### 8.3 Group Exposure Bounds

$$
L_g
\le
\sum_{i=1}^n A_{gi}w_i
\le
U_g
\qquad
\forall g\in\mathcal G.
$$

Examples of groups include asset class, region, sector, or any mutually
exclusive grouping supplied by the data.

### 8.4 Optional Minimum Expected Return

$$
\mu^\top w \ge R_{\min}.
$$

### 8.5 Optional Maximum Turnover

$$
\sum_{i=1}^n t_i \le T_{\max}.
$$

## 9. Sparse-Support Constraints

### 9.1 Support Linkage

A selected asset may receive a positive weight, while an unselected asset must
receive zero:

$$
m_i z_i \le w_i \le \hat u_i z_i.
$$

Here,

$$
m_i=\max(\ell_i,m),
\qquad
\hat u_i=\min(u_i,\bar u_i)
$$

when the optional implementation cap $\bar u_i$ is provided.

### 9.2 Exact Cardinality

$$
\sum_{i=1}^n z_i = K.
$$

This fixes the number of positive-weight holdings.

### 9.3 Eligibility

For an ineligible asset $i$,

$$
z_i=0,
\qquad
w_i=0.
$$

### 9.4 Mandatory Holdings

For a mandatory asset $i$,

$$
z_i=1,
\qquad
w_i\ge\max(m,\ell_i).
$$

Assets with a strictly positive base lower bound are also treated as
non-removable by the allocation oracle.

## 10. Optional Financial Guardrails

These constraints are enabled only when the required data are supplied.

### 10.1 Minimum Income

$$
y^\top w \ge Y_{\min}.
$$

### 10.2 Factor Exposure Bands

Define

$$
f=B^\top w.
$$

Then require

$$
f_{\min}\le f\le f_{\max}.
$$

These bounds control systematic exposures such as equity beta, duration,
inflation sensitivity, style factors, or synthetic benchmark factors.

### 10.3 Stress-Scenario Floors

Let $r_s$ be the vector of asset returns under stress scenario $s$. Require

$$
r_s^\top w \ge q_s
\qquad
\forall s\in\mathcal S_{\mathrm{stress}}.
$$

A negative floor such as $q_s=-0.12$ means the portfolio must not lose more
than 12% in that scenario.

### 10.4 Empirical CVaR Limit

Let $R_s$ be the vector of asset returns in empirical scenario $s$. Portfolio
loss is

$$
L_s(w)=-R_s^\top w.
$$

For confidence level $\alpha\in(0,1)$, the linear epigraph is

$$
\eta
+
\frac{1}{(1-\alpha)S}
\sum_{s=1}^S u_s
\le C_{\max},
$$

subject to

$$
u_s \ge L_s(w)-\eta
=
-R_s^\top w-\eta,
$$

$$
u_s\ge0.
$$

The fixed-support optimizer and the Gurobi model use this epigraph. The
validator independently recomputes empirical CVaR from the returned weights.

## 11. Related Optimization Problems

### 11.1 Continuous Full-Universe Relaxation

The relaxation temporarily removes:

- exact cardinality;
- minimum active weight;
- binary support variables.

All remaining convex constraints stay active. Its optimum is a lower bound on
the exact-cardinality minimization problem because the relaxed feasible set is
larger.

### 11.2 Fixed-Support Allocation

Given a support $\mathcal H\subseteq\mathcal A$, all weights outside
$\mathcal H$ are fixed to zero. The remaining problem is a continuous convex
QP, possibly with linear CVaR auxiliary variables.

A globally optimal fixed-support allocation is not a global optimum over all
possible supports.

### 11.3 Exact-Cardinality MIQP

The direct sparse model contains continuous $w$, binary $z$, turnover epigraph
$t$, factor exposures, and all configured guardrails. It is a mixed-integer
quadratic program.

A time-limited solver may return a feasible incumbent and a best bound without
proving global optimality.

### 11.4 Equal-Lot Discrete Baseline

For legacy small-instance comparisons, divide the budget into $M$ equal lots:

$$
\delta=\frac{B_{\mathrm{budget}}}{M},
\qquad
w_i=\delta q_i,
\qquad
q_i\in\mathbb Z_{\ge0},
$$

$$
\sum_i q_i=M.
$$

Exact lot bounds are

$$
\left\lceil\frac{\ell_i}{\delta}\right\rceil
\le q_i \le
\left\lfloor\frac{u_i}{\delta}\right\rfloor.
$$

This model discretizes portfolio percentages. It is retained as a classical
baseline and is not the main continuous-weight hybrid formulation.

## 12. Convexity and Optimality Language

- The continuous model is convex when $\Sigma$ or $\Omega$ is positive
  semidefinite.
- A successful numerical solve is accepted only after independent feasibility
  checks.
- Exact enumeration proves the optimum only for the enumerated equal-lot model.
- Gurobi proves the exact-cardinality optimum only when it returns an optimal
  status within configured tolerances.
- A fixed-support QP may be optimal conditional on its support while the support
  itself remains heuristic.
- LNS and QAOA outputs are feasible incumbents unless a separate global solver
  certifies them.

## 13. Model Scope

The current model is single-period and long-only. It does not explicitly model:

- taxes or tax lots;
- nonlinear market impact;
- bid-ask spread dynamics;
- short selling or leverage;
- multi-period recourse;
- liabilities;
- parameter uncertainty;
- live execution or future investment performance.

Those extensions require new data, equations, validation rules, and benchmark
protocols.
