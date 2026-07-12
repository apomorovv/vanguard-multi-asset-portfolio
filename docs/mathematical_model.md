# Multi-Asset Portfolio Construction Model

## 1. Purpose

The optimization problem recommends a target allocation across multiple asset classes. The portfolio should balance expected return, risk, income, and implementation cost while satisfying all mandatory investment guardrails.

The same financial model will be solved using:

1. A continuous classical optimizer.
2. A discrete classical optimizer.
3. A QUBO-compatible binary optimizer.
4. A quantum or quantum-inspired optimizer.

This separation ensures that differences between solutions arise from the optimization algorithms rather than from inconsistent problem definitions.

---

## 2. Sets

Let

$$
\mathcal A={1,\ldots,n}
$$

be the set of asset classes.

Example asset classes include:

* US equity
* International equity
* Government bonds
* Corporate bonds
* Commodities
* Cash

Let

$$
\mathcal G={1,\ldots,G}
$$

be the set of asset groups, such as equity, fixed income, alternatives, and cash.

---

## 3. Input parameters

For every asset $$(i\in\mathcal A)$$
define:

$$
\mu_i
$$

as the annual expected return,

$$
\sigma_i
$$


as the annual volatility,

$$
c_i
$$

as the transaction-cost coefficient,

$$
y_i
$$

as the annual income yield,

$$
w_i^{(0)}
$$

as the current portfolio allocation,

$$
l_i
$$

as the minimum permitted allocation, and

$$
u_i
$$

as the maximum permitted allocation.

Let

$$
\Sigma\in\mathbb R^{n\times n}
$$

be the annual covariance matrix of asset returns.

The covariance matrix is related to volatility and correlation by

$$
\Sigma_{ij}=\rho_{ij}\sigma_i\sigma_j.
$$

For each group (g), define

$$
a_{gi}=
\begin{cases}
1,&\text{if asset }i\text{ belongs to group }g,\
0,&\text{otherwise},
\end{cases}
$$

and group allocation limits

$$
L_g,\qquad U_g.
$$

---

## 4. Continuous decision variables

Let

$$
w_i\in\mathbb R
$$

be the target allocation to asset (i).

Let

$$
t_i\geq 0
$$

represent the absolute allocation change for asset (i).

The vector of target allocations is

$$
\mathbf w=(w_1,\ldots,w_n)^T.
$$

---

## 5. Portfolio quantities

### Expected return

$$
R(\mathbf w)=\boldsymbol{\mu}^T\mathbf w.
$$

### Variance

$$
V(\mathbf w)=\mathbf w^T\Sigma\mathbf w.
$$

### Volatility

$$
\sigma_p(\mathbf w) = \sqrt{\mathbf w^T\Sigma\mathbf w}.
$$

### Income yield

$$
Y(\mathbf w)=\mathbf y^T\mathbf w.
$$

### Turnover

$$
T(\mathbf w)
=

\sum_{i=1}^{n}
\left|w_i-w_i^{(0)}\right|.
$$

### Estimated transaction cost

$$
C(\mathbf w)
=

\sum_{i=1}^{n}
c_i
\left|w_i-w_i^{(0)}\right|.
$$

---

## 6. Continuous classical model

The initial classical optimization problem is

$$
\boxed{
\begin{aligned}
\min_{\mathbf w,\mathbf t}\quad
&
\lambda_{\mathrm{risk}}
\mathbf w^T\Sigma\mathbf w
-

\lambda_{\mathrm{return}}
\boldsymbol{\mu}^T\mathbf w
-

\lambda_{\mathrm{income}}
\mathbf y^T\mathbf w
+
\lambda_{\mathrm{cost}}
\sum_{i=1}^{n}c_it_i
[3pt]
\\
\text{subject to}\quad
&
\sum_{i=1}^{n}w_i=1,
\\
&
l_i\leq w_i\leq u_i,
\qquad i\in\mathcal A,
\\
&
L_g
\leq
\sum_{i=1}^{n}a_{gi}w_i
\leq
U_g,
\qquad g\in\mathcal G,
\\
&
t_i\geq w_i-w_i^{(0)},
\qquad i\in\mathcal A,
\\
&
t_i\geq w_i^{(0)}-w_i,
\qquad i\in\mathcal A,
\\
&
t_i\geq0,
\qquad i\in\mathcal A.
\end{aligned}
}
$$

The auxiliary variables (t_i) produce

$$
t_i=\left|w_i-w_i^{(0)}\right|
$$

at the optimum whenever transaction costs have positive weights.

The coefficients

$$
\lambda_{\mathrm{risk}},
\lambda_{\mathrm{return}},
\lambda_{\mathrm{income}},
\lambda_{\mathrm{cost}}
$$

control the portfolio preferences.

The budget, allocation bounds, and group exposure limits are hard constraints. Return, risk, income, and implementation cost are competing objectives.

---

## 7. Initial baseline model

The first implemented optimizer will omit the optional income preference:

$$
\lambda_{\mathrm{income}}=0.
$$

The baseline problem is therefore

$$
\boxed{
\begin{aligned}
\min_{\mathbf w,\mathbf t}\quad
&
\lambda_{\mathrm{risk}}
\mathbf w^T\Sigma\mathbf w
-

\lambda_{\mathrm{return}}
\boldsymbol{\mu}^T\mathbf w
+
\lambda_{\mathrm{cost}}
\sum_i c_it_i
\\
\text{subject to}\quad
&
\mathbf 1^T\mathbf w=1,
\\
&
\mathbf l\leq\mathbf w\leq\mathbf u,
\\
&
L_g\leq\sum_i a_{gi}w_i\leq U_g,
\\
&
t_i\geq w_i-w_i^{(0)},
\\
&
t_i\geq w_i^{(0)}-w_i,
\\
&
t_i\geq0.
\end{aligned}
}
$$

This is a convex quadratic program when the covariance matrix is positive semidefinite.

---

## 8. Discrete portfolio model

Divide the total portfolio into (M) allocation units.

Let

$$
q_i\in{0,1,\ldots,M}
$$

be the number of units allocated to asset (i).

The portfolio weight is

$$
w_i=\frac{q_i}{M}.
$$

The full-investment constraint becomes

$$
\sum_iq_i=M.
$$

Asset bounds become

$$
\left\lceil Ml_i\right\rceil
\leq
q_i
\leq
\left\lfloor Mu_i\right\rfloor.
$$

Group constraints become

$$
\left\lceil ML_g\right\rceil
\leq
\sum_i a_{gi}q_i
\leq
\left\lfloor MU_g\right\rfloor.
$$

The initial small experiment will use

$$
M=10,
$$

so one allocation unit represents ten percent of the portfolio.

The discrete model will initially be solved exactly by enumeration or by a mixed-integer quadratic optimizer. This exact discrete result will serve as the reference for the QUBO and quantum solutions.

---

## 9. Binary representation

The integer allocation will later be represented using binary variables.

For binary expansion,

$$
q_i=\sum_{k=0}^{K-1}2^k x_{ik},
\qquad
x_{ik}\in{0,1}.
$$

Alternatively, one-hot encoding can be used:

$$
q_i=\sum_{m=0}^{M}m x_{im},
$$

subject to

$$
\sum_{m=0}^{M}x_{im}=1.
$$

One-hot encoding is easier to validate, while binary expansion requires fewer binary variables.

The first QUBO experiment will use one-hot encoding on a small portfolio. Later scaling experiments will compare both encodings.

---

## 10. Hard and soft requirements

### Hard constraints

Hard constraints must never be violated:

* Full investment
* Asset minimum and maximum allocations
* Asset-group exposure limits
* Nonnegative long-only allocations
* Any explicitly required liquidity or cash allocation

A solution with a hard-constraint violation is infeasible regardless of its objective value.

### Soft objectives

Soft objectives may trade against each other:

* Expected return
* Risk
* Income
* Turnover
* Transaction cost
* Stress-scenario performance
* Diversification

---

## 11. Required output metrics

Every solver will be evaluated using the same metrics:

$$
R(\mathbf w)
=

\boldsymbol{\mu}^T\mathbf w,
$$

$$
V(\mathbf w)
=

\mathbf w^T\Sigma\mathbf w,
$$

$$
\sigma_p(\mathbf w)
=

\sqrt{\mathbf w^T\Sigma\mathbf w},
$$

$$
T(\mathbf w)
=

\sum_i|w_i-w_i^{(0)}|,
$$

$$
C(\mathbf w)
=

\sum_i c_i|w_i-w_i^{(0)}|,
$$

together with:

* Objective value
* Runtime
* Number of hard-constraint breaches
* Maximum constraint violation
* Distance from the exact discrete optimum
* Fraction of feasible samples
* Allocation difference from the continuous baseline

---

## 12. Development sequence

The project will be implemented in the following order:

1. Generate and validate synthetic asset data.
2. Implement portfolio metric calculations.
3. Solve the continuous quadratic program.
4. Solve a tiny discrete problem exactly.
5. Encode the discrete model as a QUBO.
6. Verify that QUBO energy matches the direct objective.
7. Run classical binary optimizers.
8. Run quantum or quantum-inspired optimizers.
9. Decode and validate all candidate portfolios.
10. Expose portfolio preferences through the co-pilot interface.


