# Classical Model Formulation for Multi-Asset Portfolio Construction

This document explains the classical portfolio optimization model implemented in this project in a graduate-level but accessible way. The goal is to connect the mathematical formulation to the code in the repository and to provide a roadmap for learning the underlying ideas.

## 1. The problem in plain English

We want to choose how much capital to allocate to each asset in a portfolio. The allocation should:

- earn a strong expected return,
- avoid taking too much risk,
- avoid excessive trading costs, and
- satisfy portfolio constraints such as budget, bounds, and sector exposure limits.

This is the classical mean-variance portfolio optimization problem, originally developed in modern portfolio theory.

## 2. Mathematical formulation

Let:

- $w \in \mathbb{R}^n$ be the vector of portfolio weights,
- $\mu$ be the vector of expected returns,
- $\Sigma$ be the covariance matrix of asset returns,
- $\gamma > 0$ be the risk-aversion parameter,
- $c$ be the vector of linear transaction costs,
- $w_{\text{prev}}$ be the previous portfolio allocation.

The objective is to maximize the utility

$$
U(w) = \mu^\top w - \frac{\gamma}{2} w^\top \Sigma w - \lambda_c \, c^\top |w - w_{\text{prev}}|
$$

where:

- $\mu^\top w$ rewards expected return,
- $w^\top \Sigma w$ penalizes portfolio variance (risk),
- the turnover-cost term discourages large deviations from the previous portfolio.

The optimization is subject to constraints such as:

$$
\sum_{i=1}^n w_i = \text{budget},
$$

$$
l_i \le w_i \le u_i \quad \text{for each asset } i,
$$

and optionally:

- a minimum target return,
- sector exposure limits,
- other practical portfolio rules.

## 3. Interpretation of each term

### Expected return term

The term $\mu^\top w$ measures the portfolio’s expected gain. If an asset has a high expected return and receives a larger weight, the portfolio’s expected return increases.

### Risk term

The term $w^\top \Sigma w$ measures portfolio variance. Covariance captures how assets move together. Two assets can both have high returns individually, but if they move together strongly, combining them may not diversify risk effectively.

### Transaction cost term

The term $c^\top |w - w_{\text{prev}}|$ penalizes changing the portfolio too much. This is useful when the portfolio is being rebalanced and trading costs matter.

## 4. Continuous-weight formulation

In the continuous model, weights are allowed to be any real number between lower and upper bounds. This corresponds to the classical Markowitz formulation.

The implementation in [src/vanguard_portfolio/classical_continuous.py](src/vanguard_portfolio/classical_continuous.py) does the following:

1. Stores the expected-return vector and covariance matrix.
2. Defines the utility function.
3. Builds constraints for the budget and any additional rules.
4. Uses SciPy’s SLSQP optimizer to solve the constrained nonlinear problem.

In other words, the code searches for the best continuous portfolio weights that maximize utility while satisfying the constraints.

## 5. Discrete-weight formulation

In the discrete model, the portfolio is not allowed to use arbitrary real-valued weights. Instead, the total budget is divided into a fixed number of discrete lots. If there are $n_{\text{lots}}$ lots, then each asset $i$ receives an integer number of lots $k_i$ such that:

$$
\sum_i k_i = n_{\text{lots}},
$$

and the corresponding weight is

$$
w_i = \frac{\text{budget}}{n_{\text{lots}}} k_i.
$$

This creates a discrete lattice of feasible portfolios. It is useful because it is closer to the integer or binary decision variables that appear in later quantum/QUBO formulations.

The implementation in [src/vanguard_portfolio/classical_discrete.py](src/vanguard_portfolio/classical_discrete.py) uses either:

- exhaustive search (the `brute` method), or
- simulated annealing (the `anneal` method).

## 6. Why the two formulations are both useful

The continuous model is mathematically elegant and gives the classical benchmark. The discrete model is more practical for settings where allocations must be made in discrete units, or where the later quantum formulation requires binary/integer decisions.

Together they provide a useful comparison:

- continuous optimization gives the idealized optimum,
- discrete optimization shows the cost of restricting allocations to a lattice.

## 7. How the code maps to the math

The core methods in the code correspond closely to the mathematics:

- `expected_return(w)` computes $\mu^\top w$
- `variance(w)` computes $w^\top \Sigma w$
- `turnover(w)` computes $\sum_i |w_i - w_{\text{prev},i}|$
- `cost(w)` computes $c^\top |w - w_{\text{prev}}|$
- `utility(w)` combines them into the objective

The optimizer then searches for the vector $w$ that maximizes this utility under the constraints.

## 8. A useful way to think about it

A portfolio optimizer is essentially answering this question:

> Among all portfolios that satisfy my constraints, which one gives the best tradeoff between reward and risk?

The continuous version answers that with real-valued weights. The discrete version answers a slightly more constrained version of the same question.

## 9. Recommended learning resources

### Portfolio theory

- Harry Markowitz, “Portfolio Selection” (1952)
- Bodie, Kane, and Marcus, “Investments”
- Elton, Gruber, Brown, and Goetzmann, “Modern Portfolio Theory and Investment Analysis”

### Optimization

- Stephen Boyd and Lieven Vandenberghe, “Convex Optimization”
- SciPy documentation for `scipy.optimize.minimize` and `SLSQP`

### Discrete and combinatorial optimization

- Any introductory text on integer programming or combinatorial optimization
- Materials on QUBO/Ising formulations, since these are closely related to the discrete portfolio formulation in this project

## 10. Suggested reading order for this repository

1. Read [src/vanguard_portfolio/classical_continuous.py](src/vanguard_portfolio/classical_continuous.py) to understand the continuous objective and constraints.
2. Read [src/vanguard_portfolio/classical_discrete.py](src/vanguard_portfolio/classical_discrete.py) to see how the continuous problem becomes a discrete one.
3. Review the tests in [tests/test_classical_continuous.py](tests/test_classical_continuous.py) and [tests/test_classical_discrete.py](tests/test_classical_discrete.py) to see the expected behavior.

## 11. Summary

The classical model in this project is a constrained mean-variance portfolio optimization problem. The continuous version uses real-valued weights and a nonlinear constrained solver, while the discrete version uses integer lots and is more aligned with later quantum or QUBO-based formulations.
