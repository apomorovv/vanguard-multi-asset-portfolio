# Quantum Model Formulation for Multi-Asset Portfolio Construction

This document explains the quantum (QAOA + Pauli Correlation Encoding) discrete
portfolio optimizer implemented in this project. It picks up where
[classical_model_formulation.md](classical_model_formulation.md) leaves off: the
discrete lot-allocation problem defined there is the *same* problem this module
solves, just compiled down into a form a QAOA circuit can actually run. The goal
here is to connect the QUBO math and the PCE/iterative-alpha machinery to the
code in [src/vanguard_portfolio/quantum_discrete.py](src/vanguard_portfolio/quantum_discrete.py).

## 1. The problem in plain English

We start from exactly the same discrete lot-allocation problem as the classical
discrete solver: split the budget into `n_lots` units, give each asset an
integer lot count `k_i`, and require `sum_i k_i == n_lots`. The difference is
*how* the solver searches for the best allocation:

- the classical discrete solver enumerates or anneals over integer lot
  vectors directly,
- the quantum solver rewrites the same problem as a **binary optimization
  problem (QUBO)** and searches for a good bitstring using a **QAOA circuit**,
  with the circuit compressed via **Pauli Correlation Encoding (PCE)** so the
  number of physical qubits needed stays small even as the number of decision
  bits grows.

Getting from "portfolio weights" to "something a quantum circuit can act on"
requires several translation steps. This document walks through each one.

## 2. Step 1 — encoding lot counts as bits

A QUBO can only contain binary variables, so the first job is to write each
asset's integer lot count as a small number of bits instead of one variable
per lot. This is a standard **binary expansion**:

$$
k_i = \text{lo}_i + \sum_b 2^b x_{i,b}, \qquad x_{i,b} \in \{0, 1\}
$$

where $\text{lo}_i$ is the asset's minimum feasible lot count (from its
lower bound) and each bit $x_{i,b}$ adds a power-of-two offset on top of that
floor. The number of bits needed for asset $i$ is

$$
\text{bits}_i = \lceil \log_2(\text{hi}_i - \text{lo}_i + 1) \rceil,
$$

which grows *logarithmically* with the asset's feasible lot range rather than
linearly — a 21-value range needs 5 bits, not 21. This is implemented in
`build_lot_encoding` / `_bits_needed` / `LotEncoding` in
`quantum_discrete.py`.

The corresponding weight is:

$$
w_i = \frac{\text{budget}}{n_{\text{lots}}} k_i
$$

exactly as in the classical discrete formulation.

## 3. Step 2 — rewriting the utility as a QUBO

The real objective, from `PortfolioProblem.utility()`, is

$$
U(w) = \mu^\top w - \frac{\gamma}{2} w^\top \Sigma w
       - \lambda_c \, c^\top |w - w_{\text{prev}}|
$$

QAOA/PCE can only optimize **degree-2 polynomials in binary variables**
(constant + linear + pairwise terms) — nothing higher-order, and no exact
absolute value. Two changes are made to fit this form:

**Turnover surrogate.** The true turnover cost uses $|w_i - w_{\text{prev},i}|$
(L1), which is not degree-2 in the lot bits without extra auxiliary "sign"
qubits per asset. This module substitutes a quadratic surrogate,
$(w_i - w_{\text{prev},i})^2$, which is convex, zero at $w_{\text{prev},i}$,
and expands cleanly into the QUBO. This is flagged explicitly in the
module docstring rather than hidden; it means the **QAOA loss function** and
the **final reported utility** are not identical quantities — see Section 6.

**Substitution and expansion.** With the surrogate in place, $U(w)$ becomes a
quadratic function of $w$:

$$
U(w) = L^\top w + w^\top Q w + \text{const}_0
$$

with

$$
L = \mu + 2\lambda_c\, c \odot w_{\text{prev}}, \qquad
Q = -\tfrac{\gamma}{2}\Sigma - \lambda_c\,\text{diag}(c)
$$

Substituting $w_i = \text{const}\_{w,i} + \text{lot\_size}\sum_b 2^b x_{i,b}$
(where $\text{const}_{w,i} = \text{lot\_size} \cdot \text{lo}_i$) and expanding
term by term produces a polynomial purely in the bits $x_{i,b}$, split into
three buckets:

- **constant** — terms with no bit dependence (e.g. the fixed contribution
  from each asset's lot floor),
- **linear** — terms depending on exactly one bit,
- **quadratic** — terms depending on a pair of bits (both within one asset's
  own bits, via $w_i^2$, and across assets, via $w_i w_j$ / covariance).

One extra simplification is used during expansion: because $x \in \{0,1\}$,
$x^2 = x$, so every squared-bit term collapses from quadratic into linear.
This is implemented in `build_lot_qubo`, and evaluated back into a number via
`qubo_utility`.

## 4. Step 3 — hard constraints as a weighted generalization

The classical discrete solver treats sector/group exposure limits as a
**soft, one-sided penalty** (see `_sector_penalty` in
`classical_discrete.py`): it only penalizes exceeding an upper bound, and
only discourages it rather than forbidding it.

The quantum solver instead enforces constraints as **hard**, **two-sided**
requirements on weighted sums of bits, built in `build_constraints`:

1. **Budget / full investment** — total lots must equal `n_lots` exactly
   (an equality constraint over all bits, weighted by their power-of-two
   value).
2. **Per-asset bounds** — each asset's own offset bits must stay within its
   valid binary-expansion range (guards against "overshoot" when the range
   isn't an exact power of two minus one).
3. **Group exposure** — each group's total lots must fall within
   `[group_lower, group_upper]` (both sides), computed from the group's
   weight-fraction bounds and lot-quantized.

These are a genuine generalization of a cardinality constraint (count of
1-bits in a scope) to a **weighted sum** constraint (since binary-expansion
bits carry weights $2^b$, not weight 1). When every scope weight is 1, the
formulation reduces exactly to a plain cardinality constraint.

Because these constraints aren't hard-wired into the search the way
`donors`/`receivers` bounds are in the classical annealer, they instead enter
the QAOA loss as **penalty terms** (see Section 6), scaled by a per-constraint
`beta` computed in `beta_for_constraint` from how strongly the objective
touches that constraint's variables (`variable_reach`).

## 5. Step 4 — Pauli Correlation Encoding (PCE)

A QAOA ansatz needs one qubit per logical variable in the most naive
encoding. With dozens of lot-encoding bits, that is wasteful. PCE instead
represents each logical bit as a **pairwise Pauli correlation** between two
physical qubits, rather than the state of one dedicated qubit.

For a pair of qubits $(a, b)$ and a Pauli operator $P \in \{X, Y, Z\}$,
$\langle P_a P_b \rangle$ is a single real number in $[-1, 1]$ describing how
strongly $a$ and $b$ agree or disagree in that basis — not the state of
either qubit individually. Different Pauli choices on the *same* pair of
qubits give independent numbers, so one pair of qubits can carry up to three
separate logical variables (one per Pauli type).

With $k$ physical qubits, the number of available correlation "slots" is

$$
3 \binom{k}{2} = \frac{3k(k-1)}{2},
$$

so the number of representable logical variables grows **quadratically**
with qubit count instead of linearly. `reduce_qubits_with_pce` inverts this
relationship to find the smallest $k$ that can host a given number of logical
bits, and `build_pauli_correlation_encoding` assigns each logical bit to a
specific `(pair, Pauli)` combination, split across three groups (X, Y, Z) in
`solve()`.

## 6. Step 5 — the continuous proxy and the QAOA loss

QAOA parameters are trained by gradient-free classical optimization
(COBYLA) on a **smooth, continuous proxy** of the objective — not on
collapsed 0/1 outcomes. For each logical variable $i$, its correlation
expectation $\langle P_i \rangle$ is squashed through

$$
z_i = \tanh(\alpha \langle P_i \rangle), \qquad x_i = \frac{1 - z_i}{2}
$$

so $z_i \to -1$ means "bit = 1" and $z_i \to +1$ means "bit = 0." Both the
QUBO objective and the constraint penalties are evaluated on these $z_i$
proxies rather than hard bits, keeping the whole loss landscape
differentiable-in-spirit for the optimizer:

$$
\text{loss} = -\Big(\text{utility proxy from } \tanh(\alpha\langle P\rangle)
\text{ terms}\Big) + \beta_{\text{reg}} \cdot (\text{binarization regularizer})
+ \sum_{\text{constraints}} \text{penalty}
$$

This is implemented in `loss_func_estimator` and `constraint_penalty`. Note
that this loss is a *proxy* for utility (using the quadratic turnover
surrogate and continuous correlation values); the **final reported utility**
in the results dictionary (`utility` vs. `utility_qubo_proxy`) is recomputed
from the *real* `PortfolioProblem.utility(w)` after decoding actual weights,
so the headline comparison against the classical solver is judged by the
true objective, not the proxy.

## 7. Step 6 — iterative alpha and binarization

Early on, correlation values sit near 0 — genuinely undecided. A small
$\alpha$ keeps $\tanh(\alpha \langle P \rangle)$ smooth so the optimizer can
explore. A large $\alpha$ immediately snaps everything toward $\pm 1$, but at
the cost of vanishing gradients (the curve saturates almost everywhere) and
premature commitment before the joint solution has had a chance to settle.

`_run_iterative_alpha_pce` resolves this by **growing $\alpha$ gradually**:

1. Optimize the QAOA parameters at the current $\alpha$ (inner loop, COBYLA).
2. Check which logical variables are still "unbinarized"
   ($|\tanh(\alpha \langle P_i \rangle)| < \text{threshold\_m}$, default
   0.975).
3. Pick the single variable closest to that threshold and grow $\alpha$ by
   just enough to push it over — capped by `max_alpha_growth_per_step` so no
   single step is too aggressive.
4. Repeat until every variable is binarized or `max_alpha_iters` is reached.

The code also tracks which variable gets picked as the "pivot" most often
(`pivot_history`); if one variable dominates, it's flagged as a likely
**structural bottleneck** — a sign that its assigned qubit pair is shared
with, or entangled with, other variables in a way that makes it hard to push
to an extreme value independently.

## 8. Step 7 — decoding and multi-restart

Once $\alpha$-growth finishes (or hits its iteration cap), the final
correlation values are converted to hard bits (`decode`), and then to lot
counts and weights (`LotEncoding.decode_lots`). Because PCE's sign convention
is ambiguous going in, `best_decoding` tries both polarities and keeps
whichever satisfies more hard constraints (ties broken by higher utility).

The whole process — random parameter initialization, iterative-alpha
optimization, decoding — is repeated across `n_restarts` independent runs
with different random seeds (`solve()`'s restart loop), and the best result
across restarts (by constraints satisfied, then utility) is returned. This
mirrors the multi-start structure used to guard against QAOA's non-convex
loss landscape and its sensitivity to initialization.

## 9. Why the classical and quantum solvers aren't directly comparable out of the box

As covered in the module docstring and confirmed by reading both solvers'
code: the classical discrete solver's group-exposure handling
(`_sector_penalty` in `classical_discrete.py`) is **soft and one-sided**
(upper bound only, never forbidding a violation, and never even receiving
`group_lower`), while the quantum solver enforces **hard, two-sided** group
bounds by construction. This means the two solvers are searching different
feasible regions, not just using different search strategies over the same
one — any utility comparison between them should be read with that
asymmetry in mind (see Section 10 for how to align them).

## 10. Suggested reading order for this part of the repository

1. Read [classical_model_formulation.md](classical_model_formulation.md)
   first, for the continuous and discrete objective this module builds on.
2. Read `build_lot_encoding` / `LotEncoding` in `quantum_discrete.py` to see
   the binary lot expansion.
3. Read `build_lot_qubo` to see how the utility becomes a
   constant/linear/quadratic polynomial in bits.
4. Read `build_constraints` to see how budget, per-asset, and group bounds
   become weighted hard constraints.
5. Read `reduce_qubits_with_pce` and `build_pauli_correlation_encoding` for
   the qubit-compression step.
6. Read `loss_func_estimator`, `constraint_penalty`, and
   `_run_iterative_alpha_pce` for how the continuous proxy is optimized and
   sharpened into a bitstring.
7. Read `decode`, `best_decoding`, and `QuantumMeanVarianceDiscreteOptimizer.solve`
   for how a final allocation is chosen across restarts.

## 11. Recommended resources

### QAOA and variational quantum optimization

- Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm"
  (2014)
- Qiskit documentation on `QAOAAnsatz` and variational estimators

### Pauli Correlation Encoding

- Sciorilli et al., "Pauli Correlation Encoding" — *Nature Communications*
  (2025)
- The Iterative-alpha PCE hard-constraint procedure referenced in the module
  docstring (arXiv:2602.17479, "Pauli Correlation Encoding for
  Budget-Constrained Optimization")

### QUBO / binary optimization background

- Glover, Kochenberger, Du, "Quantum Bridge Analytics I: A Tutorial on
  Formulating and Using QUBO Models"

## 12. Summary

The quantum model in this project solves the *same* discrete lot-allocation
problem as the classical discrete solver, but reaches it through a chain of
translations: integer lots become bits via binary expansion; the true
utility (with a quadratic turnover surrogate) becomes a QUBO in those bits;
hard budget/asset/group constraints become weighted penalty terms; the bits
themselves are compressed from one-qubit-each down to pairwise Pauli
correlations on a much smaller physical register via PCE; and those
correlations are optimized as a smooth continuous proxy, gradually sharpened
into hard 0/1 values via iterative alpha growth, before being decoded back
into a concrete lot allocation and re-scored against the real objective.