# VQE Model Formulation for Single-Period Discrete Portfolio Allocation

This document explains the sampling-based VQE (Variational Quantum
Eigensolver-style ansatz + PSO→NFT optimization) discrete portfolio
optimizer implemented in this project. Like
[quantum_model_formulation.md](quantum_model_formulation.md) (the QAOA/PCE
solver), it solves the *same* discrete lot-allocation problem defined in
[classical_model_formulation.md](classical_model_formulation.md), just
compiled down into a different quantum form. The goal here is to connect the
binary-encoding math, the adaptive-CVaR/PSO/NFT machinery, and the
hardware-native chain layout to the code in
[src/vanguard_portfolio/quantum_vqe_solver.py](src/vanguard_portfolio/quantum_vqe_solver.py),
with [run_quantum_vqe.py](src/vanguard_portfolio/run_quantum_vqe.py) as the
entry point that wires a `synthetic_universe.json` universe into
`PortfolioVQESolver.run()`.

This solver follows the sampling-VQA framework used for the multi-period
model in Haghighi 2026 (arXiv:2606.10098), adapted here to the single-period,
lot-based problem.

## 1. The problem in plain English

As with the QAOA solver, we start from the same discrete lot-allocation
problem as the classical discrete solver: split the budget into `n_lots`
units, assign each asset an integer lot count `k_i`, and require
`sum_i k_i == n_lots`. Where the two quantum solvers differ from each other
is in *how* the binary problem is searched:

- the QAOA/PCE solver compresses logical bits onto a small physical
  register via Pauli correlations and trains a shallow QAOA circuit with a
  continuous `tanh`-squashed proxy loss,
- the VQE solver instead gives every logical bit its **own physical qubit**,
  uses a hardware-native, one-repetition **HNDC-style ansatz**, and
  optimizes it by directly **sampling bitstrings** and scoring them against
  the real penalized cost function — no PCE compression, no `tanh` proxy.

Both share the same binary-expansion encoding of lot counts and the same
general shape of soft-constraint penalties; what changes is the circuit
architecture and the classical optimization loop wrapped around it.

## 2. Step 1 — encoding lot counts as bits

Exactly as in the QAOA formulation, a fixed-size binary register represents
each asset's lot count, but here the encoding is a plain (non-offset) binary
expansion rather than a floor-plus-offset one:

$$
k_i = \sum_b 2^b\, q_{i,b}, \qquad q_{i,b} \in \{0, 1\},
\qquad b = 0, \dots, \text{bits\_per\_asset} - 1
$$

with a single shared register width across all assets,

$$
\text{bits\_per\_asset} = \lceil \log_2(n_{\text{lots}} + 1) \rceil,
$$

so every asset gets the same number of qubits regardless of its own
feasible range. This is a deliberate simplification relative to the QAOA
solver's per-asset, range-aware `bits_i`: it costs more qubits in general
(an asset with a narrow feasible range still gets `bits_per_asset` qubits),
but it keeps the register width uniform, which matters for the fixed chain
layout described in Section 5. Per-asset feasible ranges are recovered
afterward via `_lot_bounds` (imported from `classical_discrete.py`) and
enforced as a *penalty*, not by shrinking the register — see Section 4.

Total qubit count is therefore

$$
n_{\text{qubits}} = n_{\text{assets}} \times \text{bits\_per\_asset},
$$

computed directly in `PortfolioVQESolver.__init__`. Decoding a measured
bitstring back to lot counts is a simple reshape-and-dot-product,
implemented in `decode_lots`.

## 3. Step 2 — why the budget constraint is soft here (and stricter than it sounds)

`classical_discrete.py`'s simulated annealer never actually needs to check
`sum_i k_i == n_lots`: its neighbourhood moves a lot from one donor asset to
one receiver asset, so the total is invariant by construction, and its
brute-force solver only enumerates stars-and-bars compositions that already
sum to `n_lots`. Neither classical path can produce an infeasible total.

A sampled bitstring from a gate-model circuit has no such guarantee — each
qubit is an independent binary degree of freedom, so there is no built-in
mechanism keeping a global sum fixed across measurement outcomes. The
budget constraint therefore has to be penalized after the fact, on every
sampled bitstring, in `total_cost`:

$$
\text{pen}_{\text{budget}} = \beta \Big(\textstyle\sum_i k_i - n_{\text{lots}}\Big)^2
$$

This penalty is doing more work than its QAOA-solver counterpart, because
the *uniform* register width from Section 2 means an individual asset's
`bits_per_asset`-bit register can represent lot counts larger than
`n_lots` even before you look at any other asset — so both the per-asset
bound and the cross-asset sum need penalizing, not just the sum. Per-asset
bounds (`lot_lo`, `lot_hi`, from `_lot_bounds`) are penalized the same
squared-violation way:

$$
\text{pen}_{\text{bounds}} = \beta \sum_i \Big(
  \max(0,\, k_i - \text{hi}_i)^2 + \max(0,\, \text{lo}_i - k_i)^2
\Big)
$$

## 4. Step 3 — group exposure as a two-sided penalty

`classical_discrete._sector_penalty` penalizes only `group_upper` violations
(one-sided, and it never even receives `group_lower`) — the same asymmetry
noted in the QAOA document. Since `synthetic_universe.json` supplies both
bounds, `_sector_penalty_two_sided` in this module penalizes both sides:

$$
\text{pen}_{\text{sector}} = \sum_g \Big(
  \max(0,\, e_g - \text{hi}_g)^2 + \max(0,\, \text{lo}_g - e_g)^2
\Big), \qquad e_g = \sum_{i \in g} w_i
$$

falling back to the one-sided `sector_limits` mapping if `group_upper_arr`
isn't present on the problem instance, to stay compatible with
`PortfolioProblem` objects that don't carry the two-sided JSON fields. As
with the QAOA solver, this makes the VQE solver's feasible region a strict
subset of what the classical validator checks — keep that in mind when
comparing utilities head-to-head.

## 5. Step 4 — total cost and penalty auto-calibration

The scalar objective handed to the optimizer, `total_cost`, is

$$
\text{cost}(\text{bits}) = -U(w) + \beta \cdot \big(
  \text{pen}_{\text{budget}} + \text{pen}_{\text{bounds}} + \text{pen}_{\text{sector}}
\big)
$$

where $U(w)$ is the *real* `PortfolioProblem.utility(w)` — unlike the QAOA
solver, there is no quadratic turnover surrogate here; the true utility
(including its $\ell_1$ turnover term) is evaluated directly on decoded
weights, since nothing about a sampling-based VQA requires the objective to
be a degree-2 polynomial the way a QUBO does.

The penalty weight $\beta$ is not a fixed constant — it is auto-calibrated
in `_calibrate_penalty` by drawing 500 random feasible-ish lot vectors,
averaging the magnitude of their utility, and scaling by a fixed multiplier
`ALPHA_C = 10.0`:

$$
\beta = \text{ALPHA\_C} \times \overline{|U(w)|}
$$

This mirrors the appendix-described auto-calibration approach from the
multi-period reference paper: it keeps the penalty large enough to dominate
infeasible solutions regardless of the utility scale of the specific
universe being solved, without requiring per-universe manual tuning.

## 6. Step 5 — the HNDC-style native chain layout

Rather than compressing qubits (PCE's approach), this solver spends one
physical qubit per logical bit but arranges *which* physical qubit gets
which logical bit to match the target backend's real coupling map
(`FakeMarrakesh`), so the two-qubit gates the ansatz needs are hardware-native
and don't require SWAP-network insertion at transpile time.

The construction, in `_build_chain_components` / `_build_link_edges` /
`_build_qubit_mapping`:

1. **Within-asset chains.** For each asset, its `bits_per_asset` bits are
   assigned to a *connected path* of physical qubits on the coupling graph
   (greedy path growth over unassigned neighbours). This is the same
   rationale as the multi-period model's "one chain per rebalancing
   period," just substituting "asset" for "period": the bits that jointly
   define one asset's lot count are the most strongly coupled variables
   (they set that asset's contribution to both the linear return term and
   the diagonal of the risk term), so they get the deepest, most-connected
   part of the circuit.
2. **Inter-asset link edges.** A disjoint matching of leftover native edges
   crossing between different assets' chains is collected separately
   (`_build_link_edges`), giving the ansatz a way to entangle *across*
   assets — needed because the objective's quadratic term $w^\top \Sigma w$
   and the sector penalties both couple different assets' bits — without
   requiring a fully connected qubit register.
3. **Logical↔physical permutation.** `_build_qubit_mapping` produces the
   `perm` / `inv_perm` arrays used to translate between "logical bit index
   `asset_i * bits_per_asset + bit_b`" and "physical qubit index on the
   backend," so `total_cost` and friends can stay written purely in terms of
   logical bits while sampling happens on physical ones.

## 7. Step 6 — the one-repetition ansatz

The circuit itself (`_build_ansatz`) is a single-repetition (`reps=1` by
default) hardware-efficient ansatz, structured as:

1. An initial layer of $R_y(\theta)$ rotations, one per qubit.
2. For each repetition:
   - **Deep-chain sublayers** — CZ gates walked step-by-step along every
     asset's chain in parallel (all chains advance together), entangling
     each asset's own bits with each other.
   - **Inter-asset linking sublayer** — CZ gates on the disjoint link edges
     from Section 6, entangling across assets.
   - Another layer of $R_y(\theta)$ rotations, one per qubit.

This gives $n_{\text{qubits}} \times (\text{reps} + 1)$ trainable parameters
total. The circuit is transpiled once against the real backend via
`generate_preset_pass_manager` (`optimization_level=1`) and reused for every
subsequent parameter binding, and sampling itself runs on an
`AerSimulator` configured with a matrix-product-state backend (bond
dimension capped at 64) rather than the noisy `FakeMarrakesh` model
directly — the coupling map is used for *layout*, not for injecting noise
into the simulated samples.

## 8. Step 7 — CVaR loss and the adaptive-α schedule (§3.1 analogue)

Rather than the QAOA solver's continuous `tanh`-proxy loss, this solver
optimizes a **Conditional Value-at-Risk (CVaR) loss** over sampled
bitstrings: draw a batch of shots, score every distinct bitstring with the
real penalized `total_cost`, sort, and average the best $\alpha$-fraction:

$$
\text{CVaR}_\alpha(\theta) = \frac{1}{\lceil \alpha N \rceil}
  \sum_{k=1}^{\lceil \alpha N \rceil} \text{cost}_{(k)}
$$

where $\text{cost}_{(1)} \le \text{cost}_{(2)} \le \dots$ are the sorted
per-shot costs and $N$ is the shot count. Small $\alpha$ focuses the
gradient signal on only the best-performing tail of the distribution
(useful early, when most samples are still far from feasible); $\alpha = 1$
reduces to the plain sample mean.

`adaptive_alpha` decreases $\alpha$ in discrete steps as optimization
progresses — every `l_alpha` coordinate updates, $\alpha$ drops by
`delta_alpha`, floored at `alpha_min` — and `shots_for_alpha` compensates by
increasing the shot count roughly in proportion to $1/\alpha$
($\lceil n_0 / \alpha \rceil$), so the *effective* number of samples
contributing to the CVaR average stays roughly stable even as $\alpha$
shrinks. This is the direct single-period analogue of the multi-period
paper's §3.1 adaptive-CVaR schedule.

## 9. Step 8 — two-stage optimization: PSO then NFT (§3.2 analogue)

Parameters are trained in two stages, both operating purely on sampled
CVaR-loss evaluations (no gradients):

**Stage 1 — Particle Swarm Optimization (`run_pso`).** A swarm of
`n_particles` particles explores the full parameter space at the coarsest
CVaR setting ($\alpha = \alpha_{\max}$), each particle's position updated by
the standard inertia/cognitive/social velocity rule. Every particle's
personal-best and the swarm's global-best cost are tracked; the loop runs
until either `max_pso_budget` evaluations are spent or a stagnation check
fires (relative improvement over the last `stagnation_window` iterations
falls below `stagnation_tol`, checked only once at least `min_pso_budget`
evaluations have been spent). This stage is responsible for **global
exploration** — escaping the many local minima a hardware-efficient ansatz's
non-convex landscape typically has.

**Stage 2 — Nakanishi-Fujii-Todorov (NFT) refinement (`run_nft`).** Starting
from the PSO global-best, parameters are refined one coordinate at a time.
For each coordinate $j$, three CVaR-loss evaluations at $\theta_j$,
$\theta_j + \pi/2$, and $\theta_j - \pi/2$ exactly determine the sinusoidal
dependence of the loss on that single parameter (a property of Pauli-rotation
ansätze), letting `_nft_step` jump directly to that coordinate's exact
minimum in closed form rather than taking a gradient step:

$$
A = \tfrac{1}{2}\sqrt{(f_+ - f_-)^2 + (2f_0 - f_+ - f_-)^2}, \qquad
\phi = \arctan2(f_+ - f_-,\; 2f_0 - f_+ - f_-)
$$

$$
\theta_j^{\text{new}} = (\phi + \pi) \bmod 2\pi
$$

with $\alpha$ (and thus shot count) continuing to shrink (grow) according to
the adaptive schedule as NFT sweeps through coordinates. This stage is
responsible for **local convergence** once PSO has found a good basin. The
overall evaluation budget is split so that whatever PSO doesn't spend of
`total_budget`, NFT gets (`nft_budget = total_budget - pso_evals`).

## 10. Step 9 — final sampling, bit-flip postprocessing, and decoding

Once NFT finishes, the trained parameters are sampled once more at a large,
fixed shot count (20,000) via `sample_and_evaluate`, and every distinct
bitstring observed is scored and sorted by `total_cost`.

The single best-cost bitstring (`bits_raw`) is one candidate output. A
second candidate is produced by **greedy bit-flip postprocessing**
(`bit_flip_postprocess`, the local-search analogue of the reference paper's
§4.1): starting from each of the top-10 sampled bitstrings, qubits are
visited in a random order and flipped one at a time, keeping the flip only
if it strictly improves `total_cost`; the best result across all 10
restarts (`bits_pp`) is kept. This catches the common case where sampling
noise leaves a otherwise-good bitstring one or two bit-flips away from a
better (or feasible) one — cheap to fix classically without spending more
quantum shots.

Both candidates are decoded back to lots, weights, and real utility via
`evaluate` (which also reports `budget_ok` / `bounds_ok` feasibility flags
and the raw `sector_penalty` value), and `run()` returns both the raw and
post-processed results side by side so the value of the postprocessing step
is directly visible.

## 11. Why this solver isn't directly comparable to either the classical or the QAOA/PCE solver out of the box

Three separate asymmetries stack up here, and any comparison across all
three solvers should be read with all of them in mind:

1. **Group exposure**, as in the QAOA solver: the classical discrete
   solver's sector handling is soft and one-sided; this solver's
   `_sector_penalty_two_sided` is two-sided.
2. **Register width**, unlike the QAOA solver: this solver uses a uniform
   `bits_per_asset` register for every asset (Section 2), rather than the
   QAOA solver's per-asset, range-aware bit count — so per-asset bound
   violations are possible here in a way they structurally aren't (as
   directly) on the QAOA side, and are handled purely by penalty rather than
   by construction.
3. **Objective form**, unlike the QAOA solver: this solver optimizes the
   real utility directly, including the true $\ell_1$ turnover cost — it
   does not use the QAOA solver's quadratic turnover surrogate, since a
   sampling-based CVaR loss has no need to stay degree-2. This means the
   VQE solver's reported utility and the QAOA solver's *final* reported
   utility (Section 6 of the QAOA doc, post-decoding) are computed the same
   way, but the QAOA solver's *training* loss is not — only the VQE
   solver's training loss and final utility fully agree with each other.

## 12. Suggested reading order for this part of the repository

1. Read [classical_model_formulation.md](classical_model_formulation.md)
   first, for the continuous and discrete objective this module builds on.
2. Read [quantum_model_formulation.md](quantum_model_formulation.md) for the
   QAOA/PCE solver, to see where this solver's design choices diverge.
3. Read `load_universe_from_json` in `quantum_vqe_solver.py` to see how a
   `synthetic_universe.json` universe becomes a `PortfolioProblem`, plus the
   extra two-sided group fields stashed onto the instance.
4. Read `PortfolioVQESolver.__init__`, `_calibrate_penalty`, and
   `total_cost` to see the binary encoding, auto-calibrated penalty weight,
   and full soft-constraint cost function.
5. Read `_build_chain_components`, `_build_link_edges`, and
   `_build_qubit_mapping` for the hardware-native chain layout, then
   `_build_ansatz` for the HNDC-1 circuit itself.
6. Read `cvar_loss`, `adaptive_alpha`, and `shots_for_alpha` for the
   adaptive-CVaR sampling loss.
7. Read `run_pso` and `run_nft` / `_nft_step` for the two-stage
   PSO→NFT optimization loop.
8. Read `bit_flip_postprocess` and `PortfolioVQESolver.run` for how a final
   allocation is chosen and refined, and `run_vqe` /
   [run_quantum_vqe.py](src/vanguard_portfolio/run_quantum_vqe.py) for the
   top-level entry point that ties it to a JSON universe file.

## 13. Recommended resources

### Sampling-based VQAs and CVaR optimization

- Haghighi, "Multi-Period Portfolio Optimization with a Sampling-Based
  Variational Quantum Algorithm" (2026, arXiv:2606.10098) — the direct
  reference this module's §3.1/§3.2/§3.4.4/§4.1 section numbers are drawn
  from.
- Barkoutsos et al., "Improving Variational Quantum Optimization using
  CVaR" (2020)

### Optimizers used here

- Nakanishi, Fujii, Todorov, "Sequential minimal optimization for
  quantum-classical hybrid algorithms" (NFT, 2020)
- Kennedy & Eberhart, "Particle Swarm Optimization" (1995)

### Hardware-native circuit layout

- Qiskit documentation on `generate_preset_pass_manager`, coupling maps, and
  the `FakeMarrakesh` backend

## 14. Summary

The VQE model in this project solves the *same* single-period discrete
lot-allocation problem as the classical and QAOA/PCE solvers, but reaches it
through a different chain of design choices: integer lots become bits via a
uniform-width binary expansion; the true utility (no surrogate needed) plus
squared-violation penalties for budget, per-asset bounds, and two-sided
sector exposure form a single scalar cost function with an auto-calibrated
penalty weight; qubits are laid out one-per-bit along hardware-native chains
matched to the target backend's real coupling map instead of being
compressed; a shallow, one-repetition ansatz is trained by directly sampling
bitstrings and minimizing an adaptively-shrinking CVaR loss, first globally
via PSO and then coordinate-wise via closed-form NFT updates; and the best
sampled bitstring is locally refined by classical greedy bit-flipping before
being decoded back into lots, weights, and a final, real-objective utility
score.