"""
quantum_vqe_pce_solver.py
=========================
Combines Pauli Correlation Encoding (PCE, Sciorilli et al., Nat. Commun.
2025) with the sampling-based VQA framework from `quantum_vqe_solver.py`
(adaptive CVaR schedule + two-stage PSO->NFT optimizer), on the *same*
single-period discrete portfolio problem those two modules already solve.

This is "Option B" from the earlier discussion: keep the CVaR-over-sampled-
bitstrings training loop exactly as `quantum_vqe_solver.py` already does it,
rather than switching to expectation-value (Estimator-based) training like
`quantum_discrete.py`'s QAOA+PCE module. The only thing that changes is
*how a logical lot-bit gets its value out of a shot* -- everything about
PSO, NFT, the adaptive-alpha CVaR schedule, and bit-flip postprocessing is
reused unmodified in spirit.

------------------------------------------------------------------------
Why this needs three separate circuit executions per evaluation
------------------------------------------------------------------------
PCE encodes each binary variable x_i as the SIGN of the expectation value
of a two-qubit Pauli correlator (XX, YY, or ZZ) measured on a small,
O(sqrt(n_vars))-qubit register -- see `quantum_discrete.py`'s
`build_pauli_correlation_encoding` for the expectation-value version of
this. A single projective measurement only reveals ONE Pauli basis at a
time: you can't read off an XX outcome and a ZZ outcome from the same
shot without rotating the circuit differently before each measurement.

So, exactly as in the original PCE paper (three mutually-commuting Pauli-
string subsets => three measurement settings, regardless of how many
correlators there are), this solver runs the SAME parameterized ansatz
three times per evaluation:

    * Z-setting: measure computational basis directly       -> ZZ group
    * X-setting: apply H to every qubit, then measure        -> XX group
    * Y-setting: apply S-dagger then H to every qubit        -> YY group

For a two-qubit correlator, the measured eigenvalue is (-1)^(b_a XOR b_b)
where b_a, b_b are the two qubits' measured bits in that setting -- so a
variable's decoded bit is simply `int(b_a != b_b)` (matching the same
"z<0 -> bit=1" sign convention `quantum_discrete.py`'s `decode()` uses,
just realized per-shot instead of via an expectation value).

>>> Flagged approximation: cross-group pairing <<<
A "candidate solution" needs values for ALL variables (X-group + Y-group +
Z-group) at once, but each group's bits only come from its OWN measurement
setting's shots. This solver forms one combined candidate per index i by
pairing the i-th (independently shuffled) shot from each of the three
settings. Since the three settings are three different measurements of
the same fixed variational state, this is a factorized / independent-
marginals approximation of the joint distribution over all n_vars
variables -- it does not capture correlations between an X-group variable
and a Y-group variable the way a true joint sample would (those two
observables generally don't commute, so no single measurement can ever
give a true joint sample of both anyway). This is the same approximation
implicit in expectation-value PCE training (where you also just combine
independently-estimated <Pi> values into one QUBO objective), so it's not
introducing a new source of error relative to `quantum_discrete.py` --
just making the sampling-based analogue of the same assumption explicit.

>>> Flagged simplification: no hardware-native chain layout <<<
`quantum_vqe_solver.py`'s HNDC-style deep-chain ansatz layout exists to
maximize native two-qubit interaction depth for a ~30-qubit circuit on
real hardware coupling constraints. Here the PCE-compressed register is
tiny (~5-9 qubits for this problem size), so this solver uses a plain
linear RY+CZ-chain ansatz with no forced hardware backend/transpilation.
This keeps the qubit-index bookkeeping simple (no risk of transpiler-
inserted SWAPs scrambling which physical qubit holds which logical PCE
qubit) and isn't the bottleneck at this qubit count. The hardware-aware
layout machinery in `quantum_vqe_solver.py` could be reused verbatim if
you want to push this onto real hardware later.

>>> Shot cost <<<
Every `sample_and_evaluate` call now costs 3x the shots of the
uncompressed VQE solver (one batch per basis setting) for the same
n_shots-per-setting -- but operates on ~5-9 qubits instead of ~30, and a
CVaR loss evaluation used to require sampling a huge, mostly-infeasible
~2^30 space; here it only has to resolve ~2^5-2^9 states per basis
setting, which is the whole point of combining the two ideas.

>>> Fixed bug: final answer was independent of training <<<
An earlier version picked the final answer as the single lowest-cost
state seen in a large (20,000-shot) final sample via
`sample_and_evaluate`, same as `quantum_vqe_solver.py` does. That works
fine at ~30 qubits, where 20,000 shots covers a tiny fraction of 2^30
reachable states -- but at the compressed register's ~2^5-2^9 reachable
states PER BASIS SETTING, 20,000 shots exhaustively covers every
reachable state regardless of theta. Once every state has been seen,
argmin-cost-over-everything-seen depends only on which states are
reachable at all (fixed by the qubit-pair assignment, independent of
training) -- not on theta's *probabilities* over them. Confirmed
directly: `total_budget=0` (an untrained ansatz) produced results as
good as a fully trained one under that scheme.

Fixed via `_final_candidates()`: dedupe each basis group's final shots
into (state, frequency) pairs, keep only the top-`final_top_k` most
frequent states per group, and search the cross-product of just those
for the best cost -- rather than every reachable state. A poorly-trained
theta gives a comparatively flat frequency distribution (its top-k is
close to an arbitrary sample of the reachable space); a well-trained
theta concentrates probability on good states (its top-k is actually
informative). This is what ties the final answer back to training
quality. `sample_and_evaluate()` itself is unchanged and still used
as-is for PSO/NFT training, where n0_shots is deliberately kept below
the exhaustive-coverage threshold (see `VQEPCEConfig.n0_shots`) -- this
bug only ever affected the one-off final read-out, not training.

>>> Fixed bug: total_budget=0 wasn't actually untrained <<<
The fix above wasn't enough on its own -- total_budget=0 kept producing
BETTER results than nonzero budgets, and any nonzero total_budget kept
producing the SAME result every time. Root cause, found by re-reading
run(): `run_pso` looped against its own `max_pso_budget`/`min_pso_budget`
config fields, completely independent of `total_budget` -- only NFT's
slice was ever computed as `max(total_budget - pso_evals, 0)`. So
total_budget=0 always meant "PSO still runs fully (up to 600 evals by
default), just skip NFT" -- not "skip training." And since PSO's own RNG
is seeded once from `cfg.seed` (deterministic), PSO always converged to
the same result regardless of total_budget, and with default
total_budget=300 < a typical PSO run's ~180-600 evals, NFT's budget
usually clamped to 0 anyway -- meaning most "nonzero total_budget" runs
were secretly *also* PSO-only, just as deterministic as the total_budget=0
case, and the two only differed on the rare runs where PSO happened to
stop early enough to leave NFT a nonzero slice. Fixed two ways:

  1. `run_pso` now takes an explicit `budget_cap` (`min(max_pso_budget,
     total_budget)`), so total_budget is a real, shared cap across both
     stages -- and total_budget<=0 skips PSO and NFT entirely, using the
     untrained init_params directly, for a genuine apples-to-apples
     baseline.
  2. Elitism: NFT's coordinate-wise updates use a noisy, sample-based
     analytic formula (`_nft_step`) that isn't guaranteed to improve a
     stochastic CVaR loss on every step, and NFT was trained against
     `sample_and_evaluate`'s raw cross-paired CVaR while the final answer
     is chosen by `_final_candidates`'s frequency-based top-k selection --
     two different objectives that can disagree. So a run() now tracks
     the best theta seen on a *fixed* validation metric (`_validation_score`,
     always at alpha_min, evaluated after PSO and after every accepted NFT
     step) and returns that theta, not NFT's raw final endpoint. This
     guarantees more training budget can only match or improve the
     result, never make it worse.

>>> Fixed bug: elitism alone still wasn't enough (winner's curse) <<<
Even with the fix above, results stayed non-monotonic: total_budget=0
gave the worst result (correctly, now that it's genuinely untrained),
but budget ~20-70 beat budget >70 too -- MORE training got WORSE again.
Cause: `_validation_score` is itself a noisy estimate (a CVaR over a few
thousand simulated shots). Every `_register_candidate` call is one noisy
comparison against the running "best so far." More budget means more
such comparisons -- and with enough comparisons, sampling noise alone
will eventually hand the "elite" title to some candidate that merely
GOT LUCKY on its one noisy evaluation, not one that's actually better.
This is a textbook selection-bias / winner's-curse effect: interrogating
a noisy oracle more times doesn't make its answer more reliable, it
makes it more likely that noise alone produces an apparent winner.

Fixed by separating "which candidates get collected" from "which
candidate wins": `_register_candidate` (called from both `run_pso` and
`run_nft` now, via a shared `elite` dict) still uses cheap, noisy,
frequent checks to build a POOL of "looked good at some point" thetas
(capped at `_MAX_AUDIT_CANDIDATES` so pool size doesn't grow unbounded
with total_budget) -- but the actual decision is deferred to
`_audit_best`, which runs ONCE, re-scoring every pooled candidate with a
much larger `audit_shots` sample each. Noise can still affect which
candidates make it INTO the pool, but the final choice among them comes
from one high-precision measurement instead of a chain of cheap noisy
wins, so more training budget now means "a bigger, better pool to audit"
rather than "more chances for noise to crown an impostor."
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np

from qiskit.circuit.library import efficient_su2
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer import AerSimulator

from .classical_continuous import PortfolioProblem
from .classical_discrete import _lot_bounds, _lots_to_weights
from .quantum_discrete import build_lot_encoding, reduce_qubits_with_pce
from .quantum_vqe_solver import ALPHA_C, _sector_penalty_two_sided

N_LOTS_DEFAULT = 20


# ===========================================================================
# 1. CONFIG
# ===========================================================================

@dataclass
class VQEPCEConfig:
    n_lots: int = N_LOTS_DEFAULT
    reps: int = 1                    # ansatz repetitions (kept low, per the
    #                                   paper's own finding that fewer reps
    #                                   tend to win under a fixed budget)
    total_budget: int = 300          # TRUE shared cap across PSO+NFT (fixed --
    #                                   previously PSO used its own separate
    #                                   min/max_pso_budget uncapped by this
    #                                   value, so total_budget=0 didn't
    #                                   actually mean "untrained": PSO still
    #                                   ran fully, only NFT's slice was
    #                                   zeroed. Now total_budget<=0 skips
    #                                   PSO and NFT entirely -- see run().
    min_pso_budget: int = 180
    max_pso_budget: int = 600
    stagnation_window: int = 10
    stagnation_tol: float = 0.01
    n0_shots: int = 300              # per-basis-setting shot count at alpha=1.
    #                                   Deliberately well below the coupon-
    #                                   collector threshold for the reachable
    #                                   space (roughly n*ln(2)*2**n shots to
    #                                   see every outcome at n qubits) -- if
    #                                   shots exhaustively cover every
    #                                   reachable basis state on every
    #                                   evaluation, training has nothing to
    #                                   bite into: every theta looks the same.
    alpha_max: float = 1.0
    alpha_min: float = 0.1
    delta_alpha: float = 0.1
    l_alpha: int = 24
    n_particles: int = 20
    seed: int = 42
    pce_k: int = 2                   # correlator order; this module's per-
    #                                   shot decode (bit parity of exactly
    #                                   two qubits) only supports k=2.
    n_qubits_override: Optional[int] = None
    #                                   reduce_qubits_with_pce(n_vars) returns
    #                                   the MINIMUM qubits PCE needs to fit
    #                                   n_vars variables -- nothing requires
    #                                   using that minimum. At the minimum
    #                                   (5 qubits for this problem's 30 lot-
    #                                   bits), each basis setting only has
    #                                   2**5=32 reachable outcomes, which a
    #                                   few hundred shots already covers
    #                                   exhaustively -- same failure mode as
    #                                   above, just from the qubit-count side
    #                                   instead of the shot-count side. Set
    #                                   to None to fall back to the PCE
    #                                   minimum instead.
    final_shots: int = 20_000        # shots per basis setting used to build
    #                                   the FINAL answer's frequency estimates
    #                                   (was a hardcoded literal in run());
    #                                   now feeds _final_candidates(), not a
    #                                   raw exhaustive sample -- see there.
    final_top_k: int = 20            # final answer is chosen from the cross-
    #                                   product of each basis group's top-k
    #                                   MOST FREQUENT decoded states, not from
    #                                   every state that was merely reachable.
    #                                   This is the actual fix for the "same
    #                                   result regardless of training" bug:
    #                                   with final_shots large enough to
    #                                   exhaustively cover the reachable
    #                                   space, picking argmin-cost over ALL
    #                                   reachable states makes the answer
    #                                   depend only on which states are
    #                                   reachable at all (fixed, independent
    #                                   of theta) rather than on which states
    #                                   theta actually made probable. Ranking
    #                                   by frequency first and restricting
    #                                   the search to the top-k ties the
    #                                   final answer back to training.
    audit_shots: int = 20_000        # shot count for the ONE final, low-
    #                                   noise tournament among training-time
    #                                   candidates (see _audit_best). Kept
    #                                   deliberately large -- unlike the
    #                                   per-step elitism check, this only
    #                                   runs once per candidate at the very
    #                                   end, so it can afford enough shots
    #                                   that sampling noise stops being able
    #                                   to flip the decision.


# ===========================================================================
# 2. PCE qubit-pair assignment (sampling-side counterpart to
#    quantum_discrete.build_pauli_correlation_encoding, which builds the
#    SparsePauliOp *operators*; this just needs to know which two qubit
#    indices each variable's correlator acts on, so a shot's two measured
#    bits can be turned into that variable's decoded value.)
# ===========================================================================

def _assign_pce_pairs(group_vars: list[str], n_qubits: int, k: int = 2) -> dict[str, tuple[int, int]]:
    if k != 2:
        raise NotImplementedError(
            "This module's per-shot decode only supports two-qubit "
            "(k=2) correlators -- see module docstring."
        )
    pairs: dict[str, tuple[int, int]] = {}
    for idx, comb in enumerate(combinations(range(n_qubits), k)):
        if idx >= len(group_vars):
            break
        pairs[group_vars[idx]] = comb
    return pairs


# ===========================================================================
# 3. SOLVER
# ===========================================================================

class PortfolioVQEPCESolver:
    """PCE-compressed register + adaptive-CVaR PSO->NFT sampling VQA for the
    discrete single-period portfolio problem. See module docstring for how
    this differs from both `quantum_discrete.py` (expectation-value PCE
    training) and `quantum_vqe_solver.py` (one qubit per lot-bit, no
    compression)."""

    def __init__(self, problem: PortfolioProblem, config: Optional[VQEPCEConfig] = None):
        self.problem = problem
        self.cfg = config or VQEPCEConfig()

        self.encoding = build_lot_encoding(problem, self.cfg.n_lots)
        self.var_names = self.encoding.bit_vars
        self.n_vars = len(self.var_names)
        if self.n_vars == 0:
            raise ValueError(
                "All assets collapsed to a single feasible lot value "
                "(lo == hi everywhere) -- nothing left to encode."
            )

        third = self.n_vars // 3
        self.group_x = self.var_names[:third]
        self.group_y = self.var_names[third:2 * third]
        self.group_z = self.var_names[2 * third:]

        pce_min_qubits = reduce_qubits_with_pce(self.n_vars)
        if self.cfg.n_qubits_override is not None:
            if self.cfg.n_qubits_override < pce_min_qubits:
                raise ValueError(
                    f"n_qubits_override={self.cfg.n_qubits_override} is below "
                    f"the PCE minimum ({pce_min_qubits}) needed to fit "
                    f"{self.n_vars} variables -- raise it or set it to None."
                )
            self.n_qubits = self.cfg.n_qubits_override
        else:
            self.n_qubits = pce_min_qubits

        self.pair_of: dict[str, tuple[int, int]] = {}
        for group in (self.group_x, self.group_y, self.group_z):
            self.pair_of.update(_assign_pce_pairs(group, self.n_qubits, self.cfg.pce_k))
        missing = [v for v in self.var_names if v not in self.pair_of]
        if missing:
            max_pairs = len(list(combinations(range(self.n_qubits), self.cfg.pce_k)))
            raise ValueError(
                f"{len(missing)} variable(s) couldn't be assigned a PCE qubit "
                f"pair (n_qubits={self.n_qubits} gives only {max_pairs} pairs "
                "per basis group) -- increase n_qubits or reduce group size."
            )

        self.lot_lo, self.lot_hi = _lot_bounds(problem, self.cfg.n_lots)

        # bit_weight_matrix[a, j] = place-value weight of variable j if it's
        # one of asset a's lot-count bits, else 0. Column order == var_names
        # order (== group_x + group_y + group_z, since those partition
        # var_names contiguously) so it lines up directly with the decoded
        # bit arrays built in sample_and_evaluate.
        n_assets = problem.n_assets
        var_index = {v: i for i, v in enumerate(self.var_names)}
        self.bit_weight_matrix = np.zeros((n_assets, self.n_vars))
        for a, names in enumerate(self.encoding.asset_bits):
            for v in names:
                self.bit_weight_matrix[a, var_index[v]] = self.encoding.bit_weight[v]

        self.pen_weight = self._calibrate_penalty()

        
        self.ansatz = self._build_ansatz()
        self.n_params = self.ansatz.num_parameters
        self.simulator = AerSimulator()
        self._rng = np.random.default_rng(self.cfg.seed)

    # -- penalty auto-calibration (same recipe as quantum_vqe_solver.py) ----
    def _calibrate_penalty(self, n_samples: int = 500) -> float:
        rng = np.random.default_rng(self.cfg.seed)
        vals = []
        for _ in range(n_samples):
            lots = rng.integers(0, self.cfg.n_lots + 1, size=self.problem.n_assets)
            w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
            vals.append(abs(self.problem.utility(w)))
        s_util = float(np.mean(vals)) + 1e-9
        return ALPHA_C * s_util

    # -- ansatz: plain linear RY+CZ chain over the compressed register -------
    # (see module docstring: HNDC hardware-native layout is skipped here,
    # it isn't the bottleneck at ~5-9 qubits.)
    def _build_ansatz(self) -> QuantumCircuit:
        n_params_rot = self.n_qubits * (self.cfg.reps + 1)
        theta = ParameterVector("θ", n_params_rot)
        qc = QuantumCircuit(self.n_qubits)

        idx = 0
        for q in range(self.n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
        for _ in range(self.cfg.reps):
            for q in range(self.n_qubits - 1):
                qc.cz(q, q + 1)
            for q in range(self.n_qubits):
                qc.ry(theta[idx], q)
                idx += 1
        return qc


    # -- decoding & cost ------------------------------------------------
    def _row_to_dict(self, row: np.ndarray) -> dict:
        return {v: int(b) for v, b in zip(self.var_names, row)}

    def total_cost(self, bits: dict) -> float:
        """Scalar, exact-formula cost for a single decoded candidate (a
        dict of {var_name: 0/1}) -- the source of truth used by
        bit_flip_postprocess and the final single-candidate evaluate().
        Mirrors quantum_vqe_solver.PortfolioVQESolver.total_cost exactly,
        just decoding via the logical-variable dict instead of a raw
        physical-qubit bit array (there's no 1:1 qubit<->lot-bit mapping
        anymore under PCE compression)."""
        lots = self.encoding.decode_lots(bits)
        w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
        utility = self.problem.utility(w)

        pen = self.pen_weight * (int(lots.sum()) - self.cfg.n_lots) ** 2
        over = np.clip(lots - self.lot_hi, 0, None)
        under = np.clip(self.lot_lo - lots, 0, None)
        pen += self.pen_weight * float(np.sum(over ** 2 + under ** 2))
        pen += self.pen_weight * _sector_penalty_two_sided(w, self.problem)

        return -utility + pen

    def _batch_cost(self, bits_array: np.ndarray) -> np.ndarray:
        """Vectorized reimplementation of total_cost() across many candidates
        at once -- exists purely for speed (n0_shots/alpha_min = up to
        20,000 evaluations per PSO/NFT step; total_cost's per-candidate
        problem.utility() call doesn't scale to that in a Python loop).
        `bits_array` is (N, n_vars), columns ordered as self.var_names.
        Kept numerically consistent with total_cost() -- same formula,
        same penalty terms -- but reimplemented via matrix ops."""
        problem = self.problem
        n_lots = self.cfg.n_lots
        lot_size = problem.budget / n_lots

        offsets = bits_array.astype(float) @ self.bit_weight_matrix.T   # (N, n_assets)
        lots = self.encoding.lo.astype(float)[None, :] + offsets        # (N, n_assets)
        w = lot_size * lots                                             # (N, n_assets)

        ret = w @ problem.expected_returns
        risk = np.einsum('ij,jk,ik->i', w, problem.covariance, w)
        turnover = np.sum(np.abs(w - problem.prev_weights[None, :]) * problem.transaction_cost[None, :], axis=1)
        utility = ret - 0.5 * problem.risk_aversion * risk - problem.cost_aversion * turnover

        pen = self.pen_weight * (lots.sum(axis=1) - n_lots) ** 2
        over = np.clip(lots - self.lot_hi[None, :], 0, None)
        under = np.clip(self.lot_lo[None, :] - lots, 0, None)
        pen += self.pen_weight * np.sum(over ** 2 + under ** 2, axis=1)

        sector_map = np.asarray(problem.sector_map)
        sector_pen = np.zeros(len(bits_array))
        for g_idx, (lo_g, hi_g) in enumerate(zip(problem.group_lower, problem.group_upper_arr)):
            exposure = w[:, sector_map == g_idx].sum(axis=1)
            over_g = np.clip(exposure - hi_g, 0, None)
            under_g = np.clip(lo_g - exposure, 0, None)
            sector_pen += over_g ** 2 + under_g ** 2
        pen += self.pen_weight * sector_pen

        return -utility + pen

    def evaluate(self, bits: dict) -> dict:
        lots = self.encoding.decode_lots(bits)
        w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
        return {
            "lots": lots,
            "weights": w,
            "utility": self.problem.utility(w),
            "budget_ok": int(lots.sum()) == self.cfg.n_lots,
            "bounds_ok": bool(np.all(lots >= self.lot_lo) and np.all(lots <= self.lot_hi)),
            "sector_penalty": _sector_penalty_two_sided(w, self.problem),
        }

    # -- diagnostic: how much of a final sample is actually feasible? -------
    def _shot_feasibility_stats(self, bits_array: np.ndarray) -> dict:
        """Same purpose as the diagnostic added to quantum_vqe_solver.py --
        of the sampled (combined) candidates, what fraction satisfy each
        hard constraint, and all three jointly? Vectorized since bits_array
        here is already a plain numpy array of decoded logical bits."""
        n_total = len(bits_array)
        if n_total == 0:
            return {"n_total": 0}

        unique_rows, counts = np.unique(bits_array, axis=0, return_counts=True)

        offsets = unique_rows.astype(float) @ self.bit_weight_matrix.T
        lots = np.rint(self.encoding.lo.astype(float)[None, :] + offsets).astype(int)
        lot_size = self.problem.budget / self.cfg.n_lots
        w = lot_size * lots

        budget_ok = lots.sum(axis=1) == self.cfg.n_lots
        bounds_ok = np.all((lots >= self.lot_lo[None, :]) & (lots <= self.lot_hi[None, :]), axis=1)

        sector_map = np.asarray(self.problem.sector_map)
        sector_ok = np.ones(len(unique_rows), dtype=bool)
        for g_idx, (lo_g, hi_g) in enumerate(zip(self.problem.group_lower, self.problem.group_upper_arr)):
            exposure = w[:, sector_map == g_idx].sum(axis=1)
            sector_ok &= (exposure >= lo_g - 1e-9) & (exposure <= hi_g + 1e-9)

        all_ok = budget_ok & bounds_ok & sector_ok

        n_budget_ok = int(counts[budget_ok].sum())
        n_bounds_ok = int(counts[bounds_ok].sum())
        n_sector_ok = int(counts[sector_ok].sum())
        n_all_ok = int(counts[all_ok].sum())

        return {
            "n_total": n_total,
            "n_unique": len(unique_rows),
            "n_budget_ok": n_budget_ok,
            "n_bounds_ok": n_bounds_ok,
            "n_sector_ok": n_sector_ok,
            "n_all_ok": n_all_ok,
            "frac_budget_ok": n_budget_ok / n_total,
            "frac_bounds_ok": n_bounds_ok / n_total,
            "frac_sector_ok": n_sector_ok / n_total,
            "frac_all_ok": n_all_ok / n_total,
        }

    # -- sampling: 3 basis-rotated circuit executions per evaluation --------
    def _run_and_expand(self, circuit: QuantumCircuit, n_shots: int) -> np.ndarray:
        """Run one basis-setting circuit and expand its counts dict into a
        (n_shots, n_qubits) array of measured bits -- one row per shot,
        via np.repeat rather than a per-shot Python loop.

        Explicitly seeds the simulator from self._rng (itself seeded from
        cfg.seed) -- AerSimulator draws fresh entropy per call otherwise,
        which made two runs with the same cfg.seed produce different
        results purely from simulator noise, not from anything meaningful
        about theta or training."""
        seed = int(self._rng.integers(0, 2 ** 31 - 1))
        counts = self.simulator.run(circuit, shots=n_shots, seed_simulator=seed).result().get_counts()
        rows, reps = [], []
        for bitstr, cnt in counts.items():
            rows.append(np.array([int(b) for b in reversed(bitstr)], dtype=np.int8))
            reps.append(cnt)
        rows_arr = np.array(rows, dtype=np.int8)
        return np.repeat(rows_arr, reps, axis=0)

    def _decode_group_columns(self, shots_2d: np.ndarray, group_vars: list[str]) -> np.ndarray:
        """Vectorized parity decode: for each variable in this basis group,
        its bit is 1 if its two assigned qubits' measured values differ
        (eigenvalue -1), else 0 -- see module docstring for the sign
        convention this mirrors from quantum_discrete.decode()."""
        if not group_vars:
            return np.zeros((len(shots_2d), 0), dtype=np.int8)
        cols = []
        for v in group_vars:
            qa, qb = self.pair_of[v]
            cols.append((shots_2d[:, qa] != shots_2d[:, qb]).astype(np.int8))
        return np.column_stack(cols)

    def sample_and_evaluate(self, params: np.ndarray, n_shots: int):
        bound = self.ansatz.assign_parameters(params)

        circ_z = bound.copy()
        circ_z.measure_all()

        circ_x = bound.copy()
        for q in range(self.n_qubits):
            circ_x.h(q)
        circ_x.measure_all()

        circ_y = bound.copy()
        for q in range(self.n_qubits):
            circ_y.sdg(q)
            circ_y.h(q)
        circ_y.measure_all()

        shots_x = self._run_and_expand(circ_x, n_shots)
        shots_y = self._run_and_expand(circ_y, n_shots)
        shots_z = self._run_and_expand(circ_z, n_shots)

        # Shuffle each setting's shots independently before pairing across
        # settings -- see module docstring's "flagged approximation" note.
        self._rng.shuffle(shots_x)
        self._rng.shuffle(shots_y)
        self._rng.shuffle(shots_z)

        decoded_x = self._decode_group_columns(shots_x, self.group_x)
        decoded_y = self._decode_group_columns(shots_y, self.group_y)
        decoded_z = self._decode_group_columns(shots_z, self.group_z)

        n = min(len(decoded_x), len(decoded_y), len(decoded_z))
        bits_array = np.hstack([decoded_x[:n], decoded_y[:n], decoded_z[:n]])

        costs = self._batch_cost(bits_array)
        order = np.argsort(costs)
        return bits_array[order], costs[order]

    def _top_k_states(self, decoded: np.ndarray, top_k: int):
        """Dedupe a basis group's decoded shots into unique states + observed
        frequency, and return only the top_k most frequent ones (i.e. the
        states theta actually favors), not merely the ones that appeared at
        all. An empty group (0 variables) returns one trivial all-empty
        'state' with frequency 1, so the cross-product logic below still
        works uniformly."""
        n_shots = len(decoded)
        if decoded.shape[1] == 0:
            return np.zeros((1, 0), dtype=np.int8), np.array([1.0])
        unique_rows, counts = np.unique(decoded, axis=0, return_counts=True)
        order = np.argsort(-counts)[:top_k]
        freqs = counts[order].astype(float) / n_shots
        return unique_rows[order], freqs

    def _final_candidates(self, params: np.ndarray, n_shots: int, top_k: int):
        """Build the final answer from the TRAINED DISTRIBUTION's most
        probable states per basis group, rather than from raw exhaustive
        coverage over the reachable space.

        The bug this fixes: with n_shots large enough to exhaustively cover
        a basis setting's reachable states (which is easy at only a few
        hundred reachable states per setting), sample_and_evaluate's
        argmin-cost-over-everything-seen answer stops depending on theta's
        *probabilities* and starts depending only on which states are
        reachable at all -- a fixed set, independent of training quality.
        (Confirmed directly: total_budget=0, i.e. an UNTRAINED ansatz,
        produced results as good as a trained one under the old scheme.)

        Fix: dedupe each basis group's shots into (state, frequency) pairs,
        keep only the top_k most frequent states per group -- i.e. the ones
        theta actually favors -- and search the cross-product of just those
        (top_k**3 candidates at most, not 2**n_qubits per setting) for the
        best cost. A poorly-trained theta gives a comparatively flat
        frequency distribution, so its top-k is close to an arbitrary
        sample of the reachable space; a well-trained theta concentrates
        probability on good states, so its top-k is actually informative.
        This is what ties the final answer back to training quality."""
        bound = self.ansatz.assign_parameters(params)

        circ_z = bound.copy()
        circ_z.measure_all()

        circ_x = bound.copy()
        for q in range(self.n_qubits):
            circ_x.h(q)
        circ_x.measure_all()

        circ_y = bound.copy()
        for q in range(self.n_qubits):
            circ_y.sdg(q)
            circ_y.h(q)
        circ_y.measure_all()

        shots_x = self._run_and_expand(circ_x, n_shots)
        shots_y = self._run_and_expand(circ_y, n_shots)
        shots_z = self._run_and_expand(circ_z, n_shots)

        decoded_x = self._decode_group_columns(shots_x, self.group_x)
        decoded_y = self._decode_group_columns(shots_y, self.group_y)
        decoded_z = self._decode_group_columns(shots_z, self.group_z)

        top_x, freq_x = self._top_k_states(decoded_x, top_k)
        top_y, freq_y = self._top_k_states(decoded_y, top_k)
        top_z, freq_z = self._top_k_states(decoded_z, top_k)
        nx, ny, nz = len(top_x), len(top_y), len(top_z)

        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()

        bits_array = np.hstack([top_x[ix], top_y[iy], top_z[iz]])
        combined_freq = freq_x[ix] * freq_y[iy] * freq_z[iz]

        costs = self._batch_cost(bits_array)
        order = np.argsort(costs)
        return bits_array[order], costs[order], combined_freq[order], (nx, ny, nz)

    def _validation_score(self, params: np.ndarray) -> float:
        """Consistent, tail-focused metric for comparing candidate thetas on
        equal footing -- always evaluated at alpha_min (regardless of
        whichever alpha a given training step itself used), since PSO
        trains at a fixed alpha_max=1.0 (average-cost) the whole time and
        NFT's alpha only narrows gradually. Comparing two thetas using
        whatever alpha each happened to be trained at isn't apples-to-
        apples; this is. Used purely for elitism/best-so-far tracking in
        run(), not for the training updates themselves."""
        n_shots = self.shots_for_alpha(self.cfg.alpha_min)
        return self.cvar_loss(params, self.cfg.alpha_min, n_shots)

    _MAX_AUDIT_CANDIDATES = 200

    def _register_candidate(self, elite: dict, params: np.ndarray) -> None:
        """Called whenever PSO or NFT's own (cheaper, noisier, differently-
        scaled) internal criterion flags a new local best. Evaluates
        _validation_score once (still noisy, but at least on a consistent
        metric) and appends to elite['candidates'] -- a POOL of "looked
        good at some point during training" thetas, not a final decision.
        See module docstring's "winner's curse" note: the final decision
        among this pool is deferred to _audit_best's single high-precision
        tournament, specifically so that accumulating many cheap noisy
        comparisons during training can't by itself crown a false winner.
        Pool size is capped so audit cost stays bounded regardless of how
        large total_budget is."""
        val = self._validation_score(params)
        elite["evals"] = elite.get("evals", 0) + 1
        if val < elite["val"]:
            elite["val"] = val
            elite["params"] = params.copy()
        elite["candidates"].append(params.copy())
        if len(elite["candidates"]) > self._MAX_AUDIT_CANDIDATES:
            elite["candidates"].pop(0)

    def _audit_best(self, candidates: list) -> np.ndarray:
        """The ONE decision that actually determines the returned theta.
        Re-evaluates every training-time candidate with a single, much
        larger (audit_shots) sample each -- unlike the cheap, frequent,
        noisy per-step checks in _register_candidate, this runs once per
        candidate at the very end, so it can afford enough shots that
        sampling noise stops being able to flip which candidate looks
        best. This is what should make "more training budget" monotonic
        again: more budget means a bigger, better candidate pool to audit,
        not more chances for noise to crown an impostor."""
        if len(candidates) == 1:
            return candidates[0]
        n_shots = max(self.cfg.audit_shots, self.shots_for_alpha(self.cfg.alpha_min))
        scores = [self.cvar_loss(p, self.cfg.alpha_min, n_shots) for p in candidates]
        return candidates[int(np.argmin(scores))]

    def cvar_loss(self, params: np.ndarray, alpha: float, n_shots: int) -> float:
        _, costs = self.sample_and_evaluate(params, n_shots)
        k = max(1, int(np.ceil(alpha * len(costs))))
        return float(costs[:k].mean())

    # -- adaptive CVaR schedule (identical to quantum_vqe_solver.py) --------
    def adaptive_alpha(self, r: int) -> float:
        c = self.cfg
        return max(c.alpha_min, c.alpha_max - c.delta_alpha * (r // c.l_alpha))

    def shots_for_alpha(self, alpha: float) -> int:
        return int(np.ceil(self.cfg.n0_shots / alpha))

    # -- NFT (local refinement) ----------------------------------------------
    def _nft_step(self, params: np.ndarray, j: int, alpha: float) -> np.ndarray:
        n_shots = self.shots_for_alpha(alpha)
        pp = params.copy(); pp[j] += np.pi / 2
        pm = params.copy(); pm[j] -= np.pi / 2
        f0 = self.cvar_loss(params, alpha, n_shots)
        fp = self.cvar_loss(pp, alpha, n_shots)
        fm = self.cvar_loss(pm, alpha, n_shots)

        A = np.sqrt((fp - fm) ** 2 + (2 * f0 - fp - fm) ** 2) / 2
        params_new = params.copy()
        if A < 1e-10:
            return params_new
        phi = np.arctan2(fp - fm, 2 * f0 - fp - fm)
        params_new[j] = (phi + np.pi) % (2 * np.pi)
        return params_new

    def run_nft(self, params: np.ndarray, n_evals: int, log: list, iter_offset: int = 0,
                elite: Optional[dict] = None):
        """elite, if given, is a mutable {'params', 'val', 'evals', 'candidates'}
        dict shared with run_pso -- see _register_candidate/_audit_best.
        Every accepted coordinate step gets registered as a training-time
        candidate; the actual final decision happens later, once, in
        _audit_best, not here."""
        params_cur = params.copy()
        evals_used = 0
        coord_updates = 0
        while evals_used + 3 <= n_evals:
            j = coord_updates % self.n_params
            alpha = self.adaptive_alpha(iter_offset + coord_updates)
            params_new = self._nft_step(params_cur, j, alpha)
            evals_used += 3
            coord_updates += 1

            n_shots = self.shots_for_alpha(alpha)
            _, costs = self.sample_and_evaluate(params_new, n_shots)
            log.append({"evals": evals_used, "alpha": alpha, "best_cost": float(costs[0]), "stage": "NFT"})
            params_cur = params_new

            if elite is not None:
                self._register_candidate(elite, params_cur)
        return params_cur, evals_used

    # -- PSO (global exploration) --------------------------------------------
    def run_pso(self, init_params: np.ndarray, log: list, budget_cap: Optional[int] = None,
                elite: Optional[dict] = None):
        c = self.cfg
        budget_cap = c.max_pso_budget if budget_cap is None else min(c.max_pso_budget, budget_cap)
        rng = np.random.default_rng(c.seed)
        n = c.n_particles
        v_max = 0.2 * 2 * np.pi

        pos = rng.uniform(0, 2 * np.pi, (n, self.n_params))
        pos[0] = init_params
        vel = rng.uniform(-v_max, v_max, (n, self.n_params))
        pbest, pbest_cost = pos.copy(), np.full(n, np.inf)
        gbest, gbest_cost = pos[0].copy(), np.inf

        evals_used = 0
        cost_window: list[float] = []
        while evals_used < budget_cap:
            n_shots = self.shots_for_alpha(c.alpha_max)
            for k in range(n):
                cost = self.cvar_loss(pos[k], c.alpha_max, n_shots)
                if cost < pbest_cost[k]:
                    pbest_cost[k] = cost
                    pbest[k] = pos[k].copy()
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest = pos[k].copy()
                    if elite is not None:
                        self._register_candidate(elite, gbest)
            evals_used += n
            cost_window.append(gbest_cost)
            log.append({"evals": evals_used, "alpha": c.alpha_max, "best_cost": gbest_cost, "stage": "PSO"})

            r1 = rng.uniform(0, 1, (n, self.n_params))
            r2 = rng.uniform(0, 1, (n, self.n_params))
            vel = 0.7 * vel + 1.4 * r1 * (pbest - pos) + 1.4 * r2 * (gbest[None, :] - pos)
            vel = np.clip(vel, -v_max, v_max)
            pos = (pos + vel) % (2 * np.pi)

            if evals_used >= c.min_pso_budget and len(cost_window) >= c.stagnation_window:
                window = cost_window[-c.stagnation_window:]
                rel_imp = (window[0] - window[-1]) / (abs(window[0]) + 1e-8)
                if rel_imp < c.stagnation_tol:
                    break

        return gbest, evals_used

    # -- bit-flip postprocessing (operates on the decoded logical bits) -----
    def bit_flip_postprocess(self, bits: dict, seed: Optional[int] = None):
        bits = dict(bits)
        cost = self.total_cost(bits)
        rng = np.random.default_rng(seed)
        for v in rng.permutation(np.array(self.var_names)):
            bits[v] ^= 1
            new_cost = self.total_cost(bits)
            if new_cost < cost:
                cost = new_cost
            else:
                bits[v] ^= 1
        return bits, cost

    # -- top-level run --------------------------------------------------
    def run(self) -> dict:
        c = self.cfg
        init_params = np.full(self.n_params, np.pi / 4)

        if c.total_budget <= 0:
            # Genuine untrained baseline: previously PSO ran regardless of
            # total_budget (it had its own separate min/max_pso_budget,
            # uncapped by total_budget), so total_budget=0 only ever
            # skipped NFT's slice -- PSO still ran fully, meaning it was
            # NOT actually an untrained comparison. Fixed: total_budget is
            # now the single shared cap across PSO+NFT, so 0 means skip
            # both and use the untrained init_params directly.
            final_params = init_params
            pso_log, nft_log = [], []
            total_opt_evals = 0
        else:
            elite = {
                "params": init_params.copy(),
                "val": self._validation_score(init_params),
                "evals": 1,
                "candidates": [init_params.copy()],
            }

            pso_budget = min(c.max_pso_budget, c.total_budget)
            pso_log = []
            best_pso_params, pso_evals = self.run_pso(
                init_params, pso_log, budget_cap=pso_budget, elite=elite
            )

            nft_budget = max(c.total_budget - pso_evals, 0)
            nft_log = []
            _, nft_evals = self.run_nft(best_pso_params, nft_budget, nft_log, elite=elite)

            # The actual decision: a single low-noise audit across every
            # candidate that looked good at some point during training --
            # see _audit_best/module docstring's "winner's curse" note.
            final_params = self._audit_best(elite["candidates"])
            total_opt_evals = pso_evals + nft_evals + elite["evals"] + len(elite["candidates"])

        n_final_shots = c.final_shots
        bits_sorted, costs_sorted, freq_sorted, pool_shape = self._final_candidates(
            final_params, n_final_shots, c.final_top_k
        )
        best_bits = self._row_to_dict(bits_sorted[0])

        feasibility_stats = self._shot_feasibility_stats(bits_sorted)
        reachable_per_setting = 2 ** self.n_qubits
        coupon_threshold = (
            int(np.ceil(reachable_per_setting * np.log(reachable_per_setting)))
            if reachable_per_setting > 1 else 1
        )
        print(
            f"Reachable outcomes per basis setting: 2**{self.n_qubits} = {reachable_per_setting} "
            f"(coupon-collector threshold to exhaustively cover it ~= {coupon_threshold} shots).\n"
            f"  n0_shots={c.n0_shots} (PSO/NFT baseline) per basis setting -- "
            + (
                "n0_shots is at/above the exhaustive-coverage threshold, so PSO/NFT "
                "training likely sees the same reachable states regardless of theta; "
                "raise n_qubits_override or lower n0_shots further."
                if c.n0_shots >= coupon_threshold else
                "n0_shots is below the exhaustive-coverage threshold, so theta has "
                "room to shape which states actually get sampled."
            )
        )
        print(
            f"Final answer selection: top-{c.final_top_k} most-frequent states per basis "
            f"group from {n_final_shots} shots (actual pool: "
            f"{pool_shape[0]}x{pool_shape[1]}x{pool_shape[2]} states = "
            f"{len(bits_sorted)} candidates searched) -- NOT argmin-cost over every "
            f"reachable state, so the answer depends on theta's learned probabilities, "
            f"not just on which states are reachable at all."
        )
        print(
            f"Final candidate pool feasibility ({feasibility_stats['n_total']} candidates, "
            f"{feasibility_stats['n_unique']} unique): "
            f"budget_ok={feasibility_stats['frac_budget_ok']:.1%}  "
            f"bounds_ok={feasibility_stats['frac_bounds_ok']:.1%}  "
            f"sector_ok={feasibility_stats['frac_sector_ok']:.1%}  "
            f"ALL constraints={feasibility_stats['frac_all_ok']:.1%}"
        )
        if feasibility_stats["n_all_ok"] == 0:
            print(
                "  -> zero fully-feasible candidates in the final sample; "
                "utility_pp below is scored on an infeasible portfolio."
            )

        pp_candidates = []
        for i in range(min(10, len(bits_sorted))):
            pp_bits, pp_cost = self.bit_flip_postprocess(self._row_to_dict(bits_sorted[i]), seed=i)
            pp_candidates.append((pp_cost, pp_bits))
        _, best_pp_bits = min(pp_candidates, key=lambda x: x[0])

        raw_eval = self.evaluate(best_bits)
        pp_eval = self.evaluate(best_pp_bits)

        return {
            "final_params": final_params,
            "total_evals": total_opt_evals,
            "log": pso_log + nft_log,
            "bits_raw": best_bits,
            "bits_pp": best_pp_bits,
            "lots_raw": raw_eval["lots"],
            "lots_pp": pp_eval["lots"],
            "weights_raw": raw_eval["weights"],
            "weights_pp": pp_eval["weights"],
            "utility_raw": raw_eval["utility"],
            "utility_pp": pp_eval["utility"],
            "sector_penalty_raw": raw_eval["sector_penalty"],
            "sector_penalty_pp": pp_eval["sector_penalty"],
            "budget_ok_raw": raw_eval["budget_ok"],
            "budget_ok_pp": pp_eval["budget_ok"],
            "bounds_ok_raw": raw_eval["bounds_ok"],
            "bounds_ok_pp": pp_eval["bounds_ok"],
            "n_qubits": self.n_qubits,
            "n_vars": self.n_vars,
            "feasibility_stats": feasibility_stats,
            "final_top_k": c.final_top_k,
            "final_pool_shape": pool_shape,
            "final_pool_size": len(bits_sorted),
        }


def run_pce_vqe(problem: PortfolioProblem, n_lots: int = N_LOTS_DEFAULT,
                config: Optional[VQEPCEConfig] = None) -> dict:
    """Convenience wrapper, mirrors quantum_vqe_solver.run_vqe's shape."""
    cfg = config or VQEPCEConfig(n_lots=n_lots)
    solver = PortfolioVQEPCESolver(problem, cfg)
    print(
        f"PCE-compressed VQE: {solver.n_vars} logical lot-bits compressed into "
        f"{solver.n_qubits} qubits ({solver.n_params} ansatz parameters) -- "
        f"vs {solver.n_vars} qubits for the uncompressed quantum_vqe_solver."
    )
    return solver.run()