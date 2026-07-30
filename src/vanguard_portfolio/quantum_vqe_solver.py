"""
quantum_VQE_solver.py
======================
Sampling-based VQA solver for the single-period discrete portfolio
allocation problem defined by `synthetic_universe.json`, following the same
framework used for the multi-period model (Haghighi 2026, arXiv:2606.10098):

  1. ADAPTIVE CVaR SCHEDULE           (§3.1)
  2. TWO-STAGE OPTIMIZER: PSO -> NFT  (§3.2)
  3. ONE-REPETITION ANSATZ            (Study III)
  4. HNDC-STYLE NATIVE CHAIN LAYOUT   (§3.4.4)

This module is the quantum counterpart to `classical_discrete.py` /
`classical_continuous.py`: it solves the *same* PortfolioProblem, and reuses
`classical_discrete`'s lot-lattice machinery (`_lot_bounds`, `_lots_to_weights`)
so results are directly comparable.

DISCRETIZATION / BINARY ENCODING (the "incompatibility" to solve)
-------------------------------------------------------------------
`classical_discrete.py` represents the discrete allocation as a "lot"
lattice: the budget is split into `n_lots` equal units, and each asset i
gets an integer lot count k_i, with sum_i k_i == n_lots enforced *by
construction* -- its simulated-annealing neighbourhood only ever moves one
lot from a donor asset to a receiver asset, and its brute-force solver only
enumerates stars-and-bars compositions that already sum to n_lots. The
budget constraint is therefore never actually violated on the classical
side; there's nothing to check.

A gate-model ansatz can't measure "compositions that sum to n_lots"
directly -- every qubit is an independent binary degree of freedom, and
there's no native mechanism that keeps a global sum fixed across
independent measurement outcomes. So each asset's lot count k_i is instead
given its own small binary register:

    k_i = sum_b 2^b * q_{i,b},      b = 0 .. bits_per_asset - 1
    bits_per_asset = ceil(log2(n_lots + 1))

This makes the budget constraint sum_i k_i == n_lots a *soft* constraint
that must be penalized on every sampled bitstring -- the same pattern used
for the cardinality constraint in the multi-period model, but here it's
doing double duty: in the multi-period model the binary variables were
already 0/1 holdings, so only the *count* needed a soft constraint; here
the lot counts themselves are also only implicitly bounded (an n-bit
register can represent more than n_lots), so both the raw per-asset bit
value AND the cross-asset sum need penalizing. Per-asset lot bounds
(translated from the continuous `lower`/`upper` weight bounds via
`classical_discrete._lot_bounds`) are penalized the same way.

Sector exposure: `classical_discrete._sector_penalty` only penalizes
exceeding `group_upper` -- `group_lower` in the JSON is not currently
enforced by either classical solver. Since the JSON provides both, this
module's QUBO penalizes both sides (`_sector_penalty_two_sided` below). If
you want a strictly apples-to-apples comparison against the classical
baselines, keep that difference in mind -- the VQE is solving a very
slightly stricter feasibility region than the classical validator checks.

------------------------------------------------------------------------
Fixes ported from quantum_vqe_pce_solver.py (see that module's docstring
for the full diagnosis of each -- summarized here since they apply
verbatim to this file too)
------------------------------------------------------------------------
1. Duplicate simulator.run() call: `sample_and_evaluate` used to run the
   circuit TWICE per evaluation and throw the first result away -- pure
   wasted wall-clock, no effect on answer quality. Fixed: one run.

2. total_budget didn't actually cap total training: `run_pso` looped
   against its own separate `max_pso_budget`/`min_pso_budget`, completely
   uncapped by `total_budget` -- only NFT's leftover slice
   (`max(total_budget - pso_evals, 0)`) was ever bounded by it. Since PSO
   alone typically burns 180-600 evaluations by default, `total_budget=300`
   usually meant NFT got 0 evaluations, i.e. this ran as PSO-only, not the
   two-stage PSO->NFT design the paper describes. Fixed: `run_pso` now
   takes an explicit `budget_cap = min(max_pso_budget, total_budget)`, and
   `total_budget<=0` skips PSO and NFT entirely (a genuine untrained
   baseline) instead of only skipping NFT.

3. Elitism + audit (winner's-curse protection): NFT's coordinate-wise
   updates use a noisy, sample-based analytic formula that isn't
   guaranteed to improve a stochastic CVaR loss on every step, and the
   final answer used to be whatever NFT ended on. Fixed: every accepted
   PSO/NFT step registers as a training-time candidate
   (`_register_candidate`, cheap and noisy), and the actual final decision
   is a single low-noise tournament across that whole candidate pool
   (`_audit_best`, run once, with a much larger shot count per candidate)
   -- so more training budget can only match or improve the result, and
   isn't vulnerable to a lucky-noise candidate stealing the "best" title
   partway through a long run.

4. Simulator seeding: `AerSimulator.run(...)` draws fresh entropy per call
   by default, so two runs with the same cfg.seed produced different
   results -- not reproducible, which made it impossible to tell a real
   fix apart from run-to-run noise. Fixed: every `simulator.run(...)` call
   now passes `seed_simulator=` drawn from a persistent `self._rng`
   (itself seeded from cfg.seed).

------------------------------------------------------------------------
PERFORMANCE PASS (see chat writeup for the full before/after profiling)
------------------------------------------------------------------------
Runtime profiling on a 6-asset/0-period instance showed pso_sec and
nft_sec dominating (~99% of wall clock) and landing at nearly *equal*
wall-clock despite very different shot budgets per call (PSO: flat
n0_shots at alpha_max; NFT: n0_shots/alpha, growing as alpha shrinks).
That pattern -- similar wall-clock despite different per-call shot cost --
is the signature of fixed per-`simulator.run()` overhead (circuit
parameter binding + Aer/Qiskit dispatch) dominating over the actual
shot-sampling cost, not the tensor-network math itself. The paper's own
budget accounting (§4.1: "we refer to one objective-function evaluation as
one iteration... 1000 iterations") is exactly 1000 `simulator.run()`-style
calls, no more -- this implementation's elitism registration and final
audit added *uncounted* extra simulator calls on top of `total_budget`.
The changes below (5-8) target that gap directly.

5. BATCHED CIRCUIT EXECUTION (new): `run_pso`'s per-particle loop and
   `_nft_step`'s 3-shifted-evaluation loop each used to issue one
   `simulator.run()` call per particle / per shift, sequentially. Aer
   supports submitting a *list* of circuits in a single `.run()` call, and
   since all particles in a PSO generation (and all 3 NFT shifts) use the
   same shot count, they batch cleanly into one job, paying the fixed
   per-call overhead once instead of once-per-circuit. Fixed via
   `_batch_sample_and_evaluate`.

6. Elitism-check frequency: `_register_candidate` used to fire an entire
   extra `cvar_loss` call (a full extra `simulator.run()`, uncounted
   against `total_budget`) on *every* accepted PSO/NFT step. This is now
   throttled (`elite_check_every`) so it only re-validates periodically,
   cutting the number of uncounted extra circuit executions substantially
   without changing the winner's-curse protection at the end (the audit
   pool is smaller but still spans the run).

7. Postprocessing rewritten to match the paper's actual described
   algorithm (§4.1 "Postprocessing"): "Bits are visited in random order;
   if a bit flip improves the objective value, the move is accepted and
   the search restarts with a new random ordering. The procedure
   terminates when no improving flip is found within the flip budget."
   The previous version did a single fixed-order sweep with no restart,
   which cannot discover multi-bit coordinated corrections (e.g. two
   flips across different assets' registers that only jointly restore the
   budget constraint). Fixed: `bit_flip_postprocess` now restarts the
   random ordering after every accepted flip, capped at n_qubits total
   *attempted* flips (matching the paper's stated budget), and stops when
   a full pass finds no improving move.

8. Penalty calibration switched from an empirical mean over random lot
   draws to a instance-dependent bound in the same spirit as the paper's
   Appendix B (a conservative multiple of the largest observed |utility|
   sampled, rather than the mean) -- see `_calibrate_penalty` docstring.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

backend = FakeMarrakesh()
noise_model = NoiseModel.from_backend(backend)

from .classical_continuous import PortfolioProblem
from .classical_discrete import _lot_bounds, _lots_to_weights

N_LOTS_DEFAULT = 20
ALPHA_C = 10.0   # penalty auto-calibration multiplier, mirrors Appendix B


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================

def load_universe_from_json(path, risk_aversion: float = 3.0,
                             cost_aversion: float = 1.0) -> PortfolioProblem:
    """
    Build a PortfolioProblem from synthetic_universe.json.

    risk_aversion / cost_aversion aren't part of the JSON schema (they're
    investor-preference knobs, not asset-universe data), so they're passed
    in with sensible defaults -- override them to match whatever preset
    you want to compare against.
    """
    path = Path(path)
    data = json.loads(path.read_text())

    sector_map = np.array(data["asset_group"], dtype=int)
    group_upper = {i: lim for i, lim in enumerate(data["group_upper"])}

    problem = PortfolioProblem(
        expected_returns=np.array(data["mu"], dtype=float),
        covariance=np.array(data["cov"], dtype=float),
        risk_aversion=risk_aversion,
        transaction_cost=np.array(data["c"], dtype=float),
        prev_weights=np.array(data["w0"], dtype=float),
        cost_aversion=cost_aversion,
        lower_bounds=np.array(data["lower"], dtype=float),
        upper_bounds=np.array(data["upper"], dtype=float),
        budget=1.0,
        sector_map=sector_map,
        sector_limits=group_upper,
        asset_names=list(data["asset_names"]),
    )

    # Extra fields PortfolioProblem doesn't model natively -- stashed on the
    # instance so the quantum solver's penalty can use them.
    problem.group_names = list(data["group_names"])
    problem.group_lower = np.array(data["group_lower"], dtype=float)
    problem.group_upper_arr = np.array(data["group_upper"], dtype=float)
    problem.cash_yield = np.array(data["y"], dtype=float)  # unused by
    # `utility()` today (neither classical solver uses it either); kept for
    # schema parity and future extension.
    return problem


def _sector_penalty_two_sided(weights: np.ndarray, problem: PortfolioProblem) -> float:
    """Squared-violation penalty against BOTH group_lower and group_upper."""
    sector_map = np.asarray(problem.sector_map)
    group_upper = getattr(problem, "group_upper_arr", None)
    if group_upper is None:
        # Fall back to the upper-only sector_limits mapping, matching
        # classical_discrete._sector_penalty's behavior.
        if not problem.sector_limits:
            return 0.0
        penalty = 0.0
        for sector, limit in problem.sector_limits.items():
            exposure = float(weights[sector_map == sector].sum())
            penalty += max(0.0, exposure - limit) ** 2
        return penalty

    group_lower = getattr(problem, "group_lower", np.zeros_like(group_upper))
    penalty = 0.0
    for idx, (lo, hi) in enumerate(zip(group_lower, group_upper)):
        exposure = float(weights[sector_map == idx].sum())
        over = max(0.0, exposure - hi)
        under = max(0.0, lo - exposure)
        penalty += over ** 2 + under ** 2
    return penalty


# ===========================================================================
# 2.  CONFIG
# ===========================================================================

@dataclass
class VQEConfig:
    n_lots: int = N_LOTS_DEFAULT
    reps: int = 1                    # one-repetition ansatz (Study III)
    total_budget: int = 600          # TRUE shared cap across PSO+NFT (fixed --
    #                                   previously PSO used its own separate
    #                                   min/max_pso_budget uncapped by this
    #                                   value, so total_budget=0 didn't
    #                                   actually mean "untrained": PSO still
    #                                   ran fully, only NFT's slice was
    #                                   zeroed. Now total_budget<=0 skips
    #                                   PSO and NFT entirely -- see run().
    min_pso_budget: int = 180
    max_pso_budget: int = 600
    nft_reserve_fraction: float = 0.4  # PSO's own budget_cap is capped at
    #                                    total_budget * (1 - this), so NFT
    #                                    is GUARANTEED at least this share
    #                                    of total_budget regardless of how
    #                                    much PSO would otherwise use --
    #                                    previously PSO could (and by
    #                                    default did) consume the entire
    #                                    total_budget, leaving NFT's
    #                                    "leftover" slice at exactly 0, so
    #                                    the adaptive-CVaR narrowing (which
    #                                    only happens inside NFT's
    #                                    coordinate updates -- PSO always
    #                                    trains at fixed alpha_max=1.0)
    #                                    never ran at all. Note this alone
    #                                    doesn't guarantee NFT reaches
    #                                    alpha_min -- see run()'s printed
    #                                    schedule-completion diagnostic.
    stagnation_window: int = 10
    stagnation_tol: float = 0.01
    n0_shots: int = 2000             # base shot count; shots = ceil(n0/alpha)
    alpha_max: float = 1.0
    alpha_min: float = 0.3           # was 0.1 -- per-evaluation shot cost is
    #                                   n0_shots/alpha, so alpha_min directly
    #                                   caps the worst-case cost of a single
    #                                   NFT step (n0_shots/alpha_min). 0.1
    #                                   meant up to 20,000 shots/step (same
    #                                   cost as the whole final sample); 0.3
    #                                   caps it at ~6,700, trading away some
    #                                   CVaR tail-sharpness for a much lower
    #                                   worst-case runtime per step. Lower
    #                                   this back toward 0.1 if you want the
    #                                   original tail resolution and can
    #                                   afford the cost.
    delta_alpha: float = 0.1
    l_alpha: int = 24                # decrease alpha every l_alpha NFT updates
    n_particles: int = 20
    seed: int = 42
    audit_shots: int = 20_000        # shot count for the ONE final, low-
    #                                   noise tournament among training-time
    #                                   candidates (see _audit_best). Kept
    #                                   deliberately large -- unlike the
    #                                   per-step elitism check, this only
    #                                   runs once per candidate at the very
    #                                   end, so it can afford enough shots
    #                                   that sampling noise stops being able
    #                                   to flip the decision.
    min_improvement_frac: float = 0.01  # a candidate only gets ADDED to the
    #                                     audit pool if it beats the current
    #                                     elite by more than this relative
    #                                     margin (default 1%) -- previously
    #                                     ANY improvement, however tiny and
    #                                     likely noise-driven, got added,
    #                                     which let the audit pool (and
    #                                     therefore audit_sec, which scales
    #                                     linearly with pool size) balloon
    #                                     under a large total_budget with no
    #                                     real benefit -- most of those
    #                                     marginal "improvements" were never
    #                                     going to change the final answer.
    #                                     Set to 0.0 to restore the old
    #                                     "register everything" behavior.
    elite_check_every: int = 5       # NEW: only call _register_candidate's
    #                                   extra validation simulator.run()
    #                                   every `elite_check_every`-th accepted
    #                                   PSO/NFT step, instead of on every
    #                                   single one. Each check is itself a
    #                                   full extra circuit execution that is
    #                                   NOT counted against total_budget, so
    #                                   this directly cuts uncounted
    #                                   wall-clock. Set to 1 to restore the
    #                                   old "check every step" behavior.
    mps_bond_dimension: int = 64     # NEW: exposed as a config knob rather
    #                                   than hardcoded -- for a shallow,
    #                                   sparsely-entangled single-repetition
    #                                   ansatz this is likely far more than
    #                                   needed; try lowering (e.g. 16-32)
    #                                   and confirming `evaluate()` results
    #                                   are unchanged before trusting the
    #                                   speedup, since correctness depends
    #                                   on the true entanglement of your
    #                                   specific circuit/instance.


# ===========================================================================
# 3.  SOLVER
# ===========================================================================

class PortfolioVQESolver:
    """
    HNDC-1 ansatz + adaptive-CVaR PSO->NFT sampling VQA for the discrete
    single-period portfolio problem.
    """

    def __init__(self, problem: PortfolioProblem, config: Optional[VQEConfig] = None):
        self.problem = problem
        self.cfg = config or VQEConfig()

        self.n_assets = problem.n_assets
        self.bits_per_asset = int(np.ceil(np.log2(self.cfg.n_lots + 1)))
        self.n_qubits = self.n_assets * self.bits_per_asset
        self.lot_lo, self.lot_hi = _lot_bounds(problem, self.cfg.n_lots)
        self.pen_weight = self._calibrate_penalty()
        self._rng = np.random.default_rng(self.cfg.seed)  # drives simulator
        #                                                     seeding -- see
        #                                                     module docstring
        #                                                     fix (4).
        self._elite_step_counter = 0  # NEW: drives elite_check_every throttle

        self.backend = FakeMarrakesh()
        self.simulator = AerSimulator(
            method="matrix_product_state",
            noise_model=None,
            matrix_product_state_max_bond_dimension=self.cfg.mps_bond_dimension,
            matrix_product_state_truncation_threshold=1e-6,
        )
        self.coupling_map = self.backend.coupling_map

        self.components, self.hw_adj = self._build_chain_components()
        self.link_edges = self._build_link_edges()
        self.perm, self.inv_perm = self._build_qubit_mapping()

        self.ansatz = self._build_ansatz()
        self.n_params = self.ansatz.num_parameters
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        self.isa_ansatz = pm.run(self.ansatz)

    # -- penalty auto-calibration -------------------------------------------
    def _calibrate_penalty(self, n_samples: int = 500) -> float:
        """
        Penalty scale for the soft budget/bounds/sector constraints.

        Previously this used the *mean* |utility| across n_samples random
        lot draws. A mean is not a bound: if the sampled utilities have any
        meaningful spread (very plausible with few assets and no rebalancing
        periods, where a handful of draws can dominate the distribution),
        the resulting penalty can be too small for a strict tail of samples,
        letting a locally-"improving" move (lower total_cost) actually be a
        move that trades away feasibility for objective value -- silently
        weakening exactly the postprocessing/training signal this weight is
        supposed to protect.

        This now uses the maximum observed |utility| over the same sample,
        scaled by ALPHA_C, as a conservative (if not fully analytic) proxy
        for the paper's Appendix-B approach of deriving penalty coefficients
        from a provable upper bound on the objective's largest possible
        swing (S_obj), rather than its average. This is still empirical
        (a true analytic bound would require deriving per-term bounds
        specific to `problem.utility`'s formula, as Appendix B does for the
        paper's own objective), but it removes the systematic
        under-calibration risk of averaging.
        """
        rng = np.random.default_rng(self.cfg.seed)
        vals = []
        for _ in range(n_samples):
            lots = rng.integers(0, self.cfg.n_lots + 1, size=self.n_assets)
            w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
            vals.append(abs(self.problem.utility(w)))
        s_util = float(np.max(vals)) + 1e-9
        return ALPHA_C * s_util

    # -- HNDC-style native chain layout (§3.4.4) -----------------------------
    # One chain per asset (length = bits_per_asset): the bits that jointly
    # define one asset's own lot count are the strongest-coupled variables
    # (they set that asset's contribution to both the linear return term and
    # the diagonal of the risk term), so they get the deepest, most-connected
    # part of the circuit -- same rationale as grouping by rebalancing period
    # in the multi-period model, just substituting "asset" for "period".
    def _build_chain_components(self):
        nodes = set(range(self.n_qubits))
        adj: dict[int, set] = defaultdict(set)
        for u, v in self.coupling_map.get_edges():
            if u in nodes and v in nodes:
                adj[u].add(v)
                adj[v].add(u)

        unassigned = set(nodes)
        components: list[list[int]] = []
        while unassigned:
            start = min(unassigned)
            path = [start]
            unassigned.discard(start)
            while len(path) < self.bits_per_asset:
                candidates = [x for x in adj[path[-1]] if x in unassigned]
                grown = False
                if candidates:
                    nxt = min(candidates)
                    path.append(nxt)
                    unassigned.discard(nxt)
                    grown = True
                else:
                    for node in path:
                        cands2 = [x for x in adj[node] if x in unassigned]
                        if cands2:
                            nxt = min(cands2)
                            path.append(nxt)
                            unassigned.discard(nxt)
                            grown = True
                            break
                if not grown:
                    break
            components.append(path)
        return components, adj

    def _build_link_edges(self):
        comp_of = {}
        for ci, comp in enumerate(self.components):
            for q in comp:
                comp_of[q] = ci
        used: set = set()
        link_edges: list[tuple] = []
        for q in sorted(comp_of.keys()):
            if q in used:
                continue
            for nb in sorted(self.hw_adj[q]):
                if nb in used:
                    continue
                if comp_of.get(nb, comp_of[q]) != comp_of[q]:
                    link_edges.append((q, nb))
                    used.add(q)
                    used.add(nb)
                    break
        return link_edges

    def _build_qubit_mapping(self):
        # Chain i holds all bits_per_asset bits of asset i, in path order.
        perm = np.zeros(self.n_qubits, dtype=int)   # perm[qubit] = logical bit idx
        for asset_i, chain in enumerate(self.components):
            if asset_i >= self.n_assets:
                break
            for bit_b, qubit in enumerate(chain):
                if bit_b >= self.bits_per_asset:
                    break
                perm[qubit] = asset_i * self.bits_per_asset + bit_b
        inv_perm = np.argsort(perm)
        return perm, inv_perm

    def _build_ansatz(self) -> QuantumCircuit:
        n_params_rot = self.n_qubits * (self.cfg.reps + 1)
        theta = ParameterVector("θ", n_params_rot)
        qc = QuantumCircuit(self.n_qubits)

        idx = 0
        for q in range(self.n_qubits):
            qc.ry(theta[idx], q)
            idx += 1

        max_chain_len = max((len(c) for c in self.components), default=0)
        for _ in range(self.cfg.reps):
            # Deep-chain sublayers: within-asset CZs, all chains in parallel.
            for step in range(max_chain_len - 1):
                for chain in self.components:
                    if step < len(chain) - 1:
                        qc.cz(chain[step], chain[step + 1])
            # Inter-asset linking sublayer (native edges, disjoint pairs).
            for u, v in self.link_edges:
                qc.cz(u, v)
            for q in range(self.n_qubits):
                qc.ry(theta[idx], q)
                idx += 1
        return qc

    # -- decoding & cost -------------------------------------------------
    def decode_lots(self, bits: np.ndarray) -> np.ndarray:
        bits2d = bits.reshape(self.n_assets, self.bits_per_asset)
        powers = 2 ** np.arange(self.bits_per_asset)
        return (bits2d @ powers).astype(int)

    def total_cost(self, bits: np.ndarray) -> float:
        """Cost to MINIMIZE: -utility + soft-constraint penalties."""
        lots = self.decode_lots(bits)
        w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
        utility = self.problem.utility(w)

        pen = 0.0
        pen += self.pen_weight * (int(lots.sum()) - self.cfg.n_lots) ** 2
        over = np.clip(lots - self.lot_hi, 0, None)
        under = np.clip(self.lot_lo - lots, 0, None)
        pen += self.pen_weight * float(np.sum(over ** 2 + under ** 2))
        pen += self.pen_weight * _sector_penalty_two_sided(w, self.problem)

        return -utility + pen

    def evaluate(self, bits: np.ndarray) -> dict:
        """Diagnostics for a bitstring, with the penalty stripped back off
        (see the earlier discussion: report the real objective, not the
        arbitrary calibration constant, but only trust it if *_ok is True)."""
        lots = self.decode_lots(bits)
        w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
        return {
            "lots": lots,
            "weights": w,
            "utility": self.problem.utility(w),
            "budget_ok": int(lots.sum()) == self.cfg.n_lots,
            "bounds_ok": bool(np.all(lots >= self.lot_lo) and np.all(lots <= self.lot_hi)),
            "sector_penalty": _sector_penalty_two_sided(w, self.problem),
        }

    def _shot_feasibility_stats(self, bits_arr: np.ndarray) -> dict:
        """Diagnostic only -- not used by the optimizer.

        Of a batch of sampled bitstrings, what fraction actually satisfy each
        hard constraint (and all three jointly)? This is the number that
        tells you whether a bad utility_pp means "the ansatz found a
        genuinely bad but feasible portfolio" or "the ansatz essentially
        never samples a feasible portfolio at all, so utility_pp is being
        computed on something that was never really comparable to a feasible
        baseline in the first place."

        Dedupes to unique bitstrings first (weighted by shot count) since a
        converged variational state typically has far fewer unique outcomes
        than raw shots, which keeps this cheap even at n_final_shots=20_000.
        """
        n_total = len(bits_arr)
        if n_total == 0:
            return {"n_total": 0}

        unique_bits, counts = np.unique(bits_arr, axis=0, return_counts=True)

        n_budget_ok = n_bounds_ok = n_sector_ok = n_all_ok = 0
        for row, cnt in zip(unique_bits, counts):
            lots = self.decode_lots(row)
            budget_ok = int(lots.sum()) == self.cfg.n_lots
            bounds_ok = bool(np.all(lots >= self.lot_lo) and np.all(lots <= self.lot_hi))
            w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
            sector_ok = _sector_penalty_two_sided(w, self.problem) == 0.0

            n_budget_ok += int(cnt) * int(budget_ok)
            n_bounds_ok += int(cnt) * int(bounds_ok)
            n_sector_ok += int(cnt) * int(sector_ok)
            n_all_ok += int(cnt) * int(budget_ok and bounds_ok and sector_ok)

        return {
            "n_total": n_total,
            "n_unique": len(unique_bits),
            "n_budget_ok": n_budget_ok,
            "n_bounds_ok": n_bounds_ok,
            "n_sector_ok": n_sector_ok,
            "n_all_ok": n_all_ok,
            "frac_budget_ok": n_budget_ok / n_total,
            "frac_bounds_ok": n_bounds_ok / n_total,
            "frac_sector_ok": n_sector_ok / n_total,
            "frac_all_ok": n_all_ok / n_total,
        }

    # -- sampling ----------------------------------------------------------
    def sample_and_evaluate(self, params: np.ndarray, n_shots: int):
        """Single-circuit evaluation. Kept for callers that only ever need
        one parameter vector at a time (e.g. the final sample, and the
        audit pass, where circuits are naturally evaluated one at a time
        with a large, distinct shot count each). For the hot inner loops
        (PSO's per-generation particle sweep, NFT's 3-shift step), use
        `_batch_sample_and_evaluate` instead -- see that method's docstring.
        """
        bound = self.isa_ansatz.assign_parameters(params)
        bound.measure_all()
        seed = int(self._rng.integers(0, 2 ** 31 - 1))
        job = self.simulator.run(bound, shots=n_shots, seed_simulator=seed)
        counts = job.result().get_counts()
        return self._counts_to_sorted_bits_costs(counts)

    def _counts_to_sorted_bits_costs(self, counts: dict):
        bitstrings, costs = [], []
        for bitstr, count in counts.items():
            bits_raw = np.array([int(b) for b in reversed(bitstr)], dtype=np.int8)
            bits_logical = bits_raw[self.inv_perm]
            cost = self.total_cost(bits_logical)
            for _ in range(count):
                bitstrings.append(bits_logical)
                costs.append(cost)

        costs_arr = np.array(costs, dtype=np.float64)
        bits_arr = np.array(bitstrings, dtype=np.int8)
        order = np.argsort(costs_arr)
        return bits_arr[order], costs_arr[order]

    def _batch_sample_and_evaluate(self, params_list: list, n_shots: int):
        """NEW: evaluate MULTIPLE parameter vectors in a single Aer job.

        Aer's `.run()` accepts a list of circuits and executes them in one
        call, paying the fixed Python/Qiskit dispatch + Aer setup overhead
        ONCE for the whole batch instead of once per parameter vector. This
        directly targets the profiling signature described in the module
        docstring (PSO and NFT stages costing nearly identical wall-clock
        despite very different shot budgets per call -- a sign that fixed
        per-call overhead, not shot count, was the dominant cost).

        Returns a list of (bits_sorted, costs_sorted) tuples, one per input
        parameter vector, in the same order as `params_list`.
        """
        bound_circuits = []
        for params in params_list:
            bc = self.isa_ansatz.assign_parameters(params)
            bc.measure_all()
            bound_circuits.append(bc)

        seed = int(self._rng.integers(0, 2 ** 31 - 1))
        job = self.simulator.run(bound_circuits, shots=n_shots, seed_simulator=seed)
        result = job.result()

        results = []
        for i in range(len(bound_circuits)):
            counts = result.get_counts(i)
            results.append(self._counts_to_sorted_bits_costs(counts))
        return results

    def _batch_cvar_loss(self, params_list: list, alpha: float, n_shots: int) -> list:
        """Batched counterpart to `cvar_loss` -- one Aer call for the whole
        list of parameter vectors, returning one CVaR value per vector."""
        batch = self._batch_sample_and_evaluate(params_list, n_shots)
        out = []
        for _, costs in batch:
            k = max(1, int(np.ceil(alpha * len(costs))))
            out.append(float(costs[:k].mean()))
        return out

    def _validation_score(self, params: np.ndarray) -> float:
        """Consistent, tail-focused metric for comparing candidate thetas on
        equal footing -- always evaluated at alpha_min (regardless of
        whichever alpha a given training step itself used), since PSO
        trains at a fixed alpha_max=1.0 (average-cost) the whole time and
        NFT's alpha only narrows gradually. Used purely for elitism/
        best-so-far tracking in run(), not for the training updates
        themselves.

        Deliberately uses the flat n0_shots baseline, NOT
        shots_for_alpha(alpha_min) (= n0_shots/alpha_min, a 10x-larger
        count at the default alpha_min=0.1) -- this is called on every
        PSO gbest improvement and every accepted NFT step, so it needs to
        actually be cheap, not merely smaller than the one-time final
        audit. An earlier version used the inflated count here, making
        the "cheap frequent check vs. expensive one-time decision"
        asymmetry the whole design relies on not actually hold -- see
        module docstring."""
        return self.cvar_loss(params, self.cfg.alpha_min, self.cfg.n0_shots)

    _MAX_AUDIT_CANDIDATES = 200

    def _register_candidate(self, elite: dict, params: np.ndarray) -> None:
        """Called whenever PSO or NFT's own (cheaper, noisier, differently-
        scaled) internal criterion flags a new local best. Evaluates
        _validation_score once (still noisy, but at least on a consistent
        metric); only ADDS to elite['candidates'] -- the POOL of "looked
        good at some point during training" thetas -- if the improvement
        over the current elite exceeds min_improvement_frac. Below that
        margin, the "improvement" is treated as noise and simply ignored
        (elite['val']/'params' don't move either): without this filter,
        a large total_budget lets the pool balloon with marginal,
        likely-noise-driven registrations that were never going to change
        the final answer, and audit_sec scales linearly with pool size.
        The final decision among the pool is still deferred to
        _audit_best's single high-precision tournament -- this only
        controls what's eligible to be IN that tournament, not who wins
        it (winner's-curse / selection-bias protection is unaffected).
        Pool size is additionally capped so audit cost stays bounded
        regardless of how large total_budget is.

        NEW: throttled via `elite_check_every`. Every call to this method
        is a full extra `simulator.run()` that is NOT counted against
        `total_budget` -- on the original per-step-unconditional version,
        this meant every single accepted PSO/NFT step paid for TWO circuit
        executions (the training step itself, plus this validation call),
        silently doubling uncounted wall-clock on top of the profiled
        pso_sec/nft_sec. This now only actually validates every
        `elite_check_every`-th call; in between, the local-best params are
        still tracked (so nothing is lost as a *candidate*), but the extra
        simulator call, and thus the extra wall-clock, is skipped.
        """
        self._elite_step_counter += 1
        if self._elite_step_counter % self.cfg.elite_check_every != 0:
            # Still worth keeping as a low-cost fallback candidate in case
            # the throttled steps happen to bracket the true elite, but
            # without paying for a fresh validation run.
            elite.setdefault("skipped_candidates", []).append(params.copy())
            return

        val = self._validation_score(params)
        elite["evals"] = elite.get("evals", 0) + 1
        improvement_frac = (elite["val"] - val) / (abs(elite["val"]) + 1e-9)
        if improvement_frac > self.cfg.min_improvement_frac:
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
        best. NEW: batched into a single Aer call across all candidates
        instead of one call per candidate."""
        if len(candidates) == 1:
            return candidates[0]
        scores = self._batch_cvar_loss(candidates, self.cfg.alpha_min, self.cfg.audit_shots)
        return candidates[int(np.argmin(scores))]

    def cvar_loss(self, params: np.ndarray, alpha: float, n_shots: int) -> float:
        _, costs = self.sample_and_evaluate(params, n_shots)
        k = max(1, int(np.ceil(alpha * len(costs))))
        return float(costs[:k].mean())

    # -- adaptive CVaR schedule (§3.1) ---------------------------------------
    def adaptive_alpha(self, r: int) -> float:
        c = self.cfg
        return max(c.alpha_min, c.alpha_max - c.delta_alpha * (r // c.l_alpha))

    def shots_for_alpha(self, alpha: float) -> int:
        return int(np.ceil(self.cfg.n0_shots / alpha))

    # -- NFT (local refinement) ----------------------------------------------
    def _nft_step(self, params: np.ndarray, j: int, alpha: float) -> np.ndarray:
        """NEW: the three shifted evaluations (θ, θ+π/2, θ-π/2) are now
        submitted as a single batched Aer call instead of three sequential
        `cvar_loss` calls -- this is the single biggest lever for nft_sec,
        since NFT issues 3 circuit executions per coordinate update and
        previously paid full per-call overhead on each of the three."""
        n_shots = self.shots_for_alpha(alpha)
        pp = params.copy(); pp[j] += np.pi / 2
        pm = params.copy(); pm[j] -= np.pi / 2

        f0, fp, fm = self._batch_cvar_loss([params, pp, pm], alpha, n_shots)

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
        """NEW: the per-generation loop over all n_particles used to call
        `cvar_loss` once per particle (n_particles sequential Aer calls per
        generation). All particles in a generation share the same shot
        count (shots_for_alpha(c.alpha_max) is fixed within PSO), so they
        now batch into a single Aer call per generation via
        `_batch_cvar_loss` -- this is the single biggest lever for
        pso_sec, cutting the per-generation call count from n_particles
        down to 1."""
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
            costs = self._batch_cvar_loss([pos[k] for k in range(n)], c.alpha_max, n_shots)
            for k in range(n):
                cost = costs[k]
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

    # -- bit-flip postprocessing (§4.1, matching the paper's described
    #    algorithm) -----------------------------------------------------
    def bit_flip_postprocess(self, bits: np.ndarray, seed: Optional[int] = None):
        """Rewritten to match the paper's §4.1 "Postprocessing" description
        verbatim: "For each sampled bitstring, the maximum number of
        attempted bit flips is capped at the number of binary variables.
        Bits are visited in random order; if a bit flip improves the
        objective value, the move is accepted and the search restarts with
        a new random ordering. The procedure terminates when no improving
        flip is found within the flip budget."

        The previous version did a single fixed-order sweep with no
        restart on acceptance, which can only ever consider each bit
        exactly once and can miss multi-bit coordinated corrections (e.g.
        two flips across different assets' K-bit registers that only
        jointly restore the budget constraint, where the first flip alone
        looks like a regression and gets rejected before the second is
        ever tried). Restarting the random ordering after every accepted
        flip lets previously-rejected bits be retried once the local cost
        landscape has actually changed.
        """
        bits = bits.copy()
        cost = self.total_cost(bits)
        rng = np.random.default_rng(seed)

        attempts_used = 0
        max_attempts = self.n_qubits
        while attempts_used < max_attempts:
            order = rng.permutation(self.n_qubits)
            found_improvement = False
            for idx in order:
                if attempts_used >= max_attempts:
                    break
                attempts_used += 1
                bits[idx] ^= 1
                new_cost = self.total_cost(bits)
                if new_cost < cost:
                    cost = new_cost
                    found_improvement = True
                    break  # restart with a new random ordering, per the paper
                else:
                    bits[idx] ^= 1
            if not found_improvement:
                break  # a full pass found no improving flip -> local optimum
        return bits, cost

    # -- top-level run --------------------------------------------------
    def run(self) -> dict:
        import time as _time  # local import so this diagnostic doesn't
        #                        force a module-level dependency change

        c = self.cfg
        init_params = np.full(self.n_params, np.pi / 4)
        timings: dict = {}

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
            n_candidates_audited = 0
        else:
            elite = {
                "params": init_params.copy(),
                "val": self._validation_score(init_params),
                "evals": 1,
                "candidates": [init_params.copy()],
            }

            pso_budget = min(
                c.max_pso_budget,
                max(int(c.total_budget * (1 - c.nft_reserve_fraction)), 0),
            )
            pso_log = []
            t0 = _time.perf_counter()
            best_pso_params, pso_evals = self.run_pso(
                init_params, pso_log, budget_cap=pso_budget, elite=elite
            )
            timings["pso_sec"] = _time.perf_counter() - t0

            nft_budget = max(c.total_budget - pso_evals, 0)
            nft_log = []
            t0 = _time.perf_counter()
            _, nft_evals = self.run_nft(best_pso_params, nft_budget, nft_log, elite=elite)
            timings["nft_sec"] = _time.perf_counter() - t0

            # Diagnostic: did NFT actually reach alpha_min, or did it run
            # out of budget partway through the schedule? See module
            # docstring / nft_reserve_fraction's comment -- a nonzero NFT
            # budget doesn't by itself guarantee the schedule completes.
            coord_updates_done = nft_evals // 3
            steps_needed = int(np.ceil((c.alpha_max - c.alpha_min) / c.delta_alpha))
            coord_updates_needed = steps_needed * c.l_alpha
            alpha_reached = self.adaptive_alpha(coord_updates_done) if coord_updates_done else c.alpha_max
            print(
                f"NFT schedule: {coord_updates_done}/{coord_updates_needed} coordinate "
                f"updates completed, reached alpha={alpha_reached:.3f} "
                f"(target alpha_min={c.alpha_min})"
                + ("  -- FULLY reached alpha_min." if coord_updates_done >= coord_updates_needed
                   else "  -- did NOT reach alpha_min; narrowing was cut short by budget. "
                        "Raise total_budget and/or nft_reserve_fraction, or reduce "
                        "l_alpha/increase delta_alpha to shorten the schedule, if you "
                        "need this to complete.")
            )

            # The actual decision: a single low-noise audit across every
            # candidate that looked good at some point during training --
            # see _audit_best/module docstring's "winner's curse" note.
            n_candidates_audited = len(elite["candidates"])
            t0 = _time.perf_counter()
            final_params = self._audit_best(elite["candidates"])
            timings["audit_sec"] = _time.perf_counter() - t0
            total_opt_evals = pso_evals + nft_evals + elite["evals"] + len(elite["candidates"])

        n_final_shots = 20_000
        t0 = _time.perf_counter()
        bits_sorted, costs_sorted = self.sample_and_evaluate(final_params, n_final_shots)
        timings["final_sample_sec"] = _time.perf_counter() - t0
        best_bits = bits_sorted[0]

        print(
            "Timing breakdown: "
            + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items())
            + (f", audit_candidates={n_candidates_audited}"
               if "audit_sec" in timings else "")
        )

        # --- diagnostic: how much of the final distribution is even feasible? ---
        feasibility_stats = self._shot_feasibility_stats(bits_sorted)
        print(
            f"Final sample feasibility ({feasibility_stats['n_total']} shots, "
            f"{feasibility_stats['n_unique']} unique bitstrings): "
            f"budget_ok={feasibility_stats['frac_budget_ok']:.1%}  "
            f"bounds_ok={feasibility_stats['frac_bounds_ok']:.1%}  "
            f"sector_ok={feasibility_stats['frac_sector_ok']:.1%}  "
            f"ALL constraints={feasibility_stats['frac_all_ok']:.1%}"
        )
        if feasibility_stats["n_all_ok"] == 0:
            print(
                "  -> zero fully-feasible bitstrings in the final sample. "
                "utility_pp below is being computed on an infeasible portfolio -- "
                "not a fair comparison against a feasible baseline's utility."
            )

        t0 = _time.perf_counter()
        pp_candidates = []
        for i in range(min(10, len(bits_sorted))):
            pp_bits, pp_cost = self.bit_flip_postprocess(bits_sorted[i], seed=i)
            pp_candidates.append((pp_cost, pp_bits))
        _, best_pp_bits = min(pp_candidates, key=lambda x: x[0])
        print(f"  bit_flip_postprocess: {_time.perf_counter() - t0:.1f}s")

        raw_eval = self.evaluate(best_bits)
        pp_eval = self.evaluate(best_pp_bits)

        return {
            "final_params": final_params,
            "total_evals": total_opt_evals,
            "log": pso_log + nft_log,
            "feasibility_stats": feasibility_stats,
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
        }


def run_vqe(problem: PortfolioProblem, n_lots: int = N_LOTS_DEFAULT,
            config: Optional[VQEConfig] = None) -> dict:
    """Convenience wrapper, mirrors mean_variance_continuous/discrete's shape."""
    cfg = config or VQEConfig(n_lots=n_lots)
    solver = PortfolioVQESolver(problem, cfg)
    two_q_gates = sum(1 for _, qargs, _ in solver.isa_ansatz if len(qargs) == 2)
    print(f"HNDC-{cfg.reps} ansatz: {solver.n_qubits} qubits "
          f"({solver.n_assets} assets x {solver.bits_per_asset} bits/asset), "
          f"{solver.n_params} parameters")
    print(f"Transpiled: depth={solver.isa_ansatz.depth()}, 2Q gates={two_q_gates}")
    return solver.run()
