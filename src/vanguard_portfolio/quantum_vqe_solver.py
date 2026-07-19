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
    total_budget: int = 300          # total objective evaluations
    min_pso_budget: int = 180
    max_pso_budget: int = 600
    stagnation_window: int = 10
    stagnation_tol: float = 0.01
    n0_shots: int = 2000             # base shot count; shots = ceil(n0/alpha)
    alpha_max: float = 1.0
    alpha_min: float = 0.1
    delta_alpha: float = 0.1
    l_alpha: int = 24                # decrease alpha every l_alpha NFT updates
    n_particles: int = 20
    seed: int = 42


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

        self.backend = FakeMarrakesh()
        self.simulator = AerSimulator(method="matrix_product_state", noise_model=None,  matrix_product_state_max_bond_dimension=64,   # tune down if still slow
    matrix_product_state_truncation_threshold=1e-6)
        self.coupling_map = self.backend.coupling_map

        self.components, self.hw_adj = self._build_chain_components()
        self.link_edges = self._build_link_edges()
        self.perm, self.inv_perm = self._build_qubit_mapping()

        self.ansatz = self._build_ansatz()
        self.n_params = self.ansatz.num_parameters
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        self.isa_ansatz = pm.run(self.ansatz)

    # -- penalty auto-calibration (Appendix B analogue) ----------------------
    def _calibrate_penalty(self, n_samples: int = 500) -> float:
        rng = np.random.default_rng(self.cfg.seed)
        vals = []
        for _ in range(n_samples):
            lots = rng.integers(0, self.cfg.n_lots + 1, size=self.n_assets)
            w = _lots_to_weights(lots, self.problem, self.cfg.n_lots)
            vals.append(abs(self.problem.utility(w)))
        s_util = float(np.mean(vals)) + 1e-9
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

    # -- sampling ----------------------------------------------------------
    def sample_and_evaluate(self, params: np.ndarray, n_shots: int):
        bound = self.isa_ansatz.assign_parameters(params)
        bound.measure_all()
        job = self.simulator.run(bound, shots=n_shots)
        import time
        t0 = time.time()
        job = self.simulator.run(bound, shots=n_shots)
        counts = job.result().get_counts()
        elapsed = time.time() - t0
        if elapsed > 5:
            print(f"  slow sample: {elapsed:.1f}s, alpha={alpha if 'alpha' in locals() else '?'}")  

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

    def run_nft(self, params: np.ndarray, n_evals: int, log: list, iter_offset: int = 0):
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
        return params_cur, evals_used

    # -- PSO (global exploration) --------------------------------------------
    def run_pso(self, init_params: np.ndarray, log: list):
        c = self.cfg
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
        while evals_used < c.max_pso_budget:
            n_shots = self.shots_for_alpha(c.alpha_max)
            for k in range(n):
                cost = self.cvar_loss(pos[k], c.alpha_max, n_shots)
                if cost < pbest_cost[k]:
                    pbest_cost[k] = cost
                    pbest[k] = pos[k].copy()
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest = pos[k].copy()
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

    # -- bit-flip postprocessing (§4.1 analogue) -----------------------------
    def bit_flip_postprocess(self, bits: np.ndarray, seed: Optional[int] = None):
        bits = bits.copy()
        cost = self.total_cost(bits)
        rng = np.random.default_rng(seed)
        for idx in rng.permutation(self.n_qubits):
            bits[idx] ^= 1
            new_cost = self.total_cost(bits)
            if new_cost < cost:
                cost = new_cost
            else:
                bits[idx] ^= 1
        return bits, cost

    # -- top-level run --------------------------------------------------
    def run(self) -> dict:
        c = self.cfg
        init_params = np.full(self.n_params, np.pi / 4)

        pso_log: list = []
        best_pso_params, pso_evals = self.run_pso(init_params, pso_log)

        nft_log: list = []
        nft_budget = max(c.total_budget - pso_evals, 0)
        final_params, nft_evals = self.run_nft(best_pso_params, nft_budget, nft_log)

        n_final_shots = 20_000
        bits_sorted, costs_sorted = self.sample_and_evaluate(final_params, n_final_shots)
        best_bits = bits_sorted[0]

        pp_candidates = []
        for i in range(min(10, len(bits_sorted))):
            pp_bits, pp_cost = self.bit_flip_postprocess(bits_sorted[i], seed=i)
            pp_candidates.append((pp_cost, pp_bits))
        _, best_pp_bits = min(pp_candidates, key=lambda x: x[0])

        raw_eval = self.evaluate(best_bits)
        pp_eval = self.evaluate(best_pp_bits)

        return {
            "final_params": final_params,
            "total_evals": pso_evals + nft_evals,
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