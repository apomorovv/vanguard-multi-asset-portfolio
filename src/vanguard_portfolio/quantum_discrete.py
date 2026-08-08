"""Quantum-inspired discrete mean-variance portfolio optimizer.

This is the quantum counterpart to :mod:`classical_discrete`. It solves the
*same* integer-lot allocation problem (`k_i` lots per asset, `sum_i k_i ==
n_lots`, `w_i = (budget / n_lots) * k_i`) using QAOA with Pauli Correlation
Encoding (PCE, Sciorilli et al., Nat. Commun. 2025) plus the
Iterative-alpha-PCE hard-constraint procedure from arXiv:2602.17479
("Pauli Correlation Encoding for Budget-Constrained Optimization").

------------------------------------------------------------------------
IMPORTANT: how this differs from the original standalone script, and why
------------------------------------------------------------------------
The original standalone file this was extracted from modeled a 6-asset,
12-period "hold / don't hold" problem (72 binary variables `h_{i,t}`) with
three hard-constraint families defined *per period*:

    1. full investment       -- exactly K assets held every period
    2. asset min/max alloc.  -- asset i held in [L_i, U_i] periods
    3. group exposure limit  -- group G holds in [L_G, U_G] assets/period

That data model (multi-period price paths, per-period holdings) does not
exist anywhere else in this project -- `classical_continuous.py` and
`classical_discrete.py` both work off a single-period `PortfolioProblem`
(mean `mu`, covariance `cov`, per-asset weight bounds, sector/group
exposure bounds, a starting weight `w0`, and a turnover cost). Per the
project owner's instruction, this module reads the *shared* single-period
data instead of generating its own multi-period synthetic prices, so the
constraint families above have been re-mapped onto the lot-allocation
problem instead of dropped:

    1. full investment  -> "budget" constraint: the lots must sum to
       exactly `n_lots` (there is only one "period" now, so this is a
       single equality constraint instead of one per period).
    2. asset min/max     -> each asset's lot count must land in
       `[lo_i, hi_i]`, the lot-quantized version of `lower[i] / upper[i]`.
       (Mechanically this is enforced by the *binary encoding domain* of
       each asset's lot count, plus a small "overshoot" range constraint --
       see `build_constraints` below.)
    3. group exposure     -> each group's total lots must land in
       `[lo_G, hi_G]`, the lot-quantized version of `group_lower[G] /
       group_upper[G]`.

Every asset's lot count `k_i` is written in binary (`k_i = lo_i +
sum_b 2**b * x_{i,b}`), so the *variables* PCE encodes are now lot-count
bits, not per-period holdings. This keeps the same overall machinery
(PCE compression, iterative-alpha binarization, constraint penalties
expressed on the continuous `z = tanh(alpha * <P_i>)` proxy, multi-restart)
but the constraint bookkeeping had to be generalized from "count of 1-bits
in a scope" (the original, cardinality-only version) to "weighted sum of
bits in a scope" (needed here because binary-expansion bits have weights
`2**b`, not weight 1). When every scope weight is 1 this generalization
reduces exactly to the original cardinality formulation, so nothing about
the original constraint algebra was thrown away -- it was extended.

------------------------------------------------------------------------
Objective: matches PortfolioProblem.utility(w) exactly, except turnover
------------------------------------------------------------------------
Now that `classical_continuous.py` has been provided, the real objective is:

    U(w) = mu . w  -  0.5 * risk_aversion * (w' Sigma w)
           - cost_aversion * (transaction_cost . |w - prev_weights|)

Notably there is NO income/yield term -- the shared JSON's `"y"` field is
loaded but is not part of `PortfolioProblem.utility()`, so it is not part
of this module's QAOA loss either (kept out deliberately, so the quantum
objective matches the real one rather than silently including an extra
term the classical solver doesn't optimize for).

`build_lot_qubo` below reproduces this exactly EXCEPT for the turnover
term, which uses `(w_i - prev_weights_i)**2` in place of the true
`|w_i - prev_weights_i|` L1 cost (see the surrogate note further down) --
everything else (return, risk, the 0.5 factor, cost_aversion) is wired to
match `PortfolioProblem.utility()` bit for bit.

For anything that needs the *real* ground-truth objective -- ranking
QAOA restarts against each other is fine using the closed form above, but
the final "how good is this vs. the rest of the project" comparison in
`scripts/run_quantum.py` calls `problem.utility(w)` directly, and also
calls `classical_discrete.mean_variance_discrete(...)` as an exact/near-
exact baseline (reusing the teammate-written brute-force/annealing solver
instead of re-deriving a MILP baseline here). That guarantees the
headline comparison numbers are judged by the *actual* project objective,
not by this module's closed-form stand-in.

>>> Turnover-cost simplification (flagged, not hidden) <<<
The real per-asset transaction cost is `cost_aversion * transaction_cost_i
* |w_i - prev_weights_i|` (L1, confirmed from `PortfolioProblem.cost()` /
`.utility()`). An exact L1 term is not degree-<=2 in the lot-count bits
without extra auxiliary "sign" qubits per asset, which would materially
complicate the PCE encoding for a proof-of-concept model. This module
instead penalizes `cost_aversion * transaction_cost_i * (w_i -
prev_weights_i)**2`: convex, zero at `prev_weights_i`, and cheap to fold
into the existing quadratic (risk) term exactly like the sector-penalty
code elsewhere in this project already does (squared-violation
penalties). If exact L1 parity with the classical solver's cost term
matters for your write-up, say so and I'll add the auxiliary-qubit L1
linearization -- flagging here rather than silently changing the shape of
the cost function.

>>> Noise model (flagged, not hidden) <<<
The original standalone file built `noise_model = NoiseModel.from_backend
(backend)` but then passed `"noise_model": None` into the estimator
options -- so despite the surrounding commentary about simulating a
noisy backend, the original script actually ran *noiseless*. That
behavior is preserved by default here (`apply_noise=False`) so nothing
changes silently; set `apply_noise=True` on the optimizer if you actually
want FakeMarrakesh's noise applied.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as Estimator

from .classical_continuous import PortfolioProblem
from .classical_discrete import mean_variance_discrete

# ---------------------------------------------------------------------------
# 1. Lot encoding: integer lot count k_i -> binary expansion bits
# ---------------------------------------------------------------------------


def _asset_lot_bounds(problem: PortfolioProblem, n_lots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Identical math to classical_discrete._lot_bounds, reproduced locally
    so this module doesn't depend on that private (underscored) helper."""
    lot_size = problem.budget / n_lots
    lo = np.floor(problem.lower_bounds / lot_size + 1e-9).astype(int)
    hi = np.ceil(problem.upper_bounds / lot_size - 1e-9).astype(int)
    lo = np.clip(lo, 0, n_lots)
    hi = np.clip(hi, 0, n_lots)
    hi = np.maximum(hi, lo)
    return lo, hi


def _bits_needed(range_i: int) -> int:
    """Bits required to represent offsets 0..range_i inclusive."""
    if range_i <= 0:
        return 0
    return int(np.ceil(np.log2(range_i + 1)))


@dataclass
class LotEncoding:
    """Binary expansion of each asset's integer lot count.

    k_i = lo[i] + sum_b 2**b * x_{i,b},   x_{i,b} in {0,1}
    """

    asset_names: List[str]
    lo: np.ndarray
    hi: np.ndarray
    bits_per_asset: List[int]
    bit_vars: List[str]
    bit_asset_index: Dict[str, int]
    bit_weight: Dict[str, int]
    asset_bits: List[List[str]]

    @property
    def n_bits(self) -> int:
        return len(self.bit_vars)

    def decode_lots(self, bits: Dict[str, int]) -> np.ndarray:
        k = self.lo.copy()
        for i, names in enumerate(self.asset_bits):
            offset = sum(self.bit_weight[v] * bits[v] for v in names)
            k[i] += offset
        return k


def build_lot_encoding(problem: PortfolioProblem, n_lots: int) -> LotEncoding:
    lo, hi = _asset_lot_bounds(problem, n_lots)
    n = problem.n_assets
    bits_per_asset = [_bits_needed(int(hi[i] - lo[i])) for i in range(n)]

    bit_vars: List[str] = []
    bit_asset_index: Dict[str, int] = {}
    bit_weight: Dict[str, int] = {}
    asset_bits: List[List[str]] = []
    for i in range(n):
        names = []
        for b in range(bits_per_asset[i]):
            name = f"k_{i}_{b}"
            bit_vars.append(name)
            bit_asset_index[name] = i
            bit_weight[name] = 2 ** b
            names.append(name)
        asset_bits.append(names)

    return LotEncoding(
        asset_names=list(problem.asset_names),
        lo=lo,
        hi=hi,
        bits_per_asset=bits_per_asset,
        bit_vars=bit_vars,
        bit_asset_index=bit_asset_index,
        bit_weight=bit_weight,
        asset_bits=asset_bits,
    )


# ---------------------------------------------------------------------------
# 2. Closed-form utility QUBO (see module docstring for the exact formula)
# ---------------------------------------------------------------------------


def build_lot_qubo(
    encoding: LotEncoding,
    mu: np.ndarray,
    cov: np.ndarray,
    transaction_cost: np.ndarray,
    prev_weights: np.ndarray,
    risk_aversion: float,
    cost_aversion: float,
    budget: float,
    n_lots: int,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    """Build (linear, quadratic, constant) for U(w) as a function of the
    lot-count bits, matching `PortfolioProblem.utility()`:

        U(w) = mu.w - 0.5*risk_aversion*(w'Sigma w)
               - cost_aversion * sum_i transaction_cost_i * |w_i - prev_weights_i|

    with the L1 turnover term replaced by a quadratic surrogate
    `(w_i - prev_weights_i)**2` (see module docstring). This is a UTILITY
    to be MAXIMIZED (higher is better), unlike the original file's QUBO
    which was a cost to be minimized -- flip the sign once, at the loss
    function, rather than here."""
    lot_size = budget / n_lots
    n = len(mu)
    c = transaction_cost
    w0 = prev_weights

    const_w = lot_size * encoding.lo.astype(float)  # w_i contribution from lo[i]

    L = mu.copy() + 2.0 * cost_aversion * c * w0  # linear coefficient on w_i
    Q = -0.5 * risk_aversion * cov.copy()
    for i in range(n):
        Q[i, i] -= cost_aversion * c[i]

    linear: Dict[str, float] = defaultdict(float)
    quadratic: Dict[Tuple[str, str], float] = defaultdict(float)
    constant = -cost_aversion * float(np.sum(c * w0 ** 2))
    constant += float(np.sum(L * const_w))
    for i in range(n):
        constant += Q[i, i] * const_w[i] ** 2
        for j in range(i + 1, n):
            constant += 2 * Q[i, j] * const_w[i] * const_w[j]

    # Linear-in-bits contribution of L_i * w_i
    for i in range(n):
        for name in encoding.asset_bits[i]:
            coef = encoding.bit_weight[name] * lot_size
            linear[name] += L[i] * coef

    # Quadratic-in-bits contribution of Q_ij * w_i * w_j
    for i in range(n):
        bits_i = encoding.asset_bits[i]
        # cross term with the constant offset of asset i itself: 2*Q_ii*const_i*w_i
        for name in bits_i:
            coef = encoding.bit_weight[name] * lot_size
            linear[name] += 2 * Q[i, i] * const_w[i] * coef
        # self-interaction w_i^2 fully expanded over asset i's own bits
        for a_idx, name_a in enumerate(bits_i):
            coef_a = encoding.bit_weight[name_a] * lot_size
            for name_b in bits_i[a_idx:]:
                coef_b = encoding.bit_weight[name_b] * lot_size
                if name_a == name_b:
                    linear[name_a] += Q[i, i] * coef_a * coef_a  # x^2 == x for binary
                else:
                    quadratic[(name_a, name_b)] += 2 * Q[i, i] * coef_a * coef_b
        # cross-asset terms i<j
        for j in range(i + 1, n):
            bits_j = encoding.asset_bits[j]
            qij2 = 2 * Q[i, j]
            for name_a in bits_i:
                coef_a = encoding.bit_weight[name_a] * lot_size
                linear[name_a] += qij2 * const_w[j] * coef_a
            for name_b in bits_j:
                coef_b = encoding.bit_weight[name_b] * lot_size
                linear[name_b] += qij2 * const_w[i] * coef_b
            for name_a in bits_i:
                coef_a = encoding.bit_weight[name_a] * lot_size
                for name_b in bits_j:
                    coef_b = encoding.bit_weight[name_b] * lot_size
                    quadratic[(name_a, name_b)] += qij2 * coef_a * coef_b

    return dict(linear), dict(quadratic), constant


def qubo_utility(bits: Dict[str, int], linear, quadratic, constant) -> float:
    val = constant
    for name, coeff in linear.items():
        val += coeff * bits[name]
    for (ni, nj), coeff in quadratic.items():
        val += coeff * bits[ni] * bits[nj]
    return val


# ---------------------------------------------------------------------------
# 3. Hard constraints (weighted generalization of the original file's
#    cardinality-only eq/range constraints -- see module docstring)
# ---------------------------------------------------------------------------


def build_constraints(
    encoding: LotEncoding,
    group_map: List[int],
    group_lower: List[float],
    group_upper: List[float],
    group_names: List[str],
    n_lots: int,
) -> List[dict]:
    constraints = []
    lo_sum = int(encoding.lo.sum())

    # 1. Budget / "full investment": total lots must equal n_lots exactly.
    weights_all = {v: encoding.bit_weight[v] for v in encoding.bit_vars}
    constraints.append({
        "name": "budget_full_investment",
        "scope": list(encoding.bit_vars),
        "weights": weights_all,
        "type": "eq",
        "target": n_lots - lo_sum,
    })

    # 2. Per-asset bound: offset bits for asset i must stay in [0, hi_i-lo_i]
    #    (guards against binary-expansion overshoot when hi-lo+1 isn't a
    #    power of two).
    for i, names in enumerate(encoding.asset_bits):
        if not names:
            continue
        w = {v: encoding.bit_weight[v] for v in names}
        constraints.append({
            "name": f"asset_bounds_{encoding.asset_names[i]}",
            "scope": names,
            "weights": w,
            "type": "range",
            "bounds": (0, int(encoding.hi[i] - encoding.lo[i])),
        })

    # 3. Group exposure: total lots for group G within its lot-quantized
    #    [group_lower, group_upper] weight-fraction bounds.
    n_assets = len(encoding.asset_names)
    for g_idx, g_name in enumerate(group_names):
        members = [i for i in range(n_assets) if group_map[i] == g_idx]
        if not members:
            continue
        scope: List[str] = []
        w: Dict[str, int] = {}
        base = int(sum(encoding.lo[i] for i in members))
        for i in members:
            for name in encoding.asset_bits[i]:
                scope.append(name)
                w[name] = encoding.bit_weight[name]
        lo_lots = int(np.floor(group_lower[g_idx] * n_lots + 1e-9)) - base
        hi_lots = int(np.ceil(group_upper[g_idx] * n_lots - 1e-9)) - base
        lo_lots = max(0, lo_lots)
        hi_lots = max(lo_lots, hi_lots)
        constraints.append({
            "name": f"group_{g_name}",
            "scope": scope,
            "weights": w,
            "type": "range",
            "bounds": (lo_lots, hi_lots),
        })

    return constraints


def variable_reach(bit_vars: List[str], linear_d, quadratic_d) -> Dict[str, float]:
    reach = defaultdict(float)
    for name in bit_vars:
        reach[name] = 0.0
    for name, coeff in linear_d.items():
        reach[name] += abs(coeff)
    for (ni, nj), coeff in quadratic_d.items():
        reach[ni] += abs(coeff)
        reach[nj] += abs(coeff)
    return reach


def beta_for_constraint(con: dict, reach_map: Dict[str, float]) -> float:
    """Penalty weight scaled to the objective magnitude touching the
    constraint's scope. The original file's tight "top-c reach" bound
    assumed pure cardinality constraints (c = a variable *count*); with
    weighted constraints "c" no longer has a clean meaning, so this uses
    the safe, conservative generalization: sum of ALL reach values in
    scope. Looser than the original bound, but still scales beta to the
    objective's magnitude, which is the property that actually matters."""
    return sum(reach_map[v] for v in con["scope"])


# ---------------------------------------------------------------------------
# 4. Pauli Correlation Encoding (unchanged from the original file)
# ---------------------------------------------------------------------------


def reduce_qubits_with_pce(n_logical: int) -> int:
    return int(np.ceil((1 + np.sqrt(1 + (8 / 3) * n_logical)) / 2))


def build_pauli_correlation_encoding(pauli: str, group: List[str], n_qubits: int, k: int = 2):
    ops = []
    for idx, comb in enumerate(combinations(range(n_qubits), k)):
        if idx >= len(group):
            break
        paulis = ["I"] * n_qubits
        paulis[comb[0]] = pauli
        paulis[comb[1]] = pauli
        ops.append(SparsePauliOp.from_list([("".join(paulis)[::-1], 1.0)]))
    return ops


# ---------------------------------------------------------------------------
# 5. Constraint penalty (continuous z-proxy) and QAOA loss
# ---------------------------------------------------------------------------


def constraint_penalty(alpha: float, node_exp_map: Dict[str, float], constraints: List[dict]):
    """z_i = tanh(alpha * <P_i>); x_i = (1 - z_i) / 2 (z<0 -> x=1 convention,
    matches `decode` below). Weighted sum of x over a constraint's scope is
    the continuous proxy for that constraint's true (integer) value."""
    total = 0.0
    diag = []
    for con in constraints:
        scope = con["scope"]
        w = con["weights"]
        zs = {v: np.tanh(alpha * node_exp_map[v]) for v in scope}
        value_proxy = sum(w[v] * (1.0 - zs[v]) / 2.0 for v in scope)
        beta = con["beta"]
        if con["type"] == "eq":
            viol = value_proxy - con["target"]
            pen = beta * viol ** 2
        else:
            lo, hi = con["bounds"]
            over = max(0.0, value_proxy - hi)
            under = max(0.0, lo - value_proxy)
            pen = beta * (over ** 2 + under ** 2)
        total += pen
        diag.append((con["name"], pen))
    return total, diag


def loss_func_estimator(params, ansatz, pce_ops_flat, ordered_vars, estimator, log,
                         alpha, constraints, linear, quadratic, var_names, beta_reg, v_scale):
    job = estimator.run([(ansatz, pce_ops_flat, params)])
    result = job.result()
    node_exp_map = {name: ev for name, ev in zip(ordered_vars, result[0].data.evs)}

    # utility (to maximize) -> loss (to minimize) = -utility_proxy
    utility_proxy = 0.0
    for (ni, nj), coeff in quadratic.items():
        utility_proxy += coeff * np.tanh(alpha * node_exp_map[ni]) * np.tanh(alpha * node_exp_map[nj])
    for ni, coeff in linear.items():
        utility_proxy += coeff * np.tanh(alpha * node_exp_map[ni])
    loss = -utility_proxy

    # binarization regularizer, same shape as the original file
    reg = np.mean([np.tanh(alpha * node_exp_map[name]) ** 2 for name in var_names]) ** 2
    loss += beta_reg * v_scale * reg

    con_pen, _ = constraint_penalty(alpha, node_exp_map, constraints)
    loss += con_pen

    log.append({"loss": loss, "exp_map": node_exp_map, "alpha": alpha})
    return loss


# ---------------------------------------------------------------------------
# 6. Decode: correlation expectations -> lot bits -> lots -> weights
# ---------------------------------------------------------------------------


def decode(exp_map: Dict[str, float], var_names: List[str], flip: bool = False) -> Dict[str, int]:
    bits = {}
    for name in var_names:
        z = exp_map[name]
        if flip:
            z = -z
        bits[name] = 1 if z < 0 else 0  # x=(1-z)/2 convention: z<0 -> x=1
    return bits


def constraint_violations(bits: Dict[str, int], constraints: List[dict]):
    n_satisfied = 0
    report = []
    for con in constraints:
        value = sum(con["weights"][v] * bits[v] for v in con["scope"])
        if con["type"] == "eq":
            ok = value == con["target"]
            desc = f"value={value}, target={con['target']}"
        else:
            lo, hi = con["bounds"]
            ok = lo <= value <= hi
            desc = f"value={value}, bounds=[{lo},{hi}]"
        n_satisfied += int(ok)
        report.append((con["name"], ok, desc))
    return n_satisfied, report


def best_decoding(exp_map, var_names, constraints, linear, quadratic, constant):
    """Try both PCE sign conventions, keep whichever satisfies more
    constraints (ties broken on higher utility -- this is a MAXIMIZATION,
    unlike the original minimization file)."""
    bits_a = decode(exp_map, var_names, flip=False)
    bits_b = decode(exp_map, var_names, flip=True)
    val_a = qubo_utility(bits_a, linear, quadratic, constant)
    val_b = qubo_utility(bits_b, linear, quadratic, constant)
    sat_a, report_a = constraint_violations(bits_a, constraints)
    sat_b, report_b = constraint_violations(bits_b, constraints)
    if (sat_a, val_a) >= (sat_b, val_b):
        return bits_a, val_a, "z<0 -> hold", sat_a, report_a
    return bits_b, val_b, "z>0 -> hold (sign flipped)", sat_b, report_b


# ---------------------------------------------------------------------------
# 7. Public API
# ---------------------------------------------------------------------------


@dataclass
class QuantumMeanVarianceDiscreteOptimizer:
    """QAOA + Iterative-alpha PCE discrete (integer-lot) optimizer.

    Mirrors `classical_discrete.MeanVarianceDiscreteOptimizer`'s role and
    return-dict shape so the two are drop-in comparable from a runner
    script, with group-exposure constraints layered on top (the classical
    discrete solver only has a one-sided soft sector penalty; this solver
    enforces two-sided group bounds as HARD constraints, matching the
    original quantum file's constraint families -- see module docstring).
    """

    problem: PortfolioProblem
    n_lots: int
    group_map: List[int]
    group_lower: List[float]
    group_upper: List[float]
    group_names: List[str]
    risk_aversion: Optional[float] = None  # falls back to problem.risk_aversion
    cost_aversion: Optional[float] = None  # falls back to problem.cost_aversion
    reps: int = 12
    n_restarts: int = 3
    alpha_0: float = 1.0
    threshold_m: float = 0.975
    max_alpha_iters: int = 30
    inner_maxiter: int = 60
    max_alpha_growth_per_step: float = 3.0
    beta_reg: float = 0.5
    apply_noise: bool = False  # see module docstring: original file computed
    #                            a noise model but never actually applied it
    seed: Optional[int] = None

    def solve(self, mu: np.ndarray, cov: np.ndarray,
              transaction_cost: np.ndarray, prev_weights: np.ndarray) -> Dict[str, object]:
        p = self.problem
        risk_aversion = self.risk_aversion if self.risk_aversion is not None else getattr(
            p, "risk_aversion", None)
        if risk_aversion is None:
            raise ValueError(
                "risk_aversion not found on `problem` and none was passed "
                "explicitly to QuantumMeanVarianceDiscreteOptimizer."
            )
        cost_aversion = self.cost_aversion if self.cost_aversion is not None else getattr(
            p, "cost_aversion", 1.0)

        encoding = build_lot_encoding(p, self.n_lots)
        var_names = encoding.bit_vars
        n_vars = encoding.n_bits
        if n_vars == 0:
            raise ValueError(
                "All assets collapsed to a single feasible lot value "
                "(lo == hi everywhere) -- nothing left for QAOA to decide."
            )

        linear, quadratic, constant = build_lot_qubo(
            encoding, mu, cov, transaction_cost, prev_weights,
            risk_aversion, cost_aversion, p.budget, self.n_lots,
        )

        constraints = build_constraints(
            encoding, self.group_map, self.group_lower, self.group_upper,
            self.group_names, self.n_lots,
        )
        reach = variable_reach(var_names, linear, quadratic)
        for con in constraints:
            con["beta"] = beta_for_constraint(con, reach)

        # --- PCE encoding -----------------------------------------------
        num_qubits_pce = reduce_qubits_with_pce(n_vars)
        third = n_vars // 3
        group_x = var_names[:third]
        group_y = var_names[third:2 * third]
        group_z = var_names[2 * third:]
        pce_x = build_pauli_correlation_encoding("X", group_x, num_qubits_pce)
        pce_y = build_pauli_correlation_encoding("Y", group_y, num_qubits_pce)
        pce_z = build_pauli_correlation_encoding("Z", group_z, num_qubits_pce)
        ordered_vars = group_x + group_y + group_z
        ordered_ops = pce_x + pce_y + pce_z

        labels, coeffs = [], []
        for op in ordered_ops:
            labels.extend(op.paulis.to_labels())
            coeffs.extend(op.coeffs.tolist())
        combined_hamiltonian = SparsePauliOp(labels, coeffs).simplify()

        ansatz = QAOAAnsatz(cost_operator=combined_hamiltonian, reps=self.reps)
        backend = FakeMarrakesh()
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_ansatz = pm.run(ansatz)
        pce_ops_isa_flat = [op.apply_layout(isa_ansatz.layout) for op in ordered_ops]

        noise_model = NoiseModel.from_backend(backend) if self.apply_noise else None
        estimator = Estimator(options={
            "backend_options": {"noise_model": noise_model},
            "run_options": {"shots": 4000},
        })

        v_scale = len(quadratic) / 2 + (n_vars - 1) / 4
        rng = np.random.default_rng(self.seed)
        restart_seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=self.n_restarts)]

        restart_results = []
        best_restart = None
        for r_idx, seed in enumerate(restart_seeds):
            rng_r = np.random.default_rng(seed)
            init_params = 2 * np.pi * rng_r.random(isa_ansatz.num_parameters)
            params, alpha, best_entry, evals = self._run_iterative_alpha_pce(
                constraints, init_params, isa_ansatz, pce_ops_isa_flat, ordered_vars,
                estimator, linear, quadratic, var_names, v_scale,
                label=f"[restart {r_idx} seed={seed}] ",
            )
            exp_map = best_entry["exp_map"]
            holdings, value, convention, n_sat, report = best_decoding(
                exp_map, var_names, constraints, linear, quadratic, constant)
            result_r = {
                "seed": seed, "bits": holdings, "utility": value, "convention": convention,
                "n_satisfied": n_sat, "report": report, "final_alpha": alpha, "evals": evals,
            }
            restart_results.append(result_r)
            if best_restart is None or (result_r["n_satisfied"], result_r["utility"]) > \
                    (best_restart["n_satisfied"], best_restart["utility"]):
                best_restart = result_r

        lots = encoding.decode_lots(best_restart["bits"])
        w = (p.budget / self.n_lots) * lots.astype(float)

        return {
            "weights": w,
            "lots": lots,
            "utility": p.utility(w),
            "utility_qubo_proxy": best_restart["utility"],
            "expected_return": p.expected_return(w),
            "variance": p.variance(w),
            "volatility": float(np.sqrt(max(p.variance(w), 0.0))),
            "turnover": p.turnover(w),
            "cost": p.cost(w),
            "n_lots": self.n_lots,
            "method": "quantum:qaoa_pce",
            "n_qubits": num_qubits_pce,
            "n_bit_vars": n_vars,
            "n_constraints": len(constraints),
            "n_constraints_satisfied": best_restart["n_satisfied"],
            "constraint_report": best_restart["report"],
            "decoding_convention": best_restart["convention"],
            "best_seed": best_restart["seed"],
            "restart_results": restart_results,
            "total_evals": sum(r["evals"] for r in restart_results),
        }

    def _run_iterative_alpha_pce(self, constraints, init_params, isa_ansatz, pce_ops_isa_flat,
                                  ordered_vars, estimator, linear, quadratic, var_names,
                                  v_scale, label=""):
        alpha = self.alpha_0
        params = init_params.copy()
        all_evals = 0
        best_entry = None
        pivot_history = []

        for outer_iter in range(self.max_alpha_iters):
            inner_log = []
            result = minimize(
                loss_func_estimator, params,
                args=(isa_ansatz, pce_ops_isa_flat, ordered_vars, estimator, inner_log,
                      alpha, constraints, linear, quadratic, var_names,
                      self.beta_reg, v_scale),
                method="COBYLA",
                options={"maxiter": self.inner_maxiter},
            )
            params = result.x
            all_evals += len(inner_log)
            best_entry = min(inner_log, key=lambda e: e["loss"])
            exp_map = best_entry["exp_map"]

            z = {name: np.tanh(alpha * exp_map[name]) for name in var_names}
            under_thresh = [name for name, zv in z.items() if abs(zv) < self.threshold_m]

            print(f"  {label}[outer {outer_iter:2d}] alpha={alpha:10.4g}  "
                  f"best_loss={best_entry['loss']:10.4f}  "
                  f"unbinarized={len(under_thresh)}/{len(var_names)}")

            if not under_thresh:
                print(f"  All {len(var_names)} variables binarized "
                      f"(|tanh(alpha*<P>)| >= {self.threshold_m}). Stopping.")
                break

            i_star = min(under_thresh, key=lambda name: abs(abs(z[name]) - self.threshold_m))
            z_istar = np.clip(abs(z[i_star]), 1e-9, 0.999999)
            uncapped_ratio = np.arctanh(self.threshold_m) / z_istar
            growth_ratio = min(uncapped_ratio, self.max_alpha_growth_per_step)
            alpha = alpha * growth_ratio
            pivot_history.append(i_star)
        else:
            print(f"  Reached max_alpha_iters={self.max_alpha_iters} without full "
                  f"binarization (still usable, may be partially infeasible).")

        if pivot_history:
            counts = Counter(pivot_history)
            top_var, top_count = counts.most_common(1)[0]
            if top_count >= max(3, len(pivot_history) // 2):
                print(f"  -> '{top_var}' pivoted {top_count}/{len(pivot_history)} times: "
                      f"looks like a structural bottleneck for this variable's qubit pair.")

        return params, alpha, best_entry, all_evals


def quantum_mean_variance_discrete(
    problem: PortfolioProblem,
    mu: np.ndarray,
    cov: np.ndarray,
    transaction_cost: np.ndarray,
    prev_weights: np.ndarray,
    group_map: List[int],
    group_lower: List[float],
    group_upper: List[float],
    group_names: List[str],
    n_lots: int = 20,
    **kwargs,
) -> Dict[str, object]:
    """Convenience wrapper mirroring classical_discrete.mean_variance_discrete."""
    optimizer = QuantumMeanVarianceDiscreteOptimizer(
        problem=problem, n_lots=n_lots, group_map=group_map,
        group_lower=group_lower, group_upper=group_upper, group_names=group_names,
        **kwargs,
    )
    return optimizer.solve(mu, cov, transaction_cost, prev_weights)


def _format_result(problem: PortfolioProblem, result: Dict[str, object]) -> str:
    lines = [
        f"Quantum (QAOA+PCE) discrete allocation (n_lots={result['n_lots']}, "
        f"qubits={result['n_qubits']}, bit_vars={result['n_bit_vars']})",
        "-" * 60,
    ]
    for name, w, k in zip(problem.asset_names, result["weights"], result["lots"]):
        lines.append(f"  {name:<14} {w:>8.2%}   ({int(k)} lots)")
    lines.append("-" * 60)
    lines.append(f"  Expected return {result['expected_return']:>8.2%}")
    lines.append(f"  Volatility      {result['volatility']:>8.2%}")
    lines.append(f"  Variance        {result['variance']:>8.4f}")
    lines.append(f"  Turnover        {result['turnover']:>8.2%}")
    lines.append(f"  Cost            {result['cost']:>8.4f}")
    lines.append(f"  Utility (real)  {result['utility']:>8.4f}")
    lines.append(f"  Utility (proxy) {result['utility_qubo_proxy']:>8.4f}")
    lines.append(
        f"  Constraints     {result['n_constraints_satisfied']}/{result['n_constraints']} "
        f"satisfied (decoding: {result['decoding_convention']})"
    )
    return "\n".join(lines)