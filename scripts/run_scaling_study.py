"""Run the full solver comparison across a range of asset counts, to see
how the classical baselines and the PCE-compressed VQE solver's runtime,
qubit count, and solution quality trend as the problem scales up.

Usage (from the repository root)::

    python scripts/run_scaling_study.py
    python scripts/run_scaling_study.py --sizes 6 10 15 20 30
    python scripts/run_scaling_study.py --sizes 6 12 24 --seed 7

For n_assets=6, the existing data/synthetic/synthetic_universe.json is
used (if present) so results stay comparable with earlier single-size
runs; every other size is generated fresh via generate_synthetic_universe
(same script as `python scripts/generate_synthetic_universe.py`) and
written to data/synthetic/synthetic_universe_<n>.json.

What gets skipped, and why
---------------------------
Classical brute force (exact discrete optimum) is exponential-ish in
n_assets: the number of lot compositions is C(n_lots + n_assets - 1,
n_assets - 1). At n_assets=6, n_lots=20 that's a trivial 53,130. It grows
fast -- by n_assets=15 it's already in the hundreds of millions. This
script skips brute force above BRUTE_MAX_COMPOSITIONS and falls back to
simulated annealing as the "classical reference" for the utility-gap
comparison at larger sizes, printing clearly which reference was used at
each size so the numbers stay honest (annealing is a strong heuristic,
but it's not a certified exact optimum the way brute force is).

The PCE-VQE's qubit count is NOT held fixed across sizes -- n_qubits is
recomputed at each size by probing the solver's own PCE-minimum (rather
than recomputing the formula independently in this script, which risks
silently drifting from whatever quantum_discrete.py actually implements)
and adding a fixed margin (see --qubit-margin).

>>> Config knobs now SCALE with problem size, not held fixed <<<
An earlier version of this script held every other VQEPCEConfig field
(shots, budget, top_k) constant across sizes "for a clean first look."
That produced misleadingly bad results at larger sizes for two
compounding, self-inflicted reasons, visible directly in the solver's
own printed diagnostics:
  1. The reachable-state coupon-collector threshold grows with n_qubits,
     but n0_shots didn't -- by n_assets=30 (n_qubits=11), 300 shots was
     only ~2% of what's needed for a meaningful frequency signal, so
     training was working from close to noise.
  2. The truly-feasible slice of the reachable space shrinks fast with
     more assets (more ways to violate budget/bounds/sector at once),
     while final_top_k's search pool size stayed flat -- at n_assets=15
     and 30 the pool contained ZERO fully-feasible candidates at all,
     forcing bit_flip_postprocess to do all the work via greedy repair
     from a broken starting point rather than the trained circuit
     finding anything close to a good feasible region.
Fixed via `_scaled_vqe_config()`: n0_shots, final_shots, total_budget,
and final_top_k are all now derived from n_qubits/reachable-state-count
at each size, so sampling resolution and search-pool size grow roughly
in step with the problem instead of staying fixed while the problem gets
harder. This is still a first-pass scaling heuristic, not a tuned
schedule -- if quality still degrades with these in place, that's a more
meaningful finding about the compressed representation itself.
>>> Uncompressed VQE is included too, with a time budget <<<
`quantum_vqe_solver.py`'s plain VQE (one qubit per lot-bit, no PCE
compression) is now run alongside PCE-VQE at each size, so you can see
directly what the compression is buying (or costing) at each scale --
not just against the classical references.

Its qubit count is n_assets * bits_per_asset (no compression), so its
circuits and per-evaluation shot counts grow much faster than PCE-VQE's
as n_assets increases. Rather than guess a runtime-vs-size formula the
way brute force's exact composition count allows, this script tracks
CUMULATIVE wall-clock time actually spent on the plain VQE across sizes
processed so far, via `_TimeBudget`. Once that exceeds
PLAIN_VQE_TIME_BUDGET_SEC (2 minutes by default), every subsequent size
skips the plain VQE entirely (PCE-VQE and the classical baselines keep
running as normal). This is a coarse "don't start another size once
we're over budget" gate, not a hard per-call kill switch -- a single
size's run can still finish once started even if it pushes well past the
budget, since forcibly interrupting a running qiskit/Aer computation
mid-flight is its own can of worms and not worth the complexity here. If
you need a hard timeout, look at wrapping the call in a
`multiprocessing.Process` with `.join(timeout=...)` + `.terminate()`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.synthetic_uni_scale import generate_synthetic_universe  # noqa: E402
from src.vanguard_portfolio.classical_continuous import mean_variance_continuous  # noqa: E402
from src.vanguard_portfolio.classical_discrete import mean_variance_discrete  # noqa: E402
from src.vanguard_portfolio.quantum_vqe_pce_solver import (  # noqa: E402
    N_LOTS_DEFAULT,
    PortfolioVQEPCESolver,
    VQEPCEConfig,
    run_pce_vqe,
)
from src.vanguard_portfolio.quantum_vqe_solver import (  # noqa: E402
    PortfolioVQESolver,
    VQEConfig,
    load_universe_from_json,
)

DEFAULT_SIZES = [6, 10, 15, 20, 25]
BRUTE_MAX_COMPOSITIONS = 200_000
QUBIT_MARGIN = 2  # extra qubits above the PCE minimum, at every size --
#                    see quantum_vqe_pce_solver.py's module docstring for
#                    why the minimum alone causes an exhaustive-coverage
#                    problem during training.
PLAIN_VQE_TIME_BUDGET_SEC = 150.0  # see module docstring's "uncompressed
#                                     VQE is included too" section.


class _TimeBudget:
    """Tracks cumulative wall-clock time spent on one solver across sizes;
    once exhausted, every later size skips that solver. See module
    docstring for why this is a coarse gate, not a hard per-call timeout."""

    def __init__(self, seconds: float):
        self.remaining = seconds
        self.exhausted = False

    def consume(self, seconds: float) -> None:
        self.remaining -= seconds
        if self.remaining <= 0:
            self.exhausted = True


def _universe_path_for(n_assets: int, seed: int) -> Path:
    default_6 = ROOT / "data" / "synthetic" / "synthetic_universe.json"
    if n_assets == 6 and default_6.exists():
        return default_6
    path = ROOT / "data" / "synthetic" / f"synthetic_universe_{n_assets}.json"
    if not path.exists():
        import json
        data = generate_synthetic_universe(n_assets, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        print(f"  (generated {path})")
    return path


def _n_compositions(n_lots: int, n_assets: int) -> int:
    return math.comb(n_lots + n_assets - 1, n_assets - 1)


def _scaled_vqe_config(problem, n_lots: int, seed: int, qubit_margin: int,
                        total_budget_override: Optional[int] = None):
    """Derive a VQEPCEConfig whose shot counts, training budget, and
    final-selection pool size scale with n_qubits at this problem size,
    instead of reusing the same fixed defaults regardless of scale --
    see module docstring for why that was producing misleadingly bad
    results at larger n_assets.

    n_qubits itself is obtained by PROBING the solver's own PCE-minimum
    (constructing a throwaway solver with n_qubits_override=None) rather
    than recomputing reduce_qubits_with_pce independently here -- keeps
    this script from silently drifting out of sync with whatever
    quantum_discrete.py actually implements.

    total_budget_override, if given, is used AS-IS instead of the
    max(300, n_params*15) auto-scaled floor -- previously there was no
    way to actually raise total_budget from the command line: it was
    always silently recomputed here regardless of anything the caller
    wanted, so passing e.g. --total-budget 900 had no effect at all.

    Returns (config, n_vars) -- n_vars is read off the same probe solver
    so callers don't need to construct yet another solver just to learn it.
    """
    probe_cfg = VQEPCEConfig(n_lots=n_lots, n_qubits_override=None, seed=seed)
    probe = PortfolioVQEPCESolver(problem, probe_cfg)
    pce_min_qubits = probe.n_qubits

    n_qubits = pce_min_qubits + qubit_margin
    reachable = 2 ** n_qubits
    n_params = n_qubits * 2  # reps=1 default -> n_qubits * (reps + 1)

    n0_shots = max(300, reachable * 3)          # a handful of shots/state
    #                                              on average, comfortably
    #                                              under the coupon-collector
    #                                              threshold (~reachable*ln
    #                                              (reachable)) so training
    #                                              still has room for theta
    #                                              to matter -- just no
    #                                              longer flat at 300
    #                                              regardless of how big
    #                                              reachable has gotten.
    final_shots = max(20_000, reachable * 25)   # richer resolution for the
    #                                              final frequency-based
    #                                              candidate selection.
    if total_budget_override is not None:
        total_budget = total_budget_override
    else:
        total_budget = max(300, n_params * 15)  # more ansatz parameters
        #                                          need more training
        #                                          evaluations to search
        #                                          meaningfully.
    final_top_k = min(reachable, max(20, reachable // 6))
    #                                            -- search-pool size scales
    #                                              with the reachable space,
    #                                              since the feasible
    #                                              fraction shrinks fast
    #                                              with more assets (see
    #                                              module docstring).

    return VQEPCEConfig(
        n_lots=n_lots,
        n_qubits_override=n_qubits,
        seed=seed,
        n0_shots=n0_shots,
        final_shots=final_shots,
        total_budget=total_budget,
        final_top_k=final_top_k,
    ), probe.n_vars


def run_one_size(n_assets: int, seed: int, n_lots: int = N_LOTS_DEFAULT,
                  qubit_margin: int = QUBIT_MARGIN,
                  plain_vqe_budget: Optional["_TimeBudget"] = None,
                  total_budget: Optional[int] = None) -> dict:
    print(f"\n{'=' * 70}\nn_assets = {n_assets}\n{'=' * 70}")

    path = _universe_path_for(n_assets, seed)
    problem = load_universe_from_json(path)

    row: dict = {"n_assets": n_assets}

    t0 = time.perf_counter()
    cont = mean_variance_continuous(problem)
    row["continuous_utility"] = cont["utility"]
    row["continuous_sec"] = time.perf_counter() - t0
    print(f"  continuous: utility={cont['utility']:.5f}  ({row['continuous_sec']:.2f}s)")

    n_comp = _n_compositions(n_lots, n_assets)
    brute_ran = n_comp <= BRUTE_MAX_COMPOSITIONS
    if brute_ran:
        t0 = time.perf_counter()
        brute = mean_variance_discrete(problem, n_lots=n_lots, method="brute")
        row["brute_sec"] = time.perf_counter() - t0
        row["brute_utility"] = brute["utility"]
        print(
            f"  discrete-brute: {n_comp:,} compositions, "
            f"utility={brute['utility']:.5f}  ({row['brute_sec']:.2f}s)"
        )
    else:
        print(f"  discrete-brute: SKIPPED ({n_comp:,} compositions > {BRUTE_MAX_COMPOSITIONS:,})")
        row["brute_sec"] = None
        row["brute_utility"] = None

    t0 = time.perf_counter()
    anneal = mean_variance_discrete(problem, n_lots=n_lots, method="anneal", seed=seed)
    row["anneal_sec"] = time.perf_counter() - t0
    row["anneal_utility"] = anneal["utility"]
    print(f"  discrete-anneal: utility={anneal['utility']:.5f}  ({row['anneal_sec']:.2f}s)")

    reference_name = "brute" if brute_ran else "anneal"
    reference_utility = row["brute_utility"] if brute_ran else row["anneal_utility"]
    row["reference_method"] = reference_name

    cfg, n_vars_probe = _scaled_vqe_config(
        problem, n_lots, seed, qubit_margin, total_budget_override=total_budget
    )
    t0 = time.perf_counter()
    pce_result = run_pce_vqe(problem, config=cfg)
    row["pce_sec"] = time.perf_counter() - t0
    row["n_vars"] = n_vars_probe
    row["n_qubits"] = cfg.n_qubits_override
    row["n0_shots"] = cfg.n0_shots
    row["final_shots"] = cfg.final_shots
    row["total_budget"] = cfg.total_budget
    row["final_top_k"] = cfg.final_top_k
    row["pce_utility_pp"] = pce_result["utility_pp"]
    row["pce_feasible_pp"] = bool(
        pce_result["budget_ok_pp"] and pce_result["bounds_ok_pp"]
        and pce_result["sector_penalty_pp"] < 1e-9
    )
    row["pce_total_evals"] = pce_result["total_evals"]
    row["frac_all_ok"] = pce_result["feasibility_stats"].get("frac_all_ok")

    if row["pce_feasible_pp"] and reference_utility:
        row["gap_pct"] = 100 * (reference_utility - row["pce_utility_pp"]) / abs(reference_utility)
    else:
        row["gap_pct"] = None

    print(
        f"  pce-vqe: n_vars={row['n_vars']} -> n_qubits={row['n_qubits']}  "
        f"(n0_shots={row['n0_shots']}, final_shots={row['final_shots']}, "
        f"total_budget={row['total_budget']}, final_top_k={row['final_top_k']})"
    )
    print(
        f"  utility={pce_result['utility_pp']:.5f}  "
        f"feasible={row['pce_feasible_pp']}  "
        f"frac_all_ok={row['frac_all_ok']:.1%}  ({row['pce_sec']:.2f}s)"
    )
    gap_str = f"{row['gap_pct']:.2f}%" if row["gap_pct"] is not None else "n/a (infeasible)"
    print(f"  gap vs {reference_name}: {gap_str}")

    if plain_vqe_budget is not None and plain_vqe_budget.exhausted:
        print(
            f"  vqe-plain (uncompressed): SKIPPED (already used up the "
            f"{PLAIN_VQE_TIME_BUDGET_SEC:.0f}s time budget for this solver "
            f"on smaller sizes)"
        )
        row["vqe_plain_sec"] = None
        row["vqe_plain_n_qubits"] = None
        row["vqe_plain_utility_pp"] = None
        row["vqe_plain_feasible_pp"] = None
        row["vqe_plain_gap_pct"] = None
    else:
        vqe_cfg = VQEConfig(
            n_lots=n_lots, seed=seed,
            **({"total_budget": total_budget} if total_budget is not None else {}),
        )
        vqe_solver = PortfolioVQESolver(problem, vqe_cfg)
        t0 = time.perf_counter()
        vqe_result = vqe_solver.run()
        vqe_elapsed = time.perf_counter() - t0
        if plain_vqe_budget is not None:
            plain_vqe_budget.consume(vqe_elapsed)

        row["vqe_plain_sec"] = vqe_elapsed
        row["vqe_plain_n_qubits"] = vqe_solver.n_qubits
        row["vqe_plain_utility_pp"] = vqe_result["utility_pp"]
        vqe_feasible = bool(
            vqe_result["budget_ok_pp"] and vqe_result["bounds_ok_pp"]
            and vqe_result["sector_penalty_pp"] < 1e-9
        )
        row["vqe_plain_feasible_pp"] = vqe_feasible
        if vqe_feasible and reference_utility:
            row["vqe_plain_gap_pct"] = (
                100 * (reference_utility - vqe_result["utility_pp"]) / abs(reference_utility)
            )
        else:
            row["vqe_plain_gap_pct"] = None

        print(
            f"  vqe-plain (uncompressed): n_qubits={vqe_solver.n_qubits}  "
            f"utility={vqe_result['utility_pp']:.5f}  "
            f"feasible={vqe_feasible}  ({vqe_elapsed:.2f}s, "
            f"{plain_vqe_budget.remaining:.0f}s time budget left)"
            if plain_vqe_budget is not None else
            f"  vqe-plain (uncompressed): n_qubits={vqe_solver.n_qubits}  "
            f"utility={vqe_result['utility_pp']:.5f}  "
            f"feasible={vqe_feasible}  ({vqe_elapsed:.2f}s)"
        )
        vqe_gap_str = (
            f"{row['vqe_plain_gap_pct']:.2f}%" if row["vqe_plain_gap_pct"] is not None
            else "n/a (infeasible)"
        )
        print(f"  gap vs {reference_name}: {vqe_gap_str}")

    return row


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 100}\nScaling summary\n{'=' * 100}")
    header = (
        f"{'n_assets':>8} {'n_vars':>7} {'n_qubits':>8} {'ref':>7} "
        f"{'pce_gap_%':>10} {'vqe_gap_%':>10} {'all_ok_%':>9} "
        f"{'pce_sec':>9} {'vqe_sec':>9} {'ref_sec':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ref_sec = r["brute_sec"] if r["reference_method"] == "brute" else r["anneal_sec"]
        gap_str = f"{r['gap_pct']:.2f}" if r["gap_pct"] is not None else "n/a"
        vqe_gap_str = (
            f"{r['vqe_plain_gap_pct']:.2f}" if r.get("vqe_plain_gap_pct") is not None
            else ("skip" if r.get("vqe_plain_sec") is None else "n/a")
        )
        vqe_sec_str = f"{r['vqe_plain_sec']:.2f}" if r.get("vqe_plain_sec") is not None else "skip"
        all_ok_str = f"{100 * r['frac_all_ok']:.2f}" if r.get("frac_all_ok") is not None else "n/a"
        print(
            f"{r['n_assets']:>8} {r['n_vars']:>7} {r['n_qubits']:>8} "
            f"{r['reference_method']:>7} {gap_str:>10} {vqe_gap_str:>10} {all_ok_str:>9} "
            f"{r['pce_sec']:>9.2f} {vqe_sec_str:>9} {ref_sec:>9.2f}"
        )


def _save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "n_assets", "n_vars", "n_qubits", "n0_shots", "final_shots",
        "total_budget", "final_top_k", "reference_method", "gap_pct",
        "frac_all_ok", "continuous_utility", "brute_utility", "anneal_utility",
        "pce_utility_pp", "pce_feasible_pp", "pce_total_evals",
        "vqe_plain_n_qubits", "vqe_plain_utility_pp", "vqe_plain_feasible_pp",
        "vqe_plain_gap_pct", "vqe_plain_sec",
        "continuous_sec", "brute_sec", "anneal_sec", "pce_sec",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nSaved {path}")


def _save_plot(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot; CSV/table above still has everything)")
        return

    sizes = [r["n_assets"] for r in rows]
    gaps = [r["gap_pct"] for r in rows]
    pce_sec = [r["pce_sec"] for r in rows]
    ref_sec = [r["brute_sec"] if r["reference_method"] == "brute" else r["anneal_sec"] for r in rows]
    qubits = [r["n_qubits"] for r in rows]
    frac_all_ok = [100 * r["frac_all_ok"] if r.get("frac_all_ok") is not None else None for r in rows]

    # vqe-plain series may have None entries once its time budget runs out
    # partway through -- filter those out per-series rather than breaking
    # the whole plot.
    vqe_sizes_gap = [r["n_assets"] for r in rows if r.get("vqe_plain_gap_pct") is not None]
    vqe_gaps = [r["vqe_plain_gap_pct"] for r in rows if r.get("vqe_plain_gap_pct") is not None]
    vqe_sizes_sec = [r["n_assets"] for r in rows if r.get("vqe_plain_sec") is not None]
    vqe_sec = [r["vqe_plain_sec"] for r in rows if r.get("vqe_plain_sec") is not None]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))

    axes[0].plot(sizes, gaps, marker="o", label="pce-vqe")
    if vqe_gaps:
        axes[0].plot(vqe_sizes_gap, vqe_gaps, marker="o", label="vqe-plain (uncompressed)")
    axes[0].set_xlabel("n_assets")
    axes[0].set_ylabel("utility gap vs. classical reference (%)")
    axes[0].set_title("Solution quality gap")
    axes[0].legend()

    axes[1].plot(sizes, pce_sec, marker="o", label="pce-vqe")
    if vqe_sec:
        axes[1].plot(vqe_sizes_sec, vqe_sec, marker="o", label="vqe-plain (uncompressed)")
    axes[1].plot(sizes, ref_sec, marker="o", label="classical reference")
    axes[1].set_xlabel("n_assets")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Runtime")
    axes[1].legend()

    axes[2].plot(sizes, qubits, marker="o")
    axes[2].set_xlabel("n_assets")
    axes[2].set_ylabel("qubits used")
    axes[2].set_title("PCE qubit count")

    axes[3].plot(sizes, frac_all_ok, marker="o", color="tab:red")
    axes[3].set_xlabel("n_assets")
    axes[3].set_ylabel("% of final pool fully feasible")
    axes[3].set_title("Feasible-pool fraction\n(leading indicator for the gap)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-lots", type=int, default=N_LOTS_DEFAULT)
    parser.add_argument("--qubit-margin", type=int, default=QUBIT_MARGIN)
    parser.add_argument("--total-budget", type=int, default=None,
                         help="Override total_budget for BOTH pce-vqe and vqe-plain "
                              "at every size, instead of pce-vqe's auto-scaled "
                              "max(300, n_params*15) floor and vqe-plain's flat "
                              "VQEConfig default of 300. Previously there was no "
                              "way to actually raise this from the command line -- "
                              "it was silently recomputed regardless of intent.")
    parser.add_argument("--plain-vqe-time-budget", type=float, default=PLAIN_VQE_TIME_BUDGET_SEC,
                         help="Cumulative seconds to spend on the uncompressed VQE "
                              "solver across all sizes before skipping it for the "
                              "rest of the sweep. See module docstring.")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-plot", type=str, default=None)
    args = parser.parse_args()

    plain_vqe_budget = _TimeBudget(args.plain_vqe_time_budget)

    rows = [
        run_one_size(
            n, args.seed, args.n_lots,
            qubit_margin=args.qubit_margin,
            plain_vqe_budget=plain_vqe_budget,
            total_budget=args.total_budget,
        )
        for n in args.sizes
    ]

    _print_summary(rows)

    out_csv = Path(args.out_csv) if args.out_csv else ROOT / "data" / "scaling_study.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(rows, out_csv)

    out_plot = Path(args.out_plot) if args.out_plot else ROOT / "data" / "scaling_study.png"
    _save_plot(rows, out_plot)

    print("\nDone.")


if __name__ == "__main__":
    main()
