"""
qubit_overhead_sweep.py
========================
Small sweep utility for `quantum_vqe_pce_solver.PortfolioVQEPCESolver`.

WHAT THIS SWEEPS
----------------
`VQEPCEConfig.n_qubits_override` lets you give PCE MORE qubits than the
bare minimum `reduce_qubits_with_pce(n_vars)` needs to fit all the lot-bit
variables. Nothing about PCE requires using the minimum -- it's just the
smallest register that *can* hold every variable's correlator pair.

Why "a couple extra qubits" is worth trying at all:

  * Each basis setting's reachable-state count is `2**n_qubits`. Going
    from the minimum to minimum+1 DOUBLES it; +2 quadruples it. That
    directly widens what `_final_candidates`'s frequency-based top-k
    selection has to work with, and gives `sample_and_evaluate`'s CVaR
    loss a less saturated space to shape during training.
  * More qubits also means more available two-qubit combinations
    (`C(n_qubits, 2)`) to spread each basis group's variables across --
    at the PCE minimum, pairs are packed about as tightly as they can be;
    a little slack can reduce how many variables are forced to share
    "crowded" qubits.
  * The cost of this is real but modest at these sizes: `2**n_qubits`
    reachable states per basis setting is still tiny (dozens to low
    thousands) for +0..+5 over a ~5-9 qubit PCE minimum, so a full sweep
    over this range stays cheap to run in its entirety.

This sweep is deliberately small (default: overhead 0 through 5) --
PCE's whole appeal is a small register, so this is meant to answer "does
a *little* slack help" rather than to explore how far compression can be
pushed before it stops mattering.

USAGE
-----
    from src.vanguard_portfolio.qubit_overhead_sweep import run_qubit_overhead_sweep

    results = run_qubit_overhead_sweep(problem, n_lots=20)
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable, Optional

from .classical_continuous import PortfolioProblem
from .quantum_discrete import build_lot_encoding, reduce_qubits_with_pce
from .quantum_vqe_pce_solver import N_LOTS_DEFAULT, PortfolioVQEPCESolver, VQEPCEConfig, compute_pce_min_qubits
DEFAULT_OVERHEADS = range(0, 6)   # 0, 1, 2, 3, 4, 5 extra qubits


def sweep_qubit_overhead(
    problem: PortfolioProblem,
    overheads: Iterable[int] = DEFAULT_OVERHEADS,
    n_lots: int = N_LOTS_DEFAULT,
    base_config: Optional[VQEPCEConfig] = None,
) -> list[dict]:
    """
    Run the PCE VQE solver once per qubit-overhead value in `overheads`,
    where overhead is ADDED on top of the PCE-minimum qubit count via
    `n_qubits_override`. Returns one result dict per overhead tested
    (PortfolioVQEPCESolver.run()'s normal output, plus 'overhead' and
    'n_qubits' keys).

    `base_config`, if given, is used as the template for every run --
    only `n_qubits_override` is varied. Anything else you want fixed
    across the sweep (n_lots, reps, seed, shot counts, ...) should be set
    on `base_config` directly rather than passed as a separate kwarg,
    so every point in the sweep is otherwise apples-to-apples.
    """
    base_cfg = base_config or VQEPCEConfig(n_lots=n_lots)

    # PCE minimum is fixed by the problem size alone, independent of
    # overhead -- compute it once so every sweep point offsets from the
    # same baseline (and so we can report it alongside each result).
    pce_min_qubits = compute_pce_min_qubits(problem, base_cfg.n_lots)

    results: list[dict] = []
    for overhead in overheads:
        if overhead < 0:
            raise ValueError(f"overhead must be >= 0, got {overhead}")

        n_qubits = pce_min_qubits + overhead
        cfg = replace(base_cfg, n_qubits_override=n_qubits)

        print(
            f"\n{'=' * 70}\n"
            f"Qubit overhead sweep: +{overhead} "
            f"(n_qubits={n_qubits}, PCE minimum={pce_min_qubits}, "
            f"reachable/basis-setting=2**{n_qubits}={2 ** n_qubits})\n"
            f"{'=' * 70}"
        )
        solver = PortfolioVQEPCESolver(problem, cfg)
        t0 = time.time()
        result = solver.run()
        result["runtime_s"] = time.time() - t0
        result["overhead"] = overhead
        result["n_qubits"] = n_qubits
        result["pce_min_qubits"] = pce_min_qubits
        results.append(result)

    return results


def _print_qubit_overhead_summary(results: list[dict]) -> None:
    header = (
        f"{'overhead':>8}{'n_qubits':>9}{'reachable':>11}"
        f"{'utility_pp':>12}{'feasible':>10}{'all_ok%':>9}{'evals':>8}{'runtime_s':>11}"
    )
    print("\nQubit overhead sweep summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        feasible = r["budget_ok_pp"] and r["bounds_ok_pp"]
        all_ok_frac = r["feasibility_stats"].get("frac_all_ok", float("nan"))
        reachable = 2 ** r["n_qubits"]
        print(
            f"{r['overhead']:>8}{r['n_qubits']:>9}{reachable:>11}"
            f"{r['utility_pp']:>12.5f}{str(feasible):>10}"
            f"{all_ok_frac:>9.1%}{r['total_evals']:>8}{r.get('runtime_s', float('nan')):>11.1f}"
        )

    best = max(
        (r for r in results if r["budget_ok_pp"] and r["bounds_ok_pp"]),
        key=lambda r: r["utility_pp"],
        default=None,
    )
    if best is not None:
        print(
            f"\nBest feasible result: overhead=+{best['overhead']} "
            f"(n_qubits={best['n_qubits']}), utility_pp={best['utility_pp']:.5f}"
        )
    else:
        print(
            "\nNo overhead in this sweep produced a fully-feasible "
            "postprocessed solution -- consider a wider overhead range, "
            "or check pen_weight calibration / n0_shots first."
        )


def run_qubit_overhead_sweep(
    problem: PortfolioProblem,
    n_lots: int = N_LOTS_DEFAULT,
    overheads: Iterable[int] = DEFAULT_OVERHEADS,
    config: Optional[VQEPCEConfig] = None,
) -> list[dict]:
    """Convenience wrapper: run the sweep and print a comparison table,
    mirroring the shape of run_vqe / run_pce_vqe elsewhere in this package."""
    results = sweep_qubit_overhead(problem, overheads=overheads, n_lots=n_lots, base_config=config)
    _print_qubit_overhead_summary(results)
    return results
