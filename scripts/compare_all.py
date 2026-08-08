"""compare_all.py
===================
Runtime + solution-quality comparison across the three portfolio solvers in
this project, each left at its own as-shipped default configuration, all
solving the *same* single-period problem loaded from a shared
`synthetic_universe.json`:

  1. classical_discrete.MeanVarianceDiscreteOptimizer   (method="anneal")
  2. quantum_discrete.QuantumMeanVarianceDiscreteOptimizer   (QAOA + PCE)
  3. quantum_vqe_solver.PortfolioVQESolver   (sampling VQA: PSO -> NFT)

...plus a zero-domain-knowledge baseline, `random` (see Section 5 below),
so the three real solvers' numbers have a floor to be compared against,
not just each other.

Usage
-----
    python compare_all.py [path/to/universe.json]

Runs directly from anywhere in the repo -- no `python -m`, no package
`__init__.py` needed, and no fixed location relative to
classical_continuous.py etc. required. It locates its sibling modules
automatically (see `_bootstrap_import_path` below). If no path is given,
"synthetic_universe.json" next to this file is used.
Pass --skip-qaoa / --skip-vqe / --skip-classical / --skip-random /
--skip-brute to leave a solver out.
classical:brute (exact enumeration) runs by default alongside the others
-- see --skip-brute's help text for why that's safe at this problem size.

Design notes
------------
* n_lots is pinned to 20 (--n-lots to override) for every solver so the
  comparison is apples-to-apples on lattice resolution. No other
  hyperparameter is retuned here -- reps, n_restarts, PSO/NFT budgets,
  etc. are whatever each solver's dataclass already defaults to. That is
  the whole point of this script: "what do you get today, out of the box,
  from each method, on the same problem, and how long does each take."
* The problem is built directly from JSON here (not by importing
  quantum_vqe_solver.load_universe_from_json), so that the classical-only
  path has *zero* dependency on qiskit -- if qiskit / qiskit-aer /
  qiskit-ibm-runtime aren't installed, classical still runs standalone and
  the two quantum rows are reported as skipped rather than crashing the
  whole comparison.
* Every row's headline "utility" is the *real* PortfolioProblem.utility(w)
  -- not a solver-internal training proxy (QAOA's QUBO surrogate loss, or
  VQE's penalized sampling cost) -- so all three numbers are directly
  comparable. Proxy values are still printed underneath each row for
  reference.
* Feasibility is normalized across the three solvers into a single
  feasible=True/False flag per row (see `_normalize_result` below) even
  though each solver reports it differently: classical enforces
  budget/per-asset bounds by construction and only *soft*-penalizes
  one-sided group exposure; QAOA enforces budget/per-asset/group bounds as
  hard constraints but can still land outside them after decoding, hence
  its own n_satisfied/n_constraints report; VQE penalizes all three
  (budget, per-asset, two-sided group) and reports budget_ok/bounds_ok
  booleans plus a numeric sector penalty. Because the three solvers'
  constraint sets are not identical (see the "why these aren't directly
  comparable" sections of the model-formulation docs), read the
  feasibility column as "did this solver satisfy *its own* constraint
  definition," not as a single shared notion of feasibility.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _ensure_numpy() -> None:
    """Make sure `import numpy` will succeed under *this* interpreter
    before anything else in the file runs, so a plain `python
    compare_all.py` "just works" the same way from any directory,
    regardless of which `python` happens to be first on PATH.

    Without this, running the script under an interpreter that never had
    the project's dependencies installed into it (e.g. a bare Conda
    `base` env, or system python) fails with a bare
    `ModuleNotFoundError: No module named 'numpy'` raised from deep
    inside the script -- unhelpful, and easy to misread as "the script is
    broken" rather than "wrong interpreter."

    Fix: if numpy isn't importable here, search upward from this file for
    a project virtualenv (`.venv`, `venv`, or `env`, the common names) that
    *does* have numpy installed, and transparently re-exec this exact
    invocation under that venv's python. If no such venv is found, fail
    with a short, actionable message instead of a traceback.
    """
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve().parent
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    python_name = "python.exe" if os.name == "nt" else "python"
    current = Path(sys.executable).resolve()

    for root in (here, *here.parents):
        for venv_name in (".venv", "venv", "env"):
            candidate = root / venv_name / bin_dir / python_name
            if not candidate.exists() or candidate.resolve() == current:
                continue
            probe = subprocess.run(
                [str(candidate), "-c", "import numpy"],
                capture_output=True,
            )
            if probe.returncode == 0:
                print(
                    f"[compare_all.py] '{sys.executable}' has no numpy -- "
                    f"re-running under '{candidate}' instead.",
                    file=sys.stderr,
                )
                os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])

    raise SystemExit(
        f"numpy is not installed for this interpreter ({sys.executable}).\n"
        "Fix by either:\n"
        "  1) activating this project's virtualenv before running this "
        "script, e.g.:\n"
        "       source .venv/bin/activate   (macOS/Linux)\n"
        "       .venv\\Scripts\\activate      (Windows)\n"
        "  2) installing dependencies for the interpreter you're currently "
        f"using:\n       {sys.executable} -m pip install numpy\n"
        "No .venv/venv/env directory with numpy already installed was "
        "found automatically under this file's directory tree, so this "
        "couldn't be resolved for you."
    )


_ensure_numpy()

import argparse
import importlib
import json
import time
import traceback
from typing import Optional

import numpy as np


def _bootstrap_import_path() -> tuple[Path, str]:
    """Make classical_continuous / classical_discrete / quantum_discrete /
    quantum_vqe_solver importable regardless of where this script lives in
    the repo, and return `(package_dir, package_name)`.

    Two separate problems show up when this script is moved out of the
    package (e.g. into a top-level scripts/ folder) and run directly:

    1. This file's own `from .classical_continuous import ...` fails with
       "attempted relative import with no known parent package", because a
       directly-run script has no package context at all.
    2. Even after fixing (1) by importing classical_discrete.py as a bare
       top-level module, classical_discrete.py's *own* internal
       `from .classical_continuous import PortfolioProblem` breaks the
       same way -- a bare top-level module still has no package context,
       so its relative import fails too, just one level deeper.

    Fix: don't import the sibling modules as bare top-level names. Find
    the directory containing classical_continuous.py, add its PARENT to
    sys.path, and import everything through `importlib` using
    "<package_name>.<module>" (e.g. "vanguard_portfolio.classical_discrete").
    That gives classical_discrete.py a real `__package__`, so its own
    internal relative import resolves exactly as it would if you'd run
    `python -m <package>.<module>` from the repo root -- this works
    whether or not that directory has an `__init__.py` (Python 3's
    implicit namespace packages cover the no-`__init__.py` case too).
    """
    here = Path(__file__).resolve().parent
    search_roots = [here, *here.parents]
    for root in search_roots:
        matches = list(root.rglob("classical_continuous.py"))
        if matches:
            pkg_dir = matches[0].parent
            parent_dir = pkg_dir.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            return pkg_dir, pkg_dir.name
    raise ImportError(
        "Could not locate classical_continuous.py anywhere under "
        f"{search_roots[-1]}. Either run this script with "
        "`python -m <package>.<module>` from your repo root so relative "
        "imports resolve normally, or make sure classical_continuous.py / "
        "classical_discrete.py / quantum_discrete.py / quantum_vqe_solver.py "
        "are somewhere under this file's directory tree."
    )


_PACKAGE_DIR, _PACKAGE_NAME = _bootstrap_import_path()


def _import_sibling(module_name: str):
    """importlib.import_module wrapper that gives a clearer error than the
    default ModuleNotFoundError traceback if a sibling module is missing
    or itself fails to import (e.g. a missing third-party dependency)."""
    return importlib.import_module(f"{_PACKAGE_NAME}.{module_name}")


_classical_continuous = _import_sibling("classical_continuous")
_classical_discrete = _import_sibling("classical_discrete")

PortfolioProblem = _classical_continuous.PortfolioProblem
MeanVarianceDiscreteOptimizer = _classical_discrete.MeanVarianceDiscreteOptimizer
_lot_bounds = _classical_discrete._lot_bounds
_lots_to_weights = _classical_discrete._lots_to_weights

# Prefer a synthetic_universe.json next to this script; fall back to one
# next to classical_continuous.py, since this script may now live in a
# separate scripts/ folder rather than inside the package itself.
_LOCAL_DATA_PATH = Path(__file__).with_name("synthetic_universe.json")
_PACKAGE_DATA_PATH = _PACKAGE_DIR / "synthetic_universe.json"
DEFAULT_DATA_PATH = _LOCAL_DATA_PATH if _LOCAL_DATA_PATH.exists() else _PACKAGE_DATA_PATH

DEFAULT_N_LOTS = 20
DEFAULT_RISK_AVERSION = 3.0
DEFAULT_COST_AVERSION = 1.0
DEFAULT_SEED = 0
DEFAULT_N_RANDOM_SAMPLES = 5000


# ---------------------------------------------------------------------------
# 1. Shared problem construction (no qiskit dependency -- see module note)
# ---------------------------------------------------------------------------

def build_problem_from_json(
    path: Path,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    cost_aversion: float = DEFAULT_COST_AVERSION,
) -> PortfolioProblem:
    """Mirrors quantum_vqe_solver.load_universe_from_json's construction
    exactly (including the extra group_lower / group_upper_arr / cash_yield
    fields stashed on the instance for the VQE solver's two-sided sector
    penalty), but lives here so classical-only runs don't need qiskit
    importable just to build the problem."""
    data = json.loads(Path(path).read_text())

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

    problem.group_names = list(data["group_names"])
    problem.group_lower = np.array(data["group_lower"], dtype=float)
    problem.group_upper_arr = np.array(data["group_upper"], dtype=float)
    problem.cash_yield = np.array(data["y"], dtype=float)  # unused by utility()
    return problem


def _sector_penalty_two_sided(weights: np.ndarray, problem: PortfolioProblem) -> float:
    """Squared-violation penalty against BOTH group_lower and group_upper.

    Duplicated (not imported) from quantum_vqe_solver.py so this module
    keeps its no-qiskit-required property -- see the module docstring.
    Used only for the `random` baseline's feasibility reporting below, so
    the "how feasible was this guess" column has a single consistent
    (two-sided) definition, separate from the one-sided definition
    classical:anneal reports for itself."""
    sector_map = np.asarray(problem.sector_map)
    group_upper = getattr(problem, "group_upper_arr", None)
    group_lower = getattr(problem, "group_lower", None)
    if group_upper is None or group_lower is None:
        return 0.0
    penalty = 0.0
    for idx, (lo, hi) in enumerate(zip(group_lower, group_upper)):
        exposure = float(weights[sector_map == idx].sum())
        over = max(0.0, exposure - hi)
        under = max(0.0, lo - exposure)
        penalty += over ** 2 + under ** 2
    return penalty


# ---------------------------------------------------------------------------
# 2. Per-solver runners. Each returns (elapsed_seconds, normalized_row) or
#    raises -- the caller decides how to report a failure/skip.
# ---------------------------------------------------------------------------

def _normalize_result(method: str, elapsed: float, **fields) -> dict:
    row = {"method": method, "elapsed_sec": elapsed}
    row.update(fields)
    return row


def run_classical(problem: PortfolioProblem, n_lots: int, method: str, seed: int) -> dict:
    t0 = time.perf_counter()
    optimizer = MeanVarianceDiscreteOptimizer(
        problem=problem, n_lots=n_lots, method=method, seed=seed,
    )
    result = optimizer.solve()
    elapsed = time.perf_counter() - t0

    # Budget + per-asset bounds are satisfied by construction for both the
    # brute-force and annealing paths (see classical_discrete.py docstring);
    # only the one-sided group/sector penalty is a genuine soft violation.
    feasible = result["sector_penalty"] == 0.0
    return _normalize_result(
        f"classical:{method}", elapsed,
        utility=result["utility"],
        proxy_utility=None,
        feasible=feasible,
        feasibility_detail=f"sector_penalty={result['sector_penalty']:.6g} (one-sided, upper-bound only)",
        weights=result["weights"],
        n_lots=n_lots,
        raw_result=result,
    )


def run_qaoa_pce(problem: PortfolioProblem, n_lots: int, seed: int) -> dict:
    QuantumMeanVarianceDiscreteOptimizer = _import_sibling(
        "quantum_discrete"
    ).QuantumMeanVarianceDiscreteOptimizer

    t0 = time.perf_counter()
    optimizer = QuantumMeanVarianceDiscreteOptimizer(
        problem=problem,
        n_lots=n_lots,
        group_map=problem.sector_map.tolist(),
        group_lower=problem.group_lower.tolist(),
        group_upper=problem.group_upper_arr.tolist(),
        group_names=problem.group_names,
        seed=seed,
    )
    result = optimizer.solve(
        mu=problem.expected_returns,
        cov=problem.covariance,
        transaction_cost=problem.transaction_cost,
        prev_weights=problem.prev_weights,
    )
    elapsed = time.perf_counter() - t0

    n_sat = result["n_constraints_satisfied"]
    n_con = result["n_constraints"]
    feasible = n_sat == n_con
    return _normalize_result(
        "quantum:qaoa_pce", elapsed,
        utility=result["utility"],
        proxy_utility=result["utility_qubo_proxy"],
        feasible=feasible,
        feasibility_detail=f"{n_sat}/{n_con} hard constraints satisfied after decoding",
        weights=result["weights"],
        n_lots=n_lots,
        extra=f"n_qubits={result['n_qubits']}, bit_vars={result['n_bit_vars']}, "
              f"total_evals={result['total_evals']}",
        raw_result=result,
    )


def run_vqe(problem: PortfolioProblem, n_lots: int, seed: int) -> dict:
    _vqe_mod = _import_sibling("quantum_vqe_solver")
    PortfolioVQESolver = _vqe_mod.PortfolioVQESolver
    VQEConfig = _vqe_mod.VQEConfig

    t0 = time.perf_counter()
    cfg = VQEConfig(n_lots=n_lots, seed=seed)
    solver = PortfolioVQESolver(problem, cfg)
    result = solver.run()
    elapsed = time.perf_counter() - t0

    # Report the post-processed (bit-flip-refined) candidate as the
    # headline row, same way the QAOA/classical rows report their single
    # best allocation rather than every intermediate candidate.
    feasible = bool(result["budget_ok_pp"] and result["bounds_ok_pp"]
                     and result["sector_penalty_pp"] == 0.0)
    return _normalize_result(
        "quantum:vqe_sampling", elapsed,
        utility=result["utility_pp"],
        proxy_utility=result["utility_raw"],
        feasible=feasible,
        feasibility_detail=(
            f"budget_ok={result['budget_ok_pp']}, bounds_ok={result['bounds_ok_pp']}, "
            f"sector_penalty={result['sector_penalty_pp']:.6g}"
        ),
        weights=result["weights_pp"],
        n_lots=n_lots,
        extra=f"n_qubits={solver.n_qubits}, total_evals={result['total_evals']}",
        raw_result=result,
    )


def run_random(problem: PortfolioProblem, n_lots: int, n_samples: int, seed: int) -> dict:
    """Domain-aware random baseline: every draw is a random composition of
    exactly n_lots units across assets that respects every asset's own
    [lo_i, hi_i] lot bounds *by construction* -- start every asset at its
    floor `lo`, then repeatedly hand one more lot to a uniformly-random
    asset that still has headroom (< hi) until the budget is exhausted.
    Same repair-free "each unit placed once, randomly, respecting
    per-asset caps" idea as classical_discrete._feasible_start (except
    that function seeds toward higher-expected-return assets and is only
    ever called once, as an annealing starting point); this repeats the
    random version n_samples times and keeps the best.

    This makes `random` a meaningfully harder floor to beat than pure
    i.i.d. draws would be: budget and per-asset bounds are guaranteed on
    every single sample, so a real solver has to win on *allocation
    quality*, not just on "didn't ignore the two easiest constraints."
    Group/sector exposure is deliberately left unconstrained by the
    construction (there is no cheap "repair" analogous to the lot-handout
    loop for a two-sided range over a *sum* of assets), so it's checked
    per-draw instead.

    The winner is the *best-utility sample among the ones that satisfy
    the two-sided sector bounds* -- i.e. this is the best FEASIBLE draw
    out of n_samples, not just the best draw overall. An infeasible
    sample never wins over a feasible one, no matter its utility; among
    feasible samples, ties are broken by utility as usual. If none of the
    n_samples happens to satisfy the sector bounds (possible with a small
    n_samples or narrow group ranges), this falls back to the best
    overall draw and reports it as infeasible rather than silently
    returning nothing -- `feasibility_detail` records which case
    happened."""
    lo, hi = _lot_bounds(problem, n_lots)
    if int(lo.sum()) > n_lots or int(hi.sum()) < n_lots:
        raise ValueError("Per-asset lot bounds are infeasible for the requested n_lots.")

    rng = np.random.default_rng(seed)
    n = problem.n_assets

    t0 = time.perf_counter()
    best_feasible_lots, best_feasible_utility = None, -np.inf
    best_overall_lots, best_overall_utility = None, -np.inf
    n_feasible = 0
    for _ in range(n_samples):
        lots = lo.copy()
        remaining = n_lots - int(lots.sum())
        headroom = hi - lots
        while remaining > 0:
            candidates = np.flatnonzero(headroom > 0)
            pick = rng.choice(candidates)
            lots[pick] += 1
            headroom[pick] -= 1
            remaining -= 1
        w = _lots_to_weights(lots, problem, n_lots)
        u = problem.utility(w)

        if u > best_overall_utility:
            best_overall_utility, best_overall_lots = u, lots

        if _sector_penalty_two_sided(w, problem) == 0.0:
            n_feasible += 1
            if u > best_feasible_utility:
                best_feasible_utility, best_feasible_lots = u, lots
    elapsed = time.perf_counter() - t0

    if best_feasible_lots is not None:
        best_lots, best_utility = best_feasible_lots, best_feasible_utility
        feasible = True
        detail = (
            f"budget_ok=True, bounds_ok=True (both by construction), "
            f"sector_penalty=0 -- best of {n_feasible}/{n_samples} feasible draws"
        )
    else:
        # No sample among n_samples satisfied the sector bounds; fall back
        # to the best overall draw so the baseline still returns something,
        # but report it honestly as infeasible.
        best_lots, best_utility = best_overall_lots, best_overall_utility
        w_fallback = _lots_to_weights(best_lots, problem, n_lots)
        sector_pen = _sector_penalty_two_sided(w_fallback, problem)
        feasible = False
        detail = (
            f"budget_ok=True, bounds_ok=True (both by construction), "
            f"sector_penalty={sector_pen:.6g} -- 0/{n_samples} draws were sector-feasible, "
            f"falling back to best overall draw"
        )

    w = _lots_to_weights(best_lots, problem, n_lots)
    return _normalize_result(
        "baseline:random", elapsed,
        utility=best_utility,
        proxy_utility=None,
        feasible=feasible,
        feasibility_detail=detail,
        weights=w,
        n_lots=n_lots,
        extra=f"n_samples={n_samples}",
        raw_result={"lots": best_lots},
    )


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------

def _print_row(row: Optional[dict], label: str, error: Optional[str] = None) -> None:
    if error is not None:
        print(f"{label:<24} SKIPPED / FAILED -- {error}")
        return
    proxy = "" if row["proxy_utility"] is None else f" (proxy: {row['proxy_utility']:.6f})"
    extra = f"  [{row['extra']}]" if row.get("extra") else ""
    print(
        f"{row['method']:<24} "
        f"{row['elapsed_sec']:>10.3f}s   "
        f"utility={row['utility']:>10.6f}{proxy}   "
        f"feasible={str(row['feasible']):<5}   "
        f"{row['feasibility_detail']}{extra}"
    )


def _print_weights_table(problem: PortfolioProblem, rows: list[dict]) -> None:
    col_w = max(12, max((len(r["method"]) for r in rows), default=12) + 2)
    header = f"{'asset':<14}" + "".join(f"{r['method']:>{col_w}}" for r in rows)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(problem.asset_names):
        line = f"{name:<14}"
        for r in rows:
            line += f"{r['weights'][i]:>{col_w - 1}.2%} "
        print(line)


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(DEFAULT_DATA_PATH),
                         help="Path to a synthetic_universe.json-shaped file.")
    parser.add_argument("--n-lots", type=int, default=DEFAULT_N_LOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--risk-aversion", type=float, default=DEFAULT_RISK_AVERSION)
    parser.add_argument("--cost-aversion", type=float, default=DEFAULT_COST_AVERSION)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-qaoa", action="store_true")
    parser.add_argument("--skip-vqe", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--n-random-samples", type=int, default=DEFAULT_N_RANDOM_SAMPLES,
                         help="Number of draws for the `random` baseline.")
    parser.add_argument("--skip-brute", action="store_true",
                         help="Skip classical brute-force (exact reference row). "
                              "On by default: with n_lots=20/n_assets=6 that's "
                              "C(25,5)=53,130 compositions, trivially fast. For "
                              "much larger n_lots/n_assets this can blow up "
                              "combinatorially -- skip it if you push those up.")
    args = parser.parse_args(argv)

    problem = build_problem_from_json(
        Path(args.data_path), risk_aversion=args.risk_aversion, cost_aversion=args.cost_aversion,
    )

    print(f"Universe: {args.data_path}")
    print(f"Assets: {problem.asset_names}")
    print(f"n_lots={args.n_lots}, risk_aversion={args.risk_aversion}, "
          f"cost_aversion={args.cost_aversion}, seed={args.seed}")
    print("=" * 100)

    rows: list[dict] = []

    jobs = []
    if not args.skip_random:
        jobs.append(("baseline:random", lambda: run_random(
            problem, args.n_lots, args.n_random_samples, args.seed)))
    if not args.skip_classical:
        jobs.append(("classical:anneal", lambda: run_classical(
            problem, args.n_lots, "anneal", args.seed)))
    if not args.skip_brute:
        jobs.append(("classical:brute", lambda: run_classical(
            problem, args.n_lots, "brute", args.seed)))
    if not args.skip_qaoa:
        jobs.append(("quantum:qaoa_pce", lambda: run_qaoa_pce(
            problem, args.n_lots, args.seed)))
    if not args.skip_vqe:
        jobs.append(("quantum:vqe_sampling", lambda: run_vqe(
            problem, args.n_lots, args.seed)))

    for label, job in jobs:
        print(f"Running {label} ...", flush=True)
        try:
            row = job()
        except ImportError as exc:
            _print_row(None, label, error=f"missing dependency ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - report and keep going
            tb_last_line = traceback.format_exc().strip().splitlines()[-1]
            _print_row(None, label, error=tb_last_line)
            continue
        rows.append(row)
        _print_row(row, label)

    print("=" * 100)
    if not rows:
        print("No solver completed successfully -- nothing to compare.")
        return

    print("\nRuntime ranking (fastest first):")
    for r in sorted(rows, key=lambda r: r["elapsed_sec"]):
        print(f"  {r['elapsed_sec']:>10.3f}s   {r['method']}")

    print("\nUtility ranking (real PortfolioProblem.utility(w), highest first):")
    for r in sorted(rows, key=lambda r: -r["utility"]):
        flag = "" if r["feasible"] else "  (infeasible under its own constraints)"
        print(f"  {r['utility']:>12.6f}   {r['method']}{flag}")

    print("\nAllocation comparison:")
    _print_weights_table(problem, rows)


if __name__ == "__main__":
    # Run directly, from anywhere in the repo:  python compare_all.py
    main(sys.argv[1:])