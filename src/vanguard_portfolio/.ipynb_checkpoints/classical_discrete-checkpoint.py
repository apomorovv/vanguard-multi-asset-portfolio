"""Discrete lot-allocation solvers for exact and heuristic baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from ._result import make_result
from .portfolio_model import (
    discrete_constraints_hold,
    lot_bounds,
    lots_to_weights,
    objective_value,
    swap_objective_delta,
)
from .schemas import (
    InfeasibleProblemError,
    PortfolioConstraints,
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverSkippedError,
    SolverUnavailableError,
)
from .validation import validate_weights


def bounded_lot_allocation_count(problem: PortfolioProblem, units: int) -> int:
    """Count budget-feasible lot vectors after asset bounds, before other rules.

    The dynamic program is O(nM) and avoids constructing any allocations.  It
    provides a much tighter enumeration safety estimate than the unrestricted
    stars-and-bars count while remaining conservative with respect to group,
    return, and turnover constraints.
    """
    low, high = lot_bounds(problem, units)
    if np.any(low > high) or int(low.sum()) > units or int(high.sum()) < units:
        return 0
    counts = [0] * (units + 1)
    counts[0] = 1
    for lower, upper in zip(low.tolist(), high.tolist()):
        updated = [0] * (units + 1)
        window = 0
        for total in range(units + 1):
            add_index = total - lower
            remove_index = total - upper - 1
            if add_index >= 0:
                window += counts[add_index]
            if remove_index >= 0:
                window -= counts[remove_index]
            updated[total] = window
        counts = updated
    return int(counts[units])


def enumerate_feasible_lots(problem: PortfolioProblem, units: int) -> Iterator[np.ndarray]:
    """Yield each hard-feasible integer composition exactly once."""
    low, high = lot_bounds(problem, units)
    if np.any(low > high) or low.sum() > units or high.sum() < units:
        return
    n = problem.n
    suffix_low = np.zeros(n + 1, dtype=int)
    suffix_high = np.zeros(n + 1, dtype=int)
    for i in range(n - 1, -1, -1):
        suffix_low[i] = suffix_low[i + 1] + low[i]
        suffix_high[i] = suffix_high[i + 1] + high[i]
    current = np.zeros(n, dtype=int)

    def recurse(i: int, remaining: int) -> Iterator[np.ndarray]:
        if i == n:
            if remaining == 0 and discrete_constraints_hold(current, problem, units):
                yield current.copy()
            return
        lo = max(int(low[i]), remaining - int(suffix_high[i + 1]))
        hi = min(int(high[i]), remaining - int(suffix_low[i + 1]))
        for value in range(lo, hi + 1):
            current[i] = value
            yield from recurse(i + 1, remaining - value)
        current[i] = 0

    yield from recurse(0, int(units))


def _first_feasible_lots(
    problem: PortfolioProblem,
    units: int,
    *,
    time_limit: float | None = 60.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find a feasible lot vector with a zero-objective HiGHS MILP.

    The earlier implementation recursively enumerated until it found a first
    feasible point.  That is unsafe for hundreds of assets and can exceed
    Python's recursion limit.  A dedicated feasibility MILP honors every hard
    rule and supplies scalable starts for both classical heuristics.
    """
    low, high = lot_bounds(problem, units)
    if np.any(low > high) or int(low.sum()) > units or int(high.sum()) < units:
        raise InfeasibleProblemError(
            f"no feasible discrete allocation exists at M={units}; asset lot bounds conflict"
        )

    n = problem.n
    dimension = 2 * n
    lot_size = problem.budget / units
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, n), dtype=float)
    blocks: list[sparse.spmatrix] = []
    lower_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []

    def add(matrix: sparse.spmatrix, lower: np.ndarray | float, upper: np.ndarray | float) -> None:
        rows = matrix.shape[0]
        blocks.append(matrix.tocsr())
        lower_rows.append(np.broadcast_to(np.asarray(lower, dtype=float), (rows,)).copy())
        upper_rows.append(np.broadcast_to(np.asarray(upper, dtype=float), (rows,)).copy())

    budget = sparse.csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, dimension)
    )
    add(budget, units, units)
    add(
        sparse.hstack(
            (
                lot_size * sparse.csr_matrix(problem.A),
                sparse.csr_matrix((problem.num_groups, n)),
            ),
            format="csr",
        ),
        problem.group_lower,
        problem.group_upper,
    )
    add(
        sparse.hstack((-lot_size * identity, identity), format="csr"),
        -problem.w0,
        np.inf,
    )
    add(
        sparse.hstack((lot_size * identity, identity), format="csr"),
        problem.w0,
        np.inf,
    )
    if problem.target_return is not None:
        add(
            sparse.csr_matrix(
                np.concatenate((lot_size * problem.mu, np.zeros(n)))[None, :]
            ),
            problem.target_return,
            np.inf,
        )
    if problem.max_turnover is not None:
        add(
            sparse.csr_matrix(np.concatenate((np.zeros(n), np.ones(n)))[None, :]),
            -np.inf,
            problem.max_turnover,
        )

    constraints = LinearConstraint(
        sparse.vstack(blocks, format="csc"),
        np.concatenate(lower_rows),
        np.concatenate(upper_rows),
    )
    variable_bounds = Bounds(
        np.concatenate((low.astype(float), np.zeros(n))),
        np.concatenate((high.astype(float), np.full(n, np.inf))),
    )
    options: dict[str, Any] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    start = time.perf_counter()
    result = milp(
        c=np.zeros(dimension),
        integrality=np.concatenate((np.ones(n, dtype=int), np.zeros(n, dtype=int))),
        bounds=variable_bounds,
        constraints=constraints,
        options=options,
    )
    runtime = time.perf_counter() - start
    if result.x is None:
        raise InfeasibleProblemError(
            f"no feasible discrete allocation was found at M={units}: {result.message}"
        )
    lots = np.rint(np.asarray(result.x[:n])).astype(int)
    if not discrete_constraints_hold(lots, problem, units):
        raise InfeasibleProblemError(
            "the feasibility MILP returned a candidate that failed independent validation"
        )
    return lots, {
        "feasible_start_method": "scipy_highs_milp",
        "feasible_start_status": str(result.message),
        "feasible_start_seconds": runtime,
    }


def solve_discrete_enumeration(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 10,
    max_candidates: int | None = 2_000_000,
    max_assets: int | None = 200,
) -> SolveResult:
    """Prove the tiny-instance optimum by exhaustive feasible enumeration."""
    preferences = preferences or Preferences()
    start = time.perf_counter()
    if max_assets is not None and problem.n > int(max_assets):
        raise SolverSkippedError(
            "exact enumeration recursion safety guard: "
            f"n_assets={problem.n} exceeds max_assets={int(max_assets)}"
        )
    bounded_candidates = bounded_lot_allocation_count(problem, units)
    if max_candidates is not None and bounded_candidates > int(max_candidates):
        raise SolverSkippedError(
            "exact enumeration safety guard: "
            f"{bounded_candidates:,} asset-bound-feasible lot vectors exceed "
            f"max_candidates={int(max_candidates):,}; use Gurobi/SCIP or a heuristic"
        )
    best_lots: np.ndarray | None = None
    best_objective = np.inf
    feasible_candidates = 0
    for lots in enumerate_feasible_lots(problem, units):
        feasible_candidates += 1
        weights = lots_to_weights(lots, problem, units)
        value = objective_value(weights, problem, preferences)
        if value < best_objective - 1e-15:
            best_objective = value
            best_lots = lots.copy()
    runtime = time.perf_counter() - start
    if best_lots is None:
        raise InfeasibleProblemError(f"no feasible discrete allocation exists at M={units}")
    return make_result(
        method="exact_enumeration",
        model_type="discrete",
        weights=lots_to_weights(best_lots, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status="optimal",
        success=True,
        optimal=True,
        units=units,
        metadata={
            "lots": best_lots.tolist(),
            "bounded_candidates_before_group_constraints": bounded_candidates,
            "feasible_candidates": feasible_candidates,
            "model_build_seconds": 0.0,
            "solve_seconds": runtime,
        },
    )


@dataclass
class _SwapState:
    lots: np.ndarray
    weights: np.ndarray
    cov_times_weights: np.ndarray
    group_exposure: np.ndarray
    expected_return: float
    turnover: float
    objective: float


def _make_swap_state(
    lots: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
    units: int,
) -> _SwapState:
    lots = np.asarray(lots, dtype=int).copy()
    weights = lots_to_weights(lots, problem, units)
    return _SwapState(
        lots=lots,
        weights=weights,
        cov_times_weights=problem.covariance_matvec(weights),
        group_exposure=problem.A @ weights,
        expected_return=float(problem.mu @ weights),
        turnover=float(np.sum(np.abs(weights - problem.w0))),
        objective=objective_value(weights, problem, preferences),
    )


def _turnover_change(
    state: _SwapState,
    donor: int,
    receiver: int,
    problem: PortfolioProblem,
    units: int,
) -> float:
    lot_size = problem.budget / units
    before = (
        abs(state.weights[donor] - problem.w0[donor])
        + abs(state.weights[receiver] - problem.w0[receiver])
    )
    after = (
        abs(state.weights[donor] - lot_size - problem.w0[donor])
        + abs(state.weights[receiver] + lot_size - problem.w0[receiver])
    )
    return float(after - before)


def _swap_is_feasible(
    state: _SwapState,
    donor: int,
    receiver: int,
    problem: PortfolioProblem,
    units: int,
    low: np.ndarray,
    high: np.ndarray,
    tol: float = 1e-10,
) -> bool:
    if donor == receiver:
        return False
    if state.lots[donor] <= low[donor] or state.lots[receiver] >= high[receiver]:
        return False

    lot_size = problem.budget / units
    donor_group = problem.asset_group[donor]
    receiver_group = problem.asset_group[receiver]
    if donor_group != receiver_group:
        donor_exposure = state.group_exposure[donor_group] - lot_size
        receiver_exposure = state.group_exposure[receiver_group] + lot_size
        if (
            donor_exposure < problem.group_lower[donor_group] - tol
            or donor_exposure > problem.group_upper[donor_group] + tol
            or receiver_exposure < problem.group_lower[receiver_group] - tol
            or receiver_exposure > problem.group_upper[receiver_group] + tol
        ):
            return False

    if problem.target_return is not None:
        trial_return = state.expected_return + lot_size * (
            problem.mu[receiver] - problem.mu[donor]
        )
        if trial_return < problem.target_return - tol:
            return False

    if problem.max_turnover is not None:
        trial_turnover = state.turnover + _turnover_change(
            state, donor, receiver, problem, units
        )
        if trial_turnover > problem.max_turnover + tol:
            return False
    return True


def _apply_swap(
    state: _SwapState,
    donor: int,
    receiver: int,
    delta_objective: float,
    problem: PortfolioProblem,
    units: int,
) -> None:
    lot_size = problem.budget / units
    turnover_delta = _turnover_change(state, donor, receiver, problem, units)
    donor_group = problem.asset_group[donor]
    receiver_group = problem.asset_group[receiver]

    state.lots[donor] -= 1
    state.lots[receiver] += 1
    state.weights[donor] -= lot_size
    state.weights[receiver] += lot_size
    covariance_columns = problem.covariance_submatrix(
        np.arange(problem.n),
        [receiver, donor],
    )
    state.cov_times_weights += lot_size * (
        covariance_columns[:, 0] - covariance_columns[:, 1]
    )
    if donor_group != receiver_group:
        state.group_exposure[donor_group] -= lot_size
        state.group_exposure[receiver_group] += lot_size
    state.expected_return += lot_size * (problem.mu[receiver] - problem.mu[donor])
    state.turnover += turnover_delta
    state.objective += delta_objective


def _candidate_assets(
    state: _SwapState,
    problem: PortfolioProblem,
    preferences: Preferences,
    low: np.ndarray,
    high: np.ndarray,
    candidate_pool_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    donors = np.flatnonzero(state.lots > low)
    receivers = np.flatnonzero(state.lots < high)
    if candidate_pool_size is None:
        return donors, receivers
    pool = max(1, int(candidate_pool_size))
    if donors.size <= pool and receivers.size <= pool:
        return donors, receivers

    smooth_gradient = (
        2.0 * preferences.lambda_risk * state.cov_times_weights
        - preferences.lambda_return * problem.mu
        - preferences.lambda_income * problem.y
    )
    if donors.size > pool:
        donor_order = np.argpartition(smooth_gradient[donors], -pool)[-pool:]
        donors = donors[donor_order]
    if receivers.size > pool:
        receiver_order = np.argpartition(smooth_gradient[receivers], pool - 1)[:pool]
        receivers = receivers[receiver_order]
    return donors, receivers


def _local_improve(
    initial: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
    units: int,
    max_iterations: int,
    candidate_pool_size: int | None = None,
) -> tuple[np.ndarray, int, int, bool]:
    """One-lot best-improvement search using O(1) proposal deltas.

    With ``candidate_pool_size=None`` every donor/receiver pair is checked and
    the termination point is a true one-swap local optimum.  A finite pool
    ranks assets by smooth marginal objective and keeps large runs bounded.
    """
    state = _make_swap_state(initial, problem, preferences, units)
    low, high = lot_bounds(problem, units)
    evaluations = 0
    iterations = 0
    stationary = False
    for _ in range(max_iterations):
        best_pair: tuple[int, int] | None = None
        best_delta = 0.0
        donors, receivers = _candidate_assets(
            state, problem, preferences, low, high, candidate_pool_size
        )
        for donor in donors:
            for receiver in receivers:
                donor_index = int(donor)
                receiver_index = int(receiver)
                if not _swap_is_feasible(
                    state,
                    donor_index,
                    receiver_index,
                    problem,
                    units,
                    low,
                    high,
                ):
                    continue
                evaluations += 1
                delta = swap_objective_delta(
                    state.weights,
                    state.cov_times_weights,
                    donor_index,
                    receiver_index,
                    problem,
                    preferences,
                    units,
                )
                if delta < best_delta - 1e-13:
                    best_delta = delta
                    best_pair = (donor_index, receiver_index)
        if best_pair is None:
            stationary = True
            break
        _apply_swap(state, *best_pair, best_delta, problem, units)
        iterations += 1
    return state.lots, iterations, evaluations, stationary


def solve_discrete_local_search(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    max_iterations: int = 500,
    candidate_pool_size: int | None = None,
    feasible_start_time_limit: float | None = 60.0,
    initial_lots: np.ndarray | None = None,
) -> SolveResult:
    """Run deterministic one-lot best-improvement local search."""
    preferences = preferences or Preferences()
    start = time.perf_counter()
    start_metadata: dict[str, Any] = {}
    if initial_lots is None:
        initial, start_metadata = _first_feasible_lots(
            problem, units, time_limit=feasible_start_time_limit
        )
    else:
        initial = np.asarray(initial_lots, dtype=int)
        start_metadata = {"feasible_start_method": "provided"}
    if not discrete_constraints_hold(initial, problem, units):
        raise ValueError("initial_lots is not hard-feasible")
    search_start = time.perf_counter()
    best, iterations, evaluations, stationary = _local_improve(
        initial,
        problem,
        preferences,
        units,
        max_iterations,
        candidate_pool_size=candidate_pool_size,
    )
    solve_seconds = time.perf_counter() - search_start
    runtime = time.perf_counter() - start
    if stationary and candidate_pool_size is None:
        status = "one_swap_locally_optimal"
    elif stationary:
        status = "candidate_pool_stationary"
    else:
        status = "iteration_limit"
    return make_result(
        method="swap_local_search",
        model_type="discrete",
        weights=lots_to_weights(best, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=status,
        success=True,
        optimal=False,
        units=units,
        metadata={
            **start_metadata,
            "lots": best.tolist(),
            "iterations": iterations,
            "objective_evaluations": evaluations,
            "candidate_pool_size": candidate_pool_size,
            "stationary": stationary,
            "solve_seconds": solve_seconds,
        },
    )


def solve_discrete_annealing(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    seed: int = 0,
    n_iterations: int = 20_000,
    initial_temperature: float = 0.02,
    final_temperature: float = 1e-5,
    local_polish: bool = True,
    polish_candidate_pool_size: int | None = None,
    feasible_start_time_limit: float | None = 60.0,
    initial_lots: np.ndarray | None = None,
) -> SolveResult:
    """Budget-preserving simulated annealing followed by optional swap polish."""
    if n_iterations <= 0 or initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("annealing iterations and temperatures must be positive")
    preferences = preferences or Preferences()
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    if initial_lots is None:
        initial, start_metadata = _first_feasible_lots(
            problem, units, time_limit=feasible_start_time_limit
        )
    else:
        initial = np.asarray(initial_lots, dtype=int)
        start_metadata = {"feasible_start_method": "provided"}
    if not discrete_constraints_hold(initial, problem, units):
        raise ValueError("initial_lots is not hard-feasible")
    state = _make_swap_state(initial, problem, preferences, units)
    low, high = lot_bounds(problem, units)

    # Randomize the deterministic feasible start without violating hard constraints.
    for _ in range(10 * problem.n):
        donor, receiver = rng.choice(problem.n, size=2, replace=False)
        donor = int(donor)
        receiver = int(receiver)
        if _swap_is_feasible(state, donor, receiver, problem, units, low, high):
            delta = swap_objective_delta(
                state.weights,
                state.cov_times_weights,
                donor,
                receiver,
                problem,
                preferences,
                units,
            )
            _apply_swap(state, donor, receiver, delta, problem, units)

    best = state.lots.copy()
    best_value = state.objective
    accepted = 0
    feasible_trials = 0
    ratio = final_temperature / initial_temperature
    solve_start = time.perf_counter()

    for step in range(n_iterations):
        donor, receiver = rng.choice(problem.n, size=2, replace=False)
        donor = int(donor)
        receiver = int(receiver)
        if not _swap_is_feasible(state, donor, receiver, problem, units, low, high):
            continue
        feasible_trials += 1
        delta = swap_objective_delta(
            state.weights,
            state.cov_times_weights,
            donor,
            receiver,
            problem,
            preferences,
            units,
        )
        fraction = step / max(n_iterations - 1, 1)
        temperature = initial_temperature * ratio**fraction
        if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
            _apply_swap(state, donor, receiver, delta, problem, units)
            accepted += 1
            if state.objective < best_value:
                best = state.lots.copy()
                best_value = state.objective

    polish_iterations = 0
    polish_evaluations = 0
    polish_stationary = False
    if local_polish:
        best, polish_iterations, polish_evaluations, polish_stationary = _local_improve(
            best,
            problem,
            preferences,
            units,
            max_iterations=500,
            candidate_pool_size=polish_candidate_pool_size,
        )
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - start
    return make_result(
        method="simulated_annealing_swap",
        model_type="discrete",
        weights=lots_to_weights(best, problem, units),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status="heuristic_complete",
        success=True,
        optimal=False,
        units=units,
        seed=seed,
        metadata={
            **start_metadata,
            "lots": best.tolist(),
            "iterations": n_iterations,
            "feasible_trials": feasible_trials,
            "accepted_moves": accepted,
            "polish_iterations": polish_iterations,
            "polish_evaluations": polish_evaluations,
            "polish_stationary": polish_stationary,
            "polish_candidate_pool_size": polish_candidate_pool_size,
            "solve_seconds": solve_seconds,
            "fast_swap_deltas": True,
        },
    )


def solve_discrete_gurobi(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    time_limit: float | None = None,
    mip_gap: float = 1e-9,
    threads: int | None = None,
    seed: int | None = None,
    node_limit: float | None = None,
    mip_focus: int | None = None,
    heuristics: float | None = None,
    output: bool = False,
) -> SolveResult:
    """Solve the integer-lot MIQP directly with optional Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError(
            "gurobipy is not installed; install the 'gurobi' extra"
        ) from exc

    start = time.perf_counter()
    preferences = preferences or Preferences()
    low, high = lot_bounds(problem, units)
    if np.any(low > high):
        raise InfeasibleProblemError("discrete lot bounds are infeasible")
    lot_size = problem.budget / units
    try:
        model = gp.Model("vanguard_discrete")
        model.Params.OutputFlag = int(output)
        model.Params.MIPGap = float(mip_gap)
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)
        if threads is not None:
            model.Params.Threads = int(threads)
        if seed is not None:
            model.Params.Seed = int(seed)
        if node_limit is not None:
            model.Params.NodeLimit = float(node_limit)
        if mip_focus is not None:
            model.Params.MIPFocus = int(mip_focus)
        if heuristics is not None:
            model.Params.Heuristics = float(heuristics)
        q = model.addMVar(problem.n, vtype=GRB.INTEGER, lb=low, ub=high, name="q")
        t = model.addMVar(problem.n, lb=0.0, name="t")
        model.addConstr(q.sum() == units, name="budget_lots")
        model.addConstr(lot_size * (problem.A @ q) >= problem.group_lower, name="group_lower")
        model.addConstr(lot_size * (problem.A @ q) <= problem.group_upper, name="group_upper")
        model.addConstr(t >= lot_size * q - problem.w0, name="turnover_plus")
        model.addConstr(t >= problem.w0 - lot_size * q, name="turnover_minus")
        if problem.target_return is not None:
            model.addConstr(lot_size * (problem.mu @ q) >= problem.target_return)
        if problem.max_turnover is not None:
            model.addConstr(t.sum() <= problem.max_turnover)
        if problem.has_factor_model:
            factor_exposure = lot_size * (problem.factor_loadings.T @ q)
            risk_expression = (
                factor_exposure @ problem.factor_cov @ factor_exposure
                + lot_size**2 * ((problem.idiosyncratic_var * q) @ q)
            )
        else:
            risk_expression = lot_size**2 * (q @ problem.cov @ q)
        model.setObjective(
            preferences.lambda_risk * risk_expression
            - preferences.lambda_return * lot_size * (problem.mu @ q)
            - preferences.lambda_income * lot_size * (problem.y @ q)
            + preferences.lambda_cost * (problem.c @ t),
            GRB.MINIMIZE,
        )
        build_seconds = time.perf_counter() - start
        solve_start = time.perf_counter()
        model.optimize()
    except gp.GurobiError as exc:  # pragma: no cover - license dependent
        raise SolverUnavailableError(f"Gurobi could not start or solve: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - start
    success = model.SolCount > 0
    optimal = model.Status == GRB.OPTIMAL
    lots = np.rint(np.asarray(q.X)).astype(int) if success else np.zeros(problem.n, dtype=int)
    weights = lots_to_weights(lots, problem, units) if success else np.full(problem.n, np.nan)
    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    metadata: dict[str, Any] = {
        "lots": lots.tolist(),
        "solver_runtime": float(model.Runtime),
        "nodes": float(model.NodeCount),
        "model_build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
    }
    if success:
        metadata["native_objective"] = float(model.ObjVal)
        try:
            metadata["best_bound"] = float(model.ObjBound)
            metadata["reported_mip_gap"] = float(model.MIPGap)
        except gp.GurobiError:
            pass
    return make_result(
        method="gurobi_miqp",
        model_type="discrete",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=status_names.get(model.Status, f"status_{model.Status}"),
        success=success,
        optimal=optimal,
        units=units,
        seed=seed,
        metadata=metadata,
    )


def solve_cardinality_gurobi(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    warm_start: np.ndarray | None = None,
    time_limit: float = 300.0,
    mip_gap: float = 1e-3,
    threads: int = 0,
    seed: int = 0,
    mip_focus: int = 1,
    output: bool = False,
) -> SolveResult:
    """Solve the canonical continuous-weight, exact-cardinality MIQP.

    This is the gold-standard reference for the hybrid window methods.  The
    legacy :func:`solve_discrete_gurobi` remains an equal-lot benchmark; this
    function instead uses binary support variables and exact continuous
    percentages, matching the hybrid architecture.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError(
            "gurobipy is not installed; install the 'gurobi' extra"
        ) from exc
    constraints.validate_for(problem)
    if constraints.exact_cardinality is None:
        raise ValueError("solve_cardinality_gurobi requires exact_cardinality")
    n = problem.n
    eligible = constraints.eligible_mask(n)
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(problem.lower > 1e-12).tolist()
    )
    upper = problem.upper.copy()
    if constraints.maximum_weights is not None:
        upper = np.minimum(upper, constraints.maximum_weights)
    active_lower = np.maximum(problem.lower, constraints.minimum_active_weight)
    start = time.perf_counter()
    try:
        model = gp.Model("vanguard_cardinality")
        model.Params.OutputFlag = int(output)
        model.Params.TimeLimit = float(time_limit)
        model.Params.MIPGap = float(mip_gap)
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.MIPFocus = int(mip_focus)
        w = model.addMVar(n, lb=0.0, ub=upper, name="w")
        z = model.addMVar(n, vtype=GRB.BINARY, name="z")
        t = model.addMVar(n, lb=0.0, name="t")
        model.addConstr(w.sum() == problem.budget, name="budget")
        model.addConstr(z.sum() == constraints.exact_cardinality, name="cardinality")
        model.addConstr(w <= upper * z, name="link_upper")
        model.addConstr(w >= active_lower * z, name="link_lower")
        for index in np.flatnonzero(~eligible):
            model.addConstr(z[int(index)] == 0, name=f"ineligible_{index}")
        for index in sorted(mandatory):
            model.addConstr(z[int(index)] == 1, name=f"mandatory_{index}")
        model.addConstr(problem.A @ w >= problem.group_lower, name="group_lower")
        model.addConstr(problem.A @ w <= problem.group_upper, name="group_upper")
        model.addConstr(t >= w - problem.w0, name="turnover_plus")
        model.addConstr(t >= problem.w0 - w, name="turnover_minus")
        if problem.target_return is not None:
            model.addConstr(problem.mu @ w >= problem.target_return, name="target_return")
        if problem.max_turnover is not None:
            model.addConstr(t.sum() <= problem.max_turnover, name="max_turnover")
        if constraints.minimum_income is not None:
            model.addConstr(problem.y @ w >= constraints.minimum_income, name="minimum_income")
        if constraints.factor_lower is not None:
            factor_exposure = problem.factor_loadings.T @ w
            model.addConstr(factor_exposure >= constraints.factor_lower, name="factor_lower")
            model.addConstr(factor_exposure <= constraints.factor_upper, name="factor_upper")
        if constraints.stress_scenarios is not None:
            model.addConstr(
                constraints.stress_scenarios @ w >= constraints.stress_floors,
                name="stress_floors",
            )
        if constraints.maximum_cvar is not None:
            scenario_count = constraints.scenario_returns.shape[0]
            eta = model.addVar(lb=-GRB.INFINITY, name="cvar_eta")
            excess = model.addMVar(scenario_count, lb=0.0, name="cvar_excess")
            model.addConstr(
                excess >= -(constraints.scenario_returns @ w) - eta,
                name="cvar_excess_definition",
            )
            model.addConstr(
                eta
                + excess.sum() / ((1.0 - constraints.cvar_alpha) * scenario_count)
                <= constraints.maximum_cvar,
                name="maximum_cvar",
            )
        if problem.has_factor_model:
            factor_exposure = problem.factor_loadings.T @ w
            risk_expression = (
                factor_exposure @ problem.factor_cov @ factor_exposure
                + (problem.idiosyncratic_var * w) @ w
            )
        else:
            risk_expression = w @ problem.cov @ w
        model.setObjective(
            preferences.lambda_risk * risk_expression
            - preferences.lambda_return * (problem.mu @ w)
            - preferences.lambda_income * (problem.y @ w)
            + preferences.lambda_cost * (problem.c @ t),
            GRB.MINIMIZE,
        )
        if warm_start is not None:
            warm = np.asarray(warm_start, dtype=float).reshape(n)
            w.Start = warm
            z.Start = (warm > 1e-8).astype(float)
            t.Start = np.abs(warm - problem.w0)
        build_seconds = time.perf_counter() - start
        solve_start = time.perf_counter()
        model.optimize()
    except gp.GurobiError as exc:  # pragma: no cover - license dependent
        raise SolverUnavailableError(f"Gurobi could not start or solve: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    success = model.SolCount > 0
    weights = np.asarray(w.X, dtype=float) if success else np.full(n, np.nan)
    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    result = make_result(
        method="gurobi_cardinality_miqp",
        model_type="cardinality_miqp",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=time.perf_counter() - start,
        status=status_names.get(model.Status, f"status_{model.Status}"),
        success=success,
        optimal=model.Status == GRB.OPTIMAL,
        seed=seed,
        metadata={
            "solver_runtime": float(model.Runtime),
            "nodes": float(model.NodeCount),
            "model_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "best_bound": float(model.ObjBound) if success else None,
            "reported_mip_gap": float(model.MIPGap) if success else None,
            "support": np.flatnonzero(weights > 1e-8).tolist() if success else [],
        },
    )
    report = validate_weights(weights, problem, constraints=constraints)
    result.feasible = report.feasible
    result.breaches = report.breaches
    result.max_violation = report.max_violation
    result.success = bool(result.success and report.feasible)
    result.optimal = bool(result.optimal and report.feasible)
    return result


def solve_discrete_cvxpy(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    solver_name: str = "SCIP",
    solver_options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve the MIQP through an installed CVXPY MIP-capable backend."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("CVXPY is not installed; install the 'qp' extra") from exc

    solver_name = solver_name.upper()
    if solver_name not in cp.installed_solvers():
        raise SolverUnavailableError(f"CVXPY solver {solver_name!r} is not installed")
    total_start = time.perf_counter()
    preferences = preferences or Preferences()
    build_start = time.perf_counter()
    low, high = lot_bounds(problem, units)
    lot_size = problem.budget / units
    q = cp.Variable(problem.n, integer=True, name="q")
    t = cp.Variable(problem.n, nonneg=True, name="t")
    w = lot_size * q
    if problem.has_factor_model:
        risk_expression = cp.quad_form(
            problem.factor_loadings.T @ w,
            cp.psd_wrap(problem.factor_cov),
        ) + cp.sum(cp.multiply(problem.idiosyncratic_var, cp.square(w)))
    else:
        risk_expression = cp.quad_form(w, cp.psd_wrap(problem.cov))
    objective = cp.Minimize(
        preferences.lambda_risk * risk_expression
        - preferences.lambda_return * problem.mu @ w
        - preferences.lambda_income * problem.y @ w
        + preferences.lambda_cost * problem.c @ t
    )
    constraints = [
        q >= low,
        q <= high,
        cp.sum(q) == units,
        problem.A @ w >= problem.group_lower,
        problem.A @ w <= problem.group_upper,
        t >= w - problem.w0,
        t >= problem.w0 - w,
    ]
    if problem.target_return is not None:
        constraints.append(problem.mu @ w >= problem.target_return)
    if problem.max_turnover is not None:
        constraints.append(cp.sum(t) <= problem.max_turnover)
    model = cp.Problem(objective, constraints)
    build_seconds = time.perf_counter() - build_start
    solve_start = time.perf_counter()
    try:
        model.solve(solver=solver_name, verbose=False, **dict(solver_options or {}))
    except cp.error.SolverError as exc:  # pragma: no cover - solver-specific
        raise SolverUnavailableError(f"CVXPY/{solver_name} could not run: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - total_start
    usable_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if hasattr(cp, "USER_LIMIT"):
        usable_statuses.add(cp.USER_LIMIT)
    success = model.status in usable_statuses and q.value is not None
    lots = (
        np.rint(np.asarray(q.value).reshape(-1)).astype(int)
        if success
        else np.zeros(problem.n, dtype=int)
    )
    weights = lots_to_weights(lots, problem, units) if success else np.full(problem.n, np.nan)
    return make_result(
        method=f"cvxpy_{solver_name.lower()}_miqp",
        model_type="discrete",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=str(model.status),
        success=success,
        optimal=model.status == cp.OPTIMAL,
        units=units,
        metadata={
            "lots": lots.tolist(),
            "solver_name": solver_name,
            "native_objective": float(model.value) if model.value is not None else None,
            "solver_runtime": getattr(model.solver_stats, "solve_time", None),
            "extra_stats": getattr(model.solver_stats, "extra_stats", None),
            "model_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
        },
    )


def solve_discrete(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    units: int = 20,
    backend: str = "enumeration",
    **kwargs: Any,
) -> SolveResult:
    name = backend.strip()
    lower = name.lower()
    if lower in {"enumeration", "exact", "brute"}:
        return solve_discrete_enumeration(problem, preferences, units=units, **kwargs)
    if lower in {"local_search", "swap", "swap_local_search"}:
        return solve_discrete_local_search(problem, preferences, units=units, **kwargs)
    if lower in {"annealing", "anneal", "simulated_annealing"}:
        return solve_discrete_annealing(problem, preferences, units=units, **kwargs)
    if lower == "gurobi":
        return solve_discrete_gurobi(problem, preferences, units=units, **kwargs)
    if lower.startswith("cvxpy:"):
        return solve_discrete_cvxpy(
            problem,
            preferences,
            units=units,
            solver_name=name.split(":", 1)[1],
            **kwargs,
        )
    raise ValueError(f"unknown discrete backend {backend!r}")


@dataclass
class MeanVarianceDiscreteOptimizer:
    problem: PortfolioProblem
    preferences: Preferences = Preferences()
    units: int = 20
    method: str = "annealing"
    seed: int = 0

    def solve(self, **kwargs: Any) -> SolveResult:
        if self.method.lower() in {"annealing", "anneal", "simulated_annealing"}:
            kwargs.setdefault("seed", self.seed)
        return solve_discrete(
            self.problem,
            self.preferences,
            units=self.units,
            backend=self.method,
            **kwargs,
        )


def mean_variance_discrete(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    n_lots: int | None = None,
    units: int | None = None,
    method: str = "annealing",
    **kwargs: Any,
) -> SolveResult:
    """Compatibility wrapper accepting either ``n_lots`` or ``units``."""
    resolved_units = units if units is not None else (n_lots if n_lots is not None else 20)
    return solve_discrete(
        problem,
        preferences,
        units=resolved_units,
        backend=method,
        **kwargs,
    )


__all__ = [
    "MeanVarianceDiscreteOptimizer",
    "bounded_lot_allocation_count",
    "enumerate_feasible_lots",
    "mean_variance_discrete",
    "solve_cardinality_gurobi",
    "solve_discrete",
    "solve_discrete_annealing",
    "solve_discrete_cvxpy",
    "solve_discrete_enumeration",
    "solve_discrete_gurobi",
    "solve_discrete_local_search",
]
