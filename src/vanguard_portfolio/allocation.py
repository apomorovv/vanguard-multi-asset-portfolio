"""Fixed-support allocation oracle and feasible sparse initialization.

Candidate generators choose *which* assets to hold. This module solves the
continuous convex allocation on that support, enforces every configured hard
guardrail, and caches repeated supports. No candidate reaches the final
portfolio without passing :func:`vanguard_portfolio.validation.validate_weights`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

from ._result import make_result
from .classical_continuous import solve_continuous
from .portfolio_model import risk_gradient, variance
from .schemas import (
    InfeasibleProblemError,
    PortfolioConstraints,
    PortfolioError,
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverUnavailableError,
)
from .validation import ConstraintReport, validate_weights


@dataclass
class OracleEvaluation:
    support: tuple[int, ...]
    feasible: bool
    objective: float
    weights: np.ndarray
    runtime: float
    reason: str
    result: SolveResult | None = None
    report: ConstraintReport | None = None
    cached: bool = False


@dataclass(frozen=True)
class ExtendedQPData:
    """Sparse QP representation for ``x=[w,t,f,(eta,u)]``.

    ``f`` is present only for factor-model problems. Keeping factor exposures as
    explicit variables avoids materializing an ``n_assets x n_assets`` risk
    matrix. ``eta`` and ``u`` are present only when a CVaR limit is configured.
    """

    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...]
    n_weights: int
    n_turnover: int
    n_factors: int
    n_scenarios: int


def _support_problem(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    support: tuple[int, ...],
) -> tuple[PortfolioProblem, PortfolioConstraints, np.ndarray]:
    n = problem.n
    selected = np.asarray(support, dtype=int)
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected] = True
    eligible = constraints.eligible_mask(n)
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(problem.lower > 1e-12).tolist()
    )
    if not np.all(eligible[selected_mask]):
        raise ValueError("support contains an ineligible asset")
    if not mandatory.issubset(support):
        raise ValueError("support omits a mandatory or positive-lower-bound asset")
    if (
        constraints.exact_cardinality is not None
        and len(support) != constraints.exact_cardinality
    ):
        raise ValueError("support does not satisfy exact cardinality")

    upper_cap = problem.upper.copy()
    if constraints.maximum_weights is not None:
        upper_cap = np.minimum(upper_cap, constraints.maximum_weights)
    lower = np.maximum(problem.lower[selected], constraints.minimum_active_weight)
    upper = upper_cap[selected]
    if np.any(lower > upper + 1e-12):
        raise ValueError("support conflicts with position bounds")
    if lower.sum() > problem.budget + 1e-12 or upper.sum() < problem.budget - 1e-12:
        raise ValueError("support cannot satisfy the budget within position bounds")

    outside_turnover = float(np.sum(problem.w0[~selected_mask]))
    reduced_turnover = problem.max_turnover
    if reduced_turnover is not None:
        reduced_turnover -= outside_turnover
        if reduced_turnover < -1e-12:
            raise ValueError("liquidating outside-support holdings exceeds max_turnover")
        reduced_turnover = max(reduced_turnover, 0.0)

    payload: dict[str, Any] = {
        "asset_names": [problem.asset_names[index] for index in selected],
        "group_names": list(problem.group_names),
        "asset_group": [problem.asset_group[index] for index in selected],
        "mu": problem.mu[selected].tolist(),
        "sigma": problem.sigma[selected].tolist(),
        "corr": problem.correlation_submatrix(selected).tolist(),
        "cov": problem.covariance_submatrix(selected).tolist(),
        "y": problem.y[selected].tolist(),
        "c": problem.c[selected].tolist(),
        "w0": problem.w0[selected].tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "group_lower": problem.group_lower.tolist(),
        "group_upper": problem.group_upper.tolist(),
        "budget": problem.budget,
        "target_return": problem.target_return,
        "max_turnover": reduced_turnover,
    }
    if problem.has_factor_model:
        payload.update(
            {
                "factor_names": list(problem.factor_names),
                "factor_loadings": problem.factor_loadings[selected].tolist(),
                "factor_cov": problem.factor_cov.tolist(),
                "idiosyncratic_var": problem.idiosyncratic_var[selected].tolist(),
            }
        )

    reduced_problem = PortfolioProblem.from_dict(payload)
    reduced_constraints = replace(
        constraints,
        exact_cardinality=None,
        minimum_active_weight=0.0,
        eligible_assets=None,
        mandatory_assets=(),
        maximum_weights=None,
        stress_scenarios=None
        if constraints.stress_scenarios is None
        else constraints.stress_scenarios[:, selected],
        scenario_returns=None
        if constraints.scenario_returns is None
        else constraints.scenario_returns[:, selected],
    )
    reduced_constraints.validate_for(reduced_problem)
    return reduced_problem, reduced_constraints, selected


def _has_extended_rules(constraints: PortfolioConstraints) -> bool:
    return any(
        value is not None
        for value in (
            constraints.minimum_income,
            constraints.factor_lower,
            constraints.stress_scenarios,
            constraints.maximum_cvar,
        )
    )


def _requires_extended_solver(
    constraints: PortfolioConstraints,
    backend: str,
) -> bool:
    """Keep explicit extended-backend aliases valid even without extra rules."""

    return _has_extended_rules(constraints) or backend.strip().lower() in {
        "clarabel",
        "clarabel_extended",
        "cvxpy_clarabel",
        "gurobi_extended",
        "osqp_extended",
    }


def _normalise_solver_options(options: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(options or {})
    nested = normalized.pop("solver_options", None)
    if nested is not None:
        if not isinstance(nested, dict):
            raise TypeError("solver_options must be a mapping")
        normalized.update(nested)
    return normalized


def _select_solver_options(
    options: dict[str, Any],
    accepted: set[str],
    *,
    backend: str,
) -> dict[str, Any]:
    unknown = sorted(set(options) - accepted)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"unsupported {backend} solver option(s): {names}")
    return {key: options[key] for key in accepted if key in options}


def _linear_model(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    list[tuple[float | None, float | None]],
]:
    """Build sparse linear constraints for ``x=[w,t,f,(eta,u)]``.

    The CVaR block is assembled with sparse horizontal stacks and a sparse
    identity. Memory therefore grows with the number of nonzero scenario
    coefficients, not with the square of the scenario count.
    """

    n = problem.n
    factor_count = problem.num_factors if problem.has_factor_model else 0
    scenario_count = (
        0 if constraints.maximum_cvar is None else constraints.scenario_returns.shape[0]
    )
    cvar_dim = 0 if scenario_count == 0 else 1 + scenario_count
    factor_start = 2 * n
    factor_end = factor_start + factor_count
    dim = factor_end + cvar_dim

    rows: list[sparse.csr_matrix] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []

    def add(matrix: np.ndarray | sparse.spmatrix, lo: Any, hi: Any) -> None:
        block = sparse.csr_matrix(matrix)
        count = block.shape[0]
        rows.append(block)
        lower.append(np.broadcast_to(np.asarray(lo, dtype=float), (count,)).copy())
        upper.append(np.broadcast_to(np.asarray(hi, dtype=float), (count,)).copy())

    def pad_weight_rows(values: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
        block = sparse.csr_matrix(values)
        return sparse.hstack(
            [block, sparse.csr_matrix((block.shape[0], dim - n))],
            format="csr",
        )

    budget = sparse.csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
        shape=(1, dim),
    )
    add(budget, problem.budget, problem.budget)
    add(pad_weight_rows(problem.A), problem.group_lower, problem.group_upper)

    identity = sparse.eye(n, format="csr")
    trailing = sparse.csr_matrix((n, dim - 2 * n))
    absolute_plus = sparse.hstack((-identity, identity, trailing), format="csr")
    absolute_minus = sparse.hstack((identity, identity, trailing), format="csr")
    add(absolute_plus, -problem.w0, np.inf)
    add(absolute_minus, problem.w0, np.inf)

    if problem.target_return is not None:
        add(pad_weight_rows(problem.mu[None, :]), problem.target_return, np.inf)

    if problem.max_turnover is not None:
        turnover_row = sparse.hstack(
            [
                sparse.csr_matrix((1, n)),
                sparse.csr_matrix(np.ones((1, n))),
                sparse.csr_matrix((1, dim - 2 * n)),
            ],
            format="csr",
        )
        add(turnover_row, -np.inf, problem.max_turnover)

    if constraints.minimum_income is not None:
        add(pad_weight_rows(problem.y[None, :]), constraints.minimum_income, np.inf)

    if factor_count:
        factor_definition = sparse.hstack(
            [
                -sparse.csr_matrix(problem.factor_loadings.T),
                sparse.csr_matrix((factor_count, n)),
                sparse.eye(factor_count, format="csr"),
                sparse.csr_matrix((factor_count, cvar_dim)),
            ],
            format="csr",
        )
        add(factor_definition, 0.0, 0.0)

    if constraints.factor_lower is not None:
        factor_band = sparse.hstack(
            [
                sparse.csr_matrix((factor_count, 2 * n)),
                sparse.eye(factor_count, format="csr"),
                sparse.csr_matrix((factor_count, cvar_dim)),
            ],
            format="csr",
        )
        add(factor_band, constraints.factor_lower, constraints.factor_upper)

    if constraints.stress_scenarios is not None:
        add(
            pad_weight_rows(constraints.stress_scenarios),
            constraints.stress_floors,
            np.inf,
        )

    if scenario_count:
        eta_index = factor_end
        u_start = eta_index + 1
        excess = sparse.hstack(
            [
                sparse.csr_matrix(constraints.scenario_returns),
                sparse.csr_matrix((scenario_count, n + factor_count)),
                sparse.csr_matrix(np.ones((scenario_count, 1))),
                sparse.eye(scenario_count, format="csr"),
            ],
            format="csr",
        )
        if excess.shape != (scenario_count, dim):
            raise RuntimeError("internal CVaR matrix dimension mismatch")
        add(excess, 0.0, np.inf)

        tail_scale = 1.0 / ((1.0 - constraints.cvar_alpha) * scenario_count)
        cvar = sparse.csr_matrix(
            (
                np.concatenate(([1.0], np.full(scenario_count, tail_scale))),
                (
                    np.zeros(1 + scenario_count, dtype=int),
                    np.concatenate(([eta_index], np.arange(u_start, dim))),
                ),
            ),
            shape=(1, dim),
        )
        add(cvar, -np.inf, constraints.maximum_cvar)

    bounds: list[tuple[float | None, float | None]] = [
        (float(problem.lower[index]), float(problem.upper[index])) for index in range(n)
    ]
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(None, None)] * factor_count)
    if scenario_count:
        bounds.append((None, None))
        bounds.extend([(0.0, None)] * scenario_count)

    return (
        sparse.vstack(rows, format="csr"),
        np.concatenate(lower),
        np.concatenate(upper),
        bounds,
    )


def _extended_qp_data(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
) -> ExtendedQPData:
    matrix, lower, upper, bounds = _linear_model(problem, constraints)
    n = problem.n
    factor_count = problem.num_factors if problem.has_factor_model else 0
    scenario_count = (
        0 if constraints.maximum_cvar is None else constraints.scenario_returns.shape[0]
    )
    dim = matrix.shape[1]

    if factor_count:
        weight_risk = sparse.diags(
            2.0 * preferences.lambda_risk * problem.idiosyncratic_var,
            format="csc",
        )
        turnover_risk = sparse.csc_matrix((n, n))
        factor_risk = sparse.csc_matrix(
            2.0 * preferences.lambda_risk * problem.factor_cov
        )
        remaining = dim - 2 * n - factor_count
        blocks: list[sparse.spmatrix] = [weight_risk, turnover_risk, factor_risk]
        if remaining:
            blocks.append(sparse.csc_matrix((remaining, remaining)))
        P = sparse.block_diag(blocks, format="csc")
    else:
        covariance = sparse.csc_matrix(
            2.0 * preferences.lambda_risk * problem.covariance_submatrix(np.arange(n))
        )
        remaining = dim - n
        P = sparse.block_diag(
            [covariance, sparse.csc_matrix((remaining, remaining))],
            format="csc",
        )

    q = np.zeros(dim)
    q[:n] = (
        -preferences.lambda_return * problem.mu
        - preferences.lambda_income * problem.y
    )
    q[n : 2 * n] = preferences.lambda_cost * problem.c

    return ExtendedQPData(
        P=P,
        q=q,
        A=matrix,
        lower=lower,
        upper=upper,
        bounds=tuple(bounds),
        n_weights=n,
        n_turnover=n,
        n_factors=factor_count,
        n_scenarios=scenario_count,
    )


def _bound_arrays(
    bounds: tuple[tuple[float | None, float | None], ...]
    | list[tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [-np.inf if lo is None else float(lo) for lo, _ in bounds],
        dtype=float,
    )
    upper = np.asarray(
        [np.inf if hi is None else float(hi) for _, hi in bounds],
        dtype=float,
    )
    return lower, upper


def _validated_tolerance(requested: float) -> float:
    """Return the explicit solver tolerance after basic validation."""
    value = float(requested)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("tol must be a finite positive number")
    return value


def _linear_conic_form(
    data: ExtendedQPData,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Return equality, upper, and lower rows including variable bounds."""

    variable_lower, variable_upper = _bound_arrays(data.bounds)
    matrix = sparse.vstack(
        [data.A, sparse.eye(data.q.size, format="csr")],
        format="csr",
    )
    lower = np.concatenate([data.lower, variable_lower])
    upper = np.concatenate([data.upper, variable_upper])
    equality = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isclose(lower, upper, rtol=0.0, atol=1.0e-12)
    )
    return matrix.tocsc(), lower, upper, equality


def _validated_extended_result(
    *,
    method: str,
    weights: np.ndarray,
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    runtime: float,
    status: str,
    success: bool,
    optimal: bool,
    metadata: dict[str, Any],
) -> SolveResult:
    result = make_result(
        method=method,
        model_type="extended_qp",
        weights=np.asarray(weights, dtype=float),
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=status,
        success=success,
        optimal=optimal,
        metadata=metadata,
    )
    report = validate_weights(result.weights, problem, constraints=constraints)
    result.feasible = report.feasible
    result.breaches = report.breaches
    result.max_violation = report.max_violation
    result.success = bool(result.success and report.feasible)
    result.optimal = bool(result.optimal and report.feasible)
    result.metadata["constraint_violations"] = report.details
    return result


def _solve_extended_scipy(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    tol: float = 1e-9,
    max_iter: int = 3_000,
) -> SolveResult:
    start = time.perf_counter()
    data = _extended_qp_data(problem, preferences, constraints)
    equality = np.isfinite(data.lower) & np.isfinite(data.upper) & np.isclose(
        data.lower, data.upper, atol=1e-12
    )
    inequalities = data.A[~equality]
    ineq_low = data.lower[~equality]
    ineq_high = data.upper[~equality]
    upper_rows: list[sparse.spmatrix] = []
    upper_rhs: list[np.ndarray] = []
    finite_high = np.isfinite(ineq_high)
    finite_low = np.isfinite(ineq_low)
    if np.any(finite_high):
        upper_rows.append(inequalities[finite_high])
        upper_rhs.append(ineq_high[finite_high])
    if np.any(finite_low):
        upper_rows.append(-inequalities[finite_low])
        upper_rhs.append(-ineq_low[finite_low])

    feasible_start = time.perf_counter()
    feasible = linprog(
        np.zeros(data.q.size),
        A_ub=sparse.vstack(upper_rows, format="csr") if upper_rows else None,
        b_ub=np.concatenate(upper_rhs) if upper_rhs else None,
        A_eq=data.A[equality] if np.any(equality) else None,
        b_eq=data.lower[equality] if np.any(equality) else None,
        bounds=data.bounds,
        method="highs",
    )
    feasible_seconds = time.perf_counter() - feasible_start
    if not feasible.success:
        raise InfeasibleProblemError(
            f"extended hard constraints are infeasible: {feasible.message}"
        )
    x0 = np.asarray(feasible.x, dtype=float)
    n = problem.n

    def objective(x: np.ndarray) -> float:
        return float(
            preferences.lambda_risk * variance(x[:n], problem)
            - preferences.lambda_return * (problem.mu @ x[:n])
            - preferences.lambda_income * (problem.y @ x[:n])
            + preferences.lambda_cost * (problem.c @ x[n : 2 * n])
        )

    def gradient(x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        result[:n] = (
            preferences.lambda_risk * risk_gradient(x[:n], problem)
            - preferences.lambda_return * problem.mu
            - preferences.lambda_income * problem.y
        )
        result[n : 2 * n] = preferences.lambda_cost * problem.c
        return result

    scipy_constraints: list[LinearConstraint] = []
    if np.any(equality):
        scipy_constraints.append(
            LinearConstraint(data.A[equality], data.lower[equality], data.upper[equality])
        )
    if np.any(~equality):
        scipy_constraints.append(
            LinearConstraint(data.A[~equality], data.lower[~equality], data.upper[~equality])
        )

    solve_start = time.perf_counter()
    solved = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=data.bounds,
        constraints=scipy_constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )
    solve_seconds = time.perf_counter() - solve_start
    return _validated_extended_result(
        method="scipy_extended_qp",
        weights=np.asarray(solved.x[:n], dtype=float),
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        runtime=time.perf_counter() - start,
        status=str(solved.message),
        success=bool(solved.success),
        optimal=bool(solved.success),
        metadata={
            "iterations": int(getattr(solved, "nit", 0)),
            "function_evaluations": int(getattr(solved, "nfev", 0)),
            "linear_feasibility_status": str(feasible.message),
            "feasible_start_seconds": feasible_seconds,
            "solve_seconds": solve_seconds,
            "matrix_rows": data.A.shape[0],
            "matrix_columns": data.A.shape[1],
            "matrix_nonzeros": data.A.nnz,
            "cvar_scenarios": data.n_scenarios,
        },
    )


def _solve_extended_osqp(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    time_limit: float | None = None,
    polish: bool = True,
    warm_start: bool = True,
    verbose: bool = False,
) -> SolveResult:
    try:
        import osqp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("OSQP is not installed; install the 'qp' extra") from exc

    start = time.perf_counter()
    build_start = time.perf_counter()
    data = _extended_qp_data(problem, preferences, constraints)
    matrix, lower, upper, _ = _linear_conic_form(data)
    native_tol = _validated_tolerance(tol)
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")
    if time_limit is not None and float(time_limit) <= 0.0:
        raise ValueError("time_limit must be positive when supplied")
    build_seconds = time.perf_counter() - build_start

    solver = osqp.OSQP()
    settings: dict[str, Any] = {
        "verbose": bool(verbose),
        "eps_abs": native_tol,
        "eps_rel": native_tol,
        "max_iter": int(max_iter),
    }
    if time_limit is not None:
        settings["time_limit"] = float(time_limit)
    setup_start = time.perf_counter()
    try:
        solver.setup(
            P=sparse.triu(data.P, format="csc"),
            q=data.q,
            A=matrix,
            l=lower,
            u=upper,
            polishing=bool(polish),
            **settings,
        )
    except (TypeError, ValueError):
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.triu(data.P, format="csc"),
            q=data.q,
            A=matrix,
            l=lower,
            u=upper,
            polish=bool(polish),
            **settings,
        )
    setup_seconds = time.perf_counter() - setup_start

    warm_started = False
    if warm_start:
        n = problem.n
        weights = np.clip(problem.w0, problem.lower, problem.upper)
        initial = np.zeros(data.q.size, dtype=float)
        initial[:n] = weights
        initial[n : 2 * n] = np.abs(weights - problem.w0)
        factor_end = 2 * n
        if data.n_factors:
            factor_end += data.n_factors
            initial[2 * n : factor_end] = problem.factor_loadings.T @ weights
        if data.n_scenarios:
            losses = -(constraints.scenario_returns @ weights)
            eta = float(np.quantile(losses, constraints.cvar_alpha))
            initial[factor_end] = eta
            initial[factor_end + 1 :] = np.maximum(losses - eta, 0.0)
        if np.all(np.isfinite(initial)):
            solver.warm_start(x=initial)
            warm_started = True

    solve_start = time.perf_counter()
    try:
        solved = solver.solve(raise_error=False)
    except TypeError:  # OSQP 0.6 does not expose the raise_error argument.
        solved = solver.solve()
    solve_seconds = time.perf_counter() - solve_start
    status = str(solved.info.status)
    has_primal_iterate = (
        solved.x is not None
        and np.asarray(solved.x).size >= problem.n
        and np.all(np.isfinite(np.asarray(solved.x)[: problem.n]))
    )
    success = status.lower().startswith("solved") and has_primal_iterate
    weights = (
        np.asarray(solved.x[: problem.n], dtype=float)
        if has_primal_iterate
        else np.full(problem.n, np.nan)
    )
    return _validated_extended_result(
        method="osqp_extended_qp",
        weights=weights,
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        runtime=time.perf_counter() - start,
        status=status,
        success=success,
        optimal=success and "inaccurate" not in status.lower(),
        metadata={
            "iterations": int(getattr(solved.info, "iter", 0)),
            "native_objective": float(getattr(solved.info, "obj_val", np.nan)),
            "solver_runtime": float(getattr(solved.info, "run_time", np.nan)),
            "primal_residual": float(
                getattr(solved.info, "prim_res", getattr(solved.info, "pri_res", np.nan))
            ),
            "dual_residual": float(
                getattr(solved.info, "dual_res", getattr(solved.info, "dua_res", np.nan))
            ),
            "rho_updates": int(getattr(solved.info, "rho_updates", 0)),
            "requested_tolerance": float(tol),
            "native_tolerance": native_tol,
            "time_limit": None if time_limit is None else float(time_limit),
            "polish": bool(polish),
            "warm_started": warm_started,
            "has_primal_iterate": has_primal_iterate,
            "model_build_seconds": build_seconds,
            "solver_setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "matrix_rows": matrix.shape[0],
            "matrix_columns": matrix.shape[1],
            "matrix_nonzeros": matrix.nnz,
            "cvar_scenarios": data.n_scenarios,
        },
    )


def _solve_extended_clarabel(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    verbose: bool = False,
    solver_options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve the sparse extended QP directly through Clarabel's Python API."""

    try:
        import clarabel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError(
            "Clarabel is not installed; install the 'qp' extra"
        ) from exc

    start = time.perf_counter()
    build_start = time.perf_counter()
    data = _extended_qp_data(problem, preferences, constraints)
    matrix, lower, upper, equality = _linear_conic_form(data)
    finite_upper = np.isfinite(upper) & ~equality
    finite_lower = np.isfinite(lower) & ~equality

    blocks: list[sparse.spmatrix] = []
    right_hand_sides: list[np.ndarray] = []
    cones: list[Any] = []
    if np.any(equality):
        blocks.append(matrix[equality])
        right_hand_sides.append(lower[equality])
        cones.append(clarabel.ZeroConeT(int(np.count_nonzero(equality))))
    if np.any(finite_upper):
        blocks.append(matrix[finite_upper])
        right_hand_sides.append(upper[finite_upper])
        cones.append(clarabel.NonnegativeConeT(int(np.count_nonzero(finite_upper))))
    if np.any(finite_lower):
        blocks.append(-matrix[finite_lower])
        right_hand_sides.append(-lower[finite_lower])
        cones.append(clarabel.NonnegativeConeT(int(np.count_nonzero(finite_lower))))

    conic_matrix = sparse.vstack(blocks, format="csc")
    conic_rhs = np.concatenate(right_hand_sides)
    native_tol = _validated_tolerance(tol)
    settings = clarabel.DefaultSettings()
    settings.verbose = bool(verbose)
    settings.max_iter = int(max_iter)
    settings.tol_feas = native_tol
    settings.tol_gap_abs = native_tol
    settings.tol_gap_rel = native_tol
    for key, value in dict(solver_options or {}).items():
        if key in {"verbose", "max_iter", "tol_feas", "tol_gap_abs", "tol_gap_rel"}:
            raise ValueError(
                f"Clarabel option {key!r} must be supplied through tol, max_iter, or verbose"
            )
        if not hasattr(settings, key):
            raise ValueError(f"unknown Clarabel setting {key!r}")
        setattr(settings, key, value)
    build_seconds = time.perf_counter() - build_start

    setup_start = time.perf_counter()
    try:
        solver = clarabel.DefaultSolver(
            sparse.triu(data.P, format="csc"),
            data.q,
            conic_matrix,
            conic_rhs,
            cones,
            settings,
        )
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - solver-specific
        raise PortfolioError(f"Clarabel could not build the extended QP: {exc}") from exc
    setup_seconds = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    try:
        solved = solver.solve()
    except RuntimeError as exc:  # pragma: no cover - solver-specific
        raise PortfolioError(f"Clarabel could not solve the extended QP: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    status = str(solved.status)
    success = status in {"Solved", "AlmostSolved"} and solved.x is not None
    weights = (
        np.asarray(solved.x[: problem.n], dtype=float)
        if success
        else np.full(problem.n, np.nan)
    )
    return _validated_extended_result(
        method="clarabel_extended_qp",
        weights=weights,
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        runtime=time.perf_counter() - start,
        status=status,
        success=success,
        optimal=status == "Solved",
        metadata={
            "iterations": int(getattr(solved, "iterations", 0)),
            "native_objective": float(getattr(solved, "obj_val", np.nan)),
            "native_dual_objective": float(
                getattr(solved, "obj_val_dual", np.nan)
            ),
            "solver_runtime": float(getattr(solved, "solve_time", np.nan)),
            "primal_residual": float(getattr(solved, "r_prim", np.nan)),
            "dual_residual": float(getattr(solved, "r_dual", np.nan)),
            "requested_tolerance": float(tol),
            "native_tolerance": native_tol,
            "model_build_seconds": build_seconds,
            "solver_setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "matrix_rows": conic_matrix.shape[0],
            "matrix_columns": conic_matrix.shape[1],
            "matrix_nonzeros": conic_matrix.nnz,
            "cvar_scenarios": data.n_scenarios,
        },
    )


def _solve_extended_cvxpy(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    solver_name: str = "CLARABEL",
    tol: float = 1e-8,
    max_iter: int = 100_000,
    solver_options: dict[str, Any] | None = None,
) -> SolveResult:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("CVXPY is not installed; install the 'qp' extra") from exc

    solver_name = solver_name.upper()
    if solver_name not in cp.installed_solvers():
        raise SolverUnavailableError(f"CVXPY solver {solver_name!r} is not installed")

    start = time.perf_counter()
    build_start = time.perf_counter()
    data = _extended_qp_data(problem, preferences, constraints)
    variable_lower, variable_upper = _bound_arrays(data.bounds)
    x = cp.Variable(data.q.size, name="extended_allocation")
    objective = cp.Minimize(
        0.5 * cp.quad_form(x, cp.psd_wrap(data.P)) + data.q @ x
    )
    cvx_constraints: list[Any] = []
    finite_lower = np.isfinite(data.lower)
    finite_upper = np.isfinite(data.upper)
    if np.any(finite_lower):
        cvx_constraints.append(data.A[finite_lower] @ x >= data.lower[finite_lower])
    if np.any(finite_upper):
        cvx_constraints.append(data.A[finite_upper] @ x <= data.upper[finite_upper])
    finite_variable_lower = np.isfinite(variable_lower)
    finite_variable_upper = np.isfinite(variable_upper)
    if np.any(finite_variable_lower):
        cvx_constraints.append(x[finite_variable_lower] >= variable_lower[finite_variable_lower])
    if np.any(finite_variable_upper):
        cvx_constraints.append(x[finite_variable_upper] <= variable_upper[finite_variable_upper])
    model = cp.Problem(objective, cvx_constraints)
    build_seconds = time.perf_counter() - build_start

    options = dict(solver_options or {})
    if solver_name == "CLARABEL":
        options.setdefault("tol_gap_abs", float(tol))
        options.setdefault("tol_gap_rel", float(tol))
        options.setdefault("tol_feas", float(tol))
        options.setdefault("max_iter", int(max_iter))
    else:
        options.setdefault("max_iter", int(max_iter))

    solve_start = time.perf_counter()
    try:
        model.solve(solver=solver_name, verbose=False, **options)
    except cp.error.SolverError as exc:  # pragma: no cover - solver-specific
        raise SolverUnavailableError(f"CVXPY/{solver_name} could not run: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    success = model.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None
    weights = (
        np.asarray(x.value[: problem.n], dtype=float).reshape(-1)
        if success
        else np.full(problem.n, np.nan)
    )
    stats = model.solver_stats
    return _validated_extended_result(
        method=f"cvxpy_{solver_name.lower()}_extended_qp",
        weights=weights,
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        runtime=time.perf_counter() - start,
        status=str(model.status),
        success=success,
        optimal=model.status == cp.OPTIMAL,
        metadata={
            "solver_name": solver_name,
            "native_objective": float(model.value) if model.value is not None else None,
            "solver_runtime": getattr(stats, "solve_time", None),
            "iterations": getattr(stats, "num_iters", None),
            "extra_stats": getattr(stats, "extra_stats", None),
            "model_build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "matrix_rows": data.A.shape[0],
            "matrix_columns": data.A.shape[1],
            "matrix_nonzeros": data.A.nnz,
            "cvar_scenarios": data.n_scenarios,
        },
    )


def _solve_extended_gurobi(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    tol: float = 1e-8,
    time_limit: float | None = None,
    threads: int | None = None,
    seed: int | None = None,
    method: int | None = None,
    bar_conv_tol: float | None = None,
    numeric_focus: int | None = None,
    output: bool = False,
) -> SolveResult:
    """Solve the sparse extended QP directly through optional Gurobi."""

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError(
            "gurobipy is not installed; install the 'gurobi' extra"
        ) from exc

    start = time.perf_counter()
    try:
        data = _extended_qp_data(problem, preferences, constraints)
        variable_lower, variable_upper = _bound_arrays(data.bounds)
        variable_lower = np.where(np.isfinite(variable_lower), variable_lower, -GRB.INFINITY)
        variable_upper = np.where(np.isfinite(variable_upper), variable_upper, GRB.INFINITY)
        # Gurobi's documented lower bound for these feasibility tolerances is
        # 1e-9.  Unlike the previous asset-count formula, this clamp is a
        # backend limitation and is exposed in the result metadata.
        native_tol = max(1.0e-9, _validated_tolerance(tol))

        model = gp.Model("vanguard_extended_allocation")
        model.Params.OutputFlag = int(output)
        model.Params.FeasibilityTol = native_tol
        model.Params.OptimalityTol = native_tol
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)
        if threads is not None:
            model.Params.Threads = int(threads)
        if seed is not None:
            model.Params.Seed = int(seed)
        if method is not None:
            model.Params.Method = int(method)
        if bar_conv_tol is not None:
            model.Params.BarConvTol = float(bar_conv_tol)
        if numeric_focus is not None:
            model.Params.NumericFocus = int(numeric_focus)

        x = model.addMVar(
            data.q.size,
            lb=variable_lower,
            ub=variable_upper,
            name="extended_allocation",
        )
        equality = (
            np.isfinite(data.lower)
            & np.isfinite(data.upper)
            & np.isclose(data.lower, data.upper, rtol=0.0, atol=1.0e-12)
        )
        finite_lower = np.isfinite(data.lower) & ~equality
        finite_upper = np.isfinite(data.upper) & ~equality
        if np.any(equality):
            model.addMConstr(data.A[equality], x, "=", data.lower[equality], name="equal")
        if np.any(finite_lower):
            model.addMConstr(
                data.A[finite_lower],
                x,
                ">",
                data.lower[finite_lower],
                name="lower",
            )
        if np.any(finite_upper):
            model.addMConstr(
                data.A[finite_upper],
                x,
                "<",
                data.upper[finite_upper],
                name="upper",
            )
        model.setObjective(
            0.5 * (x @ data.P @ x) + data.q @ x,
            GRB.MINIMIZE,
        )
        build_seconds = time.perf_counter() - start

        solve_start = time.perf_counter()
        model.optimize()
        solve_seconds = time.perf_counter() - solve_start
    except gp.GurobiError as exc:  # license and environment failures land here
        raise SolverUnavailableError(f"Gurobi could not start or solve: {exc}") from exc

    success = model.SolCount > 0
    optimal = model.Status == GRB.OPTIMAL
    weights = np.asarray(x.X[: problem.n]) if success else np.full(problem.n, np.nan)
    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.ITERATION_LIMIT: "iteration_limit",
    }
    metadata: dict[str, Any] = {
        "native_objective": float(model.ObjVal) if success else None,
        "solver_runtime": float(model.Runtime),
        "iterations": float(model.IterCount),
        "requested_tolerance": float(tol),
        "native_tolerance": native_tol,
        "model_build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "matrix_rows": data.A.shape[0],
        "matrix_columns": data.A.shape[1],
        "matrix_nonzeros": data.A.nnz,
        "cvar_scenarios": data.n_scenarios,
    }
    if success:
        metadata["best_bound"] = float(model.ObjBound)
    return _validated_extended_result(
        method="gurobi_extended_qp",
        weights=weights,
        problem=problem,
        preferences=preferences,
        constraints=constraints,
        runtime=time.perf_counter() - start,
        status=status_names.get(model.Status, f"status_{model.Status}"),
        success=success,
        optimal=optimal,
        metadata=metadata,
    )


def _solve_extended(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    backend: str,
    solver_options: dict[str, Any] | None,
) -> SolveResult:
    name = backend.strip()
    lower_name = name.lower()
    options = _normalise_solver_options(solver_options)

    if lower_name in {"scipy", "slsqp", "scipy_slsqp"}:
        accepted = _select_solver_options(
            options,
            {"tol", "max_iter"},
            backend="SciPy",
        )
        return _solve_extended_scipy(problem, preferences, constraints, **accepted)

    if lower_name in {"osqp", "osqp_extended"}:
        accepted = _select_solver_options(
            options,
            {"tol", "max_iter", "time_limit", "polish", "warm_start", "verbose"},
            backend="OSQP",
        )
        return _solve_extended_osqp(problem, preferences, constraints, **accepted)

    if lower_name in {"clarabel", "clarabel_extended"}:
        tol = float(options.pop("tol", 1e-8))
        max_iter = int(options.pop("max_iter", 100_000))
        verbose = bool(options.pop("verbose", False))
        return _solve_extended_clarabel(
            problem,
            preferences,
            constraints,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            solver_options=options,
        )

    if lower_name == "cvxpy_clarabel":
        tol = float(options.pop("tol", 1e-8))
        max_iter = int(options.pop("max_iter", 100_000))
        return _solve_extended_cvxpy(
            problem,
            preferences,
            constraints,
            solver_name="CLARABEL",
            tol=tol,
            max_iter=max_iter,
            solver_options=options,
        )

    if lower_name in {"gurobi", "gurobi_extended"}:
        accepted = _select_solver_options(
            options,
            {
                "tol",
                "time_limit",
                "threads",
                "seed",
                "method",
                "bar_conv_tol",
                "numeric_focus",
                "output",
            },
            backend="Gurobi",
        )
        return _solve_extended_gurobi(problem, preferences, constraints, **accepted)

    if lower_name.startswith("cvxpy:"):
        solver_name = name.split(":", 1)[1]
        tol = float(options.pop("tol", 1e-8))
        max_iter = int(options.pop("max_iter", 100_000))
        return _solve_extended_cvxpy(
            problem,
            preferences,
            constraints,
            solver_name=solver_name,
            tol=tol,
            max_iter=max_iter,
            solver_options=options,
        )

    raise ValueError(
        "extended allocation backend must be one of "
        "'scipy', 'osqp', 'clarabel', 'gurobi', or 'cvxpy:<solver>'"
    )


@dataclass
class AllocationOracle:
    problem: PortfolioProblem
    preferences: Preferences
    constraints: PortfolioConstraints
    backend: str = "scipy"
    solver_options: dict[str, Any] = field(default_factory=dict)
    cache: dict[tuple[int, ...], OracleEvaluation] = field(default_factory=dict)
    calls: int = 0
    cache_hits: int = 0

    def __post_init__(self) -> None:
        self.constraints.validate_for(self.problem)

    def evaluate(self, support: Iterable[int]) -> OracleEvaluation:
        support_tuple = tuple(sorted({int(index) for index in support}))
        if support_tuple in self.cache:
            self.cache_hits += 1
            cached = self.cache[support_tuple]
            return replace(cached, cached=True, weights=cached.weights.copy())

        self.calls += 1
        start = time.perf_counter()
        try:
            restricted, reduced_constraints, selected = _support_problem(
                self.problem, self.constraints, support_tuple
            )
            if _requires_extended_solver(reduced_constraints, self.backend):
                reduced_result = _solve_extended(
                    restricted,
                    self.preferences,
                    reduced_constraints,
                    backend=self.backend,
                    solver_options=self.solver_options,
                )
            else:
                reduced_result = solve_continuous(
                    restricted,
                    self.preferences,
                    backend=self.backend,
                    **self.solver_options,
                )

            full_weights = np.zeros(self.problem.n)
            full_weights[selected] = reduced_result.weights
            result = make_result(
                method=reduced_result.method,
                model_type="fixed_support",
                weights=full_weights,
                problem=self.problem,
                preferences=self.preferences,
                runtime=time.perf_counter() - start,
                status=reduced_result.status,
                success=reduced_result.success,
                optimal=reduced_result.optimal,
                seed=reduced_result.seed,
                metadata={
                    **reduced_result.metadata,
                    "reduced_dimension": len(support_tuple),
                    "full_dimension": self.problem.n,
                },
            )
            report = validate_weights(
                result.weights,
                self.problem,
                constraints=self.constraints,
            )
            result.model_type = "fixed_support"
            result.feasible = report.feasible
            result.breaches = report.breaches
            result.max_violation = report.max_violation
            result.success = bool(result.success and report.feasible)
            result.optimal = bool(result.optimal and report.feasible)
            result.metadata.update(
                {
                    "support": list(support_tuple),
                    "support_size": len(support_tuple),
                }
            )
            evaluation = OracleEvaluation(
                support=support_tuple,
                feasible=bool(result.success and report.feasible),
                objective=float(result.objective),
                weights=result.weights.copy(),
                runtime=time.perf_counter() - start,
                reason=(
                    result.status
                    if result.success
                    else "; ".join(report.details) or result.status
                ),
                result=result,
                report=report,
            )
        except (ValueError, PortfolioError, SolverUnavailableError) as exc:
            evaluation = OracleEvaluation(
                support=support_tuple,
                feasible=False,
                objective=np.inf,
                weights=np.full(self.problem.n, np.nan),
                runtime=time.perf_counter() - start,
                reason=str(exc),
            )

        self.cache[support_tuple] = evaluation
        return replace(evaluation, weights=evaluation.weights.copy())


def solve_relaxation(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    backend: str = "scipy",
    solver_options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve the full-universe continuous lower-bound problem."""

    constraints.validate_for(problem)
    eligible = constraints.eligible_mask(problem.n)
    payload = problem.to_dict()
    lower = problem.lower.copy()
    upper = problem.upper.copy()
    lower[~eligible] = 0.0
    upper[~eligible] = 0.0
    if constraints.maximum_weights is not None:
        upper = np.minimum(upper, constraints.maximum_weights)
    for index in constraints.mandatory_assets:
        lower[index] = max(lower[index], constraints.minimum_active_weight)
    payload["lower"] = lower.tolist()
    payload["upper"] = upper.tolist()

    relaxed_problem = PortfolioProblem.from_dict(payload)
    relaxed_rules = replace(
        constraints,
        exact_cardinality=None,
        minimum_active_weight=0.0,
        mandatory_assets=(),
    )
    if _requires_extended_solver(relaxed_rules, backend):
        result = _solve_extended(
            relaxed_problem,
            preferences,
            relaxed_rules,
            backend=backend,
            solver_options=solver_options,
        )
    else:
        result = solve_continuous(
            relaxed_problem,
            preferences,
            backend=backend,
            **dict(solver_options or {}),
        )
    result.model_type = "continuous_relaxation"
    return result


def _rank_assets(
    problem: PortfolioProblem,
    preferences: Preferences,
    relaxation_weights: np.ndarray,
) -> np.ndarray:
    smooth = (
        preferences.lambda_return * problem.mu
        + preferences.lambda_income * problem.y
        - preferences.lambda_cost * problem.c
        - preferences.lambda_risk * risk_gradient(relaxation_weights, problem)
    )
    scale = np.std(smooth)
    normalized = smooth / (scale if scale > 1e-12 else 1.0)
    return relaxation_weights + 0.02 * normalized


def find_feasible_support_milp(
    oracle: AllocationOracle,
    relaxation_weights: np.ndarray,
    *,
    time_limit: float = 20.0,
) -> OracleEvaluation:
    """Find a valid sparse support with a linear mixed-integer feasibility model."""

    if time_limit <= 0.0:
        raise ValueError("time_limit must be positive")
    problem = oracle.problem
    constraints = oracle.constraints
    if constraints.exact_cardinality is None:
        raise ValueError("support feasibility MILP requires exact_cardinality")
    constraints.validate_for(problem)

    start = time.perf_counter()
    base_matrix, base_lower, base_upper, base_bounds = _linear_model(problem, constraints)
    n = problem.n
    base_dim = base_matrix.shape[1]
    z_start = base_dim
    upper = problem.upper.copy()
    if constraints.maximum_weights is not None:
        upper = np.minimum(upper, constraints.maximum_weights)
    active_lower = np.maximum(problem.lower, constraints.minimum_active_weight)

    padded_base = sparse.hstack(
        [base_matrix, sparse.csr_matrix((base_matrix.shape[0], n))],
        format="csr",
    )
    remaining = sparse.csr_matrix((n, base_dim - n))
    link_upper = sparse.hstack(
        [sparse.eye(n, format="csr"), remaining, -sparse.diags(upper)],
        format="csr",
    )
    link_lower = sparse.hstack(
        [sparse.eye(n, format="csr"), remaining, -sparse.diags(active_lower)],
        format="csr",
    )
    cardinality = sparse.hstack(
        [sparse.csr_matrix((1, base_dim)), sparse.csr_matrix(np.ones((1, n)))],
        format="csr",
    )
    matrix = sparse.vstack(
        [padded_base, link_upper, link_lower, cardinality],
        format="csr",
    )
    lower_rows = np.concatenate(
        [
            base_lower,
            np.full(n, -np.inf),
            np.zeros(n),
            np.asarray([constraints.exact_cardinality], dtype=float),
        ]
    )
    upper_rows = np.concatenate(
        [
            base_upper,
            np.zeros(n),
            np.full(n, np.inf),
            np.asarray([constraints.exact_cardinality], dtype=float),
        ]
    )

    base_lb, base_ub = _bound_arrays(base_bounds)
    z_lb = np.zeros(n)
    z_ub = np.ones(n)
    eligible = constraints.eligible_mask(n)
    z_ub[~eligible] = 0.0
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(problem.lower > 1e-12).tolist()
    )
    for index in mandatory:
        z_lb[index] = 1.0
        z_ub[index] = 1.0

    objective = np.zeros(base_dim + n)
    objective[:n] = (
        -oracle.preferences.lambda_return * problem.mu
        - oracle.preferences.lambda_income * problem.y
    )
    objective[n : 2 * n] = oracle.preferences.lambda_cost * problem.c
    support_score = _rank_assets(problem, oracle.preferences, relaxation_weights)
    score_scale = max(float(np.max(np.abs(support_score), initial=0.0)), 1e-12)
    objective[z_start:] = -1e-3 * support_score / score_scale
    integrality = np.zeros(base_dim + n, dtype=int)
    integrality[z_start:] = 1

    solved = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(
            np.concatenate([base_lb, z_lb]),
            np.concatenate([base_ub, z_ub]),
        ),
        constraints=LinearConstraint(matrix, lower_rows, upper_rows),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-3,
            "presolve": True,
        },
    )
    if int(solved.status) == 2:
        raise InfeasibleProblemError(
            "the exact-cardinality financial guardrails are jointly infeasible"
        )
    if solved.x is None:
        raise ValueError(f"support feasibility MILP found no incumbent: {solved.message}")

    support = tuple(np.flatnonzero(solved.x[z_start:] > 0.5).tolist())
    if len(support) != constraints.exact_cardinality:
        raise ValueError("support feasibility MILP returned the wrong cardinality")
    evaluation = oracle.evaluate(support)
    if not evaluation.feasible or evaluation.result is None:
        raise ValueError(
            "the feasibility MILP support could not be allocated by the convex oracle: "
            + evaluation.reason
        )

    runtime = time.perf_counter() - start
    result = replace(
        evaluation.result,
        runtime=runtime,
        metadata={
            **evaluation.result.metadata,
            "initialization": "scipy_highs_feasibility_milp",
            "feasibility_milp_status": str(solved.message),
            "feasibility_milp_nodes": getattr(solved, "mip_node_count", None),
            "feasibility_milp_gap": getattr(solved, "mip_gap", None),
        },
    )
    return replace(
        evaluation,
        runtime=runtime,
        reason="valid support from SciPy/HiGHS feasibility MILP",
        result=result,
    )


def find_feasible_initial_support(
    oracle: AllocationOracle,
    relaxation_weights: np.ndarray,
    *,
    max_trials: int = 250,
    seed: int = 0,
    exact_enumeration_limit: int = 200_000,
    milp_time_limit: float = 20.0,
) -> OracleEvaluation:
    """Construct a valid K-asset portfolio before any local or quantum search."""

    constraints = oracle.constraints
    if constraints.exact_cardinality is None:
        raise ValueError("hybrid support construction requires exact_cardinality")
    k = constraints.exact_cardinality
    current_support = tuple(np.flatnonzero(oracle.problem.w0 > 1.0e-12).tolist())

    if len(current_support) == k:
        current = oracle.evaluate(current_support)
        if current.feasible:
            if current.result is not None:
                current.result.metadata["initialization"] = "current_valid_support"
            return current

    eligible = np.flatnonzero(constraints.eligible_mask(oracle.problem.n))
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(oracle.problem.lower > 1e-12).tolist()
    )
    score = _rank_assets(oracle.problem, oracle.preferences, relaxation_weights)

    protected = set(mandatory)
    for group in np.flatnonzero(oracle.problem.group_lower > 1e-12):
        candidates = eligible[np.asarray(oracle.problem.asset_group)[eligible] == group]
        if candidates.size:
            protected.add(int(candidates[np.argmax(score[candidates])]))
    if len(protected) > k:
        raise ValueError("mandatory/group coverage requires more than K assets")

    ordered = [int(index) for index in eligible[np.argsort(score[eligible])[::-1]]]
    deterministic = list(protected)
    deterministic.extend(index for index in ordered if index not in protected)
    first = oracle.evaluate(deterministic[:k])
    if first.feasible:
        return first

    milp_failure = ""
    try:
        return find_feasible_support_milp(
            oracle,
            relaxation_weights,
            time_limit=milp_time_limit,
        )
    except InfeasibleProblemError:
        raise
    except ValueError as exc:
        milp_failure = str(exc)

    rng = np.random.default_rng(seed)
    optional = np.asarray([index for index in eligible if index not in protected], dtype=int)
    slots = k - len(protected)
    best_failure = first
    temperature = max(float(np.std(score[optional])), 1e-8) if optional.size else 1.0
    probabilities = np.exp((score[optional] - np.max(score[optional])) / temperature)
    probabilities /= probabilities.sum() if probabilities.sum() else 1.0
    for _ in range(max(0, int(max_trials) - 1)):
        chosen = (
            rng.choice(optional, size=slots, replace=False, p=probabilities)
            if slots
            else np.empty(0, dtype=int)
        )
        candidate = oracle.evaluate((*protected, *chosen.tolist()))
        if candidate.feasible:
            return candidate
        if candidate.report is not None and (
            best_failure.report is None
            or candidate.report.max_violation < best_failure.report.max_violation
        ):
            best_failure = candidate

    combinations_count = comb(len(optional), slots) if 0 <= slots <= len(optional) else 0
    if combinations_count and combinations_count <= int(exact_enumeration_limit):
        best: OracleEvaluation | None = None
        for chosen in combinations(optional.tolist(), slots):
            candidate = oracle.evaluate((*protected, *chosen))
            if candidate.feasible and (best is None or candidate.objective < best.objective):
                best = candidate
        if best is not None:
            return best

    raise ValueError(
        "could not construct a feasible K-asset portfolio; "
        f"feasibility MILP: {milp_failure or 'not available'}; "
        f"best attempted support: {best_failure.reason}"
    )


__all__ = [
    "AllocationOracle",
    "ExtendedQPData",
    "OracleEvaluation",
    "find_feasible_initial_support",
    "find_feasible_support_milp",
    "solve_relaxation",
]
