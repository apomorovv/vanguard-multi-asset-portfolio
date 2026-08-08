"""Continuous convex-QP portfolio backends.

SciPy is the guaranteed baseline. OSQP, CVXPY solvers, and Gurobi are
optional comparison backends and are imported only when requested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, linprog, minimize

from ._result import make_result
from .portfolio_model import QPData, build_continuous_qp, risk_gradient, variance
from .schemas import (
    InfeasibleProblemError,
    PortfolioProblem,
    Preferences,
    SolveResult,
    SolverUnavailableError,
)
from .validation import validate_weights


def _validated_tolerance(requested: float) -> float:
    """Return the caller's tolerance after basic validation.

    Solver precision is a user-facing experiment parameter.  In particular,
    do not tighten it as a function of the asset count: that made otherwise
    identical scaling runs take radically different numerical paths.
    """

    value = float(requested)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("tol must be a finite positive number")
    return value


def _repair_turnover_residual(
    weights: np.ndarray,
    problem: PortfolioProblem,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Remove accumulated epigraph residual without changing the model tolerance.

    OSQP controls each ``|w_i-w_i0| <= t_i`` row independently.  Across tens of
    thousands of assets, tiny accepted row residuals can make the recomputed
    turnover exceed its cap even when the native QP reports ``solved``.  When
    the incumbent portfolio is itself feasible, a convex contraction toward it
    preserves every convex hard constraint and removes that numerical excess.
    """

    metadata: dict[str, float | bool] = {"turnover_repair_applied": False}
    if problem.max_turnover is None or not np.all(np.isfinite(weights)):
        return weights, metadata
    measured = float(np.sum(np.abs(weights - problem.w0)))
    metadata["turnover_before_repair"] = measured
    limit = float(problem.max_turnover)
    if measured <= limit or measured <= 0.0:
        metadata["turnover_after_repair"] = measured
        return weights, metadata
    if not validate_weights(problem.w0, problem).feasible:
        metadata["turnover_after_repair"] = measured
        return weights, metadata
    target = np.nextafter(limit, 0.0)
    repaired = problem.w0 + (target / measured) * (weights - problem.w0)
    metadata.update(
        {
            "turnover_repair_applied": True,
            "turnover_after_repair": float(np.sum(np.abs(repaired - problem.w0))),
        }
    )
    return repaired, metadata


def _linear_program_start(problem: PortfolioProblem, data: QPData) -> np.ndarray:
    """Find a feasible ``[w,t]`` point before invoking a QP backend."""
    equal = np.isfinite(data.lower) & np.isfinite(data.upper) & np.isclose(
        data.lower, data.upper, atol=1e-12
    )
    inequalities = data.A[~equal].tocsr()
    inequality_lower = data.lower[~equal]
    inequality_upper = data.upper[~equal]
    finite_upper = np.isfinite(inequality_upper)
    finite_lower = np.isfinite(inequality_lower)
    upper_rows: list[sparse.spmatrix] = []
    upper_rhs: list[np.ndarray] = []
    if np.any(finite_upper):
        upper_rows.append(inequalities[finite_upper])
        upper_rhs.append(inequality_upper[finite_upper])
    if np.any(finite_lower):
        upper_rows.append(-inequalities[finite_lower])
        upper_rhs.append(-inequality_lower[finite_lower])

    result = linprog(
        np.zeros(data.q.size),
        A_ub=sparse.vstack(upper_rows, format="csr") if upper_rows else None,
        b_ub=np.concatenate(upper_rhs) if upper_rhs else None,
        A_eq=data.A[equal].tocsr() if np.any(equal) else None,
        b_eq=data.lower[equal] if np.any(equal) else None,
        bounds=[(None, None)] * data.q.size,
        method="highs",
    )
    if not result.success:
        raise InfeasibleProblemError(
            f"continuous hard constraints are infeasible: {result.message}"
        )
    return np.asarray(result.x, dtype=float)


def solve_continuous_scipy(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    tol: float = 1e-9,
    max_iter: int = 2000,
    initial_weights: np.ndarray | None = None,
) -> SolveResult:
    """Solve the convex QP with SciPy SLSQP and analytic gradients."""
    total_start = time.perf_counter()
    preferences = preferences or Preferences()
    build_start = time.perf_counter()
    data = build_continuous_qp(problem, preferences)
    build_seconds = time.perf_counter() - build_start
    n = problem.n
    feasible_start_begin = time.perf_counter()
    if initial_weights is None:
        x0 = _linear_program_start(problem, data)
    else:
        w0 = np.asarray(initial_weights, dtype=float).reshape(n)
        factor_exposure = (
            problem.factor_loadings.T @ w0 if problem.has_factor_model else np.empty(0)
        )
        x0 = np.concatenate([w0, np.abs(w0 - problem.w0), factor_exposure])
    feasible_start_seconds = time.perf_counter() - feasible_start_begin

    def objective(x: np.ndarray) -> float:
        weights = x[:n]
        return float(
            preferences.lambda_risk * variance(weights, problem)
            + data.q @ x
        )

    def jacobian(x: np.ndarray) -> np.ndarray:
        gradient = data.q.copy()
        gradient[:n] += preferences.lambda_risk * risk_gradient(x[:n], problem)
        return gradient

    equality = np.isfinite(data.lower) & np.isfinite(data.upper) & np.isclose(
        data.lower, data.upper, atol=1e-12
    )
    constraints: list[LinearConstraint] = []
    if np.any(equality):
        constraints.append(
            LinearConstraint(data.A[equality], data.lower[equality], data.upper[equality])
        )
    if np.any(~equality):
        constraints.append(
            LinearConstraint(data.A[~equality], data.lower[~equality], data.upper[~equality])
        )

    solve_start = time.perf_counter()
    result = minimize(
        objective,
        x0,
        jac=jacobian,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - total_start
    weights = np.asarray(result.x[:n], dtype=float)
    return make_result(
        method="scipy_slsqp",
        model_type="continuous",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=str(result.message),
        success=bool(result.success),
        optimal=bool(result.success),
        metadata={
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "model_build_seconds": build_seconds,
            "feasible_start_seconds": feasible_start_seconds,
            "solve_seconds": solve_seconds,
        },
    )


def solve_continuous_osqp(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    time_limit: float | None = None,
    polish: bool = True,
    warm_start: bool = True,
) -> SolveResult:
    """Solve the matrix-form QP with the optional open-source OSQP backend."""
    try:
        import osqp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError("OSQP is not installed; install the 'qp' extra") from exc

    total_start = time.perf_counter()
    preferences = preferences or Preferences()
    build_start = time.perf_counter()
    data = build_continuous_qp(problem, preferences)
    build_seconds = time.perf_counter() - build_start
    native_tol = _validated_tolerance(tol)
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")
    if time_limit is not None and float(time_limit) <= 0.0:
        raise ValueError("time_limit must be positive when supplied")
    solver = osqp.OSQP()
    settings: dict[str, Any] = {
        "verbose": False,
        "eps_abs": native_tol,
        "eps_rel": native_tol,
        "max_iter": int(max_iter),
    }
    if time_limit is not None:
        settings["time_limit"] = float(time_limit)
    setup_start = time.perf_counter()
    try:
        solver.setup(
            P=data.P,
            q=data.q,
            A=data.A,
            l=data.lower,
            u=data.upper,
            polishing=bool(polish),
            **settings,
        )
    except (TypeError, ValueError):  # OSQP 0.6 uses the older setting name.
        solver = osqp.OSQP()
        solver.setup(
            P=data.P,
            q=data.q,
            A=data.A,
            l=data.lower,
            u=data.upper,
            polish=bool(polish),
            **settings,
        )
    setup_seconds = time.perf_counter() - setup_start
    warm_started = False
    if warm_start:
        weights = np.asarray(problem.w0, dtype=float)
        factor_exposure = (
            problem.factor_loadings.T @ weights
            if problem.has_factor_model
            else np.empty(0, dtype=float)
        )
        initial = np.concatenate(
            [weights, np.abs(weights - problem.w0), factor_exposure]
        )
        if initial.shape == data.q.shape and np.all(np.isfinite(initial)):
            solver.warm_start(x=initial)
            warm_started = True
    solve_start = time.perf_counter()
    try:
        solved = solver.solve(raise_error=False)
    except TypeError:  # OSQP 0.6 does not expose the raise_error argument.
        solved = solver.solve()
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - total_start
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
    turnover_metadata: dict[str, float | bool] = {}
    if success:
        weights, turnover_metadata = _repair_turnover_residual(weights, problem)
    return make_result(
        method="osqp",
        model_type="continuous",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
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
            **turnover_metadata,
            "model_build_seconds": build_seconds,
            "solver_setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
        },
    )


def solve_continuous_cvxpy(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    solver_name: str = "CLARABEL",
    solver_options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve with any installed CVXPY-compatible continuous convex solver."""
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
    n = problem.n
    w = cp.Variable(n, name="w")
    t = cp.Variable(n, nonneg=True, name="t")
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
        cp.sum(w) == problem.budget,
        w >= problem.lower,
        w <= problem.upper,
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
    except cp.error.SolverError as exc:  # pragma: no cover - solver-specific failure
        raise SolverUnavailableError(f"CVXPY/{solver_name} could not run: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - total_start
    success = model.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and w.value is not None
    weights = np.asarray(w.value).reshape(-1) if success else np.full(n, np.nan)
    stats = model.solver_stats
    return make_result(
        method=f"cvxpy_{solver_name.lower()}",
        model_type="continuous",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
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
        },
    )


def solve_continuous_gurobi(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    time_limit: float | None = None,
    threads: int | None = None,
    seed: int | None = None,
    method: int | None = None,
    bar_conv_tol: float | None = None,
    output: bool = False,
) -> SolveResult:
    """Solve the continuous QP directly with optional Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverUnavailableError(
            "gurobipy is not installed; install the 'gurobi' extra"
        ) from exc

    preferences = preferences or Preferences()
    n = problem.n
    start = time.perf_counter()
    try:
        model = gp.Model("vanguard_continuous")
        model.Params.OutputFlag = int(output)
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
        w = model.addMVar(n, lb=problem.lower, ub=problem.upper, name="w")
        t = model.addMVar(n, lb=0.0, name="t")
        model.addConstr(w.sum() == problem.budget, name="budget")
        model.addConstr(problem.A @ w >= problem.group_lower, name="group_lower")
        model.addConstr(problem.A @ w <= problem.group_upper, name="group_upper")
        model.addConstr(t >= w - problem.w0, name="turnover_plus")
        model.addConstr(t >= problem.w0 - w, name="turnover_minus")
        if problem.target_return is not None:
            model.addConstr(problem.mu @ w >= problem.target_return, name="target_return")
        if problem.max_turnover is not None:
            model.addConstr(t.sum() <= problem.max_turnover, name="max_turnover")
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
        build_seconds = time.perf_counter() - start
        solve_start = time.perf_counter()
        model.optimize()
    except gp.GurobiError as exc:  # license and environment failures land here
        raise SolverUnavailableError(f"Gurobi could not start or solve: {exc}") from exc
    solve_seconds = time.perf_counter() - solve_start
    runtime = time.perf_counter() - start
    success = model.SolCount > 0
    optimal = model.Status == GRB.OPTIMAL
    weights = np.asarray(w.X) if success else np.full(n, np.nan)
    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    metadata: dict[str, Any] = {
        "native_objective": float(model.ObjVal) if success else None,
        "solver_runtime": float(model.Runtime),
        "iterations": float(model.IterCount),
        "model_build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
    }
    if success:
        try:
            metadata["best_bound"] = float(model.ObjBound)
        except gp.GurobiError:
            pass
    return make_result(
        method="gurobi_qp",
        model_type="continuous",
        weights=weights,
        problem=problem,
        preferences=preferences,
        runtime=runtime,
        status=status_names.get(model.Status, f"status_{model.Status}"),
        success=success,
        optimal=optimal,
        seed=seed,
        metadata=metadata,
    )


def solve_continuous(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    *,
    backend: str = "scipy",
    **kwargs: Any,
) -> SolveResult:
    """Dispatch to a named continuous backend."""
    name = backend.strip()
    lower = name.lower()
    if lower in {"scipy", "slsqp", "scipy_slsqp"}:
        return solve_continuous_scipy(problem, preferences, **kwargs)
    if lower == "osqp":
        return solve_continuous_osqp(problem, preferences, **kwargs)
    if lower == "gurobi":
        return solve_continuous_gurobi(problem, preferences, **kwargs)
    if lower.startswith("cvxpy:"):
        return solve_continuous_cvxpy(
            problem, preferences, solver_name=name.split(":", 1)[1], **kwargs
        )
    raise ValueError(f"unknown continuous backend {backend!r}")


@dataclass
class MeanVarianceContinuousOptimizer:
    problem: PortfolioProblem
    preferences: Preferences = field(default_factory=Preferences)
    backend: str = "scipy"

    def solve(self, **kwargs: Any) -> SolveResult:
        return solve_continuous(
            self.problem, self.preferences, backend=self.backend, **kwargs
        )


def mean_variance_continuous(
    problem: PortfolioProblem,
    preferences: Preferences | None = None,
    **kwargs: Any,
) -> SolveResult:
    """Backward-compatible convenience wrapper returning :class:`SolveResult`."""
    return solve_continuous(problem, preferences, **kwargs)


__all__ = [
    "MeanVarianceContinuousOptimizer",
    "PortfolioProblem",
    "mean_variance_continuous",
    "solve_continuous",
    "solve_continuous_cvxpy",
    "solve_continuous_gurobi",
    "solve_continuous_osqp",
    "solve_continuous_scipy",
]
