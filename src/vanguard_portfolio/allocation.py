"""Fixed-support allocation oracle and feasible sparse initialization.

Candidate generators choose *which* assets to hold.  This module solves the
continuous convex allocation on that support, enforces every configured hard
guardrail, and caches repeated supports.  No candidate reaches the final
portfolio without passing :func:`vanguard_portfolio.validation.validate_weights`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from typing import Any, Iterable

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
        "corr": problem.corr[np.ix_(selected, selected)].tolist(),
        "cov": problem.cov[np.ix_(selected, selected)].tolist(),
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


def _linear_model(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    """Build linear constraints for ``x=[w,t,(eta,u)]``."""
    n = problem.n
    scenario_count = (
        0 if constraints.maximum_cvar is None else constraints.scenario_returns.shape[0]
    )
    cvar_dim = 0 if scenario_count == 0 else 1 + scenario_count
    dim = 2 * n + cvar_dim
    rows: list[sparse.csr_matrix] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []

    def add(matrix: np.ndarray | sparse.spmatrix, lo: Any, hi: Any) -> None:
        block = sparse.csr_matrix(matrix)
        count = block.shape[0]
        rows.append(block)
        lower.append(np.broadcast_to(np.asarray(lo, dtype=float), (count,)).copy())
        upper.append(np.broadcast_to(np.asarray(hi, dtype=float), (count,)).copy())

    budget = np.zeros((1, dim))
    budget[0, :n] = 1.0
    add(budget, problem.budget, problem.budget)

    group = np.zeros((problem.num_groups, dim))
    group[:, :n] = problem.A
    add(group, problem.group_lower, problem.group_upper)

    absolute_plus = np.zeros((n, dim))
    absolute_plus[:, :n] = -np.eye(n)
    absolute_plus[:, n : 2 * n] = np.eye(n)
    add(absolute_plus, -problem.w0, np.inf)
    absolute_minus = np.zeros((n, dim))
    absolute_minus[:, :n] = np.eye(n)
    absolute_minus[:, n : 2 * n] = np.eye(n)
    add(absolute_minus, problem.w0, np.inf)

    if problem.target_return is not None:
        row = np.zeros((1, dim))
        row[0, :n] = problem.mu
        add(row, problem.target_return, np.inf)
    if problem.max_turnover is not None:
        row = np.zeros((1, dim))
        row[0, n : 2 * n] = 1.0
        add(row, -np.inf, problem.max_turnover)
    if constraints.minimum_income is not None:
        row = np.zeros((1, dim))
        row[0, :n] = problem.y
        add(row, constraints.minimum_income, np.inf)
    if constraints.factor_lower is not None:
        factor = np.zeros((problem.num_factors, dim))
        factor[:, :n] = problem.factor_loadings.T
        add(factor, constraints.factor_lower, constraints.factor_upper)
    if constraints.stress_scenarios is not None:
        stress = np.zeros((constraints.stress_scenarios.shape[0], dim))
        stress[:, :n] = constraints.stress_scenarios
        add(stress, constraints.stress_floors, np.inf)

    if scenario_count:
        eta_index = 2 * n
        u_start = eta_index + 1
        # u_s >= loss_s - eta, with loss_s = -r_s @ w.
        excess = np.zeros((scenario_count, dim))
        excess[:, :n] = constraints.scenario_returns
        excess[:, eta_index] = 1.0
        excess[:, u_start:] = np.eye(scenario_count)
        add(excess, 0.0, np.inf)
        cvar = np.zeros((1, dim))
        cvar[0, eta_index] = 1.0
        cvar[0, u_start:] = 1.0 / ((1.0 - constraints.cvar_alpha) * scenario_count)
        add(cvar, -np.inf, constraints.maximum_cvar)

    bounds: list[tuple[float | None, float | None]] = [
        (float(problem.lower[index]), float(problem.upper[index])) for index in range(n)
    ]
    bounds.extend([(0.0, None)] * n)
    if scenario_count:
        bounds.append((None, None))
        bounds.extend([(0.0, None)] * scenario_count)
    return (
        sparse.vstack(rows, format="csr"),
        np.concatenate(lower),
        np.concatenate(upper),
        bounds,
    )


def _solve_extended_scipy(
    problem: PortfolioProblem,
    preferences: Preferences,
    constraints: PortfolioConstraints,
    *,
    tol: float = 1e-9,
    max_iter: int = 3_000,
) -> SolveResult:
    start = time.perf_counter()
    matrix, lower, upper, bounds = _linear_model(problem, constraints)
    equality = np.isfinite(lower) & np.isfinite(upper) & np.isclose(
        lower, upper, atol=1e-12
    )
    inequalities = matrix[~equality]
    ineq_low = lower[~equality]
    ineq_high = upper[~equality]
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
    feasible = linprog(
        np.zeros(matrix.shape[1]),
        A_ub=sparse.vstack(upper_rows, format="csr") if upper_rows else None,
        b_ub=np.concatenate(upper_rhs) if upper_rhs else None,
        A_eq=matrix[equality] if np.any(equality) else None,
        b_eq=lower[equality] if np.any(equality) else None,
        bounds=bounds,
        method="highs",
    )
    if not feasible.success:
        raise ValueError(f"fixed support is linearly infeasible: {feasible.message}")
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

    solve_start = time.perf_counter()
    scipy_constraints: list[LinearConstraint] = []
    if np.any(equality):
        scipy_constraints.append(
            LinearConstraint(matrix[equality], lower[equality], upper[equality])
        )
    if np.any(~equality):
        scipy_constraints.append(
            LinearConstraint(matrix[~equality], lower[~equality], upper[~equality])
        )
    solved = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )
    solve_seconds = time.perf_counter() - solve_start
    result = make_result(
        method="scipy_extended_qp",
        model_type="fixed_support",
        weights=np.asarray(solved.x[:n], dtype=float),
        problem=problem,
        preferences=preferences,
        runtime=time.perf_counter() - start,
        status=str(solved.message),
        success=bool(solved.success),
        optimal=bool(solved.success),
        metadata={
            "iterations": int(getattr(solved, "nit", 0)),
            "solve_seconds": solve_seconds,
            "linear_feasibility_status": str(feasible.message),
        },
    )
    return result


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
            has_extended_rules = any(
                value is not None
                for value in (
                    reduced_constraints.minimum_income,
                    reduced_constraints.factor_lower,
                    reduced_constraints.stress_scenarios,
                    reduced_constraints.maximum_cvar,
                )
            )
            if has_extended_rules:
                result = _solve_extended_scipy(
                    restricted,
                    self.preferences,
                    reduced_constraints,
                    **{
                        key: value
                        for key, value in self.solver_options.items()
                        if key in {"tol", "max_iter"}
                    },
                )
            else:
                reduced_result = solve_continuous(
                    restricted,
                    self.preferences,
                    backend=self.backend,
                    **self.solver_options,
                )
                result = reduced_result
            full_weights = np.zeros(self.problem.n)
            full_weights[selected] = result.weights
            result = make_result(
                method=result.method,
                model_type="fixed_support",
                weights=full_weights,
                problem=self.problem,
                preferences=self.preferences,
                runtime=time.perf_counter() - start,
                status=result.status,
                success=result.success,
                optimal=result.optimal,
                seed=result.seed,
                metadata={
                    **result.metadata,
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
    has_extended_rules = any(
        value is not None
        for value in (
            relaxed_rules.minimum_income,
            relaxed_rules.factor_lower,
            relaxed_rules.stress_scenarios,
            relaxed_rules.maximum_cvar,
        )
    )
    if has_extended_rules:
        result = _solve_extended_scipy(
            relaxed_problem,
            preferences,
            relaxed_rules,
            **{
                key: value
                for key, value in dict(solver_options or {}).items()
                if key in {"tol", "max_iter"}
            },
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
    """Find a valid sparse support with a linear mixed-integer feasibility model.

    The MILP contains the exact continuous weights, binary support decisions,
    turnover auxiliaries, and every configured linear guardrail. Its objective
    is only a lightweight quality tie-breaker; the returned support is always
    reallocated by the canonical convex oracle before it can be accepted.
    """
    if time_limit <= 0.0:
        raise ValueError("time_limit must be positive")
    problem = oracle.problem
    constraints = oracle.constraints
    if constraints.exact_cardinality is None:
        raise ValueError("support feasibility MILP requires exact_cardinality")
    constraints.validate_for(problem)
    start = time.perf_counter()
    base_matrix, base_lower, base_upper, base_bounds = _linear_model(
        problem,
        constraints,
    )
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
        [sparse.csr_matrix((1, base_dim)), np.ones((1, n))],
        format="csr",
    )
    matrix = sparse.vstack(
        [padded_base, link_upper, link_lower, cardinality],
        format="csr",
    )
    lower = np.concatenate(
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

    base_lb = np.asarray(
        [-np.inf if lo is None else lo for lo, _ in base_bounds],
        dtype=float,
    )
    base_ub = np.asarray(
        [np.inf if hi is None else hi for _, hi in base_bounds],
        dtype=float,
    )
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
        constraints=LinearConstraint(matrix, lower, upper_rows),
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
    eligible = np.flatnonzero(constraints.eligible_mask(oracle.problem.n))
    mandatory = set(constraints.mandatory_assets) | set(
        np.flatnonzero(oracle.problem.lower > 1e-12).tolist()
    )
    score = _rank_assets(oracle.problem, oracle.preferences, relaxation_weights)

    # Reserve one strong asset for every group that has a positive weight floor.
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
    "OracleEvaluation",
    "find_feasible_initial_support",
    "find_feasible_support_milp",
    "solve_relaxation",
]
