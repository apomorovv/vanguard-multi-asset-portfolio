"""Interactive Streamlit Copilot for the final hybrid optimizer.

Run with::

    streamlit run src/vanguard_portfolio/copilot_app.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vanguard_portfolio.data_generation import (
    generate_factor_universe,
    generate_return_scenarios,
    generate_synthetic_universe,
)
from vanguard_portfolio.hybrid import HybridConfig, run_hybrid_optimizer
from vanguard_portfolio.quantum_solver import XYQAOAConfig
from vanguard_portfolio.schemas import PortfolioConstraints, Preferences
from vanguard_portfolio.validation import validate_weights


def _run(
    n_assets: int,
    cardinality: int,
    seed: int,
    preferences: Preferences,
    minimum_weight: float,
    maximum_weight: float,
    target_return: float | None,
    maximum_turnover: float | None,
    minimum_income: float | None,
    factor_tolerance: float | None,
    maximum_stress_loss: float | None,
    maximum_cvar: float | None,
    mode: str,
) -> Any:
    problem = (
        generate_synthetic_universe()
        if n_assets == 6
        else generate_factor_universe(
            n_assets=n_assets,
            n_groups=min(8, max(2, cardinality // 2)),
            n_factors=min(8, max(2, n_assets // 20)),
            seed=seed,
        )
    )
    problem.target_return = target_return
    problem.max_turnover = maximum_turnover
    factor_lower = None
    factor_upper = None
    if factor_tolerance is not None:
        if not problem.has_factor_model:
            raise ValueError("factor bands require a factor-model universe")
        current_factor = problem.factor_loadings.T @ problem.w0
        factor_lower = current_factor - factor_tolerance
        factor_upper = current_factor + factor_tolerance
    scenarios = None
    if maximum_stress_loss is not None or maximum_cvar is not None:
        scenarios = generate_return_scenarios(problem, 250, seed=seed + 101)
    stress_scenarios = None
    stress_floors = None
    if maximum_stress_loss is not None:
        current_scenario_returns = scenarios @ problem.w0
        worst = np.argsort(current_scenario_returns)[:5]
        stress_scenarios = scenarios[worst]
        stress_floors = np.full(len(worst), -maximum_stress_loss)
    constraints = PortfolioConstraints(
        exact_cardinality=cardinality,
        minimum_active_weight=minimum_weight,
        maximum_weights=np.minimum(problem.upper, maximum_weight),
        minimum_income=minimum_income,
        factor_lower=factor_lower,
        factor_upper=factor_upper,
        stress_scenarios=stress_scenarios,
        stress_floors=stress_floors,
        scenario_returns=scenarios if maximum_cvar is not None else None,
        maximum_cvar=maximum_cvar,
    ).validate_for(problem)
    quantum = mode == "Quantum hybrid"
    certified = mode == "Certified classical"
    config = HybridConfig(
        iterations=2 if quantum else 1,
        window_size=min(14, n_assets),
        run_quantum=quantum,
        run_penalty_qaoa=False,
        run_gurobi_reference=certified,
        use_topology=True,
        classical_tabu_iterations=35,
        seed=seed,
        quantum=XYQAOAConfig(
            depth=1,
            shots=2_048,
            optimizer_maxiter=40,
            optimizer_starts=2,
            seed=seed,
            backend="subspace",
        ),
    )
    return run_hybrid_optimizer(problem, preferences, constraints, config)


def main() -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("Install the app extra with: pip install -e '.[app]'") from exc

    st.set_page_config(page_title="Vanguard Portfolio Copilot", layout="wide")
    st.title("Constraint-Safe Multi-Asset Portfolio Copilot")
    st.caption(
        "Classical optimization assigns exact percentages and validates every guardrail. "
        "XY-QAOA is an optional fixed-cardinality swap proposer."
    )

    with st.sidebar:
        st.header("Portfolio request")
        n_assets = st.select_slider("Asset universe", options=[6, 25, 50, 100, 250], value=50)
        maximum_k = min(50, n_assets - 1)
        cardinality = st.slider("Exact holdings", 2, maximum_k, min(15, maximum_k))
        mode = st.radio(
            "Solver mode",
            ["Fast classical", "Quantum hybrid", "Certified classical"],
            help="Certified mode requires a working Gurobi installation and license.",
        )
        st.subheader("Goal sliders")
        growth = st.slider("Growth", 0.0, 3.0, 1.0, 0.1)
        risk = st.slider("Risk control", 0.0, 20.0, 5.0, 0.5)
        income = st.slider("Income", 0.0, 3.0, 0.5, 0.1)
        costs = st.slider("Trading-cost sensitivity", 0.0, 10.0, 1.0, 0.5)
        with st.expander("Advanced hard guardrails"):
            minimum_weight = st.number_input(
                "Minimum active position", 0.0, 0.20, min(0.01, 0.8 / cardinality), 0.005
            )
            maximum_weight = st.number_input(
                "Maximum position", 0.01, 1.0, max(0.10, 1.5 / cardinality), 0.01
            )
            use_return = st.checkbox("Minimum expected return")
            target_return = (
                st.number_input("Return floor", 0.0, 0.20, 0.03, 0.005) if use_return else None
            )
            use_turnover = st.checkbox("Maximum turnover")
            maximum_turnover = (
                st.number_input("Turnover cap", 0.0, 2.0, 1.0, 0.05)
                if use_turnover
                else None
            )
            use_income = st.checkbox("Minimum income")
            minimum_income = (
                st.number_input("Income floor", 0.0, 0.10, 0.01, 0.002)
                if use_income
                else None
            )
            use_factors = st.checkbox(
                "Factor-exposure bands",
                disabled=n_assets == 6,
                help="Keep each factor exposure near the current portfolio.",
            )
            factor_tolerance = (
                st.number_input("Factor tolerance", 0.01, 1.0, 0.25, 0.01)
                if use_factors
                else None
            )
            use_stress = st.checkbox("Stress-loss limit")
            maximum_stress_loss = (
                st.number_input("Maximum stress loss", 0.01, 1.0, 0.25, 0.01)
                if use_stress
                else None
            )
            use_cvar = st.checkbox("Empirical CVaR limit")
            maximum_cvar = (
                st.number_input("Maximum 95% CVaR", 0.01, 1.0, 0.30, 0.01)
                if use_cvar
                else None
            )
        seed = st.number_input("Reproducibility seed", 0, 10_000_000, 7)
        optimize = st.button("Optimize and validate", type="primary", use_container_width=True)

    if not optimize:
        st.info("Set goals and guardrails, then select **Optimize and validate**.")
        return
    preferences = Preferences(
        lambda_return=growth,
        lambda_risk=risk,
        lambda_income=income,
        lambda_cost=costs,
    )
    try:
        with st.spinner("Solving the complete financial model..."):
            run = _run(
                n_assets,
                cardinality,
                int(seed),
                preferences,
                minimum_weight,
                maximum_weight,
                target_return,
                maximum_turnover,
                minimum_income,
                factor_tolerance,
                maximum_stress_loss,
                maximum_cvar,
                mode,
            )
    except Exception as exc:
        st.error(f"The requested guardrails could not produce a valid portfolio: {exc}")
        st.caption(
            "Try widening the position/group limits or relaxing return, income, or turnover."
        )
        return

    best = run.best
    columns = st.columns(6)
    metrics = [
        ("Expected return", best.metrics["expected_return"], ".2%"),
        ("Volatility", best.metrics["volatility"], ".2%"),
        ("Income", best.metrics["income"], ".2%"),
        ("Turnover", best.metrics["turnover"], ".2%"),
        ("Holdings", int(np.count_nonzero(best.weights > 1e-8)), "d"),
        ("Hard breaches", best.breaches, "d"),
    ]
    for column, (label, value, formatting) in zip(columns, metrics):
        column.metric(label, format(value, formatting))

    left, right = st.columns([1.4, 1.0])
    selected = np.flatnonzero(best.weights > 1e-8)
    order = selected[np.argsort(best.weights[selected])[::-1]]
    allocation = go.Figure(
        go.Bar(
            x=[run.problem.asset_names[index] for index in order],
            y=best.weights[order],
            marker_color="#0B5CAD",
        )
    )
    allocation.update_layout(
        title="Recommended allocation",
        xaxis_title="Asset",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        height=460,
    )
    left.plotly_chart(allocation, use_container_width=True)

    exposure = run.problem.A @ best.weights
    group = go.Figure()
    group.add_bar(name="Recommended", x=run.problem.group_names, y=exposure, marker_color="#00A6A6")
    group.add_scatter(
        name="Lower bound",
        x=run.problem.group_names,
        y=run.problem.group_lower,
        mode="markers",
        marker={"color": "#D1495B", "symbol": "triangle-up", "size": 10},
    )
    group.add_scatter(
        name="Upper bound",
        x=run.problem.group_names,
        y=run.problem.group_upper,
        mode="markers",
        marker={"color": "#D1495B", "symbol": "triangle-down", "size": 10},
    )
    group.update_layout(title="Group exposure and guardrails", yaxis_tickformat=".0%", height=460)
    right.plotly_chart(group, use_container_width=True)

    st.subheader("Method comparison")
    st.dataframe(run.summary_records(), use_container_width=True, hide_index=True)
    report = validate_weights(best.weights, run.problem, constraints=run.constraints)
    if report.feasible:
        st.success("Independent validation passed: zero hard-constraint breaches.")
    else:
        st.error("Independent validation found a breach; this result must not be used.")

    st.subheader("Why this portfolio")
    added = np.flatnonzero((best.weights > 1e-8) & (run.problem.w0 <= 1e-8))
    st.write(
        f"The optimizer selected {cardinality} assets while balancing growth, risk, income, "
        f"and trading cost. The binding constraints are shown below; positive slack means the "
        f"guardrail is satisfied. {len(added)} new assets entered the portfolio."
    )
    checks = sorted(report.checks, key=lambda item: item.slack)[:10]
    st.dataframe(
        [
            {
                "guardrail": check.name,
                "value": check.lhs,
                "limit": check.rhs,
                "sense": check.sense,
                "slack": check.slack,
                "status": "binding" if abs(check.slack) < 1e-5 else "satisfied",
            }
            for check in checks
        ],
        use_container_width=True,
        hide_index=True,
    )
    if run.skipped:
        skipped = "; ".join(f"{key}: {value}" for key, value in run.skipped.items())
        st.warning("Optional components skipped: " + skipped)


if __name__ == "__main__":
    main()
