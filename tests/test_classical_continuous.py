"""Tests for the continuous mean-variance optimizer."""

import numpy as np
import pytest

from src.vanguard_portfolio.classical_continuous import (
    MeanVarianceContinuousOptimizer,
    PortfolioProblem,
    mean_variance_continuous,
)


@pytest.fixture
def simple_problem():
    """A small, well-conditioned 3-asset problem, long-only, fully invested."""
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    return PortfolioProblem(mu, cov, risk_aversion=3.0)


def test_budget_constraint_is_satisfied(simple_problem):
    res = mean_variance_continuous(simple_problem)
    assert res["success"]
    assert res["weights"].sum() == pytest.approx(simple_problem.budget, abs=1e-6)


def test_long_only_bounds_respected(simple_problem):
    res = mean_variance_continuous(simple_problem)
    w = res["weights"]
    assert np.all(w >= -1e-6)
    assert np.all(w <= 1.0 + 1e-6)


def test_higher_risk_aversion_reduces_variance():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    low = mean_variance_continuous(PortfolioProblem(mu, cov, risk_aversion=1.0))
    high = mean_variance_continuous(PortfolioProblem(mu, cov, risk_aversion=25.0))
    assert high["variance"] <= low["variance"] + 1e-9


def test_target_return_constraint():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    problem = PortfolioProblem(mu, cov, risk_aversion=10.0)
    target = 0.08
    res = MeanVarianceContinuousOptimizer(problem, target_return=target).solve()
    assert res["success"]
    assert res["expected_return"] >= target - 1e-5


def test_sector_limit_not_breached():
    mu = np.array([0.12, 0.11, 0.04, 0.03])
    cov = np.diag([0.05, 0.05, 0.01, 0.01])
    problem = PortfolioProblem(
        mu,
        cov,
        risk_aversion=1.0,
        sector_map=["equity", "equity", "bond", "bond"],
        sector_limits={"equity": 0.5},
    )
    res = MeanVarianceContinuousOptimizer(problem).solve()
    equity_exposure = res["weights"][:2].sum()
    assert equity_exposure <= 0.5 + 1e-5


def test_transaction_cost_reduces_turnover():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    prev = np.array([1 / 3, 1 / 3, 1 / 3])
    no_cost = mean_variance_continuous(
        PortfolioProblem(mu, cov, risk_aversion=3.0, prev_weights=prev)
    )
    with_cost = mean_variance_continuous(
        PortfolioProblem(
            mu,
            cov,
            risk_aversion=3.0,
            prev_weights=prev,
            transaction_cost=np.full(3, 0.05),
            cost_aversion=5.0,
        )
    )
    assert with_cost["turnover"] <= no_cost["turnover"] + 1e-6


def test_invalid_covariance_shape_raises():
    mu = np.array([0.1, 0.05])
    bad_cov = np.eye(3)
    with pytest.raises(ValueError):
        PortfolioProblem(mu, bad_cov)