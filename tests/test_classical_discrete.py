"""Tests for the discrete (integer-lot) mean-variance optimizer."""

import numpy as np
import pytest

from src.vanguard_portfolio.classical_continuous import PortfolioProblem, mean_variance_continuous
from src.vanguard_portfolio.classical_discrete import (
    MeanVarianceDiscreteOptimizer,
    mean_variance_discrete,
)


@pytest.fixture
def simple_problem():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    return PortfolioProblem(mu, cov, risk_aversion=3.0)


@pytest.mark.parametrize("method", ["brute", "anneal"])
def test_budget_preserved_by_lots(simple_problem, method):
    res = mean_variance_discrete(simple_problem, n_lots=10, method=method, seed=0)
    assert int(res["lots"].sum()) == 10
    assert res["weights"].sum() == pytest.approx(simple_problem.budget, abs=1e-9)


@pytest.mark.parametrize("method", ["brute", "anneal"])
def test_weights_are_on_the_lot_grid(simple_problem, method):
    n_lots = 10
    res = mean_variance_discrete(simple_problem, n_lots=n_lots, method=method, seed=0)
    lot_size = simple_problem.budget / n_lots
    ratios = res["weights"] / lot_size
    assert np.allclose(ratios, np.round(ratios), atol=1e-9)


def test_brute_is_optimal_versus_anneal(simple_problem):
    brute = mean_variance_discrete(simple_problem, n_lots=8, method="brute")
    anneal = mean_variance_discrete(simple_problem, n_lots=8, method="anneal", seed=1)
    # The exhaustive search can never be beaten by annealing.
    assert brute["utility"] >= anneal["utility"] - 1e-9


def test_discrete_approaches_continuous_as_resolution_grows():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    problem = PortfolioProblem(mu, cov, risk_aversion=3.0)

    cont = mean_variance_continuous(problem)
    coarse = mean_variance_discrete(problem, n_lots=4, method="brute")
    fine = mean_variance_discrete(problem, n_lots=40, method="anneal", seed=0)

    gap_coarse = abs(cont["utility"] - coarse["utility"])
    gap_fine = abs(cont["utility"] - fine["utility"])
    # Finer lattice should track the continuous optimum at least as well.
    assert gap_fine <= gap_coarse + 1e-6


def test_lot_bounds_are_respected():
    mu = np.array([0.10, 0.07, 0.03])
    cov = np.diag([0.04, 0.02, 0.005])
    # Cap each asset at 50% of the budget.
    problem = PortfolioProblem(mu, cov, risk_aversion=3.0, upper_bounds=0.5)
    res = mean_variance_discrete(problem, n_lots=10, method="brute")
    assert np.all(res["weights"] <= 0.5 + 1e-9)


def test_sector_penalty_discourages_breach():
    mu = np.array([0.12, 0.11, 0.04, 0.03])
    cov = np.diag([0.05, 0.05, 0.01, 0.01])
    problem = PortfolioProblem(
        mu,
        cov,
        risk_aversion=1.0,
        sector_map=["equity", "equity", "bond", "bond"],
        sector_limits={"equity": 0.5},
    )
    res = mean_variance_discrete(
        problem, n_lots=10, method="brute", penalty_weight=1000.0
    )
    equity_exposure = res["weights"][:2].sum()
    assert equity_exposure <= 0.5 + 1e-9
    assert res["sector_penalty"] == pytest.approx(0.0, abs=1e-9)


def test_unknown_method_raises(simple_problem):
    with pytest.raises(ValueError):
        MeanVarianceDiscreteOptimizer(simple_problem, method="quantum").solve()