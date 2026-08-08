"""Generate a synthetic_universe.json-shaped portfolio dataset for a given
number of assets, so the solvers can be scale-tested beyond the original
fixed 6-asset dataset.

Usage (from the repository root)::

    python scripts/generate_synthetic_universe.py 15 --seed 1
    # writes data/synthetic/synthetic_universe_15.json

    python scripts/generate_synthetic_universe.py 15 --out my_universe.json

Schema produced matches exactly what build_problem_from_json /
load_universe_from_json already expect (see compare_all.py /
quantum_vqe_solver.py): asset_names, group_names, asset_group, mu, cov,
c, w0, lower, upper, group_lower, group_upper, y.

Design notes
------------
* Every asset is assigned to one of 4 themed groups (Equity, FixedIncome,
  Alternatives, Cash), same as the original dataset -- just with more
  assets stuffed into each group as n_assets grows, rather than inventing
  new groups. This keeps group-exposure constraints meaningfully binding
  at any scale instead of becoming trivial as n_assets grows.
* Expected returns and volatilities are drawn from group-themed base
  values (equity-like assets get higher return/vol than cash-like ones)
  plus small random noise, so the resulting problem stays realistic-
  looking rather than being pure unstructured noise.
* The covariance matrix is built as diag(vol) @ corr @ diag(vol) where
  corr comes from normalizing a random Gram matrix (A @ A.T) -- this is
  guaranteed positive semi-definite (a congruence transform of a PSD
  matrix is PSD), so PortfolioProblem's variance term is always
  well-defined regardless of n_assets or seed.
* Per-asset upper bounds are randomized but rescaled if needed so they
  sum to at least 1.1x the budget -- otherwise the budget constraint
  (lots must sum to exactly n_lots) could become infeasible outright for
  small n_assets with tight caps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_GROUP_NAMES = ["Equity", "FixedIncome", "Alternatives", "Cash"]

# (group_lower, group_upper) as a fraction of total budget -- fixed,
# generous bands regardless of n_assets, since these are aggregate
# exposure limits, not per-asset ones.
DEFAULT_GROUP_BOUNDS = {
    "Equity": (0.05, 0.70),
    "FixedIncome": (0.05, 0.70),
    "Alternatives": (0.00, 0.30),
    "Cash": (0.00, 0.50),
}

BASE_MU = {"Equity": 0.07, "FixedIncome": 0.03, "Alternatives": 0.05, "Cash": 0.01}
BASE_VOL = {"Equity": 0.18, "FixedIncome": 0.06, "Alternatives": 0.12, "Cash": 0.01}


def generate_synthetic_universe(
    n_assets: int,
    seed: int = 0,
    group_names: Optional[list[str]] = None,
) -> dict:
    group_names = group_names or DEFAULT_GROUP_NAMES
    n_groups = len(group_names)
    if n_assets < n_groups:
        raise ValueError(
            f"n_assets={n_assets} is fewer than the {n_groups} themed groups "
            f"({group_names}) -- every group needs at least one asset."
        )

    rng = np.random.default_rng(seed)

    # Round-robin group assignment guarantees every group gets >=1 asset,
    # then shuffle so the group pattern isn't a suspiciously regular cycle.
    asset_group = np.array([i % n_groups for i in range(n_assets)])
    rng.shuffle(asset_group)

    counters = {g: 0 for g in range(n_groups)}
    asset_names = []
    for g in asset_group:
        counters[g] += 1
        asset_names.append(f"{group_names[g]}_{counters[g]}")

    mu = np.array([
        BASE_MU.get(group_names[g], 0.04) + rng.normal(0, 0.01)
        for g in asset_group
    ])

    vol = np.array([
        max(0.005, BASE_VOL.get(group_names[g], 0.10) + rng.normal(0, 0.02))
        for g in asset_group
    ])
    A = rng.normal(size=(n_assets, n_assets))
    gram = A @ A.T
    d = np.sqrt(np.diag(gram))
    corr = gram / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol, vol) * corr

    c = np.round(rng.uniform(0.0005, 0.004, size=n_assets), 5)
    w0 = np.full(n_assets, 1.0 / n_assets)

    lower = np.zeros(n_assets)
    upper = np.clip(rng.uniform(0.15, 0.35, size=n_assets), 0.10, 0.90)
    if upper.sum() < 1.1:
        upper = upper * (1.1 / upper.sum())

    group_lower = np.array([DEFAULT_GROUP_BOUNDS.get(name, (0.0, 1.0))[0] for name in group_names])
    group_upper = np.array([DEFAULT_GROUP_BOUNDS.get(name, (0.0, 1.0))[1] for name in group_names])

    y = np.array([0.01 if group_names[g] == "Cash" else 0.0 for g in asset_group])

    return {
        "asset_names": asset_names,
        "group_names": group_names,
        "asset_group": asset_group.tolist(),
        "mu": mu.tolist(),
        "cov": cov.tolist(),
        "c": c.tolist(),
        "w0": w0.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "group_lower": group_lower.tolist(),
        "group_upper": group_upper.tolist(),
        "y": y.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_assets", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                         help="Output path. Default: "
                              "data/synthetic/synthetic_universe_<n_assets>.json")
    args = parser.parse_args()

    data = generate_synthetic_universe(args.n_assets, seed=args.seed)

    if args.out:
        out_path = Path(args.out)
    else:
        root = Path(__file__).resolve().parents[1]
        out_path = root / "data" / "synthetic" / f"synthetic_universe_{args.n_assets}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out_path}  ({args.n_assets} assets, seed={args.seed})")


if __name__ == "__main__":
    main()
