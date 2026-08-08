"""Lightweight market-community signals for diverse window construction.

Communities influence candidate ordering only.  They never remove an asset or
override a financial constraint, so an imperfect graph cannot make the final
portfolio infeasible.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from .schemas import PortfolioProblem


def _feature_matrix(problem: PortfolioProblem, dimensions: int) -> np.ndarray:
    if problem.has_factor_model:
        features = problem.factor_loadings.copy()
        if features.shape[1] > 1:
            # Remove the common factor before forming communities.  A market
            # factor has large same-signed loadings but can have less
            # cross-sectional variance than a style factor, so variance alone
            # is not a reliable identifier.
            names = [name.lower() for name in (problem.factor_names or [])]
            common = (
                names.index("market")
                if "market" in names
                else int(np.argmax(np.mean(np.abs(features), axis=0)))
            )
            features = np.delete(features, common, axis=1)
        return features[:, : max(1, dimensions)]
    count = min(max(1, dimensions), problem.n - 1)
    values, vectors = eigsh(problem.corr, k=count, which="LA")
    order = np.argsort(values)[::-1]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))


def market_communities(
    problem: PortfolioProblem,
    *,
    n_communities: int | None = None,
    dimensions: int = 6,
    max_iterations: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Cluster factor/residual features with deterministic farthest-first k-means."""
    if problem.n < 2:
        return np.zeros(problem.n, dtype=int)
    groups = int(n_communities or max(2, min(12, problem.num_groups * 2)))
    groups = min(groups, problem.n)
    features = _feature_matrix(problem, dimensions)
    features = features - features.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(scale, 1e-12)
    rng = np.random.default_rng(seed)
    centroids = [features[int(rng.integers(problem.n))]]
    distance = np.sum((features - centroids[0]) ** 2, axis=1)
    for _ in range(1, groups):
        index = int(np.argmax(distance))
        centroids.append(features[index])
        distance = np.minimum(distance, np.sum((features - features[index]) ** 2, axis=1))
    centers = np.asarray(centroids)
    labels = np.zeros(problem.n, dtype=int)
    for _ in range(int(max_iterations)):
        squared = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = np.argmin(squared, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for group in range(groups):
            members = features[labels == group]
            if members.size:
                centers[group] = members.mean(axis=0)
            else:
                centers[group] = features[int(np.argmax(np.min(squared, axis=1)))]
    unique = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([unique[int(label)] for label in labels], dtype=int)


def community_summary(problem: PortfolioProblem, labels: np.ndarray) -> list[dict[str, object]]:
    labels = np.asarray(labels, dtype=int).reshape(problem.n)
    rows: list[dict[str, object]] = []
    for community in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == community)
        rows.append(
            {
                "community": int(community),
                "assets": int(indices.size),
                "average_return": float(problem.mu[indices].mean()),
                "average_volatility": float(problem.sigma[indices].mean()),
                "dominant_group": problem.group_names[
                    int(np.bincount(np.asarray(problem.asset_group)[indices]).argmax())
                ],
            }
        )
    return rows


__all__ = ["community_summary", "market_communities"]
