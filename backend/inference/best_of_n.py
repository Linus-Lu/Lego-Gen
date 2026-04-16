# backend/inference/best_of_n.py
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans

from backend.brick.parser import Brick


def rank_candidates(candidates: Iterable[dict]) -> list[dict]:
    """Sort candidates descending by (stable, brick_count).

    Preserves input order as the final tiebreaker so ranking is deterministic
    when two candidates are otherwise equal — important for the eval plot.
    """
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            1 if pair[1]["stable"] else 0,
            pair[1]["brick_count"],
            -pair[0],  # earlier index wins ties
        ),
        reverse=True,
    )
    return [c for _, c in indexed]


def structural_features(bricks: list[Brick]) -> np.ndarray:
    """Fixed 9-dim descriptor for clustering candidates without rendering.

    Layout:
      0  brick count
      1  x extent (max(x+h) - min(x))
      2  y extent (max(y+w) - min(y))
      3  z extent (max(z) - min(z) + 1)
      4  center-of-mass x (weighted by footprint area)
      5  center-of-mass y
      6  center-of-mass z
      7  mean footprint area
      8  unique colors
    """
    if not bricks:
        return np.zeros(9, dtype=np.float32)

    xs_lo = min(b.x for b in bricks)
    xs_hi = max(b.x + b.h for b in bricks)
    ys_lo = min(b.y for b in bricks)
    ys_hi = max(b.y + b.w for b in bricks)
    zs_lo = min(b.z for b in bricks)
    zs_hi = max(b.z for b in bricks)

    areas = np.array([b.h * b.w for b in bricks], dtype=np.float32)
    com_x = float(np.sum([(b.x + b.h / 2) * b.h * b.w for b in bricks]) / areas.sum())
    com_y = float(np.sum([(b.y + b.w / 2) * b.h * b.w for b in bricks]) / areas.sum())
    com_z = float(np.sum([b.z * b.h * b.w for b in bricks]) / areas.sum())

    return np.array([
        float(len(bricks)),
        float(xs_hi - xs_lo),
        float(ys_hi - ys_lo),
        float(zs_hi - zs_lo + 1),
        com_x, com_y, com_z,
        float(areas.mean()),
        float(len({b.color for b in bricks})),
    ], dtype=np.float32)


def cluster_and_pick(candidates: list[dict], k: int = 2, seed: int = 0) -> dict:
    """Filter to stable candidates, cluster by structural features, return the
    candidate closest to the centroid of the largest cluster.

    Falls back to rank_candidates when there aren't strictly more stable
    candidates than clusters — at equality each point becomes its own
    singleton and KMeans contributes no signal.
    """
    stable = [c for c in candidates if c["stable"]]
    if len(stable) <= max(k, 2):
        ranked = rank_candidates(candidates)
        return ranked[0]

    feats = np.stack([structural_features(c["bricks"]) for c in stable])
    # Normalize so brick-count doesn't dominate.
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-6
    feats_z = (feats - mean) / std

    k_eff = min(k, len(stable))
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto").fit(feats_z)
    labels = km.labels_

    # Largest cluster wins.
    counts = np.bincount(labels)
    winner = int(np.argmax(counts))
    members = [i for i, lbl in enumerate(labels) if lbl == winner]

    # Centroid in normalized space -> pick closest member.
    centroid = km.cluster_centers_[winner]
    dists = np.linalg.norm(feats_z[members] - centroid, axis=1)
    best = members[int(np.argmin(dists))]
    return stable[best]
