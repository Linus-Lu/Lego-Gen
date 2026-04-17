# tests/test_best_of_n.py
import numpy as np

from backend.brick.parser import Brick
from backend.inference.best_of_n import rank_candidates, structural_features, cluster_and_pick

def test_rank_prefers_stable_then_brickcount():
    stable_big = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 8,  "stable": True,  "brick_count": 8}
    stable_small = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 3, "stable": True,  "brick_count": 3}
    unstable_big = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 20,"stable": False, "brick_count": 20}

    ranked = rank_candidates([unstable_big, stable_small, stable_big])

    assert ranked[0] is stable_big       # stable + largest
    assert ranked[1] is stable_small     # stable + smaller
    assert ranked[2] is unstable_big     # unstable last regardless of size


def test_structural_features_dim_and_monotonicity():
    small = [Brick(1, 1, 0, 0, 0, "C91A09")]
    bigger = [
        Brick(2, 4, 0, 0, 0, "C91A09"),
        Brick(2, 4, 0, 0, 1, "C91A09"),
        Brick(2, 4, 0, 0, 2, "C91A09"),
    ]

    vs, vb = structural_features(small), structural_features(bigger)

    assert vs.shape == vb.shape == (9,)
    # Feature 0 is brick count — bigger has more.
    assert vb[0] > vs[0]
    # Feature 3 is z-extent — bigger is taller.
    assert vb[3] > vs[3]
    # Feature 8 is unique-color count — both equal 1.
    assert vs[8] == vb[8] == 1.0


def test_cluster_and_pick_returns_centroid_of_largest_valid_cluster():
    big_cluster = [
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 10, "stable": True, "brick_count": 10},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 11, "stable": True, "brick_count": 11},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 9,  "stable": True, "brick_count": 9},
    ]
    singleton_outlier = [
        {"bricks": [Brick(1, 1, 5, 5, 0, "0055BF")] * 1,  "stable": True, "brick_count": 1}
    ]
    all_unstable = [
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 12, "stable": False, "brick_count": 12},
    ]

    picked = cluster_and_pick(big_cluster + singleton_outlier + all_unstable, k=2, seed=0)

    # Must be drawn from the majority stable cluster.
    assert picked in big_cluster
    # Prefer the centroid — should be the candidate closest to cluster mean count (10).
    assert picked["brick_count"] == 10


def test_cluster_and_pick_falls_back_to_rank_when_stable_equals_k():
    """When len(stable) <= k, KMeans degenerates to singleton clusters and
    the centroid pick is driven only by init seed. Fall back to rank instead."""
    cands = [
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 5, "stable": True, "brick_count": 5},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 8, "stable": True, "brick_count": 8},
    ]
    picked = cluster_and_pick(cands, k=2, seed=0)
    # Rank picks the larger stable build.
    assert picked["brick_count"] == 8


def test_structural_features_empty_bricks_returns_zero_vector():
    import numpy as np
    out = structural_features([])
    assert out.shape == (9,)
    assert np.allclose(out, 0.0)


def test_cluster_and_pick_all_unstable_falls_back_to_rank():
    """Zero stable candidates → fall through rank_candidates. Ranking
    prefers stable, so this returns the unstable candidate with the most
    bricks (all are unstable so stability is a tie)."""
    cands = [
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 3, "stable": False, "brick_count": 3},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 7, "stable": False, "brick_count": 7},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 2, "stable": False, "brick_count": 2},
    ]
    picked = cluster_and_pick(cands, k=2, seed=0)
    assert picked["brick_count"] == 7


def test_rank_candidates_is_deterministic_on_ties():
    """Identical candidates → original order preserved (earlier wins)."""
    a = {"bricks": [], "stable": True, "brick_count": 5}
    b = {"bricks": [], "stable": True, "brick_count": 5}
    ranked = rank_candidates([a, b])
    assert ranked[0] is a
    assert ranked[1] is b
