# tests/test_best_of_n.py
import numpy as np

from backend.brick.parser import Brick
from backend.inference.best_of_n import rank_candidates, structural_features

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
