# tests/test_best_of_n.py
from backend.brick.parser import Brick
from backend.inference.best_of_n import rank_candidates

def test_rank_prefers_stable_then_brickcount():
    stable_big = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 8,  "stable": True,  "brick_count": 8}
    stable_small = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 3, "stable": True,  "brick_count": 3}
    unstable_big = {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 20,"stable": False, "brick_count": 20}

    ranked = rank_candidates([unstable_big, stable_small, stable_big])

    assert ranked[0] is stable_big       # stable + largest
    assert ranked[1] is stable_small     # stable + smaller
    assert ranked[2] is unstable_big     # unstable last regardless of size
