import pytest
from backend.brick.parser import Brick
from backend.brick.occupancy import VoxelGrid
from backend.brick.reliability import ReliabilityScorer, BrickScore


def _make_scorer() -> tuple[ReliabilityScorer, VoxelGrid]:
    grid = VoxelGrid()
    return ReliabilityScorer(grid), grid


# ── Ground bricks ─────────────────────────────────────────────────────


def test_ground_brick_fully_reliable():
    scorer, grid = _make_scorer()
    brick = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    score = scorer.add_brick(brick)
    grid.place(brick)

    assert score.connectivity == 1.0
    assert score.support_ratio == 1.0  # z=0 -> ground support
    assert score.score > 0


def test_ground_brick_neighbor_count_zero():
    scorer, grid = _make_scorer()
    brick = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    score = scorer.add_brick(brick)

    assert score.neighbor_count == 0  # no prior bricks


# ── Supported bricks ──────────────────────────────────────────────────


def test_supported_brick_connected():
    scorer, grid = _make_scorer()
    base = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(base)
    grid.place(base)

    top = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    score = scorer.add_brick(top)

    assert score.connectivity == 1.0
    assert score.support_ratio > 0.0
    assert score.neighbor_count >= 1


def test_fully_supported_brick_ratio_one():
    """A 2x4 brick on top of another 2x4 at the same position: full support."""
    scorer, grid = _make_scorer()
    base = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(base)
    grid.place(base)

    top = Brick(h=2, w=4, x=0, y=0, z=1, color="05131D")
    score = scorer.add_brick(top)

    assert score.support_ratio == 1.0


# ── Floating bricks ──────────────────────────────────────────────────


def test_floating_brick_not_connected():
    scorer, grid = _make_scorer()
    base = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(base)
    grid.place(base)

    floating = Brick(h=1, w=1, x=15, y=15, z=3, color="237841")
    score = scorer.add_brick(floating)

    assert score.connectivity == 0.0
    assert score.support_ratio == 0.0
    assert score.neighbor_count == 0


def test_isolated_floating_brick():
    scorer, grid = _make_scorer()
    floating = Brick(h=2, w=2, x=5, y=5, z=5, color="C91A09")
    score = scorer.add_brick(floating)

    assert score.connectivity == 0.0


# ── Partial overhang ──────────────────────────────────────────────────


def test_overhang_partial_support():
    """A 2x4 brick half-resting on a 2x2 base: support_ratio ~0.25."""
    scorer, grid = _make_scorer()
    base = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(base)
    grid.place(base)

    overhang = Brick(h=2, w=4, x=0, y=0, z=1, color="05131D")
    score = scorer.add_brick(overhang)

    assert score.connectivity == 1.0
    # 2x2 base supports 4 cells of the 2x4 (8 cells) -> ratio = 0.5
    assert 0.0 < score.support_ratio < 1.0


# ── Aggregate score ──────────────────────────────────────────────────


def test_aggregate_score_empty():
    scorer, _ = _make_scorer()
    assert scorer.aggregate_score() == 1.0


def test_aggregate_score_multiple_bricks():
    scorer, grid = _make_scorer()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=4, x=0, y=0, z=1, color="05131D")
    scorer.add_brick(b1)
    grid.place(b1)
    scorer.add_brick(b2)
    grid.place(b2)

    agg = scorer.aggregate_score()
    assert 0.0 < agg <= 1.0
    assert agg == pytest.approx(
        (scorer.scores[0].score + scorer.scores[1].score) / 2
    )


def test_min_score():
    scorer, grid = _make_scorer()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=4, x=0, y=0, z=1, color="05131D")
    scorer.add_brick(b1)
    grid.place(b1)
    scorer.add_brick(b2)
    grid.place(b2)

    assert scorer.min_score() == min(s.score for s in scorer.scores)


def test_min_score_empty():
    scorer, _ = _make_scorer()
    assert scorer.min_score() == 1.0


# ── Rollback (remove_last) ───────────────────────────────────────────


def test_remove_last_restores_state():
    scorer, grid = _make_scorer()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    scorer.add_brick(b1)
    grid.place(b1)
    scorer.add_brick(b2)
    grid.place(b2)

    assert len(scorer.scores) == 2

    scorer.remove_last(1)
    assert len(scorer.scores) == 1

    # The remaining brick should be the ground brick.
    assert scorer.scores[0].connectivity == 1.0


def test_remove_last_more_than_exists():
    scorer, grid = _make_scorer()
    b1 = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(b1)
    grid.place(b1)

    scorer.remove_last(5)  # remove more than available
    assert len(scorer.scores) == 0
    assert scorer.aggregate_score() == 1.0


# ── Reset ─────────────────────────────────────────────────────────────


def test_reset_clears_state():
    scorer, grid = _make_scorer()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(b1)
    grid.place(b1)

    scorer.reset()
    assert len(scorer.scores) == 0
    assert scorer.aggregate_score() == 1.0


# ── Ground propagation ────────────────────────────────────────────────


def test_ground_propagation_bridges():
    """Adding a bridging brick should propagate ground-reachability."""
    scorer, grid = _make_scorer()

    base = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    scorer.add_brick(base)
    grid.place(base)

    mid = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    score_mid = scorer.add_brick(mid)
    grid.place(mid)

    assert score_mid.connectivity == 1.0

    top = Brick(h=2, w=2, x=0, y=0, z=2, color="237841")
    score_top = scorer.add_brick(top)
    grid.place(top)

    assert score_top.connectivity == 1.0


# ── Score dataclass ───────────────────────────────────────────────────


def test_brick_score_is_frozen():
    score = BrickScore(connectivity=1.0, support_ratio=0.5, neighbor_count=2, score=0.8)
    with pytest.raises(AttributeError):
        score.connectivity = 0.0  # type: ignore[misc]
