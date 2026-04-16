import pytest
from backend.brick.parser import Brick
from backend.brick.stability import is_stable, find_first_unstable


# ── Legacy tests — physically reasonable, preserved verbatim ────────────

def test_single_ground_brick_stable():
    bricks = [Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")]
    assert is_stable(bricks)


def test_floating_brick_unstable():
    bricks = [Brick(h=2, w=2, x=0, y=0, z=3, color="C91A09")]
    assert not is_stable(bricks)


def test_supported_brick_stable():
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
    ]
    assert is_stable(bricks)


def test_unsupported_second_layer():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=10, y=10, z=1, color="05131D"),
    ]
    assert not is_stable(bricks)


def test_find_first_unstable():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
        Brick(h=1, w=1, x=15, y=15, z=1, color="237841"),
    ]
    assert find_first_unstable(bricks) == 2


def test_find_first_unstable_all_stable():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
    ]
    assert find_first_unstable(bricks) == -1


# ── Physics-specific tests (fail under LP, pass under BFS-only) ─────────

def test_top_heavy_inverted_pyramid_unstable():
    """A 2x2 on a single 1x1 stud — moment arm blows up the LP even though
    BFS connectivity would happily accept it."""
    bricks = [
        Brick(h=1, w=1, x=1, y=1, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
    ]
    assert not is_stable(bricks)


def test_single_stud_cantilever_unstable():
    """A 2x4 cantilevered three studs past its 2x4 support — the overhang
    moment exceeds what two rows of stud tension can resist."""
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=4, x=0, y=3, z=1, color="05131D"),
    ]
    assert not is_stable(bricks)


def test_centered_stack_stable():
    """A centered 2x4 on a 2x4 is obviously stable — all moments cancel."""
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=4, x=0, y=0, z=1, color="05131D"),
    ]
    assert is_stable(bricks)


def test_small_overhang_balanced_stable():
    """A 2x4 offset by a single stud — six studs of support carry the
    small moment via uneven force distribution."""
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=4, x=0, y=1, z=1, color="05131D"),
    ]
    assert is_stable(bricks)


def test_counterweighted_cantilever_stable():
    """A beam that tips on its own is rescued by a lid brick bridging it to
    an anchor column. The lid's stud grip on the beam's overhang side holds
    the beam down; the LP finds the tension distribution that closes the
    moment balance."""
    bricks = [
        Brick(h=2, w=8, x=0, y=0, z=0, color="C91A09"),   # base
        Brick(h=2, w=4, x=0, y=6, z=1, color="05131D"),   # beam — 2-stud overhang
        Brick(h=2, w=4, x=0, y=0, z=1, color="237841"),   # anchor column
        Brick(h=2, w=8, x=0, y=2, z=2, color="F2CD37"),   # lid — spans anchor & beam
    ]
    assert is_stable(bricks)
