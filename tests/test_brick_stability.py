import pytest
from backend.brick.parser import Brick
from backend.brick.stability import is_stable, find_first_unstable, is_brick_connected

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


# ── is_brick_connected ───────────────────────────────────────────────


def test_is_brick_connected_ground_brick():
    brick = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    assert is_brick_connected(brick, [], set()) is True


def test_is_brick_connected_supported():
    existing = [Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")]
    ground_set = {0}
    brick = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    assert is_brick_connected(brick, existing, ground_set) is True


def test_is_brick_connected_floating():
    existing = [Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")]
    ground_set = {0}
    brick = Brick(h=1, w=1, x=15, y=15, z=3, color="237841")
    assert is_brick_connected(brick, existing, ground_set) is False


def test_is_brick_connected_not_in_ground_set():
    """A neighbor exists at z+-1 but it is not in the ground set."""
    existing = [Brick(h=2, w=2, x=0, y=0, z=1, color="C91A09")]
    ground_set: set[int] = set()  # nothing reachable from ground
    brick = Brick(h=2, w=2, x=0, y=0, z=2, color="05131D")
    assert is_brick_connected(brick, existing, ground_set) is False


def test_is_brick_connected_below():
    """Brick connects to a grounded brick that sits on top of it (z+1)."""
    existing = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=4, x=0, y=0, z=1, color="05131D"),
    ]
    ground_set = {0, 1}
    # Brick at z=0 but in a different position — connected via ground.
    brick = Brick(h=2, w=2, x=4, y=0, z=0, color="237841")
    assert is_brick_connected(brick, existing, ground_set) is True
