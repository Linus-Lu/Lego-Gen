import pytest
from backend.brick.parser import Brick
from backend.brick.stability import is_stable, find_first_unstable

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
