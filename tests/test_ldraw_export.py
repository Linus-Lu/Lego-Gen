"""Tests for LDraw (.ldr) export."""

import pytest
from backend.brick.parser import Brick
from backend.brick.ldraw import brick_to_ldr, bricks_to_ldr, bricks_text_to_ldr


def test_brick_to_ldr_basic():
    brick = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    ldr = brick_to_ldr(brick)
    assert "3001.dat" in ldr  # 2x4 LDraw part ID
    assert "4 " in ldr  # Red colour ID
    assert "0 STEP" in ldr


def test_brick_to_ldr_orientation_landscape():
    """h <= w uses the landscape rotation matrix."""
    brick = Brick(h=2, w=4, x=0, y=0, z=0, color="FFFFFF")
    ldr = brick_to_ldr(brick)
    assert "0 0 1 0 1 0 -1 0 0" in ldr


def test_brick_to_ldr_orientation_portrait():
    """h > w uses the portrait rotation matrix."""
    brick = Brick(h=4, w=2, x=0, y=0, z=0, color="FFFFFF")
    ldr = brick_to_ldr(brick)
    assert "-1 0 0 0 1 0 0 0 -1" in ldr


def test_brick_to_ldr_colour_mapping():
    # Black
    brick = Brick(h=1, w=1, x=0, y=0, z=0, color="05131D")
    ldr = brick_to_ldr(brick)
    assert " 0 " in ldr  # LDraw colour 0 = Black

    # Blue
    brick = Brick(h=1, w=2, x=0, y=0, z=0, color="0055BF")
    ldr = brick_to_ldr(brick)
    assert " 1 " in ldr  # LDraw colour 1 = Blue


def test_brick_to_ldr_unknown_colour_defaults_white():
    brick = Brick(h=2, w=2, x=0, y=0, z=0, color="ABCDEF")
    ldr = brick_to_ldr(brick)
    assert " 15 " in ldr  # White fallback


def test_brick_to_ldr_vertical_position():
    """Z coordinate should map to negative LDraw Y."""
    brick = Brick(h=2, w=4, x=0, y=0, z=3, color="C91A09")
    ldr = brick_to_ldr(brick)
    assert "-72" in ldr  # 3 * -24 = -72


def test_bricks_to_ldr_header():
    bricks = [Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")]
    ldr = bricks_to_ldr(bricks, title="Test Model")
    assert "0 Test Model" in ldr
    assert "0 Author: LegoGen" in ldr
    assert "0 !LDRAW_ORG Unofficial_Model" in ldr


def test_bricks_to_ldr_multiple_bricks():
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="0055BF"),
    ]
    ldr = bricks_to_ldr(bricks)
    assert ldr.count("0 STEP") == 2


def test_bricks_text_to_ldr():
    text = "2x4 (0,0,0) #C91A09\n2x2 (0,0,1) #0055BF"
    ldr = bricks_text_to_ldr(text)
    assert "3001.dat" in ldr
    assert "3003.dat" in ldr


def test_bricks_text_to_ldr_empty():
    ldr = bricks_text_to_ldr("")
    assert "0 LegoGen Model" in ldr  # header still present
    assert "STEP" not in ldr
