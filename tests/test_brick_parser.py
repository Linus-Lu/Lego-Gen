import pytest
from backend.brick.parser import parse_brick, serialize_brick, parse_brick_sequence, Brick

def test_parse_single_brick():
    b = parse_brick("2x4 (5,3,0) #C91A09")
    assert b == Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09")

def test_parse_brick_no_hash():
    b = parse_brick("1x1 (0,0,0) #05131D")
    assert b.color == "05131D"

def test_parse_invalid_format():
    with pytest.raises(ValueError, match="Invalid brick"):
        parse_brick("garbage")

def test_serialize_brick():
    b = Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09")
    assert serialize_brick(b) == "2x4 (5,3,0) #C91A09"

def test_roundtrip():
    line = "1x6 (10,12,3) #237841"
    assert serialize_brick(parse_brick(line)) == line

def test_parse_sequence():
    raw = "2x4 (5,3,0) #C91A09\n1x2 (3,7,1) #05131D\n"
    bricks = parse_brick_sequence(raw)
    assert len(bricks) == 2
    assert bricks[0].h == 2
    assert bricks[1].color == "05131D"

def test_parse_sequence_empty():
    assert parse_brick_sequence("") == []
    assert parse_brick_sequence("  \n  ") == []
