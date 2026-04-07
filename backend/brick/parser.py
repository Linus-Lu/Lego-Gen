"""Parse and serialize the brick text format: ``HxW (x,y,z) #RRGGBB``."""

import re
from dataclasses import dataclass

# Compiled regex: "2x4 (5,3,0) #C91A09"
_BRICK_RE = re.compile(
    r"^(\d+)x(\d+)\s+\((-?\d+),(-?\d+),(-?\d+)\)\s+#([0-9A-Fa-f]{6})$"
)


@dataclass(frozen=True, slots=True)
class Brick:
    h: int
    w: int
    x: int
    y: int
    z: int
    color: str  # 6-char hex, uppercase, no '#'


def parse_brick(line: str) -> Brick:
    """Parse a single brick line, e.g. ``"2x4 (5,3,0) #C91A09"``."""
    m = _BRICK_RE.fullmatch(line.strip())
    if m is None:
        raise ValueError(f"Invalid brick line: {line!r}")
    h, w, x, y, z = (int(m.group(i)) for i in range(1, 6))
    color = m.group(6).upper()
    return Brick(h=h, w=w, x=x, y=y, z=z, color=color)


def serialize_brick(b: Brick) -> str:
    """Serialize a Brick back to its canonical text form."""
    return f"{b.h}x{b.w} ({b.x},{b.y},{b.z}) #{b.color}"


def parse_brick_sequence(raw: str) -> list[Brick]:
    """Parse a newline-separated sequence of brick lines.

    Empty or whitespace-only lines are silently skipped.
    """
    bricks: list[Brick] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            bricks.append(parse_brick(stripped))
    return bricks
