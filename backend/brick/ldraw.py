"""LDraw (.ldr) export for LEGO brick structures.

Converts the internal brick representation to LDraw format, which is compatible
with LEGO-building tools like LDraw editors, Studio 2.0, and Blender via
ImportLDraw.
"""

from __future__ import annotations

from backend.brick.parser import Brick
from backend.brick.constants import LDRAW_IDS


# Mapping from 6-char hex (uppercase) to LDraw colour IDs.
# Covers the most common LEGO colours; unknown colours fall back to 15 (white).
_HEX_TO_LDRAW_COLOR: dict[str, int] = {
    "05131D": 0,    # Black
    "FFFFFF": 15,   # White
    "C91A09": 4,    # Red
    "0055BF": 1,    # Blue
    "237841": 2,    # Green
    "FEC401": 14,   # Yellow
    "F5CD2F": 14,   # Yellow (alt)
    "462F1D": 6,    # Brown
    "A0A5A9": 7,    # Light Grey
    "6C6E68": 8,    # Dark Grey
    "FE8A18": 25,   # Orange
    "C870A0": 13,   # Pink
    "81007B": 5,    # Magenta
    "069D9F": 3,    # Dark Turquoise
    "E4CD9E": 19,   # Tan
    "352100": 308,  # Dark Brown
    "AA7D55": 28,   # Nougat
    "9BA19D": 135,  # Sand Green
    "F2CD37": 226,  # Bright Light Yellow
    "D67572": 29,   # Salmon
}

_DEFAULT_COLOR = 15  # White


def _ldraw_color(hex_color: str) -> int:
    """Map a 6-char hex color to the closest LDraw colour ID."""
    return _HEX_TO_LDRAW_COLOR.get(hex_color.upper(), _DEFAULT_COLOR)


def _normalize_dims(h: int, w: int) -> tuple[int, int]:
    """Return (h, w) with h <= w for LDRAW_IDS lookup."""
    return (h, w) if h <= w else (w, h)


def brick_to_ldr(brick: Brick) -> str:
    """Convert a single :class:`Brick` to an LDraw part line + STEP command.

    Coordinate mapping (internal -> LDraw):
      LDraw X = (brick.x + h/2) * 20
      LDraw Y = -brick.z * 24          (LDraw Y is inverted vertical)
      LDraw Z = (brick.y + w/2) * 20
    """
    norm = _normalize_dims(brick.h, brick.w)
    part_id = LDRAW_IDS.get(norm)
    if part_id is None:
        return ""

    color = _ldraw_color(brick.color)

    # Centre of the brick footprint in LDraw stud units.
    ldr_x = (brick.x + brick.h * 0.5) * 20
    ldr_y = brick.z * -24
    ldr_z = (brick.y + brick.w * 0.5) * 20

    # Rotation matrix depends on whether the brick was rotated (h > w means
    # portrait orientation in the internal grid).
    if brick.h <= brick.w:
        matrix = "0 0 1 0 1 0 -1 0 0"
    else:
        matrix = "-1 0 0 0 1 0 0 0 -1"

    part_line = f"1 {color} {ldr_x:g} {ldr_y:g} {ldr_z:g} {matrix} {part_id}.dat"
    return part_line + "\n0 STEP\n"


def bricks_to_ldr(bricks: list[Brick], title: str = "LegoGen Model") -> str:
    """Convert a list of bricks to a complete LDraw (.ldr) file string.

    The returned string can be written directly to a ``.ldr`` file.
    """
    lines: list[str] = [
        f"0 {title}\n",
        "0 Name: model.ldr\n",
        "0 Author: LegoGen\n",
        "0 !LDRAW_ORG Unofficial_Model\n",
        "\n",
    ]
    for brick in bricks:
        ldr = brick_to_ldr(brick)
        if ldr:
            lines.append(ldr)
    return "".join(lines)


def bricks_text_to_ldr(bricks_text: str, title: str = "LegoGen Model") -> str:
    """Convenience: parse a newline-separated brick string and return LDR."""
    from backend.brick.parser import parse_brick_sequence
    return bricks_to_ldr(parse_brick_sequence(bricks_text), title=title)
