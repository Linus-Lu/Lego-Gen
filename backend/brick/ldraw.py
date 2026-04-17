"""LDraw export helpers for LEGOGen brick sequences.

The project models each placement as a simple cuboid on a stud grid. Export
uses the canonical LDraw brick part IDs already defined in ``constants.py``
and places them on a coarse stud-aligned lattice so generated builds can be
opened in downstream LEGO CAD tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.brick.constants import LDRAW_IDS
from backend.brick.parser import Brick

_COLORS_JSON = Path(__file__).resolve().parents[2] / "data" / "cache" / "colors.json"
_IDENTITY = "1 0 0 0 1 0 0 0 1"
_ROTATE_Y_90 = "0 0 -1 0 1 0 1 0 0"
_STUD_LDU = 20
_BRICK_HEIGHT_LDU = 24


def _load_ldraw_color_codes() -> dict[str, int]:
    with _COLORS_JSON.open(encoding="utf-8") as fh:
        raw: dict[str, dict] = json.load(fh)

    palette: dict[str, int] = {}
    for code, entry in raw.items():
        if entry.get("is_trans", True):
            continue
        rgb = entry.get("rgb", "")
        if len(rgb) == 6:
            palette[rgb.upper()] = int(code)
    return palette


def _color_code(color: str) -> int:
    try:
        palette = _load_ldraw_color_codes()
    except OSError:
        return 16
    return palette.get(color.upper(), 16)


def _normalized_dims(brick: Brick) -> tuple[tuple[int, int], str]:
    dims = (min(brick.h, brick.w), max(brick.h, brick.w))
    matrix = _IDENTITY if brick.h <= brick.w else _ROTATE_Y_90
    return dims, matrix


def _ldr_line(brick: Brick) -> str:
    dims, matrix = _normalized_dims(brick)
    part_id = LDRAW_IDS[dims]
    color_code = _color_code(brick.color)
    x = int((brick.x + brick.h / 2.0) * _STUD_LDU)
    y = int(-(brick.z * _BRICK_HEIGHT_LDU))
    z = int((brick.y + brick.w / 2.0) * _STUD_LDU)
    return f"1 {color_code} {x} {y} {z} {matrix} {part_id}.dat"


def _safe_model_name(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", title.strip()).strip("-")
    return cleaned or "legogen-build"


def export_ldr(bricks: list[Brick], title: str = "LEGOGen Build") -> str:
    model_name = _safe_model_name(title)
    header = [
        f"0 FILE {model_name}.ldr",
        f"0 Name: {model_name}.ldr",
        "0 Author: LEGOGen",
        "0 !LDRAW_ORG Model",
        "0 BFC CERTIFY CCW",
    ]
    return "\n".join(header + [_ldr_line(brick) for brick in bricks]) + "\n"
