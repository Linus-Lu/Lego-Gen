"""Brick library, color palette, and grid constants."""

import json
import os

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
WORLD_DIM = 20

# ---------------------------------------------------------------------------
# Brick shapes
# ---------------------------------------------------------------------------
BRICK_SHAPES: set[tuple[int, int]] = {
    (1, 1), (1, 2), (2, 1), (2, 2),
    (1, 4), (4, 1), (2, 4), (4, 2),
    (1, 6), (6, 1), (2, 6), (6, 2),
    (1, 8), (8, 1),
}

ALLOWED_DIMS: list[str] = sorted(f"{h}x{w}" for h, w in BRICK_SHAPES)

# ---------------------------------------------------------------------------
# LDraw part IDs (keyed by normalized (h,w) with h <= w)
# ---------------------------------------------------------------------------
LDRAW_IDS: dict[tuple[int, int], str] = {
    (1, 1): "3005",
    (1, 2): "3004",
    (2, 2): "3003",
    (1, 4): "3010",
    (2, 4): "3001",
    (1, 6): "3009",
    (2, 6): "2456",
    (1, 8): "3008",
}

# ---------------------------------------------------------------------------
# Color palette loaded from data/cache/colors.json
# ---------------------------------------------------------------------------
_COLORS_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cache", "colors.json"
)

def _load_color_palette(path: str) -> dict[str, str]:
    """Return a dict mapping 6-char uppercase hex -> color name.

    Excludes transparent colors and entries whose name starts with '[Unknown'.
    """
    with open(path, encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    palette: dict[str, str] = {}
    for entry in raw.values():
        if entry.get("is_trans", True):
            continue
        name: str = entry.get("name", "")
        if name.startswith("[Unknown"):
            continue
        rgb: str = entry.get("rgb", "")
        if len(rgb) == 6:
            palette[rgb.upper()] = name
    return palette


try:
    COLOR_PALETTE: dict[str, str] = _load_color_palette(_COLORS_JSON)
except FileNotFoundError:
    COLOR_PALETTE = {}
ALLOWED_COLORS: list[str] = sorted(COLOR_PALETTE.keys())
