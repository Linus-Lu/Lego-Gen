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


def _lazy_palette() -> dict[str, str]:
    """Load the palette on first access; cache it for subsequent calls.

    Kept lazy so that importing this module for ``BRICK_SHAPES`` / ``WORLD_DIM``
    (the inference path) doesn't crash when ``data/cache/colors.json`` is
    absent. Consumers of the palette (decoder.py) fail loudly at call time.
    """
    cached = getattr(_lazy_palette, "_cache", None)
    if cached is None:
        cached = _load_color_palette(_COLORS_JSON)
        _lazy_palette._cache = cached  # type: ignore[attr-defined]
    return cached


class _LazyMapping:
    """dict-like facade that defers loading until first access."""

    def _load(self) -> dict[str, str]:
        return _lazy_palette()

    def __contains__(self, key: object) -> bool:
        return key in self._load()

    def __getitem__(self, key: str) -> str:
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def get(self, key: str, default=None):
        return self._load().get(key, default)

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()


class _LazyColors:
    """list-like facade that defers palette load until first access."""

    def _load(self) -> list[str]:
        return sorted(_lazy_palette().keys())

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, idx):
        return self._load()[idx]

    def __contains__(self, item: object) -> bool:
        return item in self._load()


COLOR_PALETTE: dict[str, str] = _LazyMapping()  # type: ignore[assignment]
ALLOWED_COLORS: list[str] = _LazyColors()  # type: ignore[assignment]
