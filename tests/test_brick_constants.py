"""Smoke coverage for backend/brick/constants.py.

The module is mostly data, but the lazy palette facades have custom dunder
methods that must behave like the dict / list they model."""

import pytest

from backend.brick import constants as const
from backend.brick.constants import (
    ALLOWED_DIMS,
    BRICK_SHAPES,
    LDRAW_IDS,
    WORLD_DIM,
)


def test_world_dim_positive():
    assert WORLD_DIM == 20


def test_allowed_dims_match_brick_shapes():
    shape_set = {f"{h}x{w}" for h, w in BRICK_SHAPES}
    assert set(ALLOWED_DIMS) == shape_set


def test_allowed_dims_sorted():
    assert ALLOWED_DIMS == sorted(ALLOWED_DIMS)


def test_ldraw_ids_cover_canonical_shapes():
    # LDRAW_IDS only stores h<=w form; spot-check a few.
    assert LDRAW_IDS[(2, 4)] == "3001"
    assert LDRAW_IDS[(1, 1)] == "3005"


def test_color_palette_lazy_mapping_behaves_like_dict(seeded_palette):
    from backend.brick.constants import COLOR_PALETTE
    assert "C91A09" in COLOR_PALETTE
    assert COLOR_PALETTE["C91A09"] == "Red"
    assert COLOR_PALETTE.get("C91A09") == "Red"
    assert COLOR_PALETTE.get("missing", "fallback") == "fallback"
    assert len(COLOR_PALETTE) == len(seeded_palette)
    assert set(COLOR_PALETTE.keys()) == set(seeded_palette.keys())
    assert set(COLOR_PALETTE.values()) == set(seeded_palette.values())
    assert dict(COLOR_PALETTE.items()) == seeded_palette
    # Iteration works.
    keys = list(COLOR_PALETTE)
    assert set(keys) == set(seeded_palette)


def test_allowed_colors_list_facade(seeded_palette):
    from backend.brick.constants import ALLOWED_COLORS
    assert len(ALLOWED_COLORS) == len(seeded_palette)
    assert "C91A09" in ALLOWED_COLORS
    # Sorted for deterministic grammar regex generation.
    assert list(ALLOWED_COLORS) == sorted(seeded_palette.keys())
    # Indexing works.
    first = ALLOWED_COLORS[0]
    assert first == sorted(seeded_palette.keys())[0]


def test_lazy_palette_caches(seeded_palette):
    """Second call must not reload — cache is stored on the function itself."""
    from backend.brick.constants import _lazy_palette
    a = _lazy_palette()
    b = _lazy_palette()
    assert a is b


def test_load_color_palette_reads_json(tmp_path):
    """Exercise the real file-read path with a synthetic colors.json."""
    import json
    colors_file = tmp_path / "colors.json"
    colors_file.write_text(json.dumps({
        "1": {"rgb": "c91a09", "name": "Red", "is_trans": False},
        "2": {"rgb": "0055BF", "name": "Blue", "is_trans": False},
        # Transparent entries are filtered.
        "3": {"rgb": "FFFFFF", "name": "Glass", "is_trans": True},
        # Unknown-prefixed names are filtered.
        "4": {"rgb": "ABCDEF", "name": "[Unknown] foo", "is_trans": False},
        # Missing/short rgb is filtered.
        "5": {"rgb": "ab", "name": "Short", "is_trans": False},
    }))
    palette = const._load_color_palette(str(colors_file))
    assert palette == {"C91A09": "Red", "0055BF": "Blue"}


def test_lazy_palette_cold_load_reads_json(tmp_path, monkeypatch):
    """Cold cache → _lazy_palette calls _load_color_palette(_COLORS_JSON)."""
    import json
    colors_file = tmp_path / "colors.json"
    colors_file.write_text(json.dumps({
        "1": {"rgb": "C91A09", "name": "Red", "is_trans": False},
    }))
    monkeypatch.setattr(const, "_COLORS_JSON", str(colors_file))
    # Clear any existing cache so the cold-load path runs.
    if hasattr(const._lazy_palette, "_cache"):
        monkeypatch.delattr(const._lazy_palette, "_cache")
    result = const._lazy_palette()
    assert result == {"C91A09": "Red"}
