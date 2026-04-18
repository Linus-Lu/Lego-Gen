import pytest
from backend.data_pipeline.prepare_brick_dataset import (
    parse_st2b_bricks,
    pick_color_for_brick,
    colorize_structure,
    format_training_example,
    _extract_caption_color,
    _extract_caption_colors,
    _component_color,
    _first_non_black,
    _runtime_palette,
    CATEGORY_PALETTES,
    DEFAULT_PALETTE,
    audit_palette,
    build_arg_parser,
    canary_training_examples,
    _write_jsonl_example,
)
from backend.brick.parser import Brick


def test_parse_st2b_bricks():
    raw = "1x1 (15,17,0)\n1x4 (15,13,0)\n"
    bricks = parse_st2b_bricks(raw)
    assert len(bricks) == 2
    assert bricks[0] == (1, 1, 15, 17, 0)


def test_parse_st2b_bricks_empty():
    assert parse_st2b_bricks("") == []


def test_pick_color_returns_valid_hex():
    color = pick_color_for_brick(caption="A red car", category="car", z=0, seed=42)
    assert len(color) == 6


def test_colorize_structure():
    raw = [(2, 4, 5, 3, 0), (1, 2, 3, 7, 1)]
    colored = colorize_structure(raw, caption="A blue table", category="table", seed=0)
    assert len(colored) == 2
    assert all(isinstance(b, Brick) for b in colored)


def test_format_training_example():
    bricks = [
        Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09"),
        Brick(h=1, w=2, x=3, y=7, z=1, color="05131D"),
    ]
    example = format_training_example("A red chair", bricks)
    assert example["messages"][0]["role"] == "system"
    assert "A red chair" in example["messages"][1]["content"]
    assert "2x4 (5,3,0) #C91A09" in example["messages"][2]["content"]
    assert example["messages"][2]["content"].endswith("\nDONE")


def test_parse_st2b_bricks_skips_blank_lines():
    """Blank/whitespace-only lines must be silently dropped."""
    raw = "\n\n   \n1x4 (15,13,0)\n\n"
    bricks = parse_st2b_bricks(raw)
    assert bricks == [(1, 4, 15, 13, 0)]


def test_parse_st2b_bricks_skips_unparseable_lines():
    """Lines that don't match the ST2B regex are dropped, valid ones kept."""
    raw = "notabrick line\n2x4 (0,0,0)\ngarbage\n"
    bricks = parse_st2b_bricks(raw)
    assert bricks == [(2, 4, 0, 0, 0)]


def test_extract_caption_color_returns_none_when_no_color_word():
    """Fall through the COLOR_WORDS loop returns None."""
    assert _extract_caption_color("a mysterious shape") is None
    assert _extract_caption_color("") is None


def test_extract_caption_colors_uses_word_boundaries_and_order():
    assert _extract_caption_colors("a blueprint for a red and white house") == [
        "C91A09",
        "FFFFFF",
    ]
    assert _extract_caption_color("a redwood colored model") is None
    assert _extract_caption_colors("red red blue") == ["C91A09", "0055BF"]


def test_runtime_palette_returns_none_when_palette_file_missing(monkeypatch):
    from backend.brick import constants as const

    monkeypatch.delattr(const._lazy_palette, "_cache", raising=False)
    monkeypatch.setattr(const, "_COLORS_JSON", "/definitely/not/colors.json")

    assert _runtime_palette() is None


def test_runtime_palette_reads_seeded_palette(seeded_palette):
    assert _runtime_palette() == set(seeded_palette)


def test_component_color_detects_component_before_color():
    assert _component_color("a tower with a roof red", ("roof",)) == "C91A09"
    assert _component_color("a plain tower", ("roof",)) is None


def test_first_non_black_falls_back_when_all_black():
    assert _first_non_black(["05131D"]) == "05131D"


def test_multicolor_house_caption_assigns_component_colors():
    raw = [
        (2, 4, 0, 0, 0),
        (2, 4, 0, 0, 1),
        (2, 4, 0, 0, 2),
    ]

    bricks = colorize_structure(
        raw,
        caption="a red house with white walls and yellow roof",
        category="birdhouse",
        seed=0,
    )

    assert [brick.color for brick in bricks] == ["C91A09", "FFFFFF", "F2CD37"]
    assert len({brick.color for brick in bricks}) == 3


def test_vehicle_caption_assigns_wheels_windows_and_body():
    assert pick_color_for_brick(
        "a blue car with black wheels and white windows",
        category="car",
        z=0,
        seed=0,
        max_z=2,
        brick_index=0,
        h=1,
        w=1,
    ) == "05131D"
    assert pick_color_for_brick(
        "a blue car with black wheels and white windows",
        category="car",
        z=2,
        seed=0,
        max_z=2,
        brick_index=1,
    ) == "FFFFFF"
    assert pick_color_for_brick(
        "a blue car with black wheels and white windows",
        category="car",
        z=1,
        seed=0,
        max_z=2,
        brick_index=1,
    ) == "0055BF"


def test_tree_tower_single_and_generic_multicolor_paths():
    assert pick_color_for_brick("a green tree with dark gray trunk", "tree", 0, 0, max_z=2) == "6D6E5C"
    assert pick_color_for_brick("a green tree with dark gray trunk", "tree", 1, 0, max_z=2) == "237841"
    assert pick_color_for_brick("a gray tower with a red top", "tower", 2, 0, max_z=2) == "C91A09"
    assert pick_color_for_brick("a red and blue sculpture", "unknown", 1, 0, max_z=3, brick_index=2) == "0055BF"
    assert pick_color_for_brick("a red chair", "chair", 0, 1) in CATEGORY_PALETTES["chair"]


def test_pick_color_falls_back_to_category_palette_at_z0():
    """No caption color + z=0 → use category palette with dark-weighting."""
    caption = "a mysterious shape"  # no color words
    color = pick_color_for_brick(caption, category="car", z=0, seed=0)
    assert color in CATEGORY_PALETTES["car"]


def test_pick_color_uses_flat_weights_above_ground():
    """No caption color + z>0 → flat weights (no dark boost)."""
    caption = "a mysterious shape"
    color = pick_color_for_brick(caption, category="car", z=1, seed=0)
    assert color in CATEGORY_PALETTES["car"]


def test_pick_color_falls_back_to_default_palette_for_unknown_category():
    """Unknown category triggers the DEFAULT_PALETTE fallback."""
    caption = "a mysterious shape"
    color = pick_color_for_brick(caption, category="not_a_category", z=0, seed=0)
    assert color in DEFAULT_PALETTE


def test_palette_audit_passes_when_runtime_palette_contains_training_colors(monkeypatch):
    import backend.data_pipeline.prepare_brick_dataset as prep

    allowed = set(DEFAULT_PALETTE)
    monkeypatch.setattr(prep, "_runtime_palette", lambda: allowed)

    assert audit_palette("strict") == []


def test_palette_audit_off_skips_runtime_palette(monkeypatch):
    import backend.data_pipeline.prepare_brick_dataset as prep

    monkeypatch.setattr(prep, "_runtime_palette", lambda: (_ for _ in ()).throw(AssertionError("should not load")))

    assert audit_palette("off") == []


def test_palette_audit_strict_requires_available_palette(monkeypatch):
    import backend.data_pipeline.prepare_brick_dataset as prep

    monkeypatch.setattr(prep, "_runtime_palette", lambda: None)

    with pytest.raises(RuntimeError, match="runtime palette unavailable"):
        audit_palette("strict")


def test_palette_audit_warns_when_runtime_palette_unavailable(monkeypatch, capsys):
    import backend.data_pipeline.prepare_brick_dataset as prep

    monkeypatch.setattr(prep, "_runtime_palette", lambda: None)

    assert audit_palette("warn") == []
    assert "runtime palette unavailable" in capsys.readouterr().out


def test_palette_audit_warns_for_missing_runtime_colors(monkeypatch):
    import backend.data_pipeline.prepare_brick_dataset as prep

    monkeypatch.setattr(prep, "_runtime_palette", lambda: {"C91A09"})

    missing = audit_palette("warn")

    assert "F2CD37" in missing


def test_palette_audit_strict_rejects_missing_runtime_colors(monkeypatch):
    import backend.data_pipeline.prepare_brick_dataset as prep

    monkeypatch.setattr(prep, "_runtime_palette", lambda: {"C91A09"})

    with pytest.raises(RuntimeError, match="training colors not present"):
        audit_palette("strict")


def test_prepare_dataset_arg_parser_v2_options(tmp_path):
    args = build_arg_parser().parse_args([
        "--output-dir",
        str(tmp_path),
        "--include-canary",
        "--canary-repeat",
        "20",
        "--palette-audit",
        "strict",
    ])

    assert args.output_dir == tmp_path
    assert args.include_canary is True
    assert args.canary_repeat == 20
    assert args.palette_audit == "strict"


def test_write_jsonl_example_writes_one_line(tmp_path):
    path = tmp_path / "examples.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        _write_jsonl_example(fh, {"messages": []})

    assert path.read_text(encoding="utf-8") == '{"messages": []}\n'


def test_canary_training_examples_are_done_terminated_and_repeated():
    examples = canary_training_examples(repeat=2)

    assert len(examples) == 8
    assert all(example["messages"][-1]["content"].endswith("\nDONE") for example in examples)
