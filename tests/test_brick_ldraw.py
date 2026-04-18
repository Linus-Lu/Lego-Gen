import backend.brick.ldraw as ldraw
from backend.brick.ldraw import export_ldr
from backend.brick.parser import Brick


def test_export_ldr_emits_header_and_part_lines(tmp_path, monkeypatch):
    colors = tmp_path / "colors.json"
    colors.write_text('{"4":{"rgb":"C91A09","is_trans":false}}', encoding="utf-8")
    monkeypatch.setattr(ldraw, "_COLORS_JSON", colors)

    out = export_ldr(
        [Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")],
        title="Test Build",
    )
    assert "0 FILE Test-Build.ldr" in out
    assert "0 Author: LEGOGen" in out
    assert "1 4 20 0 40 1 0 0 0 1 0 0 0 1 3001.dat" in out


def test_export_ldr_rotates_transposed_parts():
    out = export_ldr(
        [Brick(h=4, w=1, x=0, y=0, z=0, color="0055BF")],
        title="Rotated",
    )
    assert "3010.dat" in out
    assert "0 0 -1 0 1 0 1 0 0" in out


def test_export_ldr_falls_back_to_main_color_for_unknown_hex():
    out = export_ldr(
        [Brick(h=1, w=1, x=0, y=0, z=0, color="ABCDEF")],
        title="Unknown Color",
    )
    assert "1 16" in out


def test_ldraw_color_loader_skips_transparent_and_invalid_entries(tmp_path, monkeypatch):
    colors = tmp_path / "colors.json"
    colors.write_text(
        '{"4":{"rgb":"C91A09","is_trans":false},'
        '"15":{"rgb":"FFFFFF","is_trans":true},'
        '"99":{"rgb":"BAD","is_trans":false}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(ldraw, "_COLORS_JSON", colors)

    assert ldraw._load_ldraw_color_codes() == {"C91A09": 4}


def test_export_ldr_uses_main_color_when_palette_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ldraw, "_COLORS_JSON", tmp_path / "missing.json")

    out = export_ldr(
        [Brick(h=1, w=1, x=0, y=0, z=0, color="C91A09")],
        title="Missing Palette",
    )

    assert "1 16" in out


def test_export_ldr_uses_main_color_when_palette_file_is_malformed(tmp_path, monkeypatch):
    colors = tmp_path / "colors.json"
    colors.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(ldraw, "_COLORS_JSON", colors)

    out = export_ldr(
        [Brick(h=1, w=1, x=0, y=0, z=0, color="C91A09")],
        title="Bad Palette",
    )

    assert "1 16" in out
