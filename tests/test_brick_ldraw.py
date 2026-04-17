from backend.brick.ldraw import export_ldr
from backend.brick.parser import Brick


def test_export_ldr_emits_header_and_part_lines():
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
