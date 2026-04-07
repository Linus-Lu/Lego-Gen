import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_pipeline.add_grid_pos import parse_brick_width, parse_brick_dims, compute_grid_positions


# ---------------------------------------------------------------------------
# parse_brick_width
# ---------------------------------------------------------------------------

def test_parse_brick_width():
    assert parse_brick_width("Brick 2x2") == 2
    assert parse_brick_width("Plate 1x4") == 1
    assert parse_brick_width("Something weird") == 2  # default


# ---------------------------------------------------------------------------
# compute_grid_positions
# ---------------------------------------------------------------------------

def test_single_part_gets_origin():
    layer = {
        "name": "layer_0",
        "parts": [
            {"part_id": "3003", "name": "Brick 2x2", "color": "Red", "quantity": 1}
        ],
    }
    result = compute_grid_positions(layer)
    assert result["parts"][0]["grid_pos"] == [0, 0]


def test_multiple_parts_pack_left_to_right():
    layer = {
        "name": "layer_0",
        "parts": [
            {"part_id": "3003", "name": "Brick 2x2", "color": "Red", "quantity": 1},
            {"part_id": "3004", "name": "Brick 1x2", "color": "Blue", "quantity": 1},
        ],
    }
    result = compute_grid_positions(layer)
    first_pos = result["parts"][0]["grid_pos"]
    second_pos = result["parts"][1]["grid_pos"]
    # The second part must be offset from the first
    assert first_pos != second_pos
    # First part is at origin
    assert first_pos == [0, 0]
    # Second part starts at cursor_x = width_of_first * quantity_of_first = 2*1 = 2
    # layer_width = max(4, ceil(sqrt(2*1 + 1*1))) = max(4, ceil(sqrt(3))) = 4
    # second grid_pos = [2 % 4, 2 // 4] = [2, 0]
    assert second_pos == [2, 0]


def test_preserves_existing_fields():
    layer = {
        "name": "layer_0",
        "type": "Bricks",
        "parts": [
            {
                "part_id": "3003",
                "name": "Brick 2x2",
                "category": "Bricks",
                "color": "Black",
                "color_hex": "#05131D",
                "is_trans": False,
                "quantity": 3,
            }
        ],
        "spatial": {"position": "bottom"},
    }
    result = compute_grid_positions(layer)
    part = result["parts"][0]
    # All original fields must still be present
    assert part["part_id"] == "3003"
    assert part["name"] == "Brick 2x2"
    assert part["category"] == "Bricks"
    assert part["color"] == "Black"
    assert part["color_hex"] == "#05131D"
    assert part["is_trans"] is False
    assert part["quantity"] == 3
    # grid_pos must have been added
    assert "grid_pos" in part
    # Non-parts fields on layer must be preserved
    assert result["name"] == "layer_0"
    assert result["type"] == "Bricks"
    assert result["spatial"] == {"position": "bottom"}


def test_empty_parts_list():
    layer = {"name": "layer_0", "parts": []}
    result = compute_grid_positions(layer)
    assert result["parts"] == []


def test_parse_brick_dims():
    assert parse_brick_dims("Brick 2x4") == (2, 4)
    assert parse_brick_dims("Plate 1x6") == (1, 6)
    assert parse_brick_dims("Something weird") == (2, 2)  # default


def test_depth_aware_wrapping():
    """When bricks wrap to the next row, z advances by the max depth of the
    completed row — not by 1 — to prevent z-overlap."""
    layer = {
        "name": "layer_0",
        "parts": [
            {"part_id": "3001", "name": "Brick 2x4", "color": "Red", "quantity": 3},
            {"part_id": "3002", "name": "Brick 2x2", "color": "Blue", "quantity": 1},
        ],
    }
    result = compute_grid_positions(layer)
    first_pos = result["parts"][0]["grid_pos"]
    second_pos = result["parts"][1]["grid_pos"]
    assert first_pos == [0, 0]
    # 2x4 bricks: total_width = 2*3 = 6, target_width = max(4, ceil(sqrt(8)*1.2)) = 4
    # 6 > 4, so first part placed at [0,0], then cursor_x=6
    # second part: 6+2=8 > 4 and cursor_x>0 => wrap. cursor_z += 4 (depth of first brick)
    # second part placed at [0, 4]
    assert second_pos == [0, 4]
