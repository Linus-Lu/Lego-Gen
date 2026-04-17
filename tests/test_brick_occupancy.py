import pytest
from backend.brick.parser import Brick
from backend.brick.occupancy import VoxelGrid

def test_empty_grid():
    g = VoxelGrid()
    assert g.is_empty(0, 0, 0)

def test_place_brick():
    g = VoxelGrid()
    b = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    assert g.can_place(b)
    g.place(b)
    assert not g.is_empty(0, 0, 0)
    assert not g.is_empty(1, 3, 0)
    assert g.is_empty(2, 0, 0)

def test_collision():
    g = VoxelGrid()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=2, x=1, y=1, z=0, color="05131D")
    g.place(b1)
    assert not g.can_place(b2)

def test_out_of_bounds():
    g = VoxelGrid()
    b = Brick(h=2, w=4, x=19, y=0, z=0, color="C91A09")  # x+h=21 > 20
    assert not g.can_place(b)

def test_stacking():
    g = VoxelGrid()
    b1 = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    g.place(b1)
    assert g.can_place(b2)
    g.place(b2)
    assert not g.is_empty(0, 0, 1)

def test_invalid_dims():
    g = VoxelGrid()
    b = Brick(h=3, w=3, x=0, y=0, z=0, color="C91A09")  # 3x3 not in library
    assert not g.can_place(b)


def test_can_place_rejects_negative_coords():
    """Any coordinate < 0 short-circuits can_place."""
    g = VoxelGrid()
    assert not g.can_place(Brick(h=2, w=2, x=-1, y=0, z=0, color="FFFFFF"))
    assert not g.can_place(Brick(h=2, w=2, x=0, y=-1, z=0, color="FFFFFF"))
    assert not g.can_place(Brick(h=2, w=2, x=0, y=0, z=-1, color="FFFFFF"))


def test_can_place_rejects_x_extending_out_of_world():
    """x + h > WORLD_DIM must return False."""
    g = VoxelGrid()
    # x=19, h=2 -> x+h=21 > 20
    assert not g.can_place(Brick(h=2, w=2, x=19, y=0, z=0, color="FFFFFF"))


def test_can_place_rejects_y_extending_out_of_world():
    """y + w > WORLD_DIM must return False."""
    g = VoxelGrid()
    # y=19, w=2 -> y+w=21 > 20
    assert not g.can_place(Brick(h=2, w=2, x=0, y=19, z=0, color="FFFFFF"))


def test_can_place_rejects_z_out_of_world():
    """z >= WORLD_DIM must return False."""
    g = VoxelGrid()
    assert not g.can_place(Brick(h=2, w=2, x=0, y=0, z=20, color="FFFFFF"))


def test_remove_brick_frees_voxels():
    """VoxelGrid.remove clears the previously-occupied footprint."""
    g = VoxelGrid()
    b = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    g.place(b)
    assert not g.is_empty(0, 0, 0)
    g.remove(b)
    assert g.is_empty(0, 0, 0)
    assert g.is_empty(1, 3, 0)


def test_clear_resets_entire_grid():
    """VoxelGrid.clear zeroes every voxel."""
    g = VoxelGrid()
    g.place(Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"))
    g.place(Brick(h=2, w=2, x=5, y=5, z=3, color="05131D"))
    assert g.grid.any()
    g.clear()
    assert not g.grid.any()
