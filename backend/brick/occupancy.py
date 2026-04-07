"""Voxel occupancy grid for brick collision detection."""

import numpy as np

from backend.brick.constants import WORLD_DIM, BRICK_SHAPES
from backend.brick.parser import Brick


class VoxelGrid:
    """A WORLD_DIM x WORLD_DIM x WORLD_DIM boolean voxel grid.

    Each cell records whether that unit cube is occupied by a brick.
    Axes: grid[x, y, z] — x spans brick height (h), y spans brick width (w),
    z spans the vertical stacking direction.
    """

    def __init__(self) -> None:
        self.grid = np.zeros((WORLD_DIM, WORLD_DIM, WORLD_DIM), dtype=np.bool_)

    # ------------------------------------------------------------------
    # Low-level queries
    # ------------------------------------------------------------------

    def is_empty(self, x: int, y: int, z: int) -> bool:
        """Return True when the voxel at (x, y, z) is unoccupied."""
        return not self.grid[x, y, z]

    # ------------------------------------------------------------------
    # Placement validation
    # ------------------------------------------------------------------

    def can_place(self, brick: Brick) -> bool:
        """Return True when *brick* can be placed without violating any rule.

        Rules checked:
        1. (h, w) must be a recognised shape in BRICK_SHAPES.
        2. All coordinates must be >= 0.
        3. x + h <= WORLD_DIM  (fits in x dimension)
        4. y + w <= WORLD_DIM  (fits in y dimension)
        5. z < WORLD_DIM       (fits in z dimension)
        6. No voxel in the footprint is already occupied.
        """
        if (brick.h, brick.w) not in BRICK_SHAPES:
            return False
        if brick.x < 0 or brick.y < 0 or brick.z < 0:
            return False
        if brick.x + brick.h > WORLD_DIM:
            return False
        if brick.y + brick.w > WORLD_DIM:
            return False
        if brick.z >= WORLD_DIM:
            return False
        # Collision check: any occupied voxel in footprint?
        if self.grid[brick.x:brick.x + brick.h, brick.y:brick.y + brick.w, brick.z].any():
            return False
        return True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def place(self, brick: Brick) -> None:
        """Mark the footprint of *brick* as occupied.

        Does not check validity — call :meth:`can_place` first if needed.
        """
        self.grid[brick.x:brick.x + brick.h, brick.y:brick.y + brick.w, brick.z] = True

    def remove(self, brick: Brick) -> None:
        """Clear the footprint of *brick*, marking those voxels as free."""
        self.grid[brick.x:brick.x + brick.h, brick.y:brick.y + brick.w, brick.z] = False

    def clear(self) -> None:
        """Reset every voxel to unoccupied."""
        self.grid[:] = False
