"""Per-brick reliability scoring with incremental stability tracking.

Inspired by BrickGPT's approach of checking structural reliability at every
generation step rather than only after the full sequence is produced.  Uses
lightweight heuristics (connectivity, support ratio, neighbour count) instead
of a full physics solver so it can run in the inner generation loop without
significant overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.brick.parser import Brick
from backend.brick.stability import _overlaps_xy, is_brick_connected
from backend.brick.occupancy import VoxelGrid
from backend.config import (
    RELIABILITY_WEIGHT_CONNECTIVITY,
    RELIABILITY_WEIGHT_SUPPORT,
    RELIABILITY_WEIGHT_NEIGHBORS,
)


@dataclass(frozen=True, slots=True)
class BrickScore:
    """Reliability scores for a single brick.

    Attributes:
        connectivity: 1.0 if the brick can reach the ground through the
            adjacency graph, 0.0 otherwise.
        support_ratio: Fraction of the brick's bottom footprint (z-1) that
            rests on existing bricks or the ground plane (0.0-1.0).
        neighbor_count: Number of existing bricks that are vertically adjacent
            (|dz|==1) and overlap in XY.
        score: Weighted aggregate reliability score (0.0-1.0, higher is better).
    """

    connectivity: float
    support_ratio: float
    neighbor_count: int
    score: float


def _compute_support_ratio(brick: Brick, grid: VoxelGrid) -> float:
    """Return the fraction of the brick's footprint supported from below.

    A brick at z=0 is fully supported by the ground plane (ratio 1.0).
    For z>0, count how many cells in the XY footprint at z-1 are occupied.
    """
    if brick.z == 0:
        return 1.0
    footprint_size = brick.h * brick.w
    below_z = brick.z - 1
    occupied = int(
        grid.grid[brick.x : brick.x + brick.h, brick.y : brick.y + brick.w, below_z].sum()
    )
    return occupied / footprint_size


def _count_neighbors(brick: Brick, bricks: list[Brick]) -> int:
    """Count existing bricks that are vertically adjacent and overlap in XY."""
    count = 0
    for other in bricks:
        if abs(brick.z - other.z) == 1 and _overlaps_xy(brick, other):
            count += 1
    return count


def _weighted_score(connectivity: float, support_ratio: float, neighbor_count: int) -> float:
    """Compute the weighted aggregate reliability score."""
    neighbor_norm = min(neighbor_count / 2.0, 1.0)
    return (
        RELIABILITY_WEIGHT_CONNECTIVITY * connectivity
        + RELIABILITY_WEIGHT_SUPPORT * support_ratio
        + RELIABILITY_WEIGHT_NEIGHBORS * neighbor_norm
    )


class ReliabilityScorer:
    """Incremental per-brick reliability scorer.

    Maintains internal state (brick list, ground-reachable set) so that each
    call to :meth:`add_brick` runs in O(n) rather than rebuilding the full
    adjacency graph from scratch.
    """

    def __init__(self, grid: VoxelGrid | None = None) -> None:
        self._bricks: list[Brick] = []
        self._scores: list[BrickScore] = []
        # Indices of bricks reachable from the ground.
        self._ground_set: set[int] = set()
        self._grid = grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_brick(self, brick: Brick) -> BrickScore:
        """Score *brick*, add it to internal state, and return its score.

        The brick **must** already have been placed in the shared
        :class:`VoxelGrid` (if one was provided) before calling this method,
        because :func:`_compute_support_ratio` reads the grid.  However the
        grid should reflect the state *before* this brick is placed — i.e.
        call ``add_brick`` first, then ``grid.place``.
        """
        connectivity = 1.0 if is_brick_connected(
            brick, self._bricks, self._ground_set
        ) else 0.0

        support_ratio = (
            _compute_support_ratio(brick, self._grid) if self._grid is not None else 0.0
        )

        neighbor_count = _count_neighbors(brick, self._bricks)

        score = _weighted_score(connectivity, support_ratio, neighbor_count)
        brick_score = BrickScore(
            connectivity=connectivity,
            support_ratio=support_ratio,
            neighbor_count=neighbor_count,
            score=score,
        )

        idx = len(self._bricks)
        self._bricks.append(brick)
        self._scores.append(brick_score)
        if connectivity == 1.0:
            self._ground_set.add(idx)
            # A newly grounded brick may connect previously floating bricks.
            self._propagate_ground(idx)

        return brick_score

    def remove_last(self, count: int = 1) -> None:
        """Remove the last *count* bricks and their scores (rollback support)."""
        for _ in range(min(count, len(self._bricks))):
            idx = len(self._bricks) - 1
            self._bricks.pop()
            self._scores.pop()
            self._ground_set.discard(idx)
        # After removal the ground set may be stale — rebuild it.
        if count > 0:
            self._rebuild_ground_set()

    def reset(self) -> None:
        """Clear all state."""
        self._bricks.clear()
        self._scores.clear()
        self._ground_set.clear()

    @property
    def scores(self) -> list[BrickScore]:
        """Return a copy of the per-brick score list."""
        return list(self._scores)

    def aggregate_score(self) -> float:
        """Return the mean reliability score across all bricks, or 1.0 if empty."""
        if not self._scores:
            return 1.0
        return sum(s.score for s in self._scores) / len(self._scores)

    def min_score(self) -> float:
        """Return the lowest per-brick reliability score, or 1.0 if empty."""
        if not self._scores:
            return 1.0
        return min(s.score for s in self._scores)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate_ground(self, new_idx: int) -> None:
        """BFS from *new_idx* to mark any adjacent bricks as ground-reachable."""
        queue = [new_idx]
        while queue:
            current = queue.pop()
            current_brick = self._bricks[current]
            for idx, other in enumerate(self._bricks):
                if idx in self._ground_set:
                    continue
                if abs(current_brick.z - other.z) == 1 and _overlaps_xy(current_brick, other):
                    self._ground_set.add(idx)
                    queue.append(idx)

    def _rebuild_ground_set(self) -> None:
        """Rebuild the ground-reachable set from scratch after a rollback."""
        from collections import deque

        self._ground_set.clear()
        n = len(self._bricks)
        if n == 0:
            return

        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self._bricks[i].z - self._bricks[j].z) == 1 and _overlaps_xy(
                    self._bricks[i], self._bricks[j]
                ):
                    adj[i].append(j)
                    adj[j].append(i)

        queue: deque[int] = deque()
        for i, b in enumerate(self._bricks):
            if b.z == 0:
                self._ground_set.add(i)
                queue.append(i)

        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in self._ground_set:
                    self._ground_set.add(neighbor)
                    queue.append(neighbor)
