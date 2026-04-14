"""Connectivity-based stability checker for LEGO brick structures.

A brick is considered stable if it can reach the ground (z=0) through a chain
of vertically adjacent bricks that have overlapping X-Y footprints.
"""

from collections import deque

from backend.brick.parser import Brick


def _overlaps_xy(a: Brick, b: Brick) -> bool:
    """Check if two bricks overlap in the X-Y plane."""
    return (a.x < b.x + b.h and a.x + a.h > b.x and
            a.y < b.y + b.w and a.y + a.w > b.y)


def is_stable(bricks: list[Brick]) -> bool:
    """Return True if all bricks are connected to the ground via adjacency graph.

    Two bricks are adjacent when their z-values differ by exactly 1 and their
    X-Y footprints overlap. A brick at z=0 is considered grounded. Every brick
    must be reachable from a grounded brick to make the structure stable.
    """
    if not bricks:
        return True

    n = len(bricks)

    # Build adjacency list: edges between bricks that are vertically adjacent
    # (|z difference| == 1) and overlap in XY.
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(bricks[i].z - bricks[j].z) == 1 and _overlaps_xy(bricks[i], bricks[j]):
                adj[i].append(j)
                adj[j].append(i)

    # BFS from all ground-level (z=0) bricks.
    visited = [False] * n
    queue: deque[int] = deque()

    for i, brick in enumerate(bricks):
        if brick.z == 0:
            visited[i] = True
            queue.append(i)

    while queue:
        current = queue.popleft()
        for neighbor in adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return all(visited)


def find_first_unstable(bricks: list[Brick]) -> int:
    """Return the index of the first brick that makes the structure unstable.

    Checks prefixes of increasing length: [bricks[0]], [bricks[0], bricks[1]],
    etc. Returns the index of the first brick whose addition causes instability,
    or -1 if the full structure is stable.
    """
    for i in range(1, len(bricks) + 1):
        if not is_stable(bricks[:i]):
            return i - 1
    return -1


def is_brick_connected(
    brick: Brick, existing_bricks: list[Brick], ground_set: set[int]
) -> bool:
    """Check whether *brick* connects to the ground-reachable set.

    A brick is connected when:
    - it sits at z=0 (on the ground), **or**
    - at least one brick in *existing_bricks* whose index is in *ground_set*
      is vertically adjacent (|z diff| == 1) and overlaps in XY.

    Runs in O(n) where n = len(existing_bricks), much faster than rebuilding
    the full adjacency graph with :func:`is_stable`.
    """
    if brick.z == 0:
        return True
    for idx, other in enumerate(existing_bricks):
        if idx not in ground_set:
            continue
        if abs(brick.z - other.z) == 1 and _overlaps_xy(brick, other):
            return True
    return False
