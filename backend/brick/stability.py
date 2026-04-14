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

    Uses an incremental approach: maintains an adjacency list and a set of
    grounded brick indices as bricks are added one at a time. When a new
    brick is added, only its edges to existing bricks are computed, and a
    BFS propagates ground-reachability through any newly connected bricks.

    This is O(n²) overall instead of the naive O(n³) prefix-check approach.

    Returns the index of the first brick whose addition causes instability,
    or -1 if the full structure is stable.
    """
    if not bricks:
        return -1

    adj: list[list[int]] = []
    grounded: set[int] = set()

    for i, brick in enumerate(bricks):
        # Add adjacency entries for the new brick
        adj.append([])
        for j in range(i):
            if abs(bricks[j].z - brick.z) == 1 and _overlaps_xy(bricks[j], brick):
                adj[i].append(j)
                adj[j].append(i)

        # Check if the new brick is grounded (directly or via neighbors)
        newly_grounded = False
        if brick.z == 0:
            newly_grounded = True
        else:
            for j in adj[i]:
                if j in grounded:
                    newly_grounded = True
                    break

        if not newly_grounded:
            return i

        # BFS: the new brick is grounded, propagate to any previously
        # ungrounded neighbors that can now reach ground through it.
        grounded.add(i)
        queue: deque[int] = deque([i])
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in grounded:
                    grounded.add(neighbor)
                    queue.append(neighbor)

    return -1
