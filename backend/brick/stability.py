"""Force-equilibrium stability checker for LEGO brick structures.

Every brick is one unit tall, so all contacts lie on the horizontal plane
between z and z+1. Stability reduces to a linear program: for each brick
(except those resting on the ground), net vertical force and net moment
about its center of mass must be zero under gravity.

Per-stud contact variables carry the normal force from the lower brick to
the upper brick, bounded below by ``-STUD_STRENGTH`` (tension pull-off
limit of one stud) and unbounded above (compression). Ground reaction is
modelled as one compressive-only contact per stud cell in each ground-level
brick's footprint.

Public API is unchanged — :func:`is_stable` and :func:`find_first_unstable`
preserve the signatures used by the inference pipeline.
"""

from collections import deque

import numpy as np
from scipy.optimize import linprog

from backend.brick.parser import Brick

# Per-stud tension limit (arbitrary force units).
STUD_STRENGTH = 1.0
# Weight per unit footprint area.
BRICK_DENSITY = 1.0
# LP tolerance slack for equality constraints — linprog uses HiGHS which is
# numerically tight, but we leave room for floating-point rounding.
_LP_METHOD = "highs"


def _overlaps_xy(a: Brick, b: Brick) -> bool:
    return (a.x < b.x + b.h and a.x + a.h > b.x and
            a.y < b.y + b.w and a.y + a.w > b.y)


def _stud_positions(a: Brick, b: Brick) -> list[tuple[float, float]]:
    """Integer stud centers (x+0.5, y+0.5) where a and b overlap in XY."""
    x_lo = max(a.x, b.x)
    x_hi = min(a.x + a.h, b.x + b.h)
    y_lo = max(a.y, b.y)
    y_hi = min(a.y + a.w, b.y + b.w)
    if x_hi <= x_lo or y_hi <= y_lo:
        return []
    return [
        (sx + 0.5, sy + 0.5)
        for sx in range(x_lo, x_hi)
        for sy in range(y_lo, y_hi)
    ]


def _ground_studs(b: Brick) -> list[tuple[float, float]]:
    """Ground reaction stud centers under b's full footprint."""
    return [
        (sx + 0.5, sy + 0.5)
        for sx in range(b.x, b.x + b.h)
        for sy in range(b.y, b.y + b.w)
    ]


def _brick_com(b: Brick) -> tuple[float, float]:
    return (b.x + b.h / 2.0, b.y + b.w / 2.0)


def _brick_weight(b: Brick) -> float:
    return BRICK_DENSITY * b.h * b.w


def _is_connected(bricks: list[Brick]) -> bool:
    """Every non-ground brick must be reachable from z=0 through overlapping layers."""
    n = len(bricks)
    if n == 0:
        return True

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(bricks[i].z - bricks[j].z) == 1 and _overlaps_xy(bricks[i], bricks[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    queue: deque[int] = deque()
    for i, b in enumerate(bricks):
        if b.z == 0:
            visited[i] = True
            queue.append(i)

    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    return all(visited)


def _solve_equilibrium(bricks: list[Brick]) -> bool:
    """Return True when the LP for static equilibrium is feasible."""
    n = len(bricks)
    if n == 0:
        return True

    # Cheap prefilter — any disconnected brick is unstable regardless of the LP.
    if not _is_connected(bricks):
        return False

    # Brick-brick contacts: one variable per stud position in each XY overlap
    # between a brick at z and a brick at z+1. Tension bound = -STUD_STRENGTH,
    # compression unbounded above.
    bb_contacts: list[tuple[int, int, float, float]] = []  # (lower, upper, cx, cy)
    for i in range(n):
        for j in range(n):
            if i == j or bricks[j].z != bricks[i].z + 1:
                continue
            for cx, cy in _stud_positions(bricks[i], bricks[j]):
                bb_contacts.append((i, j, cx, cy))

    # Ground contacts: one compressive-only variable per stud under each ground-level brick.
    g_contacts: list[tuple[int, float, float]] = []  # (brick_idx, cx, cy)
    for bi, b in enumerate(bricks):
        if b.z == 0:
            for cx, cy in _ground_studs(b):
                g_contacts.append((bi, cx, cy))

    num_bb = len(bb_contacts)
    num_g = len(g_contacts)
    num_vars = num_bb + num_g

    if num_vars == 0:  # pragma: no cover — unreachable: n>0 with no contacts implies no z=0 brick, so _is_connected would have returned False above.
        return False

    # Index bricks -> contacts that touch them, to avoid O(n*num_contacts) scans.
    bb_by_upper: list[list[int]] = [[] for _ in range(n)]
    bb_by_lower: list[list[int]] = [[] for _ in range(n)]
    for ci, (L, U, _, _) in enumerate(bb_contacts):
        bb_by_upper[U].append(ci)
        bb_by_lower[L].append(ci)
    g_by_brick: list[list[int]] = [[] for _ in range(n)]
    for gi, (gb, _, _) in enumerate(g_contacts):
        g_by_brick[gb].append(gi)

    # Equilibrium: for every brick, three equations — vertical force, moment-x,
    # moment-y — all evaluated at the brick's own center of mass. Three rows
    # per brick gives ``3n`` equality constraints.
    A_eq = np.zeros((3 * n, num_vars), dtype=np.float64)
    b_eq = np.zeros(3 * n, dtype=np.float64)

    for bi, b in enumerate(bricks):
        com_x, com_y = _brick_com(b)
        weight = _brick_weight(b)
        fz_row = 3 * bi
        mx_row = 3 * bi + 1
        my_row = 3 * bi + 2
        b_eq[fz_row] = weight

        # Contacts below b (b is the upper): +f on b.
        for ci in bb_by_upper[bi]:
            _, _, cx, cy = bb_contacts[ci]
            A_eq[fz_row, ci] = 1.0
            A_eq[mx_row, ci] = cy - com_y
            A_eq[my_row, ci] = -(cx - com_x)
        # Contacts above b (b is the lower): -f on b.
        for ci in bb_by_lower[bi]:
            _, _, cx, cy = bb_contacts[ci]
            A_eq[fz_row, ci] = -1.0
            A_eq[mx_row, ci] = -(cy - com_y)
            A_eq[my_row, ci] = (cx - com_x)
        # Ground contacts (pushes up on b, compressive only).
        for gi in g_by_brick[bi]:
            _, cx, cy = g_contacts[gi]
            col = num_bb + gi
            A_eq[fz_row, col] = 1.0
            A_eq[mx_row, col] = cy - com_y
            A_eq[my_row, col] = -(cx - com_x)

    # Bounds: stud grip tension for brick-brick, compression-only for ground.
    bounds = [(-STUD_STRENGTH, None)] * num_bb + [(0.0, None)] * num_g

    # Pure feasibility — minimize 0.
    c = np.zeros(num_vars)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=_LP_METHOD)
    return bool(result.success)


def is_stable(bricks: list[Brick]) -> bool:
    """Return True when the structure is in static equilibrium under gravity."""
    return _solve_equilibrium(bricks)


def find_first_unstable(bricks: list[Brick]) -> int:
    """Return the smallest index ``i`` such that ``bricks[:i+1]`` is unstable, or -1.

    Uses a linear scan rather than binary search: instability is *not*
    monotone in prefix length because stud contacts allow tension
    (``-STUD_STRENGTH`` lower bound). A brick above can pull a cantilevered
    brick down and stabilize a prefix that was unstable one brick earlier —
    see ``test_counterweighted_cantilever_stable``. Binary search on a
    non-monotone predicate can return a prefix that's either too short or
    too long.
    """
    n = len(bricks)
    if n == 0 or is_stable(bricks):
        return -1
    for i in range(n):
        if not is_stable(bricks[: i + 1]):
            return i
    return -1  # pragma: no cover — unreachable: is_stable(bricks) returned False above, so some prefix must be unstable.
