"""Geometry tests for the fleet path planner. No simulation, runs in a second.

    npy newton_track/test_fleet_route.py

Each case is a shape that broke an earlier version of `route()` in a real
mission. A point robot follows `route()` at walking pace and the test checks
three things the planner can actually promise from what it knows (hazard
discs, never the true ice):

  - it reaches the target;
  - it never gets closer to a hazard's centre than the disc edge, except as
    close as it started (a robot that has just backed off ice starts inside the
    keep-out zone) or as close as the target itself is;
  - it does not wander: the distance walked stays within 15 % of the length of
    the plan it made at the start. Dithering costs several times the plan.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fleet_search import HAZARD_RADIUS, keep_out_discs, plan_path, route  # noqa: E402

R = HAZARD_RADIUS


def walk(pos, target, hazards, teammates=(), steps: int = 1200, v: float = 0.05):
    """Follow route() from pos. Returns (reached, too close by [m], walked / planned)."""
    pos = np.array(pos, dtype=float)
    # Length of the plan as made at the start. Following it should cost about
    # that much; dithering costs several times more.
    plan = plan_path(tuple(pos), target, keep_out_discs(target, hazards, teammates))
    pts = [tuple(pos)] + list(plan)
    straight = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    # The planner's promise per hazard: never closer to its centre than the disc
    # edge, except that it may be as close as it started (it has just backed off
    # ice) or as close as the target itself is (a beacon beside mapped ice).
    floor = [
        min(hr, math.hypot(pos[0] - hx, pos[1] - hy), math.hypot(target[0] - hx, target[1] - hy) - 0.2)
        for hx, hy, hr in hazards
    ]
    memory, worst, walked = {}, 0.0, 0.0
    for _ in range(steps):
        if np.linalg.norm(np.array(target) - pos) < 0.1:
            return True, worst, walked / max(straight, 1e-9)
        wp = np.array(route(tuple(pos), target, hazards, memory=memory, teammates=teammates))
        d = wp - pos
        n = float(np.linalg.norm(d))
        step = min(v, n)
        pos = pos + d / max(n, 1e-9) * step
        walked += step
        for k, (hx, hy, _hr) in enumerate(hazards):
            worst = max(worst, floor[k] - math.hypot(pos[0] - hx, pos[1] - hy))
    return False, worst, walked / max(straight, 1e-9)


CASES = {
    # name: (start, target, hazards, teammates)
    "open ground": ((0, 0), (0, 6), [], ()),
    "hazard off to the side": ((0, 0), (0, 6), [(5, 3, R)], ()),
    "hazard dead ahead": ((0, 0), (0, 6), [(0, 3, R)], ()),
    "just backed off it, inside the keep-out": ((0, 1.4), (0, 6), [(0, 3, R)], ()),
    # seed 8: two discs side by side. Version one alternated left and right for 36 s.
    "two discs side by side": ((4.8, 1.2), (4.5, 6.0), [(4.3, 2.58, R), (5.3, 2.58, R)], ()),
    # seed 21: ice dead ahead with the sweep target just behind it. Version two skipped the hazard.
    "target just behind the hazard": ((10.5, 2.3), (10.5, 4.8), [(10.7, 3.8, R)], ()),
    # seed 40, independent, robot 2 (geometry replayed by the studies agent). Disc A was passed on
    # the right; the detour clipped a second patch, which added disc B; B inherited "right" and the
    # waypoint went to the far side of B, straight across the ice. Nine reflex events on one patch.
    "second disc found during a detour": ((8.2, 5.7), (7.5, 8.2), [(7.33, 5.60 + 0.6 * R, R), (8.88 + 0.6 * R, 5.73, R)], ()),
    # a beacon 1.6 m from a hazard centre: inside the keep-out zone, must still be reachable
    "target beside a hazard": ((0, 0), (1.6, 3.0), [(0, 3, R)], ()),
    # seed 17: a teammate parked on the way to a dig slot. Version one queued behind it for 80 s.
    "parked teammate in the way": ((4.6, 6.2), (2.6, 7.3), [], [(2.9, 6.4, 0.45), (3.8, 6.4, 0.45)]),
    "parked teammate beside the slot": ((5.0, 6.9), (2.7, 6.9), [], [(1.6, 6.9, 0.45)]),
}


def main() -> int:
    ok = True
    print(f"{'case':<42} {'reached':<8} {'too close m':<12} walked / planned")
    for name, (start, target, hazards, teammates) in CASES.items():
        reached, worst, ratio = walk(start, target, hazards, teammates)
        good = reached and worst < 0.02 and ratio < 1.15
        ok &= good
        print(f"{name:<42} {reached!s:<8} {worst:<12.3f} {ratio:<5.2f} {'' if good else 'FAIL'}")
    # A ring of hazards with the target inside it and no gap: there is no path, and plan_path says so.
    wall = [(3 * math.cos(a), 3 * math.sin(a), 1.75) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    boxed = plan_path((8.0, 0.0), (0.0, 0.0), wall) is None
    ok &= boxed
    print(f"{'target walled in: no path reported':<42} {boxed!s:<8}")
    print("\nall passed" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
