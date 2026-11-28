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
    the plan it made at the start. Dithering costs several times the plan;
  - it stays out of a parked teammate's disc on the same terms as a hazard's.

The second table is the dig positions (`RobotAgent._dig_slot`): which point a
helper is sent to, given what its roster says. No walking, only the choice.
The last lines are two things found on seed 80: how narrow a gap a plan may
thread, and a helper that set off with a waypoint committed for another goal.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fleet_search import DIG_RADIUS, HAZARD_RADIUS, RobotAgent, keep_out_discs, plan_path, route, steer  # noqa: E402

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
    # The same promise for a parked teammate, whose disc is its 0.45 m plus 0.3:
    # never inside it, except as close as the walk started or as close as the
    # target is (dig positions are next to each other by design).
    floor_mate = [
        min(r + 0.3, math.hypot(pos[0] - ox, pos[1] - oy), math.hypot(target[0] - ox, target[1] - oy) - 0.15)
        for ox, oy, r in teammates
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
        for k, (ox, oy, _r) in enumerate(teammates):
            worst = max(worst, floor_mate[k] - math.hypot(pos[0] - ox, pos[1] - oy))
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
    # 3 m radio, seed 93: a dig slot 0.7 m beyond the robot standing on the beacon, claimed from the
    # far side. A teammate that close to the target was left out of the plan, and the helper walked
    # through it.
    "slot just beyond a parked teammate": ((0.0, 0.0), (0.2, 3.7), [], [(0.0, 3.0, 0.45)]),
    # the same slot from its open side: nothing to walk round, and the fix must not invent a detour
    "slot beside a parked teammate, open side": ((3.0, 4.5), (0.2, 3.7), [], [(0.0, 3.0, 0.45)]),
    # seed 80, robot 3: a pocket between three hazard discs whose only short way out is a 1 cm gap
    # between two of them. The point robot here could thread it; the way that can be walked is round.
    "pocket with a 1 cm gap in its wall": ((9.84, 4.21), (9.84, 8.21), [(10.66, 2.56, R), (10.26, 6.05, R), (7.3, 7.12, R), (5.47, 7.08, R)], ()),
    # the whole ring from behind: a robot 0.3 m off the beacon on my slot's side, helpers parked two slots either side
    # of mine, and I come from the far side. Their discs overlap into a wall; the way is round it.
    "slot on the far side of a parked dig team": ((-3.0, 0.2), (1.1, 0.0), [], [(0.3, 0.1, 0.45), (0.55, 0.95, 0.45), (0.55, -0.95, 0.45)]),
}


def helper(me, roster, slot=None, holding=False, hazards=()):
    """A helper (id 2 of 6) at `me` with this roster, asked for its dig position round a beacon at (0, 0).

    roster: id -> (x, y, role, slot). Returns (point, slot index or None)."""
    a = RobotAgent(2, 6, 0.0, True)
    a.roster = {r: {"x": x, "y": y, "role": role, "slot": k, "t": 0} for r, (x, y, role, k) in roster.items()}
    a.slot, a.holding, a.hazards = slot, holding, list(hazards)
    return a._dig_slot(1, (me[0], me[1], 0.0), (0.0, 0.0)), a.slot


def slot_cases() -> list:
    """(name, passed, what was chosen). Distances are from the point chosen."""
    out = []

    def apart(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    # A robot parked on the ground my nearest slot is on, holding no slot number: one that stopped at
    # the far end out of earshot of the beacon, or the pinpointing robot well off its mark. Slots
    # were shared out by number alone, and I was sent to stand 0.3 m from it.
    there = (1.1 + 0.25, 0.15)
    p, k = helper((3.0, 0.3), {0: (*there, "hold", None)})
    out.append(("a robot is parked on the slot", k is not None and apart(p, there) >= 0.55, f"slot {k}, {apart(p, there):.2f} m from it"))
    # two helpers that took their bearings from estimates of the centre 0.7 m apart: the teammate is
    # parked on its "slot 3", which is 0.4 m from where my slot 5 is. Its number blocks 2 to 4 only.
    stood = (-0.953 + 0.3, 0.55 + 0.25)
    p, k = helper((-3.0, 2.0), {3: (*stood, "hold", 3)})
    out.append(("teammate parked off its slot number", k is not None and apart(p, stood) >= 0.55, f"slot {k}, {apart(p, stood):.2f} m from it"))
    # ...but a robot that is already parked on its slot does not shuffle because someone stopped near it
    mine = (1.1, 0.0)
    p, k = helper((1.0, 0.1), {0: (0.75, 0.2, "hold", None)}, slot=0, holding=True)
    out.append(("already parked: stay put", k == 0 and apart(p, mine) < 1e-9, f"slot {k}"))
    # team of six: four claims with lower ids or parked fill the ring. The fallback was the centre,
    # which is where the pinpointing robot stands.
    ring = {0: (0.1, 0.0, "hold", None), 1: (1.1, 0.0, "hold", 0), 3: (0.0, 1.1, "hold", 3), 4: (-1.1, 0.0, "hold", 6), 5: (0.0, -1.1, "hold", 9)}
    p, k = helper((4.0, 3.0), ring)
    nearest = min(apart(p, (i[0], i[1])) for i in ring.values())
    out.append(("ring full: wait outside it, not on the beacon", k is None and apart(p, (0.0, 0.0)) > DIG_RADIUS + 0.55 and nearest >= 0.55,
                f"{apart(p, (0.0, 0.0)):.2f} m from the beacon, {nearest:.2f} m from the nearest robot"))  # fmt: skip
    # seed 89, robot 1: claimed slot 8 from the south-west, walked round a hazard and came over the
    # top of the ring. Slot 6 is held, slot 4 is free and 0.75 m ahead; its own is 2.1 m away across
    # the dig team.
    over_the_top = (-1.3, 1.0)
    dug_in = {0: (-1.1, 0.05, "hold", 6), 3: (0.15, 0.1, "hold", None)}
    p, k = helper(over_the_top, dug_in, slot=8)
    out.append(("came round to the other side of the ring", k == 4, f"slot {k}, {apart(p, over_the_top):.2f} m away"))
    # ...but half a metre is not worth a change, and a parked robot never trades
    p, k = helper((-2.6, -1.9), dug_in, slot=9)
    out.append(("a slightly nearer slot: keep the claim", k == 9, f"slot {k}"))
    p, k = helper((0.0, -1.2), {0: (0.1, 0.0, "hold", None)}, slot=6, holding=True)
    out.append(("parked: never trades", k == 6, f"slot {k}"))
    # unchanged behaviour: nearest free slot, and a slot on mapped ice is never taken
    p, k = helper((3.0, 0.3), {0: (0.1, 0.0, "hold", None), 1: (0.0, 1.1, "converge", 3)})
    out.append(("nearest free slot", k == 0, f"slot {k}"))
    p, k = helper((3.0, 0.3), {0: (0.1, 0.0, "hold", None)}, hazards=[(2.0, 0.0, R)])
    out.append(("slot on mapped ice", k is not None and apart(p, (2.0, 0.0)) >= R, f"slot {k}, {apart(p, (2.0, 0.0)):.2f} m from the hazard centre"))
    return out


SEED_80 = [(10.66, 2.56, R), (10.26, 6.05, R), (7.3, 7.12, R), (5.47, 7.08, R)]  # the four hazard discs every robot had mapped


def threads_a_gap(start, goal, hazards, gap: float = 0.1) -> bool:
    """Does any leg of the plan pass between two discs that are less than `gap` from touching?"""
    discs = keep_out_discs(goal, hazards)
    pts = [tuple(start)] + list(plan_path(start, goal, discs))

    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    for i, (ax, ay, ar) in enumerate(discs):
        for bx, by, br in discs[i + 1:]:
            if 0.0 <= math.hypot(ax - bx, ay - by) - ar - br < gap:
                for p, q in zip(pts, pts[1:]):
                    if side((ax, ay), (bx, by), p) * side((ax, ay), (bx, by), q) < 0 and side(p, q, (ax, ay)) * side(p, q, (bx, by)) < 0:
                        return True
    return False


def boxed_in_walk(start, yaw, target, hazards, teammates, seconds: float = 90.0):
    """Follow route() from a start that plan_path can find no way out of, as a robot does: `steer`
    turns on the spot before it walks, so a heading that keeps reversing gets it nowhere. Returns
    (reached, deeper into any hazard zone than it started by [m], seconds taken)."""
    x, y = start
    floor = [min(hr, math.hypot(x - hx, y - hy)) for hx, hy, hr in hazards]
    memory, worst, dt = {}, 0.0, 0.02
    for k in range(int(seconds / dt)):
        if math.hypot(target[0] - x, target[1] - y) < 0.3:
            return True, worst, k * dt
        wp = route((x, y), target, hazards, memory=memory, teammates=teammates)
        final = math.hypot(wp[0] - target[0], wp[1] - target[1]) < 1e-6
        v, _, wz = steer((x, y, yaw), wp, 0.5, stop_within=0.25 if final else 0.0)
        yaw += wz * dt
        x, y = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt
        for j, (hx, hy, _hr) in enumerate(hazards):
            worst = max(worst, floor[j] - math.hypot(x - hx, y - hy))
    return False, worst, seconds


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
    print(f"\n{'dig position':<46} {'ok':<6} chosen")
    for name, good, what in slot_cases():
        ok &= good
        print(f"{name:<46} {good!s:<6} {what} {'' if good else 'FAIL'}")
    # seed 80, robot 3: the plan out of the pocket went through a gap 1 cm wide
    good = not threads_a_gap((9.84, 4.21), (9.84, 8.21), SEED_80)
    ok &= good
    print(f"\n{'plan stays out of a 1 cm gap':<46} {good!s:<6} {'' if good else 'FAIL'}")
    # 3 m radio, seed 97, robot 0: inside three keep-out zones at once after backing off ice three times,
    # with no plan out. The fallback turned to and fro between "away from this one" and "away from that one".
    reached, worst, took = boxed_in_walk((5.13, 7.35), 0.8, (8.72, 7.62), [(5.83, 6.07, R), (6.51, 8.13, R), (4.64, 8.71, R)],
                                         [(9.14, 6.53, 0.45), (9.28, 8.38, 0.45), (10.06, 7.61, 0.45)])  # fmt: skip
    good = reached and worst < 0.05
    ok &= good
    print(f"{'boxed in inside three zones':<46} {good!s:<6} reached {reached} in {took:.0f} s, {worst:.3f} m deeper {'' if good else 'FAIL'}")
    print("\nall passed" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
