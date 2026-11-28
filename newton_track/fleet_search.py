"""A team of G1s runs an avalanche search across a snowfield with hidden ice.

    npy newton_track/fleet_search.py --seeds 20            # paired: coordinated vs independent
    npy newton_track/fleet_search.py --film 3              # one coordinated mission to demo/newton_fleet_seed3.mp4

The task. Someone is buried somewhere in a debris field with a transceiver
beacon. Four robots line up along the bottom edge and must (1) sweep the field
so no strip is left unsearched, (2) pinpoint the beacon once any of them picks
up its signal, and (3) assemble a dig team of three at the burial point. The
field has patches of ice under the snow that nobody can see, and this policy
falls on ice (NEWTON_TRACK.md: the cliff is at friction 0.15 to 0.2).

Each robot is its own agent. It knows its own pose, its own transceiver reading
and whatever arrived in its inbox, and it decides its own command. There is no
central planner: agents agree on lanes and dig positions because they all apply
the same deterministic rule to the same shared roster. What they share goes
over `Bus`, a broadcast channel with a fixed message vocabulary:

  POSE     heartbeat, 5 Hz: who I am, where I am, what I am doing, and what I
           know of the team and the mission, so news crosses more than one hop.
           For the second pass it also says since when I have been parked,
           whether I am on my second pass, and the point I am searching
  DOWN     I fell, here
  HAZARD   there is ice here (from a fall, or from the slip reflex if enabled)
  SIGNAL   my transceiver reads the beacon: estimated position and range
  FOUND    pinpointed: dig here

Coordination, concretely:
  - lanes are an equal split of the field width among robots still standing, in
    left-to-right order, so when one goes down the line closes up;
  - the line stays abreast: whoever is ahead of the pack slows down;
  - a hazard one robot pays for is routed around by all of them;
  - on a signal, the robot with the shortest range pinpoints, the others take
    dig positions around it without crossing paths;
  - when the whole team stands at the far end with no signal, each robot goes
    back for the ground its own map says nobody has been in range of (the
    second pass, `RobotAgent._second_pass`).

Scoring. "Located": a robot declared the burial point within 1 m of the truth.
"Assembled": three robots standing, holding, within 1.7 m of the true beacon for
a full second. Three is fixed: a team that has lost two of four cannot assemble.

`--mode independent` is the ablation: identical robots, sensors, controller and
ice, but the bus is off. Each robot sweeps the lane it started in and reacts
only to what it senses itself. The two modes run on the same seeds, so every
row of the comparison is paired.

What a seed fixes: beacon position, ice layout, transceiver noise, and the
robots' starting joint noise. Physics is the deterministic CPU engine.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fleet_sim import ICE_MU, SNOW_MU, FleetSim, IcePatch, Overlay  # noqa: E402
from g1_newton import POLICY_HZ  # noqa: E402

# -- mission constants ----------------------------------------------------
FIELD = (0.0, 12.0, 0.0, 14.0)  # xmin, xmax, ymin, ymax, metres
SENSE_RANGE = 2.0  # transceiver range. Lanes must be narrower than twice this
# Walking speed, m/s commanded, everywhere a robot chooses its own pace: the
# sweep, a helper converging on the dig, the second pass. `--sweep-speed` sets
# it. 0.5 holds on the default ice, which stands 2 mm proud of the snow. On ice
# flush with the snow it loses about a robot a mission, and 0.35 loses none:
# in the reflex harness on flush ice at friction 0.08, square-on, falls with
# the reflex on are 0 of 64 at 0.35, 2 of 64 at 0.38, 19 of 64 at 0.40 and 32
# of 32 at 0.45. The price of 0.35 is a dig team about 16 s later and little of
# the 100 s limit to spare, which is why it is not the default (FLEET.md,
# "Flush ice, and how fast to sweep").
V_SWEEP = 0.5
V_PINPOINT = 0.35  # the robot walking onto the beacon. Never faster than V_SWEEP
# m/s commanded: a sweeper ahead of the line slows down to this and no further.
# Not scaled with V_SWEEP: at 0.35, a floor of 0.3 * V_SWEEP was no faster and
# cost one mission in twenty its dig team (seeds 180-199, flush ice).
PACE_FLOOR = 0.15
DIG_RADIUS = 1.1  # dig positions sit on a ring this far from the burial point
ASSEMBLED_WITHIN = 1.7  # a robot this close to the true beacon and holding counts as on the dig team
TEAM_SIZE = 3
HAZARD_RADIUS = 1.3  # how much ground one bad experience condemns
HEARTBEAT = POLICY_HZ // 5
LOST_AFTER = int(1.5 * POLICY_HZ)  # a heartbeat older than this is too stale to pace against
# Silence this long means a teammate is gone. It was 1.5 s, the same as the
# pacing limit, and that one number broke the team under packet loss: at 95 %
# loss live teammates were dropped 68 % of the time, a robot that heard nobody
# took the whole field as its lane, and four of twelve beacons near the field
# edge were never found, where radio-off robots found all twelve
# (results/fleet_comms.json). Ten seconds takes that from 8/12 to 12/12.
ROSTER_AFTER = 10 * POLICY_HZ
T_MAX = 100 * POLICY_HZ  # mission time limit, ticks

# -- second pass: bookkeeping constants, none of them a property of the task --
CELL = 0.25  # the coverage map is the field in squares this size
# A square counts as searched once all of it has been in range of a standing
# robot: half its diagonal inside SENSE_RANGE. Three lanes on 12 m are 4 m
# apart, which is exactly twice SENSE_RANGE, so with this margin the seam
# between two such lanes is NOT taken as searched. That is the point: on seed
# 30 the beacon sat on such a seam, 2.19 m from the nearest robot.
COVER_RANGE = SENSE_RANGE - 0.18
REACH = COVER_RANGE - 0.35  # what a robot counts on covering from a point it walks to: it stops within 0.3 m of it
VANTAGE_STEP = 0.5  # the points a robot may choose to walk to are a lattice this fine
# ...none of them within this of a hazard disc. From 1.2 m outside it a robot's
# REACH still takes in everything up to the disc's edge, and it stands 0.75 m
# clear of route()'s keep-out zone. At that zone's edge (0.6) the second pass
# met the same patch again from a new side: 6 of 14 slip-reflex events on seeds
# 20-39 were beside ice the robot already had on its map.
VANTAGE_KEEP_OUT = 1.2
COV_EVERY = 5  # the map rides on every fifth heartbeat
TAKE_OVER = 1.0  # a robot this much nearer to a teammate's claim than the teammate is may claim that ground itself
LOCAL = 3.0  # a robot finishes what is within this distance before it looks further afield
MIN_GAIN = 4  # squares (0.25 m2) of unsearched ground that make a point worth walking to
# How long the first robot to reach the far end waits there for teammates still
# short of it. With the beacon out of reach and the reflex on (seeds 140-159,
# n = 20) the last robot parked within 4.5 s of the first in 14 layouts and
# 8 to 42 s after it in the other 6: pulled along the far line by the lane
# split, queueing behind a parked teammate, or working round ice.
WAIT = 6 * POLICY_HZ
HOVER = 0.05  # a sweeper this close to the line where it parks is as good as parked
PART = (0.7, 1.0)  # closer than the first to a teammate, walk straight apart until the second
CLOSE_UP = 2.0  # a robot with nothing to search waits among the searchers, no nearer than this to where any of them is heading


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclasses.dataclass
class Scenario:
    beacon: tuple[float, float]
    patches: list
    starts: list

    @staticmethod
    def sample(seed: int, n_robots: int, n_patches: int = 3) -> "Scenario":
        # Three patches of 1.2 to 2.0 m on a side: about 5 % of the field is ice.
        # Fixed before any evaluation. The first draft had four patches up to
        # 2.8 m, which put ice across four lanes in five; with a policy that
        # cannot cross ice that measures the ice, not the team.
        rng = np.random.default_rng(1000 + seed)
        x0, x1, y0, y1 = FIELD
        patches = []
        for _ in range(200):
            if len(patches) == n_patches:
                break
            hx, hy = rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)
            p = IcePatch(float(rng.uniform(x0 + hx, x1 - hx)), float(rng.uniform(y0 + 2.5, y1 - 1.5)), float(hx), float(hy))
            if all(abs(p.x - q.x) > p.hx + q.hx + 0.8 or abs(p.y - q.y) > p.hy + q.hy + 0.8 for q in patches):
                patches.append(p)
        # The burial point is on snow with room for a dig team around it. A
        # victim under the ice is a different problem (nobody can stand there).
        for _ in range(500):
            b = (float(rng.uniform(x0 + 1.0, x1 - 1.0)), float(rng.uniform(y0 + 5.0, y1 - 1.0)))
            if not any(p.contains(*b, margin=1.6) for p in patches):
                break
        lane = (x1 - x0) / n_robots
        starts = [(x0 + (k + 0.5) * lane, y0 - 1.0, math.pi / 2) for k in range(n_robots)]
        return Scenario(b, patches, starts)


class Transceiver:
    """Range and bearing to the beacon inside SENSE_RANGE, noisier with distance."""

    def __init__(self, beacon, seed: int):
        self.beacon = beacon
        self.rng = np.random.default_rng(5000 + seed)

    def read(self, x: float, y: float):
        dx, dy = self.beacon[0] - x, self.beacon[1] - y
        d = math.hypot(dx, dy)
        noise = self.rng.normal(size=2)  # drawn every call so the stream does not depend on who is in range
        if d > SENSE_RANGE:
            return None
        frac = d / SENSE_RANGE
        rng_meas = max(0.05, d * (1.0 + 0.10 * frac * noise[0]))
        brg = math.atan2(dy, dx) + 0.15 * frac * noise[1]
        return rng_meas, (x + rng_meas * math.cos(brg), y + rng_meas * math.sin(brg))


class Bus:
    """Broadcast radio. `on=False` drops everything: the independent ablation."""

    def __init__(self, n: int, on: bool = True):
        self.n, self.on = n, on
        self.inbox = [[] for _ in range(n)]
        self.links = []  # (sender, receiver) for non-heartbeat traffic this tick, for the film
        self.sent = 0

    def begin_tick(self):
        self.links = []

    def send(self, sender: int, msg: dict, alive):
        if not self.on:
            return
        msg = {**msg, "from": sender}
        self.sent += 1
        for r in range(self.n):
            if r != sender:
                self.inbox[r].append(msg)
                if msg["type"] != "POSE" and alive[r]:
                    self.links.append((sender, r))

    def drain(self, r: int) -> list:
        out, self.inbox[r] = self.inbox[r], []
        return out


def steer(pose, target, v_max: float, stop_within: float = 0.25) -> tuple[float, float, float]:
    """Body-frame command that walks toward `target`. Turn first, then go."""
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist < stop_within:
        return 0.0, 0.0, 0.0
    err = wrap(math.atan2(dy, dx) - yaw)
    wz = float(np.clip(1.5 * err, -1.0, 1.0))
    if abs(err) > 1.0:
        return 0.0, 0.0, wz
    v = min(v_max, 0.15 + 0.8 * dist) * max(0.0, math.cos(err))
    return float(v), 0.0, wz


def _seg_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from each centre c[k] to each segment a[i]-b[i]. Shapes (m, 2), (m, 2), (h, 2) -> (m, h)."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ac = c[None, :, :] - a[:, None, :]
    t = np.clip((ac * ab[:, None, :]).sum(axis=2) / denom[:, None], 0.0, 1.0)
    closest = a[:, None, :] + t[:, :, None] * ab[:, None, :]
    return np.linalg.norm(c[None, :, :] - closest, axis=2)


def plan_path(start, goal, discs, ring: int = 12, ring_gap: float = 0.15):
    """Shortest path from start to goal that stays out of every disc (x, y, R).

    A small visibility graph: the nodes are start, goal and `ring` points just
    outside each disc; two nodes are joined only if the whole straight leg
    between them clears every disc. Returns the list of points after `start`,
    or None if there is no way through.

    A start that is already inside a disc (a robot that has just backed off
    ice stands inside that hazard's keep-out zone) may leave, but may not go
    deeper: on legs from the start, each disc counts only up to the start's own
    distance from its centre.
    """
    s, g = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if not discs:
        return [tuple(g)]
    c = np.array([[d[0], d[1]] for d in discs], dtype=float)
    R = np.array([d[2] for d in discs], dtype=float)
    R_from_start = np.minimum(R, np.linalg.norm(c - s, axis=1))
    if (_seg_dist(s[None], g[None], c)[0] >= R_from_start - 1e-9).all():
        return [tuple(g)]
    ang = np.linspace(0.0, 2 * math.pi, ring, endpoint=False)
    pts = (c[:, None, :] + (R[:, None, None] + ring_gap) * np.stack([np.cos(ang), np.sin(ang)], axis=1)[None]).reshape(-1, 2)
    keep = (np.linalg.norm(pts[:, None, :] - c[None], axis=2) >= R[None] - 1e-9).all(axis=1)  # not inside another disc
    nodes = np.vstack([s[None], g[None], pts[keep]])
    n = len(nodes)
    ii, jj = np.triu_indices(n, k=1)
    dist = _seg_dist(nodes[ii], nodes[jj], c)
    limit = np.where((ii == 0)[:, None], R_from_start[None], R[None])
    ok = (dist >= limit - 1e-9).all(axis=1)
    length = np.linalg.norm(nodes[ii] - nodes[jj], axis=1)
    cost = np.full((n, n), np.inf)
    cost[ii[ok], jj[ok]] = length[ok]
    cost[jj[ok], ii[ok]] = length[ok]
    best = np.full(n, np.inf)
    best[0] = 0.0
    prev = np.full(n, -1)
    done = np.zeros(n, dtype=bool)
    for _ in range(n):  # Dijkstra on a dense matrix: n is a few dozen
        u = int(np.argmin(np.where(done, np.inf, best)))
        if not np.isfinite(best[u]) or u == 1:
            break
        done[u] = True
        alt = best[u] + cost[u]
        better = alt < best
        best[better] = alt[better]
        prev[better] = u
    if not np.isfinite(best[1]):
        return None
    path, k = [], 1
    while k != 0:
        path.append((float(nodes[k][0]), float(nodes[k][1])))
        k = int(prev[k])
    return path[::-1]


TEAMMATE_BODY = 0.3  # a keep-out disc smaller than this round a teammate keeps nobody off it
GAP_MIN = 0.1  # two hazard zones closer than this to touching leave no way between them: a robot gets within about 5 cm of a waypoint
PLUG = 0.3  # radius of the disc that closes such a gap


def keep_out_discs(target, hazards, teammates=(), clearance: float = 0.45) -> list:
    """The discs route() plans around, for one target. See route() for the rules."""
    tx, ty = target
    discs = []
    for hx, hy, hr in hazards:
        d = math.hypot(hx - tx, hy - ty)
        full = hr + clearance
        discs.append((hx, hy, min(full, max(0.3, d - 0.2)) if d < full else full))
    # Two hazard zones that do not overlap but are less than GAP_MIN from
    # touching get the gap between them plugged. The shortest path would
    # otherwise thread it: on seed 80 two hazards 3.51 m apart left 1 cm
    # between their zones, the plan went through it, and the leg was clear only
    # from the exact waypoint before it. A robot stands within a few
    # centimetres of a waypoint at best, so from where it stood the leg was
    # blocked, the waypoint was underfoot, and it turned on the spot there for
    # the last 60 s of the mission. Zones are taken at full size here (one
    # that has shrunk for the target has not opened a gap), and the plug is
    # left out if the target is in it. With no such pair of hazards the list
    # is what it was. GAP_MIN was 0.3 at first; gaps of 0.23 m turned out to
    # be walked without trouble (seeds 262 and 272), and closing them only
    # sent robots the long way round.
    for i, (ax, ay, ar) in enumerate(hazards):
        for bx, by, br in hazards[i + 1 :]:
            apart = math.hypot(bx - ax, by - ay)
            gap = apart - ar - br - 2 * clearance
            if 0.0 <= gap < GAP_MIN:
                u = (ar + clearance + gap / 2) / apart  # the middle of the gap, as a fraction of the way from a to b
                px, py = ax + u * (bx - ax), ay + u * (by - ay)
                if math.hypot(px - tx, py - ty) >= PLUG + 0.2:
                    discs.append((px, py, PLUG))
    for ox, oy, r in teammates:
        # A parked teammate beside the target used to be left out altogether
        # (anything within r + clearance of it), so that neighbouring dig
        # positions stay reachable. That also let the leg to the target run
        # straight THROUGH the teammate when the target was on its far side:
        # on a 3 m radio, seed 93, a helper walked through the robot standing
        # on the beacon. Its disc now shrinks, as a hazard's does, to what
        # leaves the target `ring_gap` outside it. Beyond r + clearance nothing
        # changes. A target closer to the teammate than its own body is not a
        # routing question (`_dig_slot` does not hand one out).
        d = math.hypot(ox - tx, oy - ty)
        keep = min(r + 0.3, d - 0.15)
        if keep >= TEAMMATE_BODY:
            discs.append((ox, oy, keep))
    return discs


def route(pos, target, hazards, clearance: float = 0.45, memory: dict | None = None, teammates=()):
    """Next point to walk to: `target`, or the first leg of a path round what is in the way.

    Hazards are inflated by `clearance` and are never ignored. If the target
    lies inside a hazard's keep-out zone (a beacon next to mapped ice) the zone
    shrinks just enough to leave the target reachable from the far side. A
    parked teammate beside the target shrinks the same way and is not ignored:
    dig slots are next to each other by design, but the way to one is round
    the robot standing next to it, not through it.

    `memory` (per agent) holds the committed waypoint. It is kept until it is
    reached or the leg to it stops being clear, so the plan does not flip sides
    between ticks.

    History, because two simpler versions failed in instructive ways. Version
    one stepped to one side of the first disc in the way and remembered the
    side: with two discs side by side it alternated for 36 s (seed 8), and a
    newly found disc inherited the side remembered for the cluster it joined
    even when the robot stood on the other side of it (seed 40: thirteen reflex
    events on one patch). Neither version checked that the leg to its waypoint
    was itself clear. Here that is a property of every edge in the graph.
    """
    discs = keep_out_discs(target, hazards, teammates, clearance)
    if not discs:
        return target
    s = np.asarray(pos, dtype=float)
    c = np.array([[d[0], d[1]] for d in discs])
    # No slack here. With 5 cm of slack, re-applied every tick, a robot could
    # ratchet 5 cm deeper into a keep-out zone per replan and walk through it.
    R_from_here = np.minimum(np.array([d[2] for d in discs]), np.linalg.norm(c - s, axis=1))

    def leg_clear(to) -> bool:
        return bool((_seg_dist(s[None], np.asarray(to, dtype=float)[None], c)[0] >= R_from_here - 1e-9).all())

    if leg_clear(target):
        if memory is not None:
            memory.pop("waypoint", None)
        return target
    if memory is not None:
        wp = memory.get("waypoint")
        if wp is not None and math.hypot(wp[0] - pos[0], wp[1] - pos[1]) > 0.4 and leg_clear(wp):
            return wp
    path = plan_path(pos, target, discs)
    if path is None:
        # Boxed in. Head straight away from the nearest hazard and replan next tick.
        k = int(np.argmin(np.linalg.norm(c - s, axis=1)))
        away = s - c[k]
        away = away / max(float(np.linalg.norm(away)), 1e-6)
        # Inside more than one keep-out zone "the nearest" changes from tick
        # to tick and so did the way out: on a 3 m radio, seed 97, a robot
        # that had backed off ice three times stood inside three zones and
        # turned to and fro between two of these headings for the rest of the
        # mission. There the way out is away from all of them at once.
        dist_c = np.linalg.norm(c - s, axis=1)
        inside = dist_c < np.array([d[2] for d in discs])
        if inside.sum() > 1:
            out = ((s - c[inside]) / np.maximum(dist_c[inside], 1e-6)[:, None]).sum(axis=0)
            if float(np.linalg.norm(out)) > 0.2:  # two zones pushing in opposite directions leave nothing to follow
                away = out / float(np.linalg.norm(out))
        wp = (float(s[0] + away[0]), float(s[1] + away[1]))
    else:
        wp = path[0]
        # Do not hand back a node that is already underfoot when the leg to the
        # next one is clear: the walking controller treats "within 25 cm" as
        # arrived and would stand on it.
        k = 0
        while k + 1 < len(path) and math.hypot(path[k][0] - pos[0], path[k][1] - pos[1]) < 0.5 and leg_clear(path[k + 1]):
            k += 1
        wp = path[k]
    if memory is not None:
        memory["waypoint"] = wp
    return wp


class _SearchGrid:
    """Geometry of the coverage map. The same for every robot, built once."""

    _built = None

    def __init__(self):
        x0, x1, y0, y1 = FIELD
        gx, gy = np.meshgrid(np.arange(x0 + CELL / 2, x1, CELL), np.arange(y0 + CELL / 2, y1, CELL))
        self.cx, self.cy = gx.ravel(), gy.ravel()  # square centres
        vx, vy = np.meshgrid(np.arange(x0 + VANTAGE_STEP / 2, x1, VANTAGE_STEP), np.arange(y0 + VANTAGE_STEP / 2, y1, VANTAGE_STEP))
        self.vx, self.vy = vx.ravel(), vy.ravel()  # points a robot may walk to
        d2 = (self.vx[:, None] - self.cx[None]) ** 2 + (self.vy[:, None] - self.cy[None]) ** 2
        self.reach = (d2 <= REACH**2).astype(np.float32)  # (points, squares): what standing at a point is counted on to cover

    @classmethod
    def get(cls) -> "_SearchGrid":
        if cls._built is None:
            cls._built = cls()
        return cls._built

    def near_leg(self, a, b, r: float) -> np.ndarray:
        """Squares within r of the straight leg a-b."""
        ax, ay = a
        dx, dy = b[0] - ax, b[1] - ay
        u = np.clip(((self.cx - ax) * dx + (self.cy - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        return (self.cx - ax - u * dx) ** 2 + (self.cy - ay - u * dy) ** 2 <= r * r


N_SLOTS = 12  # dig positions around the burial point; neighbours of a taken slot stay empty
RECLAIM = 1.0  # a helper on its way gives up its slot for a free one this much nearer
SLOT_TAKEN = 0.55  # a dig position this close to where a parked robot stands is taken, whatever number that robot holds: two G1s 0.47 m apart touch hands
STRAGGLER_GAP = 3.0  # a sweeper this far behind the front is not holding a lane


class RobotAgent:
    """One robot's decision-making. Sees only its own sensors and its inbox.

    Relaying. A radio that reaches 3 m does not reach across a 12 m line, and
    the rule "everyone applies the same rule to the same roster" needs the same
    roster. So the heartbeat also carries what its sender knows (`_news`), and
    whoever hears it merges that in (`_merge_news`):

      team     every teammate the sender has in its roster, with the AGE of that
               entry. A receiver back-dates it, so a position that came through
               two robots is exactly as old as it is and never beats a newer one
               heard directly. Entries die at ROSTER_AFTER wherever they are, so
               nothing circulates for longer than that.
      lost     ids known to be down. A set that only grows.
      hazards  mapped ice. `_add_hazard` already drops anything within 0.6 m of
               a hazard it has, so repeats cost nothing.
      sig      the best beacon signal the sender holds, with who measured it.
               Shorter range wins, as for SIGNAL, so an old one cannot displace
               a better one.
      found    the burial point and who pinpointed it.
      cov      the sender's map of searched ground (see "Second pass"), one bit
               a square, 336 bytes, once a second and only from a robot that
               is parked at the far end or on its second pass. Maps are merged
               by OR, so they can only grow. Without it a robot that was out of
               earshot of a teammate's lane takes that lane for unsearched and
               walks the width of the field to search it again.

    A fallen robot keeps a heartbeat going with role "down" and the place it
    fell: its one DOWN reaches nobody if nobody is in range at that moment.

    Everything relayed is something a robot in range would already have heard
    first-hand, so on a radio that reaches everyone the merge changes nothing.
    `relay = False` is the agent as it was before.

    Second pass. The sweep leaves holes: when a robot drops out of the line the
    rest re-split the field into fewer, wider lanes, and three lanes on 12 m
    touch with no overlap (two do not touch at all). The team used to park at
    the far end with the beacon in such a hole and most of the clock unused.
    Now each robot keeps a map of the ground it believes has been searched
    (`cov`): every square that has been wholly within range of a position it
    stood at itself or heard a standing teammate report. When it stands at the
    far end with no signal, and every teammate it knows of has got there too
    or has been waited for long enough (`_first_pass_over`), it goes back:

      - if a teammate stands within PART of it, it first walks straight
        away from that teammate, forward or back, without turning
        (`_step_apart`);
      - it picks the point with the most unsearched ground in reach per metre
        of walking, looking within LOCAL of itself first (`_choose_claim`),
        never a point within VANTAGE_KEEP_OUT of a hazard disc, and walks
        there by the usual planner. The heartbeat carries the point as its
        `claim`;
      - ground within range of the straight leg from a teammate to that
        teammate's claim is spoken for and counts for nothing, unless this
        robot is TAKE_OVER nearer to that claim than the teammate is. If two
        claims want the same ground, the robot nearer to its own claim keeps
        it (lower id on a tie) and the other chooses again: the same rule on
        the same roster, as for dig slots. Ranked by id alone, the two
        left-hand robots walked across the field to ground the two right-hand
        robots were standing next to (seed 30);
      - a claim is kept until its ground is covered, so the choice does not
        flip between ticks. One that cannot be reached in twice its walking
        time plus ten seconds is given up and its ground written off;
      - with no point left that is worth walking to, the robot looks again
        five times a second, and meanwhile closes up (`_close_up`): it walks
        to the middle of the claims its teammates still hold, no nearer than
        CLOSE_UP to any of them, and waits there, so that a beacon found late
        in the clock has a dig team nearby. When nobody holds a claim it
        holds where it stands (or, if that is inside the keep-out zone of
        mapped ice, at the nearest point outside), and when every standing
        robot does that the mission ends as it always did.

    Mapped ice. A hazard disc is not searched: nobody can stand on it to dig,
    and the robot that reported it was standing within a metre of its centre,
    so all but its far edge was in range then. Ground beside a disc is searched
    from the side, from a point at least VANTAGE_KEEP_OUT outside it. What a
    robot still believes unsearched when it stops (`unsearched_m2`) is those
    discs, plus anything it wrote off.

    The second pass starts only when no robot is still walking up the field,
    so a mission that hears the beacon on the first pass runs exactly as it
    did (the one exception, a robot more than WAIT behind the first to
    arrive, is in `_first_pass_over`). `Mission._score` needed one change for
    a second pass to exist: a robot at the far end that has not yet ruled out
    going back (`pending`) does not count as parked.
    """

    relay = True

    def __init__(self, rid: int, n: int, start_x: float, coordinated: bool):
        self.id, self.n, self.coordinated = rid, n, coordinated
        self.role = "sweep"
        self.home_lane_x = start_x
        self.roster = {}  # id -> dict(x, y, role, slot, t) from heartbeats
        self.lost = set()
        self.hazards = []  # (x, y, r)
        self.best_signal = None  # (range, (x, y), reporter)
        self.found = None  # (x, y) once pinpointed
        self.target = None
        self.holding = False
        self.stuck_ticks = 0
        self.slot = None  # index of the dig position this robot has claimed
        self.sides = {}  # route() memory: the waypoint this robot is committed to
        self.yield_ticks = 0
        self.said = math.inf  # the shortest beacon range I have ever broadcast as my own
        self.fell_at = None  # where I went down
        # second pass
        self.cov = np.zeros(_SearchGrid.get().cx.shape, dtype=bool) if coordinated else None  # squares I believe searched
        self.pass2 = False  # the first pass is over and I am searching what it missed
        self.pending = False  # parked at the far end but not finished: read by Mission._score
        self.claim = None  # the point I am walking to in the second pass
        self.claim_k = None  # ...its index on the lattice
        self.claim_until = 0  # ...and the tick by which I should have covered it
        self.written_off = None  # squares I gave up on reaching
        self.rest = None  # where I wait when nothing is left that is worth walking to
        self.resting = False  # ...and whether I have got there
        self.rest_k = None  # ...its index on the lattice, when it is a lattice point
        self.parting = 0  # ticks spent walking straight away from a teammate that is too close
        self.following = False  # that point was chosen to be near teammates who are still searching
        self.parked_since = None  # first tick of my current unbroken spell parked ("hold", "down", or hovering on the line)
        self.lost_seen = {}  # id -> tick I learned it was down
        self.arrived = set()  # teammates whose first pass is over: heard parked at the far end, or on a second pass
        self.far_since = None  # the tick I first stood at the far end myself
        self.unsearched_m2 = None  # by my map, at my last look

    # ---- shared rules: every agent computes these from the same roster ----

    def _heard(self, t: int) -> dict:
        return {r: i for r, i in self.roster.items() if r not in self.lost and t - i["t"] <= ROSTER_AFTER}

    def _in_line(self, t: int, me) -> list:
        """Sweepers level with the front, left to right. A robot more than
        STRAGGLER_GAP behind (backing off ice, working round a hazard) is left
        out, so the others close up over its lane instead of waiting for it:
        on seed 8 they swept on in their old lanes and walked past the beacon."""
        ys = {self.id: me[1]}
        xs = {self.id: me[0]}
        for r, i in self._heard(t).items():
            if i["role"] in ("sweep", "backoff"):
                ys[r], xs[r] = i["y"], i["x"]
        front = max(ys.values())
        return sorted((r for r in ys if ys[r] >= front - STRAGGLER_GAP), key=lambda r: (xs[r], r))

    def _lane_x(self, t: int, me) -> float:
        if not self.coordinated:
            return self.home_lane_x
        order = self._in_line(t, me)
        if self.id not in order:
            return float(np.clip(me[0], FIELD[0] + 0.5, FIELD[1] - 0.5))  # straggler: straight ahead until it catches up
        x0, x1, _, _ = FIELD
        return x0 + (order.index(self.id) + 0.5) * (x1 - x0) / len(order)

    def _line_y(self, t: int, me) -> float:
        order = self._in_line(t, me)
        if self.id not in order:
            return me[1]  # a straggler does not slow down for anyone
        heard = self._heard(t)
        # Remember a quiet teammate, but only keep pace with positions that are fresh.
        # A relayed position carries its true age (`_merge_news`), so one that
        # crossed two robots has that much less of its 1.5 s left.
        ys = [me[1]] + [heard[r]["y"] for r in order if r != self.id and t - heard[r]["t"] <= LOST_AFTER]
        return float(np.mean(ys))

    def _slot_xy(self, k: int, centre) -> tuple[float, float]:
        a = 2 * math.pi * k / N_SLOTS
        return centre[0] + DIG_RADIUS * math.cos(a), centre[1] + DIG_RADIUS * math.sin(a)

    def _dig_slot(self, t: int, me, centre) -> tuple[float, float]:
        """Claim a dig position and keep it.

        Claims travel in the heartbeat. A teammate's claim outranks mine if it
        is already holding there or has a lower id, and a taken slot also
        blocks its two neighbours so nobody stands shoulder to shoulder. The
        first version shared slots out by bearing every tick: assignments
        flipped as robots moved, and on seed 17 the first to park stood in
        the way of the two behind it for 80 seconds.

        A slot is also taken if a parked robot stands on it (`clear`), and
        with no slot left the answer is a place to wait outside the ring,
        never the centre. A claim is given up for a free slot that is more
        than RECLAIM nearer, which happens to a helper that had to come round
        to another side of the ring.
        """
        blocked = set()
        parked = []
        for r, i in self._heard(t).items():
            k = i.get("slot")
            if k is not None and (i["role"] == "hold" or r < self.id):
                blocked.update({(k - 1) % N_SLOTS, k, (k + 1) % N_SLOTS})
            if i["role"] == "hold":
                parked.append((i["x"], i["y"]))

        def clear(p, k=None) -> bool:
            """Not on mapped ice, and not where a parked robot is standing.

            The second half is new. A slot number says where a teammate means
            to stand, counted from ITS idea of the centre; its heartbeat says
            where it does stand. Two helpers that took their bearings from
            different estimates can hold different numbers for the same
            ground, and a robot parked at the far end that has not heard of
            the beacon holds no number at all. SLOT_TAKEN is below the 0.6 m
            that the robot on the beacon may be from a slot when it stops its
            permitted half metre short, so that robot never trips this. The
            position I already stand on is exempt: parked robots do not
            shuffle.
            """
            if any(math.hypot(p[0] - hx, p[1] - hy) < hr for hx, hy, hr in self.hazards):
                return False
            if k is not None and k == self.slot and self.holding:
                return True
            return all(math.hypot(p[0] - ox, p[1] - oy) >= SLOT_TAKEN for ox, oy in parked)

        def reach(k) -> float:
            x, y = self._slot_xy(k, centre)
            return math.hypot(x - me[0], y - me[1])

        free = [k for k in range(N_SLOTS) if k not in blocked and clear(self._slot_xy(k, centre), k)]
        if self.slot not in free:
            if not free:
                # The ring is full. This used to send the robot to the centre,
                # which is where the pinpointing robot stands. It waits one
                # ring further out instead, on the nearest spoke that is clear,
                # and looks again every tick: a claim lapses when its holder
                # goes down.
                self.slot = None
                spokes = [(centre[0] + 2 * (x - centre[0]), centre[1] + 2 * (y - centre[1])) for x, y in (self._slot_xy(k, centre) for k in range(N_SLOTS))]
                spokes = [p for p in spokes if clear(p)]
                if not spokes:
                    return (me[0], me[1])
                return min(spokes, key=lambda p: math.hypot(p[0] - me[0], p[1] - me[1]))
            self.slot = min(free, key=reach)
        elif not self.holding:
            # Kept, but not at any price. A helper that walks round ice arrives
            # on another side of the ring than the one it claimed from, and its
            # slot is then across the dig team: on seed 89, clean radio, one
            # came over the top with a free slot 1.1 m ahead of it and walked
            # on round the robot on the beacon to its own, 14 s more. A free
            # slot more than RECLAIM nearer is taken instead. A metre is two
            # slots, so walking up to the ring cannot set it off, and a parked
            # robot never does this.
            nearest = min(free, key=reach)
            if reach(nearest) < reach(self.slot) - RECLAIM:
                self.slot = nearest
        return self._slot_xy(self.slot, centre)

    def _teammate_discs(self, t: int) -> list:
        """Teammates that are parked: walk round them, do not queue behind them."""
        return [(i["x"], i["y"], 0.45) for i in self._heard(t).values() if i["role"] in ("hold", "locate", "down")]

    def _dig_discs(self, t: int, slot) -> list:
        """The teammates a helper walks round on its way to `slot`.

        `_teammate_discs`, less a teammate that is still walking onto the
        beacon (role "locate") within 0.9 m of the slot. `keep_out_discs`
        shrinks a teammate that close to the target instead of dropping it,
        which is right for one that stands still. The pinpointing robot does
        not: a plan round where it is now flips sides as it walks. On seed 214
        a helper set off to pass in front of it, was re-planned behind it two
        seconds later, turned round on the spot and was walked into by the
        helper following it. Until it parks it is left out, as it was before.
        """
        return [
            (i["x"], i["y"], 0.45)
            for i in self._heard(t).values()
            if i["role"] in ("hold", "down") or (i["role"] == "locate" and math.hypot(i["x"] - slot[0], i["y"] - slot[1]) >= 0.9)
        ]

    def _too_close_to_turn(self, t: int, me, goal, was_parked: bool):
        """A parked robot that must turn round with a teammate at its shoulder walks straight apart first. A command, or None.

        The sweep parks robots at the far end as little as 0.4 m apart (0.38 m
        on a 3 m radio, seed 76), and a beacon heard after that is behind
        them, so both turn on the spot where they stand. The second pass met
        the same thing and has a rule for it (`_step_apart`); this is that
        rule on the way to a dig position. It starts only on the tick a robot
        leaves a parked state, and only where `steer` would turn on the spot
        (more than 1 rad off the bearing to the goal); once started it runs
        until the two are PART[1] apart. A helper already walking is left
        alone: with any close teammate as the trigger, one that stopped to
        turn at a waypoint while another helper walked up behind it went
        straight on for 9 s with the other at its heels (3 m radio, seed 265).
        """
        if not self.parting and not (was_parked and abs(wrap(math.atan2(goal[1] - me[1], goal[0] - me[0]) - me[2])) > 1.0):
            return None
        return self._step_apart(t, me)

    # ---- one tick ----

    def step(self, t: int, pose, down: bool, reading, inbox: list, ice_event=None):
        """Returns (command, outbox)."""
        out = []
        me = pose
        # `role` is still what Mission._score saw at the end of the last tick.
        # Hovering counts as parked: a robot that stops exactly on the parking
        # line drifts back over it and steps forward again, several times a
        # second, for as long as the mission lasts (seed 91 on a 3 m radio).
        hovering = self.role == "sweep" and me[1] >= FIELD[3] - 0.3 - HOVER
        if self.role not in ("hold", "down") and not hovering:
            self.parked_since = None
        elif self.parked_since is None:
            self.parked_since = t - 1
        self.pending = False
        news = []
        for m in inbox:
            src = m["from"]
            if m["type"] == "POSE":
                self.roster[src] = {"x": m["x"], "y": m["y"], "role": m["role"], "slot": m.get("slot"), "t": t,
                                    "since": m.get("since"), "p2": m.get("p2", False), "claim": m.get("claim")}  # fmt: skip
                if m["role"] == "down":
                    self.lost.add(src)
                elif self.cov is not None:
                    self._mark(m["x"], m["y"])
                news.append(m)
            elif m["type"] == "DOWN":
                self.lost.add(src)
            elif m["type"] == "HAZARD":
                self._add_hazard(m["x"], m["y"])
            elif m["type"] == "SIGNAL":
                if self.best_signal is None or m["range"] < self.best_signal[0] - 0.05:
                    self.best_signal = (m["range"], (m["x"], m["y"]), src)
            elif m["type"] == "FOUND":
                self.found = (m["x"], m["y"])
                self.best_signal = (0.0, self.found, src)
        if self.relay:
            for m in news:  # after everything first-hand: second-hand news never overrides what I heard myself this tick
                self._merge_news(t, m)
        for r in self.lost:
            self.lost_seen.setdefault(r, t)
        if self.cov is not None:
            self._watch()

        beat = t % HEARTBEAT == self.id % HEARTBEAT
        if down:
            if self.role != "down":
                self.role = "down"
                self.fell_at = (me[0], me[1])
                out.append({"type": "DOWN", "x": me[0], "y": me[1]})
                out.append({"type": "HAZARD", "x": me[0], "y": me[1]})
                self._add_hazard(me[0], me[1])
            elif beat and self.relay:
                out.append({"type": "POSE", "x": me[0], "y": me[1], "role": "down", "slot": None, "hazards": [self.fell_at]})
            self.target = None
            return (0.0, 0.0, 0.0), out

        if ice_event is not None:  # own slip reflex confirmed ice
            # The reflex reports where the EDGE is. The ice is beyond it, in the
            # direction the robot was walking (it backs straight out, so its
            # heading still points at the ice). Centre the hazard there: a disc
            # on the edge condemns as much good snow as ice.
            hx = ice_event["x"] + 0.6 * HAZARD_RADIUS * math.cos(me[2])
            hy = ice_event["y"] + 0.6 * HAZARD_RADIUS * math.sin(me[2])
            self._add_hazard(hx, hy)
            out.append({"type": "HAZARD", "x": hx, "y": hy})

        if reading is not None:
            rng_meas, est = reading
            if self.best_signal is None or rng_meas < self.best_signal[0] - 0.05 or self.best_signal[2] == self.id:
                self.best_signal = (rng_meas, est, self.id)
                self.said = min(self.said, rng_meas)
                out.append({"type": "SIGNAL", "x": est[0], "y": est[1], "range": rng_meas})

        if self.best_signal is not None:
            self.claim = None  # the beacon outranks the search
        if beat:
            if self.cov is not None:
                self._mark(me[0], me[1])  # exactly the positions my teammates hear, so on a clean radio all maps agree
            pose_msg = {"type": "POSE", "x": me[0], "y": me[1], "role": self.role, "slot": self.slot,
                        "since": self.parked_since, "p2": self.pass2, "claim": self.claim}  # fmt: skip
            if self.relay:
                pose_msg.update(self._news(t))
            out.append(pose_msg)

        _x0, _x1, _y0, y1 = FIELD
        pos = (me[0], me[1])

        # -- pinpoint / dig: someone has the signal --
        if self.best_signal is not None:
            _rng, est, who = self.best_signal
            if who == self.id:
                # The pinpointing robot stands on the beacon, not on the ring. If it
                # claimed a dig slot earlier as a helper, that claim would go on
                # blocking the slot and both neighbours (radio study, loss 0.95 seed 50).
                self.slot = None
                goal = self.found or est
                if self.found is not None and self._arrived(pos, self.found, 0.5, 0.9):
                    self.role = "hold"
                    return self._keep_station(me, self.found), out
                self.role = "locate"
                cmd = self._go(me, goal, min(V_PINPOINT, V_SWEEP), 0.3)
                close = reading is not None and reading[0] < 0.6
                self.stuck_ticks = self.stuck_ticks + 1 if cmd[0] == 0.0 and abs(cmd[2]) < 0.2 else 0
                if self.found is None and (close or (self.stuck_ticks > POLICY_HZ and reading is not None)):
                    self.found = reading[1] if reading is not None else est
                    out.append({"type": "FOUND", "x": self.found[0], "y": self.found[1]})
                return cmd, out
            # a teammate is pinpointing: take a dig position
            slot = self._dig_slot(t, me, self.found or est)
            if self._arrived(pos, slot, 0.35, 0.9):
                self.role = "hold"
                self.target = slot
                return self._keep_station(me, slot), out
            was_parked = self.role == "hold"  # still last tick's role here
            self.role = "converge"
            apart = self._too_close_to_turn(t, me, slot, was_parked)
            if apart is not None:
                self.target = None
                return apart, out
            return self._yield_to_others(t, me, self._go(me, slot, V_SWEEP, 0.3, self._dig_discs(t, slot))), out

        # -- second pass: the team reached the far end without a signal --
        if self.pass2:
            cmd = self._second_pass(t, me, beat)
            if beat:
                # Say what I decided on this tick, not on the last one. A claim
                # one heartbeat stale let two robots each give way to a claim
                # the other had already dropped: on seed 158 the two survivors
                # both left the 2.4 m seam between their lanes for the field
                # edges and came back for it 19 s later.
                pose_msg["claim"] = self.claim
            return cmd, out

        # -- sweep --
        if me[1] >= y1 - 0.3:
            self.role = "hold"
            self.target = None
            if self.coordinated:
                self.pending = True  # not finished: the mission must not end on my account
                if self.far_since is None:
                    self.far_since = t
                self.pass2 = self._first_pass_over(t)
            return (0.0, 0.0, 0.0), out
        self.role = "sweep"
        lane_x = self._lane_x(t, me)
        ahead_y = min(y1, me[1] + 2.5)
        while ahead_y < y1 and any(math.hypot(lane_x - hx, ahead_y - hy) < hr + 0.45 for hx, hy, hr in self.hazards):
            ahead_y = min(y1, ahead_y + 0.5)  # aim past mapped ice, never into it
        v = V_SWEEP
        if self.coordinated:
            ahead = me[1] - self._line_y(t, me)
            v = float(np.clip(V_SWEEP - 0.6 * ahead, PACE_FLOOR, V_SWEEP))
        return self._yield_to_others(t, me, self._go(me, (lane_x, ahead_y), v, 0.25)), out

    # ---- relaying: what rides on the heartbeat, and what is done with it ----

    def _news(self, t: int) -> dict:
        """What I know that a teammate out of earshot of the source may not.

        `age` is ticks since the robot it describes said it, counting one tick
        of radio latency per hop, so that after any number of hops a receiver
        dates the entry to the tick it would have heard it first-hand.
        """
        news = {
            "team": [(r, i["x"], i["y"], i["role"], i.get("slot"), t - i["t"] + 1, i.get("since"), i.get("p2", False), i.get("claim"))
                     for r, i in self._heard(t).items()],  # fmt: skip
            "lost": sorted(self.lost),
            "hazards": [(hx, hy) for hx, hy, _ in self.hazards],
        }
        if self.found is not None:
            news["found"] = (self.found[0], self.found[1], self.best_signal[2])
        elif self.best_signal is not None:
            rng_meas, est, who = self.best_signal
            news["sig"] = (rng_meas, est[0], est[1], who)
        elif self.cov is not None and (self.pass2 or self.parked_since is not None) and t % (COV_EVERY * HEARTBEAT) == self.id % HEARTBEAT:
            news["cov"] = np.packbits(self.cov).tobytes().hex()
        return news

    def _merge_news(self, t: int, m: dict):
        for r, x, y, role, slot, age, *more in m.get("team", ()):
            seen = t - age  # the tick I would have heard this myself, had I been in range
            if r == self.id or t - seen > ROSTER_AFTER:
                continue
            if r not in self.roster or self.roster[r]["t"] < seen:  # strictly newer only: an echo of what I have changes nothing
                since, p2, claim = more if len(more) == 3 else (None, False, None)
                self.roster[r] = {"x": x, "y": y, "role": role, "slot": slot, "t": seen, "since": since, "p2": p2, "claim": claim}
                if role != "down" and self.cov is not None:
                    self._mark(x, y)
        self.lost.update(r for r in m.get("lost", ()) if r != self.id)
        for hx, hy in m.get("hazards", ()):
            self._add_hazard(hx, hy)
        if "cov" in m and self.cov is not None:
            self.cov |= np.unpackbits(np.frombuffer(bytes.fromhex(m["cov"]), dtype=np.uint8))[: self.cov.size].astype(bool)
        if self.found is not None:
            return  # pinpointed: nothing second-hand improves on that
        found = m.get("found")
        if found is not None and found[2] != self.id:
            self.found = (found[0], found[1])
            self.best_signal = (0.0, self.found, found[2])
            return
        sig = m.get("sig")
        if sig is not None and sig[3] != self.id:
            # A relayed signal is old. If I am the one pinpointing, my teammates
            # may still hold the shortest range I ever announced, so that is
            # what it has to beat, not my latest reading: readings rise with
            # noise, and an old signal that beat a noisy one would send me off
            # to a dig slot while everyone else still waits for me.
            mine = self.best_signal
            bar = math.inf if mine is None else min(mine[0], self.said) if mine[2] == self.id else mine[0]
            if sig[0] < bar - 0.05:
                self.best_signal = (sig[0], (sig[1], sig[2]), sig[3])

    # ---- second pass: the ground the sweep missed ----

    def _mark(self, x: float, y: float):
        """A standing robot was here: every square wholly within its range is searched."""
        g = _SearchGrid.get()
        self.cov |= (g.cx - x) ** 2 + (g.cy - y) ** 2 <= COVER_RANGE**2

    def _watch(self):
        """Note which teammates have finished their first pass. Kept as a set
        that only grows: a robot parked at the far end steps off the line and
        back onto it as it drifts, and its first pass is no less over."""
        for r, i in self.roster.items():
            if i.get("p2") or (i.get("since") is not None and i["role"] != "down" and i["y"] >= FIELD[3] - 0.5):
                self.arrived.add(r)

    def _first_pass_over(self, t: int) -> bool:
        """Is the first pass over for the team?

        Called while I am parked at the far end with no signal. A teammate is
        done once I have heard it parked at the far end (each heartbeat says
        since when its sender has been parked, `parked_since`) or on its
        second pass. From there it has nothing left to sweep: it stands, or
        steps back up to the line it drifted off, and no ground comes into its
        range that was not in range on the way up. So nothing the first pass
        could still find is lost by going back the moment the last robot
        arrives, and a mission that hears the beacon on the first pass runs as
        it did: nobody has moved early.

        This used to ask for more: all of us parked at the same time for three
        seconds running, which was the rule that ended the old mission. That
        cost 3 s on every second pass, and more whenever one robot stepped off
        the line and back inside the three seconds (seed 214: 7.4 s between
        the last arrival and the first robot going back, with the dig team
        assembled at 97.1 s of 100).

        The wait is bounded. A robot that never arrived used to hold the
        whole team at the far end until the clock ran out, and one that took
        30 s over the last two metres held it for 30 s. Now a robot goes back
        once it has stood at the far end for WAIT, whoever is missing, and so
        does every robot that hears a teammate on its second pass, so the
        team leaves together on the clock of whoever arrived first. A robot
        left behind carries on with its own first pass and joins in when it
        gets to the far end. This is the one case where a mission that finds
        the beacon on the first pass can run differently: the late robot
        finds it more than WAIT after the first robot parked.

        On a radio with a range a teammate can be out of earshot. One that has
        been silent for ROSTER_AFTER, or was never heard, is treated as gone,
        as the sweep already treats it.
        """
        if self.parked_since is None:
            return False
        go = t - self.far_since > WAIT or any(i.get("p2") for i in self._heard(t).values())
        for r in range(self.n):
            if r == self.id or r in self.lost or r in self.arrived:
                continue
            i = self.roster.get(r)
            if i is None or t - i["t"] > ROSTER_AFTER:
                if t <= ROSTER_AFTER:
                    return False  # too early to call a robot I never heard gone
                continue
            if not go:
                return False  # still on its way up, and not yet waited for long enough
        return True

    def _second_pass(self, t: int, me, beat: bool):
        pos = (me[0], me[1])
        apart = self._step_apart(t, me)
        if apart is not None:
            self.role = "search"
            self.target = None
            return apart
        if beat or (self.claim is None and self.rest is None):
            self._choose_claim(t, pos)
            if self.claim is None:
                self._close_up(t, pos)
        if self.claim is None:
            # Nothing I can usefully walk to. Wait at `rest`; I look again on every heartbeat.
            # Its own hysteresis flag: `holding` belongs to the dig, and a stale True
            # there would let a helper stop 0.9 m short of its dig position.
            self.resting = math.hypot(pos[0] - self.rest[0], pos[1] - self.rest[1]) < (0.9 if self.resting else 0.35)
            if self.resting:
                self.role = "hold"
                self.target = None
                return self._keep_station(me, self.rest)
            self.role = "search"
            return self._yield_to_others(t, me, self._go(me, self.rest, V_SWEEP, 0.3, self._walk_round(t, pos)))
        self.rest = self.rest_k = None
        self.resting = self.following = False
        self.role = "search"
        return self._yield_to_others(t, me, self._go(me, self.claim, V_SWEEP, 0.3, self._walk_round(t, pos)))

    def _step_apart(self, t: int, me):
        """Too close to a teammate to turn round: walk straight apart first. A command, or None.

        The sweep can park two robots half a metre apart (lanes are re-split as
        robots arrive at the far end). Nothing moved after that until now. On
        seed 32 the robots at x 7.5 and 8.1 both turned on the spot to start
        their second pass, locked shoulders, and one went down on plain snow.
        So inside PART[0] of a teammate a robot neither turns nor steers: it
        walks straight forward or straight back, whichever takes it away, at
        the pace and with the ramp the slip reflex backs out with, until they
        are PART[1] apart.
        """
        near = None
        for r, i in self._heard(t).items():
            if t - i["t"] <= LOST_AFTER:
                d = math.hypot(i["x"] - me[0], i["y"] - me[1])
                if near is None or d < near[0]:
                    near = (d, r, i)
        if near is None or near[0] > (PART[1] if self.parting else PART[0]):
            self.parting = 0
            return None
        self.parting += 1
        d, r, i = near
        ahead = math.cos(math.atan2(me[1] - i["y"], me[0] - i["x"]) - me[2])  # how far "straight ahead" points away from it
        if abs(ahead) < 0.2:  # side by side: the lower id goes forward, the other back
            ahead = 1.0 if self.id < r else -1.0
        v = 0.25 * min(1.0, self.parting / 25)
        return (v if ahead > 0 else -v, 0.0, 0.0)

    def _walk_round(self, t: int, pos) -> list:
        """Teammates to route round on the second pass: the parked ones, and
        anyone within 1.5 m whatever it is doing, with more room than a parked
        one gets."""
        discs = self._teammate_discs(t)
        for i in self._heard(t).values():
            if i["role"] not in ("hold", "locate", "down") and t - i["t"] <= LOST_AFTER and math.hypot(i["x"] - pos[0], i["y"] - pos[1]) < 1.5:
                discs.append((i["x"], i["y"], 0.6))
        return discs

    def _off_ice(self, p, margin: float) -> bool:
        """Is p at least `margin` outside every hazard disc?"""
        return all(math.hypot(p[0] - hx, p[1] - hy) >= hr + margin for hx, hy, hr in self.hazards)

    def _close_up(self, t: int, pos):
        """Choose where to wait while I have nothing to search: in among the teammates that have.

        The spot nearest the middle of the claims I have heard, at least
        CLOSE_UP from each of them (a searcher is about to walk there) and
        outside every keep-out zone. With one claim that is beside it, on my
        side; with two it is between them, so that whichever of them finds the
        beacon has a helper a few metres away.
        """
        g = _SearchGrid.get()
        claims = [i["claim"] for i in self._heard(t).values() if i.get("claim") is not None]
        if not claims:
            # Nobody is searching: stop here. But not on the edge of ice. A robot
            # whose reflex has just fired stands where it met the ice; told to
            # hold THAT spot, it backed out, walked back to it and went down
            # (seed 106, beacon out of reach). Inside a keep-out zone the place
            # to wait is the nearest lattice point outside all of them.
            if self.following or self.rest is None or not self._off_ice(self.rest, 0.45):
                self.rest, self.rest_k, self.following = pos, None, False
                if not self._off_ice(pos, 0.45):
                    ok = np.ones(g.vx.shape, dtype=bool)
                    for hx, hy, hr in self.hazards:
                        ok &= (g.vx - hx) ** 2 + (g.vy - hy) ** 2 >= (hr + 0.45 + 0.15) ** 2
                    if ok.any():
                        k = int(np.argmin(np.where(ok, np.hypot(g.vx - pos[0], g.vy - pos[1]), np.inf)))
                        self.rest, self.rest_k = (float(g.vx[k]), float(g.vy[k])), k
            return
        self.following = True
        ok = np.ones(g.vx.shape, dtype=bool)
        for c in claims:
            ok &= (g.vx - c[0]) ** 2 + (g.vy - c[1]) ** 2 >= CLOSE_UP**2
        for hx, hy, hr in self.hazards:
            ok &= (g.vx - hx) ** 2 + (g.vy - hy) ** 2 >= (hr + VANTAGE_KEEP_OUT) ** 2
        for ox, oy, _r in self._teammate_discs(t):
            ok &= (g.vx - ox) ** 2 + (g.vy - oy) ** 2 >= 1.0
        for r, i in self._heard(t).items():  # nor where a teammate with a lower id is, or is about to be
            if r < self.id and t - i["t"] <= LOST_AFTER:
                ok &= (g.vx - i["x"]) ** 2 + (g.vy - i["y"]) ** 2 >= 1.2**2
        if not ok.any():
            if self.rest is None:
                self.rest, self.rest_k = pos, None
            return
        mx, my = float(np.mean([c[0] for c in claims])), float(np.mean([c[1] for c in claims]))
        cost = np.hypot(g.vx - mx, g.vy - my) + 0.25 * np.hypot(g.vx - pos[0], g.vy - pos[1])
        k = int(np.argmin(np.where(ok, cost, np.inf)))
        if self.rest_k is not None and ok[self.rest_k] and math.hypot(g.vx[k] - self.rest[0], g.vy[k] - self.rest[1]) <= 1.0:
            return  # where I am already heading is as good: do not chase every half metre
        self.rest, self.rest_k = (float(g.vx[k]), float(g.vy[k])), k

    def _choose_claim(self, t: int, pos):
        """Keep my claim while it is still worth having, otherwise pick the best point left."""
        g = _SearchGrid.get()
        if self.written_off is None:
            self.written_off = np.zeros(g.cx.shape, dtype=bool)
        todo = ~self.cov & ~self.written_off
        self.unsearched_m2 = round(float((~self.cov).sum()) * CELL * CELL, 2)
        for hx, hy, hr in self.hazards:  # mapped ice is not searched: nobody could stand there to dig
            todo &= (g.cx - hx) ** 2 + (g.cy - hy) ** 2 > hr * hr
        dist = np.hypot(g.vx - pos[0], g.vy - pos[1])
        k = self.claim_k
        mine = (dist[k], self.id) if self.claim is not None else None
        spoken, outranks = np.zeros(g.cx.shape, dtype=bool), np.zeros(g.cx.shape, dtype=bool)
        for r, i in self._heard(t).items():
            c = i.get("claim")
            if c is None:
                continue
            leg = g.near_leg((i["x"], i["y"]), c, COVER_RANGE)
            theirs = math.hypot(c[0] - i["x"], c[1] - i["y"])
            if math.hypot(c[0] - pos[0], c[1] - pos[1]) > theirs - TAKE_OVER:
                spoken |= leg  # unless I am clearly the nearer of us to it
            if mine is not None and (theirs, r) < mine:
                outranks |= leg
        ok = np.ones(g.vx.shape, dtype=bool)
        for hx, hy, hr in self.hazards:
            ok &= (g.vx - hx) ** 2 + (g.vy - hy) ** 2 >= (hr + VANTAGE_KEEP_OUT) ** 2
        for ox, oy, _r in self._teammate_discs(t):  # nor on top of a parked or fallen teammate
            ok &= (g.vx - ox) ** 2 + (g.vy - oy) ** 2 >= 1.0
        if self.claim is not None:
            if t >= self.claim_until:
                self.written_off |= g.reach[k] > 0  # could not get there: do not keep trying
                todo &= ~self.written_off
            elif ok[k] and dist[k] > 0.4 and float(g.reach[k] @ (todo & ~outranks)) >= MIN_GAIN:
                return
        gain = g.reach @ (todo & ~spoken).astype(np.float32)
        gain[~ok | (gain < MIN_GAIN)] = 0.0
        # Nearby ground first. Ranked on gain per metre alone, a robot left a
        # corner of a strip behind for a bigger patch across the field and
        # walked back for it later.
        local = gain * (dist <= LOCAL)
        if local.any():
            gain = local
        best = int(np.argmax(gain / (1.0 + dist)))
        if gain[best] <= 0.0:
            self.claim = self.claim_k = None
            return
        if best != k or self.claim is None or t >= self.claim_until:
            self.claim_until = t + int(POLICY_HZ * (10.0 + 2.0 * dist[best] / V_SWEEP))
        self.claim, self.claim_k = (float(g.vx[best]), float(g.vy[best])), best

    def _go(self, me, goal, v_max: float, stop_within: float, teammates=()):
        """Command toward `goal` round whatever is mapped in between.

        `stop_within` applies to the goal only. An intermediate waypoint is
        walked through, never stopped at: with one stop distance for both, a
        robot handed a waypoint 20 cm away stood on it for 70 seconds (seed 25,
        two robots at once, mission lost with nobody down).
        """
        self.target = route((me[0], me[1]), goal, self.hazards, memory=self.sides, teammates=teammates)
        final = math.hypot(self.target[0] - goal[0], self.target[1] - goal[1]) < 1e-6
        return steer(me, self.target, v_max, stop_within=stop_within if final else 0.0)

    def _arrived(self, pos, goal, enter: float, leave: float) -> bool:
        """Hold with hysteresis. A standing robot drifts a few centimetres a second;
        without the band it flips between walking and holding and never settles."""
        d = math.hypot(pos[0] - goal[0], pos[1] - goal[1])
        self.holding = d < (leave if self.holding else enter)
        return self.holding

    def _keep_station(self, me, goal):
        """Holding is not a zero command. This policy wanders when told to stand:
        a robot parked on the beacon drifted half a metre in six seconds. Inside
        25 cm it stands; beyond that it steps back at a crawl."""
        return steer(me, goal, 0.2, stop_within=0.25)

    def _add_hazard(self, x: float, y: float):
        if all(math.hypot(x - hx, y - hy) > 0.6 for hx, hy, _ in self.hazards):
            self.hazards.append((x, y, HAZARD_RADIUS))

    def _yield_to_others(self, t: int, me, cmd):
        """Give way to a MOVING teammate with a lower id that is right in front.
        Parked teammates are obstacles for route(), not something to wait for,
        and no wait lasts more than two seconds."""
        blocked = False
        for rid, info in self._heard(t).items():
            if rid > self.id or info["role"] in ("hold", "locate", "down"):
                continue
            dx, dy = info["x"] - me[0], info["y"] - me[1]
            if math.hypot(dx, dy) < 0.8 and abs(wrap(math.atan2(dy, dx) - me[2])) < 1.0:
                blocked = True
        self.yield_ticks = self.yield_ticks + 1 if blocked else 0
        if blocked and self.yield_ticks % (4 * POLICY_HZ) < 2 * POLICY_HZ:
            return (0.0, 0.0, cmd[2])
        return cmd


class Mission:
    """Physics + agents + bus for one seed. `tick()` until `done`."""

    def __init__(self, seed: int, mode: str = "coordinated", n_robots: int = 4, reflex: bool = False,
                 ice_mu: float = ICE_MU, snow_mu: float = SNOW_MU, bus: "Bus | None" = None):  # fmt: skip
        if mode not in ("coordinated", "independent"):
            raise ValueError(mode)
        self.seed, self.mode = seed, mode
        self.scenario = Scenario.sample(seed, n_robots)
        self.sim = FleetSim(self.scenario.starts, self.scenario.patches, seed=seed, ice_mu=ice_mu, snow_mu=snow_mu)
        self.n = n_robots
        coordinated = mode == "coordinated"
        # `bus` lets a study swap in a degraded radio (range limit, packet loss)
        # without touching the agents. It must keep Bus's interface.
        self.bus = bus if bus is not None else Bus(n_robots, on=coordinated)
        self.agents = [RobotAgent(r, n_robots, self.scenario.starts[r][0], coordinated) for r in range(n_robots)]
        self.sensor = Transceiver(self.scenario.beacon, seed)
        self.reflex = None
        if reflex:
            from fleet_reflex import IceReflex  # noqa: PLC0415

            self.reflex = IceReflex(self.sim)
        self._pending_ice = [None] * n_robots
        self.t = 0
        self.done = False
        self.t_found = None
        self.t_assembled = None
        self.found_error = None
        self.track = []  # (t, [(x, y)...]) every 10 ticks, for coverage
        self._team_ticks = 0
        self._parked_ticks = 0
        self._last_wz = np.zeros(n_robots, dtype=np.float32)
        self._link_until = {}  # (sender, receiver) -> tick until which the film keeps drawing it

    def tick(self):
        sim, t = self.sim, self.t
        self.bus.begin_tick()
        poses = [sim.pose(r) for r in range(self.n)]
        alive = ~sim.down
        cmds = np.zeros((self.n, 3), dtype=np.float32)
        outboxes = []
        for r, agent in enumerate(self.agents):
            reading = None if sim.down[r] else self.sensor.read(poses[r][0], poses[r][1])
            cmd, out = agent.step(t, poses[r], bool(sim.down[r]), reading, self.bus.drain(r), self._pending_ice[r])
            self._pending_ice[r] = None
            cmds[r] = cmd
            outboxes.append(out)
        # Yaw-rate slew limit. The policy falls on plain snow within a second of a
        # hard yaw reversal at speed (3 of 9 one-minute runs flipping +-1 rad/s at
        # 0.5 m/s, measured while building the slip reflex). A full reversal now
        # takes two thirds of a second.
        step = 3.0 / POLICY_HZ
        cmds[:, 2] = np.clip(cmds[:, 2], self._last_wz - step, self._last_wz + step)
        self._last_wz = cmds[:, 2].copy()
        for r, out in enumerate(outboxes):  # delivered next tick: one tick of radio latency
            for m in out:
                self.bus.send(r, m, alive)
        if self.reflex is not None:
            cmds, events = self.reflex.filter(cmds)
            for e in events:
                self._pending_ice[e["robot"]] = e
            for r in range(self.n):
                if self.reflex.active[r] and not sim.down[r]:
                    self.agents[r].role = "backoff"
        for pair in self.bus.links:  # a radio event lasts one tick; hold it half a second so it is visible
            self._link_until[pair] = self.t + POLICY_HZ // 2
        sim.step(cmds)
        self.t += 1
        self._score(cmds)

    def _score(self, cmds):
        sim, b = self.sim, self.scenario.beacon
        poses = [sim.pose(r) for r in range(self.n)]
        if self.t % 10 == 0:
            self.track.append([(p[0], p[1]) if not sim.down[r] else None for r, p in enumerate(poses)])
        if self.t_found is None:
            finders = [a for a in self.agents if a.found is not None and a.best_signal[2] == a.id]
            if finders:
                self.t_found = self.t
                self.found_error = round(math.hypot(finders[0].found[0] - b[0], finders[0].found[1] - b[1]), 2)
        # On the dig team: standing, holding position, close to the TRUE burial point.
        team = [
            r for r, p in enumerate(poses)
            if not sim.down[r] and self.agents[r].role == "hold" and math.hypot(p[0] - b[0], p[1] - b[1]) < ASSEMBLED_WITHIN
        ]  # fmt: skip
        standing = int((~sim.down).sum())
        self._team_ticks = self._team_ticks + 1 if self.t_found is not None and len(team) >= TEAM_SIZE else 0
        if self.t_assembled is None and self._team_ticks >= POLICY_HZ:  # held for a full second
            self.t_assembled = self.t
        # A robot parked at the far end that has not yet ruled out a second pass
        # (`pending`) is not parked for good. Without this the mission ends
        # three seconds after the last robot reaches the far end, and the test
        # for starting a second pass needs those three seconds.
        all_parked = all(a.role in ("hold", "down") and not a.pending for a in self.agents)
        self._parked_ticks = self._parked_ticks + 1 if all_parked else 0
        if self.t_assembled is not None or self.t >= T_MAX or standing == 0 or self._parked_ticks > 3 * POLICY_HZ:
            self.done = True

    def overlay(self) -> Overlay:
        agents = self.agents
        known = {}
        for a in agents:
            for h in a.hazards:
                known[(round(h[0], 1), round(h[1], 1))] = h
        cap = f"{self.mode}  t={self.t / POLICY_HZ:4.1f}s  standing {int((~self.sim.down).sum())}/{self.n}"
        if self.t_found is not None:
            cap += f"  beacon pinpointed at {self.t_found / POLICY_HZ:.1f}s"
        if self.t_assembled is not None:
            cap += f"  dig team assembled at {self.t_assembled / POLICY_HZ:.1f}s"
        return Overlay(
            field=FIELD,
            beacon=self.scenario.beacon,
            beacon_found=self.t_found is not None,
            hazards=list(known.values()),
            links=[pair for pair, until in self._link_until.items() if until >= self.t],
            targets=[a.target for a in agents],
            roles=[a.role for a in agents],
            caption=cap,
        )

    def coverage(self) -> float:
        """Fraction of the field that came within transceiver range of a standing robot."""
        x0, x1, y0, y1 = FIELD
        gx, gy = np.meshgrid(np.arange(x0 + 0.25, x1, 0.5), np.arange(y0 + 0.25, y1, 0.5))
        seen = np.zeros(gx.shape, dtype=bool)
        for row in self.track:
            for p in row:
                if p is not None:
                    seen |= (gx - p[0]) ** 2 + (gy - p[1]) ** 2 <= SENSE_RANGE**2
        return float(seen.mean())

    def result(self) -> dict:
        sim = self.sim
        lost = int(sim.down.sum())
        return {
            "seed": self.seed,
            "mode": self.mode,
            "reflex": self.reflex is not None,
            "located": self.t_found is not None and self.found_error < 1.0,
            "t_located_s": None if self.t_found is None else round(self.t_found / POLICY_HZ, 1),
            "locate_error_m": self.found_error,
            "assembled": self.t_assembled is not None,
            "t_assembled_s": None if self.t_assembled is None else round(self.t_assembled / POLICY_HZ, 1),
            "robots_lost": lost,
            "coverage": round(self.coverage(), 3),
            "messages": self.bus.sent,
            "duration_s": round(self.t / POLICY_HZ, 1),
            "beacon": [round(v, 2) for v in self.scenario.beacon],
        }


def run(seed: int, mode: str, n_robots: int, reflex: bool, film: str | None = None) -> dict:
    mission = Mission(seed, mode, n_robots, reflex)
    writer = None
    if film:
        from film_fleet import FleetFilm  # noqa: PLC0415

        writer = FleetFilm(mission.sim, film, field=FIELD)
    tail = 0
    while tail < 2 * POLICY_HZ:  # keep filming two seconds past the end
        if not mission.done:
            mission.tick()
        elif writer is None:
            break
        else:
            mission.sim.step(np.zeros((n_robots, 3), dtype=np.float32))
            tail += 1
        if writer is not None and mission.sim.t % 2 == 0:
            writer.frame(mission.overlay())
    r = mission.result()
    if writer is not None:
        r["frames"] = writer.close()
    return r


def summarise(rows: list) -> dict:
    n = len(rows)
    t_loc = [r["t_located_s"] for r in rows if r["located"]]
    t_asm = [r["t_assembled_s"] for r in rows if r["assembled"]]
    return {
        "missions": n,
        "located": sum(r["located"] for r in rows),
        "assembled": sum(r["assembled"] for r in rows),
        "robots_lost_mean": round(float(np.mean([r["robots_lost"] for r in rows])), 2),
        "coverage_mean": round(float(np.mean([r["coverage"] for r in rows])), 3),
        "t_located_median_s": None if not t_loc else round(float(np.median(t_loc)), 1),
        "t_assembled_median_s": None if not t_asm else round(float(np.median(t_asm)), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed; 20 onward were never used to debug the planner")
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--mode", default="both", choices=("both", "coordinated", "independent"))
    ap.add_argument("--reflex", action="store_true", help="enable the slip reflex (fleet_reflex.py) on every robot")
    ap.add_argument("--film", type=int, default=None, metavar="SEED", help="render one mission instead of evaluating")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep-speed", type=float, default=None, metavar="M_S",
                    help=f"walking speed commanded, default {V_SWEEP}. 0.35 is the safe speed on flush ice")  # fmt: skip
    args = ap.parse_args()
    if args.sweep_speed is not None:
        globals()["V_SWEEP"] = args.sweep_speed

    if args.film is not None:
        mode = "coordinated" if args.mode == "both" else args.mode
        out = args.out or f"demo/newton_fleet_seed{args.film}_{mode}.mp4"
        print(json.dumps(run(args.film, mode, args.robots, args.reflex, film=out)))
        print(f"wrote {out}")
        return 0

    modes = ("coordinated", "independent") if args.mode == "both" else (args.mode,)
    rows = []
    t0 = time.time()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for mode in modes:
            r = run(seed, mode, args.robots, args.reflex)
            rows.append(r)
            print(json.dumps(r), flush=True)
    summary = {m: summarise([r for r in rows if r["mode"] == m]) for m in modes}
    print()
    for m, s in summary.items():
        print(f"{m:<12} {s}")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parent / "results" / (
        ("fleet_search_reflex" if args.reflex else "fleet_search") + (f"_from{args.seed_start}" if args.seed_start else "") + ".json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "robots": args.robots, "field": FIELD, "sense_range": SENSE_RANGE, "ice_mu": ICE_MU, "snow_mu": SNOW_MU,
        "reflex": args.reflex, "seed_start": args.seed_start, "v_sweep": V_SWEEP, "summary": summary, "rows": rows,
    }, indent=1))  # fmt: skip
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
