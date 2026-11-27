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

  POSE     heartbeat, 5 Hz: who I am, where I am, what I am doing
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
    dig positions around it without crossing paths.

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
V_SWEEP = 0.5  # m/s commanded
DIG_RADIUS = 1.1  # dig positions sit on a ring this far from the burial point
ASSEMBLED_WITHIN = 1.7  # a robot this close to the true beacon and holding counts as on the dig team
TEAM_SIZE = 3
HAZARD_RADIUS = 1.3  # how much ground one bad experience condemns
HEARTBEAT = POLICY_HZ // 5
LOST_AFTER = int(1.5 * POLICY_HZ)  # silence this long means a teammate is gone
T_MAX = 100 * POLICY_HZ  # mission time limit, ticks


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


def route(pos, target, obstacles, clearance: float = 0.45, memory: dict | None = None):
    """Next point to walk to: `target`, or a waypoint that passes the obstacles in the way.

    `obstacles` are discs (x, y, r). The ones blocking the straight line are
    grown into a cluster with every disc that touches them, and the whole
    cluster is passed on ONE side. `memory` (per agent) remembers which side
    was chosen for each disc, so the choice survives replanning every tick.
    Without it a robot facing two discs side by side picked left, then right,
    then left, for 36 seconds, and ended up back on the ice (seed 8).
    """
    px, py = pos
    tx, ty = target
    seg = np.array([tx - px, ty - py])
    length = float(np.linalg.norm(seg))
    if length < 1e-6 or not obstacles:
        return target
    u = seg / length
    left = np.array([-u[1], u[0]])
    geo = []  # (along, across, r, key) with across > 0 meaning the disc centre is left of the path
    for ox, oy, r in obstacles:
        rel = np.array([ox - px, oy - py])
        geo.append((float(rel @ u), float(left @ rel), r, (round(ox, 2), round(oy, 2))))
    blocking = [
        k for k, (along, across, r, _key) in enumerate(geo)
        if -r < along < length - 0.2 and abs(across) < r + clearance
        and math.hypot(obstacles[k][0] - tx, obstacles[k][1] - ty) > r + clearance  # cannot avoid what the target sits beside
    ]  # fmt: skip
    if not blocking:
        return target
    cluster = {min(blocking, key=lambda k: geo[k][0])}
    grew = True
    while grew:
        grew = False
        for k, (ox, oy, r) in enumerate(obstacles):
            if k in cluster:
                continue
            if any(math.hypot(ox - obstacles[c][0], oy - obstacles[c][1]) < r + obstacles[c][2] + 2 * clearance for c in cluster):
                cluster.add(k)
                grew = True
    margin = clearance + 0.3
    go_left = max(geo[k][1] + geo[k][2] for k in cluster) + margin  # lateral offset that clears them all on the left
    go_right = min(geo[k][1] - geo[k][2] for k in cluster) - margin
    side = None
    if memory is not None:
        side = next((memory[geo[k][3]] for k in cluster if geo[k][3] in memory), None)
    if side is None:
        side = "left" if abs(go_left) <= abs(go_right) else "right"
    if memory is not None:
        for k in cluster:
            memory[geo[k][3]] = side
    lateral = go_left if side == "left" else go_right
    along = max(0.6, min(geo[k][0] for k in cluster))
    w = np.array([px, py]) + u * along + left * lateral
    return float(w[0]), float(w[1])


N_SLOTS = 12  # dig positions around the burial point; neighbours of a taken slot stay empty
STRAGGLER_GAP = 3.0  # a sweeper this far behind the front is not holding a lane


class RobotAgent:
    """One robot's decision-making. Sees only its own sensors and its inbox."""

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
        self.sides = {}  # route() memory: which side each obstacle is passed on
        self.yield_ticks = 0

    # ---- shared rules: every agent computes these from the same roster ----

    def _heard(self, t: int) -> dict:
        return {r: i for r, i in self.roster.items() if r not in self.lost and t - i["t"] <= LOST_AFTER}

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
        return float(np.mean([me[1] if r == self.id else heard[r]["y"] for r in order]))

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
        """
        blocked = set()
        for r, i in self._heard(t).items():
            k = i.get("slot")
            if k is not None and (i["role"] == "hold" or r < self.id):
                blocked.update({(k - 1) % N_SLOTS, k, (k + 1) % N_SLOTS})
        free = [
            k for k in range(N_SLOTS)
            if k not in blocked
            and not any(math.hypot(self._slot_xy(k, centre)[0] - hx, self._slot_xy(k, centre)[1] - hy) < hr for hx, hy, hr in self.hazards)
        ]  # fmt: skip
        if self.slot not in free:
            if not free:
                self.slot = None
                return centre
            self.slot = min(free, key=lambda k: math.hypot(self._slot_xy(k, centre)[0] - me[0], self._slot_xy(k, centre)[1] - me[1]))
        return self._slot_xy(self.slot, centre)

    def _teammate_discs(self, t: int) -> list:
        """Teammates that are parked: walk round them, do not queue behind them."""
        return [(i["x"], i["y"], 0.45) for i in self._heard(t).values() if i["role"] in ("hold", "locate", "down")]

    # ---- one tick ----

    def step(self, t: int, pose, down: bool, reading, inbox: list, ice_event=None):
        """Returns (command, outbox)."""
        out = []
        me = pose
        for m in inbox:
            src = m["from"]
            if m["type"] == "POSE":
                self.roster[src] = {"x": m["x"], "y": m["y"], "role": m["role"], "slot": m.get("slot"), "t": t}
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

        if down:
            if self.role != "down":
                self.role = "down"
                out.append({"type": "DOWN", "x": me[0], "y": me[1]})
                out.append({"type": "HAZARD", "x": me[0], "y": me[1]})
                self._add_hazard(me[0], me[1])
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
                out.append({"type": "SIGNAL", "x": est[0], "y": est[1], "range": rng_meas})

        if t % HEARTBEAT == self.id % HEARTBEAT:
            out.append({"type": "POSE", "x": me[0], "y": me[1], "role": self.role, "slot": self.slot})

        _x0, _x1, _y0, y1 = FIELD
        pos = (me[0], me[1])

        # -- pinpoint / dig: someone has the signal --
        if self.best_signal is not None:
            _rng, est, who = self.best_signal
            if who == self.id:
                goal = self.found or est
                if self.found is not None and self._arrived(pos, self.found, 0.5, 0.9):
                    self.role = "hold"
                    return self._keep_station(me, self.found), out
                self.role = "locate"
                self.target = route(pos, goal, self.hazards, memory=self.sides)
                cmd = steer(me, self.target, 0.35, stop_within=0.3)
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
            self.role = "converge"
            self.target = route(pos, slot, self.hazards + self._teammate_discs(t), memory=self.sides)
            return self._yield_to_others(t, me, steer(me, self.target, V_SWEEP, stop_within=0.3)), out

        # -- sweep --
        if me[1] >= y1 - 0.3:
            self.role = "hold"
            self.target = None
            return (0.0, 0.0, 0.0), out
        self.role = "sweep"
        lane_x = self._lane_x(t, me)
        self.target = route(pos, (lane_x, min(y1, me[1] + 2.5)), self.hazards, memory=self.sides)
        v = V_SWEEP
        if self.coordinated:
            ahead = me[1] - self._line_y(t, me)
            v = float(np.clip(V_SWEEP - 0.6 * ahead, 0.15, V_SWEEP))
        return self._yield_to_others(t, me, steer(me, self.target, v)), out

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
        all_parked = all(a.role in ("hold", "down") for a in self.agents)
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
    args = ap.parse_args()

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
        "reflex": args.reflex, "seed_start": args.seed_start, "summary": summary, "rows": rows,
    }, indent=1))  # fmt: skip
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
