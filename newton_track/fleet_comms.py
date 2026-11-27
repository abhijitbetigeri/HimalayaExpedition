"""Two studies on the fleet search: how much radio it needs, and how many robots.

    npy newton_track/fleet_comms.py --study comms                 # loss and range sweep, 4 robots, slip reflex on
    npy newton_track/fleet_comms.py --study scale                 # 3 to 6 robots, reflex on; 4 and 6 with it off
    npy newton_track/fleet_comms.py --study fix                   # candidate agent changes on the cells that break
    npy newton_track/fleet_comms.py --study comms --summarise     # rebuild the JSON and table from the mission log

Nothing in `fleet_search.py` is changed. `LossyBus` is passed to `Mission` through
its `bus` argument; the agents cannot tell it from `Bus` except by what arrives.
Seeds start at 40 by default: 0 to 39 were used to debug the planner.

Study 1, radio. The coordinated mission on the same seeds with
  - packet loss p in {0, 0.5, 0.8, 0.95}: every (message, receiver) pair is
    dropped independently with probability p;
  - a range limit R in {5, 3} m: a message reaches a receiver only if the two
    robots are within R at send time;
  - the independent mode (bus off) as the floor.
The loss draws are common random numbers: one uniform per (message, receiver) is
drawn whatever p is, so a packet dropped at p = 0.5 is also dropped at p = 0.8
for as long as the two runs are still sending the same messages.

Study 2, team size. 3, 4, 5 and 6 robots on the unchanged 12 x 14 m field with
the slip reflex on, and 4 and 6 with it off. The dig team stays at three.
NOTE: `Scenario.sample` draws the ice and the beacon from the seed alone, so a
seed is the same field at every team size. Only the start lanes move.

Each mission prints one JSON line and appends it to a `.jsonl` log as it
finishes, and a restarted job skips the missions already in the log.

`Probe` is read-only instrumentation. It looks at what each agent believes
(who it can hear, its lane, its dig-slot claim, its hazards) against what is
true in the sim, so the tables can say why a cell failed and not just that it
did. It never writes to an agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fleet_search import (  # noqa: E402
    FIELD,
    HAZARD_RADIUS,
    HEARTBEAT,
    LOST_AFTER,
    N_SLOTS,
    STRAGGLER_GAP,
    TEAM_SIZE,
    Bus,
    Mission,
    RobotAgent,
)
from g1_newton import POLICY_HZ  # noqa: E402

ONE_SHOT = ("DOWN", "HAZARD", "FOUND")  # sent once by the stock agent; SIGNAL and POSE repeat
WARMUP = 2 * HEARTBEAT  # ticks before every roster has had its first heartbeat
FIRST_SEED = 40


class LossyBus(Bus):
    """`Bus` with independent packet loss and a range limit.

    loss      probability that one receiver misses one message, independent
              across receivers and messages, from a generator seeded by `seed`.
    range_m   None for unlimited. Otherwise a message is delivered only to
              receivers within this many metres of the sender when it is sent.

    Positions come from `attach(sim)` (read at send time, cached per tick) or,
    without a sim, from `set_positions()` each tick.
    """

    def __init__(self, n: int, loss: float = 0.0, range_m: float | None = None, seed: int = 0, sim=None):
        super().__init__(n, on=True)
        if not 0.0 <= loss <= 1.0:
            raise ValueError(loss)
        self.loss, self.range_m = float(loss), range_m
        self.rng = np.random.default_rng([9000, seed])
        self.sim = sim
        self._pos = None
        self._pos_tick = None
        self.tick = -1
        # per message type, counted over receivers that were standing
        self.stats = {}
        self.one_shots = []  # (tick, type, sender, [standing receivers that missed it], standing receivers)

    def attach(self, sim):
        self.sim = sim

    def set_positions(self, xy):
        self._pos, self._pos_tick = [tuple(p[:2]) for p in xy], self.tick

    def begin_tick(self):
        super().begin_tick()
        self.tick += 1

    def _positions(self):
        if self.sim is not None and self._pos_tick != self.tick:
            self._pos = [self.sim.pose(r)[:2] for r in range(self.n)]
            self._pos_tick = self.tick
        if self._pos is None:
            raise RuntimeError("range-limited bus needs attach(sim) or set_positions()")
        return self._pos

    def send(self, sender: int, msg: dict, alive):
        if not self.on:
            return
        msg = {**msg, "from": sender}
        self.sent += 1
        u = self.rng.random(self.n)  # drawn whatever the loss rate is: common random numbers across cells
        pos = self._positions() if self.range_m is not None else None
        st = self.stats.setdefault(msg["type"], {"sent": 0, "delivered": 0, "lost": 0, "out_of_range": 0})
        st["sent"] += 1
        missed = []
        for r in range(self.n):
            if r == sender:
                continue
            if pos is not None and math.hypot(pos[r][0] - pos[sender][0], pos[r][1] - pos[sender][1]) > self.range_m:
                if alive[r]:
                    st["out_of_range"] += 1
                    missed.append(r)
                continue
            if u[r] < self.loss:
                if alive[r]:
                    st["lost"] += 1
                    missed.append(r)
                continue
            self.inbox[r].append(msg)
            if alive[r]:
                st["delivered"] += 1
                if msg["type"] != "POSE":
                    self.links.append((sender, r))
        if msg["type"] in ONE_SHOT:
            others = sum(1 for r in range(self.n) if r != sender and alive[r])
            self.one_shots.append((self.tick, msg["type"], sender, missed, others))


# -- candidate agent changes, as subclasses so they can be measured ---------


class RepeatingAgent(RobotAgent):
    """One-shot facts ride on the 5 Hz heartbeat.

    The stock agent says DOWN, HAZARD and FOUND exactly once. Here:
      - a fallen robot keeps repeating DOWN and HAZARD for the place it fell;
      - every robot that knows the burial point repeats FOUND with `by` set to
        the robot that pinpointed it, so a teammate out of the finder's radio
        range can get it second-hand;
      - every robot repeats the hazards it knows, one per heartbeat in rotation.
    Receiving is unchanged except that FOUND is credited to `by`, an echo of my
    own FOUND is ignored, and of two competing FOUNDs the lower id wins.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fell_at = None
        self._hz_turn = 0

    def step(self, t, pose, down, reading, inbox, ice_event=None):
        kept = []
        for m in inbox:
            if m["type"] == "FOUND" and "by" in m:
                by = m["by"]
                if by == self.id:
                    continue
                if self.found is not None and by > self.best_signal[2]:
                    continue
                m = {**m, "from": by}
            kept.append(m)
        cmd, out = super().step(t, pose, down, reading, kept, ice_event)
        beat = t % HEARTBEAT == self.id % HEARTBEAT
        if down:
            if self.fell_at is None:
                self.fell_at = (pose[0], pose[1])
            elif beat:
                out.append({"type": "DOWN", "x": self.fell_at[0], "y": self.fell_at[1]})
                out.append({"type": "HAZARD", "x": self.fell_at[0], "y": self.fell_at[1]})
            return cmd, out
        for m in out:
            if m["type"] == "FOUND":
                m["by"] = self.id
        if beat:
            if self.found is not None and not any(m["type"] == "FOUND" for m in out):
                out.append({"type": "FOUND", "x": self.found[0], "y": self.found[1], "by": self.best_signal[2]})
            if self.hazards:
                hx, hy, _ = self.hazards[self._hz_turn % len(self.hazards)]
                self._hz_turn += 1
                out.append({"type": "HAZARD", "x": hx, "y": hy})
        return cmd, out


class _LongRoster:
    """Silence alone does not drop a teammate for 10 s.

    The stock agent forgets a teammate after 1.5 s without a heartbeat, which
    is seven heartbeats in a row, and everything it shares hangs on that one
    test (`_heard`): the lane split, the pace of the line, dig-slot claims,
    who to walk round. At 80 % loss seven misses in a row happen a fifth of
    the time.

    The abreast rule (`_line_y`) keeps the stock 1.5 s: pacing against a
    position that is several seconds old makes every robot think it is ahead.
    """

    ROSTER_AFTER = 10 * POLICY_HZ

    def _heard(self, t: int) -> dict:
        return {r: i for r, i in self.roster.items() if r not in self.lost and t - i["t"] <= self.ROSTER_AFTER}

    def _line_y(self, t: int, me) -> float:
        order = self._in_line(t, me)
        if self.id not in order:
            return me[1]
        ys = [me[1]] + [self.roster[r]["y"] for r in order if r != self.id and t - self.roster[r]["t"] <= LOST_AFTER]
        return float(np.mean(ys))


class LongMemoryAgent(_LongRoster, RobotAgent):
    """The stock agent with the longer roster memory and nothing else: the smallest candidate."""


class PatientAgent(_LongRoster, RepeatingAgent):
    """Longer roster memory AND one-shot facts repeated on the heartbeat.

    A fallen robot says DOWN once and then goes quiet, so with the stock agent
    the short timeout is what closes the line when that one DOWN is missed.
    Repeating DOWN is what makes a long timeout safe when robots do fall.
    """


class SteadyAgent(PatientAgent):
    """PatientAgent, and a teammate that has gone quiet is assumed to be where the plan put it.

    Two things still move the lanes at 95 % loss with the longer memory. At the
    start nobody has been heard yet (a first heartbeat takes 4 s on average), so
    every robot believes it is alone and heads for the middle of the field. And
    `_in_line` drops a sweeper that is 3 m behind the front: a heartbeat 6 s old
    reads as 3 m behind. Here
      - the roster starts filled with the start line (every agent knows `n` and
        the rule that put the team in equal lanes);
      - a sweeper whose last heartbeat is older than 1.5 s is taken to have kept
        pace with me, until the roster memory runs out.
    """

    def __init__(self, rid: int, n: int, start_x: float, coordinated: bool):
        super().__init__(rid, n, start_x, coordinated)
        x0, x1, y0, _ = FIELD
        for k in range(n):
            if k != rid:
                self.roster[k] = {"x": x0 + (k + 0.5) * (x1 - x0) / n, "y": y0 - 1.0, "role": "sweep", "slot": None, "t": 0}

    def _in_line(self, t: int, me) -> list:
        ys = {self.id: me[1]}
        xs = {self.id: me[0]}
        for r, i in self._heard(t).items():
            if i["role"] in ("sweep", "backoff"):
                ys[r] = i["y"] if t - i["t"] <= LOST_AFTER else max(i["y"], me[1])
                xs[r] = i["x"]
        front = max(ys.values())
        return sorted((r for r in ys if ys[r] >= front - STRAGGLER_GAP), key=lambda r: (xs[r], r))


class RelayAgent(PatientAgent):
    """PatientAgent, and the heartbeat carries what I know of the team.

    Under a range limit the planner's premise fails: agents no longer hold the
    same roster, so the same rule gives them different lanes and different
    views of who claimed which dig slot. Here every heartbeat also carries the
    sender's roster (with the age of each entry), the ids it knows are down,
    and the best beacon signal it has heard. News crosses the line hop by hop
    at 5 Hz as long as the chain of neighbours is connected.
    """

    def step(self, t, pose, down, reading, inbox, ice_event=None):
        relayed = []
        for m in inbox:
            if m["type"] != "POSE":
                continue
            for rid, x, y, role, slot, age in m.get("roster", ()):
                seen = t - age
                if rid != self.id and (rid not in self.roster or self.roster[rid]["t"] < seen):
                    self.roster[rid] = {"x": x, "y": y, "role": role, "slot": slot, "t": seen}
            for rid in m.get("lost", ()):
                if rid != self.id:
                    self.lost.add(rid)
            sig = m.get("sig")
            if sig is not None and sig[3] != self.id:
                relayed.append({"type": "SIGNAL", "range": sig[0], "x": sig[1], "y": sig[2], "from": sig[3]})
        # relayed signals first: anything heard directly this tick is newer and is applied after them
        cmd, out = super().step(t, pose, down, reading, relayed + list(inbox), ice_event)
        if not down:
            for m in out:
                if m["type"] == "POSE":
                    m["roster"] = [(rid, i["x"], i["y"], i["role"], i.get("slot"), t - i["t"]) for rid, i in self.roster.items()
                                   if rid not in self.lost and t - i["t"] <= self.ROSTER_AFTER]  # fmt: skip
                    m["lost"] = sorted(self.lost)
                    if self.best_signal is not None and self.found is None:
                        rng_meas, est, who = self.best_signal
                        m["sig"] = (rng_meas, est[0], est[1], who)
        return cmd, out


AGENTS = {"stock": RobotAgent, "repeat": RepeatingAgent, "memory": LongMemoryAgent, "patient": PatientAgent, "steady": SteadyAgent, "relay": RelayAgent}


# -- instrumentation --------------------------------------------------------


class Probe:
    """Read-only: what each agent believes against what is true. Call `after_tick()` once per tick."""

    def __init__(self, mission: Mission):
        self.m = mission
        n = mission.n
        self.pair_ticks = 0  # (standing agent, standing teammate) pairs, summed over ticks after warm-up
        self.false_lost_ticks = 0  # ...of which the agent's `_heard` does not contain the teammate
        self.false_lost_events = 0  # times a teammate that had been heard dropped out of `_heard` while standing
        self.never_heard_ticks = 0  # false-lost because no heartbeat ever arrived (range limit)
        self._was_false = set()
        self.sweep_ticks = 0
        self.lane_wrong_ticks = 0  # sweeping agent whose lane differs from the lane the true line gives
        self.lane_err_sum = 0.0
        self.lane_changes = 0  # times a sweeping agent's lane moved by more than 5 cm between ticks
        self._lane = [None] * n
        self._down = np.zeros(n, dtype=bool)
        self.falls = []
        self.ice_events = []  # slip-reflex detections
        self.backoff_ticks = 0
        self.first_signal_tick = None
        # dig-slot claims: pairs of standing helpers whose claimed positions are the same or adjacent
        self.claim_pair_ticks = 0  # pairs of helpers that both hold a claim, summed over ticks
        self.conflict_ticks = 0
        self.conflict_same_ticks = 0  # ...of which the two claimed positions coincide (under 0.3 m apart)
        self.conflict_parked_ticks = 0  # ...of which both robots are already holding
        self._conflict_run = {}
        self.conflict_longest = 0
        self.conflict_events = 0
        self.conflict_log = []
        self.stale_finder_claim_ticks = 0  # the pinpointing robot still advertising a dig slot from before
        self.min_sep = float("inf")  # closest two standing robots ever got, metres
        self.close_ticks = 0  # pair-ticks under 0.45 m: bodies touching or about to

    def _patch(self, x, y, margin=0.6):
        best = None
        for k, p in enumerate(self.m.scenario.patches):
            if p.contains(x, y, margin=margin):
                d = math.hypot(x - p.x, y - p.y)
                if best is None or d < best[0]:
                    best = (d, k)
        return None if best is None else best[1]

    def _ideal_lane(self, rid: int, poses, standing) -> float:
        """`RobotAgent._lane_x` evaluated on the true poses and current roles of every standing robot."""
        x0, x1, _, _ = FIELD
        agents = self.m.agents
        line = [r for r in standing if r == rid or agents[r].role in ("sweep", "backoff")]
        front = max(poses[r][1] for r in line)
        order = sorted((r for r in line if poses[r][1] >= front - STRAGGLER_GAP), key=lambda r: (poses[r][0], r))
        if rid not in order:
            return float(np.clip(poses[rid][0], x0 + 0.5, x1 - 0.5))
        return x0 + (order.index(rid) + 0.5) * (x1 - x0) / len(order)

    def after_tick(self):
        m, sim = self.m, self.m.sim
        t = m.t
        alive = ~sim.down
        poses = [sim.pose(r) for r in range(m.n)]
        agents = m.agents

        for r in range(m.n):  # falls, and whether the ice had already been paid for
            if sim.down[r] and not self._down[r]:
                self._down[r] = True
                x, y = poses[r][0], poses[r][1]
                patch = self._patch(x, y)
                earlier = [f for f in self.falls + self.ice_events if patch is not None and f["patch"] == patch and f["robot"] != r]
                knew = any(math.hypot(h[0] - x, h[1] - y) < h[2] + 0.5 for h in agents[r].hazards)
                self.falls.append({
                    "robot": r, "t_s": round(t / POLICY_HZ, 1), "x": round(x, 2), "y": round(y, 2), "patch": patch,
                    "role": agents[r].role, "repeat": bool(earlier), "knew_hazard": bool(knew),
                })  # fmt: skip
        for r, e in enumerate(m._pending_ice):  # this tick's reflex detections, not yet seen by the agent
            if e is not None:
                patch = self._patch(e["x"], e["y"], margin=0.8)
                earlier = [f for f in self.falls + self.ice_events if patch is not None and f["patch"] == patch and f["robot"] != r]
                knew = any(math.hypot(h[0] - e["x"], h[1] - e["y"]) < h[2] + 0.5 for h in agents[r].hazards)
                self.ice_events.append({
                    "robot": r, "t_s": round(t / POLICY_HZ, 1), "x": round(e["x"], 2), "y": round(e["y"], 2), "patch": patch,
                    "repeat": bool(earlier), "knew_hazard": bool(knew),
                })  # fmt: skip
        if m.reflex is not None:
            self.backoff_ticks += int(m.reflex.active.sum())

        if self.first_signal_tick is None and any(a.best_signal is not None for a in agents):
            self.first_signal_tick = t
        standing = [r for r in range(m.n) if alive[r]]
        for i, r in enumerate(standing):
            for q in standing[i + 1:]:
                d = math.hypot(poses[r][0] - poses[q][0], poses[r][1] - poses[q][1])
                self.min_sep = min(self.min_sep, d)
                self.close_ticks += d < 0.45
        if t <= WARMUP:
            return

        for a in agents:
            if not alive[a.id] or not a.coordinated:
                continue
            heard = a._heard(t)
            for r in standing:
                if r == a.id:
                    continue
                self.pair_ticks += 1
                key = (a.id, r)
                if r in heard:
                    self._was_false.discard(key)
                    continue
                self.false_lost_ticks += 1
                if r not in a.roster:
                    self.never_heard_ticks += 1
                elif key not in self._was_false:
                    self.false_lost_events += 1
                self._was_false.add(key)
            if a.role == "sweep":
                self.sweep_ticks += 1
                lane = a._lane_x(t, poses[a.id])
                err = abs(lane - self._ideal_lane(a.id, poses, standing))
                self.lane_err_sum += err
                self.lane_wrong_ticks += err > 0.05
                if self._lane[a.id] is not None and abs(lane - self._lane[a.id]) > 0.05:
                    self.lane_changes += 1
                self._lane[a.id] = lane
            else:
                self._lane[a.id] = None

        # dig-slot claims. Each helper measures its slot from its own idea of the centre, so compare positions.
        claims = {}
        for a in agents:
            if not alive[a.id] or not a.coordinated or a.slot is None or a.best_signal is None:
                continue
            if a.best_signal[2] == a.id:
                self.stale_finder_claim_ticks += 1
                continue
            if a.role in ("converge", "hold"):
                claims[a.id] = a._slot_xy(a.slot, a.found or a.best_signal[1])
        spacing = 2 * 1.1 * math.sin(math.pi / N_SLOTS)  # neighbouring slots, 0.57 m; a legal pair is two slots apart, 1.1 m
        ids = sorted(claims)
        live = set()
        for i, r in enumerate(ids):
            for q in ids[i + 1:]:
                self.claim_pair_ticks += 1
                d = math.hypot(claims[r][0] - claims[q][0], claims[r][1] - claims[q][1])
                if d < 1.5 * spacing:  # same or neighbouring position
                    key = (r, q)
                    live.add(key)
                    self.conflict_ticks += 1
                    self.conflict_same_ticks += d < 0.3
                    self.conflict_parked_ticks += agents[r].role == "hold" and agents[q].role == "hold"
                    if key not in self._conflict_run:
                        self._conflict_run[key] = [t, 0, d]
                        self.conflict_events += 1
                    run = self._conflict_run[key]
                    run[1] += 1
                    run[2] = min(run[2], d)
                    self.conflict_longest = max(self.conflict_longest, run[1])
        for key in [k for k in self._conflict_run if k not in live]:
            self._close_conflict(key)

    def _close_conflict(self, key):
        t0, ticks, dmin = self._conflict_run.pop(key)
        if ticks >= POLICY_HZ // 2 and len(self.conflict_log) < 20:
            self.conflict_log.append({"pair": list(key), "from_s": round(t0 / POLICY_HZ, 1), "for_s": round(ticks / POLICY_HZ, 1), "closest_m": round(dmin, 2)})

    def report(self) -> dict:
        m, sim = self.m, self.m.sim
        b = m.scenario.beacon
        at_end = sorted(self._conflict_run)
        for key in at_end:
            self._close_conflict(key)
        ends = []
        for r, a in enumerate(m.agents):
            x, y, _ = sim.pose(r)
            ends.append({
                "role": a.role, "x": round(x, 2), "y": round(y, 2), "to_beacon": round(math.hypot(x - b[0], y - b[1]), 2),
                "has_signal": a.best_signal is not None, "has_found": a.found is not None, "hazards": len(a.hazards), "slot": a.slot,
            })  # fmt: skip
        metres = 0.0
        for r in range(m.n):
            pts = [row[r] for row in m.track if row[r] is not None]
            metres += sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))
        out = {
            "robot_metres": round(metres, 1),
            "falls": self.falls,
            "falls_off_ice": sum(f["patch"] is None for f in self.falls),
            "ice_events": self.ice_events,
            "ice_repeat_uninformed": sum(e["repeat"] and not e["knew_hazard"] for e in self.ice_events + self.falls),
            "backoff_s": round(self.backoff_ticks / POLICY_HZ, 1),
            "ends": ends,
            "standing_no_signal": sum(1 for r, e in enumerate(ends) if not sim.down[r] and not e["has_signal"]),
            "t_first_signal_s": None if self.first_signal_tick is None else round(self.first_signal_tick / POLICY_HZ, 1),
            "min_sep_m": None if math.isinf(self.min_sep) else round(self.min_sep, 2),
            "close_s": round(self.close_ticks / POLICY_HZ, 1),
        }
        if m.mode == "coordinated":
            out.update({
                "pair_ticks": self.pair_ticks,
                "false_lost_ticks": self.false_lost_ticks,
                "never_heard_ticks": self.never_heard_ticks,
                "false_lost_events": self.false_lost_events,
                "sweep_ticks": self.sweep_ticks,
                "lane_wrong_ticks": int(self.lane_wrong_ticks),
                "lane_err_mean_m": round(self.lane_err_sum / self.sweep_ticks, 3) if self.sweep_ticks else None,
                "lane_changes": self.lane_changes,
                "slot": {
                    "claim_pair_s": round(self.claim_pair_ticks / POLICY_HZ, 1),
                    "conflict_s": round(self.conflict_ticks / POLICY_HZ, 1),
                    "conflict_same_s": round(int(self.conflict_same_ticks) / POLICY_HZ, 1),
                    "conflict_parked_s": round(int(self.conflict_parked_ticks) / POLICY_HZ, 1),
                    "conflict_events": self.conflict_events,
                    "conflict_longest_s": round(self.conflict_longest / POLICY_HZ, 1),
                    "conflict_at_end": [list(k) for k in at_end],
                    "conflicts": self.conflict_log,
                    "stale_finder_claim_s": round(self.stale_finder_claim_ticks / POLICY_HZ, 1),
                },
            })  # fmt: skip
        return out


def run_mission(seed: int, mode: str = "coordinated", n_robots: int = 4, loss: float = 0.0, range_m: float | None = None,
                agent: str = "stock", bus: str = "lossy", reflex: bool = True) -> dict:  # fmt: skip
    """One mission. `bus="stock"` uses fleet_search.Bus untouched (the control for LossyBus at p=0, R=inf)."""
    t0 = time.time()
    radio = None
    if mode == "coordinated" and bus == "lossy":
        radio = LossyBus(n_robots, loss=loss, range_m=range_m, seed=seed)
    mission = Mission(seed, mode, n_robots, reflex=reflex, bus=radio)
    if radio is not None:
        radio.attach(mission.sim)
    if agent != "stock":
        mission.agents = [AGENTS[agent](r, n_robots, mission.scenario.starts[r][0], mode == "coordinated") for r in range(n_robots)]
    t_build = time.time() - t0
    probe = Probe(mission)
    while not mission.done:
        mission.tick()
        probe.after_tick()
    r = mission.result()
    r.update({"n_robots": n_robots, "loss": loss, "range_m": range_m, "agent": agent, "bus": bus if mode == "coordinated" else "off"})
    r.update(probe.report())
    if radio is not None:
        r["radio"] = radio.stats
        if agent == "stock":  # the candidate agents repeat these, so "missed once" means nothing for them
            shots = {k: {"sent": 0, "missed_by_someone": 0, "missed_by_all": 0} for k in ONE_SHOT}
            for _tick, typ, _sender, missed, others in radio.one_shots:
                s = shots[typ]
                s["sent"] += 1
                s["missed_by_someone"] += bool(missed)
                s["missed_by_all"] += bool(missed) and len(missed) >= others
            r["one_shots"] = shots
    r["wall_s"] = round(time.time() - t0, 1)
    r["build_s"] = round(t_build, 1)
    return r


# -- cells -----------------------------------------------------------------


def cells_for(study: str) -> list[dict]:
    c = []
    if study == "comms":
        for p in (0.0, 0.5, 0.8, 0.95):
            c.append({"cell": f"loss {p:g}", "mode": "coordinated", "n_robots": 4, "loss": p, "range_m": None})
        for rng in (5.0, 3.0):
            c.append({"cell": f"range {rng:g} m", "mode": "coordinated", "n_robots": 4, "loss": 0.0, "range_m": rng})
        c.append({"cell": "independent", "mode": "independent", "n_robots": 4})
    elif study == "scale":
        # Stock Bus throughout. With the comms study's "loss 0" (LossyBus) at n = 4 this is also the control for LossyBus.
        for n in (3, 4, 5, 6):
            c.append({"cell": f"n={n} coordinated", "mode": "coordinated", "n_robots": n, "bus": "stock"})
        for n in (4, 6):
            c.append({"cell": f"n={n} coordinated, no reflex", "mode": "coordinated", "n_robots": n, "bus": "stock", "reflex": False})
        for n in (3, 4, 5, 6):
            c.append({"cell": f"n={n} independent", "mode": "independent", "n_robots": n})
    elif study == "fix":
        def fix(agent, loss=0.0, rng=None):
            radio = f"range {rng:g} m" if rng is not None else f"loss {loss:g}"
            c.append({"cell": f"{radio} +{agent}", "mode": "coordinated", "n_robots": 4, "loss": loss, "range_m": rng, "agent": agent})

        fix("memory", loss=0.8)
        fix("patient", loss=0.8)
        fix("relay", rng=5.0)
        fix("memory", loss=0.95)
        fix("patient", loss=0.95)
        fix("relay", rng=3.0)
        fix("relay", loss=0.0)
        fix("steady", loss=0.95)
        fix("steady", loss=0.8)
        fix("steady", loss=0.0)
    else:
        raise ValueError(study)
    return c


def references(row: dict) -> dict:
    """What a cell is paired against, as {label: predicate on rows}. Same seeds on both sides."""
    refs = {}
    if row["cell"].startswith("n="):  # team-size study
        n, mode, reflex = row["n_robots"], row["mode"], row["reflex"]
        if mode == "coordinated" and reflex:
            refs["independent, same team size"] = lambda r: r["cell"] == f"n={n} independent"
            if n != 4:
                refs["four robots"] = lambda r: r["cell"] == "n=4 coordinated"
        if mode == "coordinated" and not reflex:
            refs["reflex on, same team size"] = lambda r: r["cell"] == f"n={n} coordinated"
            if n != 4:
                refs["four robots, no reflex"] = lambda r: r["cell"] == "n=4 coordinated, no reflex"
    elif row["mode"] == "coordinated":
        loss, rng = row["loss"], row["range_m"]
        if row["agent"] != "stock":
            refs["stock agent, same radio"] = lambda r: (r["mode"] == "coordinated" and r["agent"] == "stock" and r["bus"] == "lossy"
                                                         and r["loss"] == loss and r["range_m"] == rng)  # fmt: skip
        if row["agent"] != "stock" or loss != 0.0 or rng is not None:
            refs["stock agent, perfect radio"] = lambda r: r["cell"] == "loss 0"
    return refs


def sign_p(k: int, m: int) -> float | None:
    """Two-sided exact sign test on k against m discordant pairs."""
    n = k + m
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(k, m) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 3)


def code_state() -> dict:
    repo = HERE.parent

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e}"

    frozen = repo / "COMMIT"  # a `git archive` of one commit unpacked outside the repo, so later edits cannot move it
    return {
        "commit": frozen.read_text().strip() if frozen.exists() else git("rev-parse", "HEAD"),
        "frozen_copy": frozen.exists(),
        "git_branch": "" if frozen.exists() else git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_diff_stat": "" if frozen.exists() else git("diff", "--stat"),
        "git_status_short": "" if frozen.exists() else git("status", "--short"),
        # the modules this process imported, hashed when it started: one process is one code state
        "sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16]
                   for f in ("fleet_search.py", "fleet_sim.py", "fleet_reflex.py", "g1_newton.py", "g1_collision.py", "fleet_comms.py")
                   if (HERE / f).exists()},  # fmt: skip
        "launched": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def med(xs):
    return None if not xs else round(float(np.median(xs)), 1)


def bounce(row: dict) -> int:
    """Most reflex detections by one robot on one patch. One is discovery; three or more is a robot
    walking back into ice it has already mapped."""
    count = {}
    for e in row["ice_events"]:
        key = (e["robot"], e["patch"])
        count[key] = count.get(key, 0) + 1
    return max(count.values(), default=0)


def mean(xs, nd=2):
    return None if not xs else round(float(np.mean(xs)), nd)


def summarise(rows: list, refs: dict | None = None) -> dict:
    n = len(rows)
    t_loc = [r["t_located_s"] for r in rows if r["located"]]
    t_asm = [r["t_assembled_s"] for r in rows if r["assembled"]]
    s = {
        "missions": n,
        "seeds": sorted(r["seed"] for r in rows),
        "located": sum(r["located"] for r in rows),
        "assembled": sum(r["assembled"] for r in rows),
        "robots_lost_mean": mean([r["robots_lost"] for r in rows]),
        "robots_lost_total": int(sum(r["robots_lost"] for r in rows)),
        "t_located_median_s": med(t_loc),
        "t_assembled_median_s": med(t_asm),
        "coverage_mean": mean([r["coverage"] for r in rows], 3),
        "robot_metres_mean": mean([r["robot_metres"] for r in rows], 1),
        "duration_mean_s": mean([r["duration_s"] for r in rows], 1),
        "timed_out": sum(r["duration_s"] >= 100.0 for r in rows),
        "messages_mean": mean([r["messages"] for r in rows], 0),
        "falls_off_ice": int(sum(r["falls_off_ice"] for r in rows)),
        "ice_events_mean": mean([len(r["ice_events"]) for r in rows]),
        "ice_repeat_uninformed": int(sum(r["ice_repeat_uninformed"] for r in rows)),
        "missions_with_bounce": sum(bounce(r) >= 3 for r in rows),
        "bounce_max": max(bounce(r) for r in rows),
        "seeds_with_bounce": sorted(r["seed"] for r in rows if bounce(r) >= 3),
        "backoff_mean_s": mean([r["backoff_s"] for r in rows], 1),
        "min_sep_min_m": min((r["min_sep_m"] for r in rows if r["min_sep_m"] is not None), default=None),
        "close_mean_s": mean([r["close_s"] for r in rows], 1),
        # missions that located the beacon, kept three robots standing and still did not assemble: a coordination failure, not attrition
        "located_3up_not_assembled": sum(1 for r in rows if r["located"] and not r["assembled"] and r["n_robots"] - r["robots_lost"] >= TEAM_SIZE),
        "under_3_standing": sum(1 for r in rows if r["n_robots"] - r["robots_lost"] < TEAM_SIZE),
    }
    co = [r for r in rows if "pair_ticks" in r]
    if co:
        pt = sum(r["pair_ticks"] for r in co)
        sw = sum(r["sweep_ticks"] for r in co)
        s.update({
            "false_lost_events_mean": mean([r["false_lost_events"] for r in co], 1),
            "false_lost_frac": round(sum(r["false_lost_ticks"] for r in co) / pt, 4) if pt else None,
            "never_heard_frac": round(sum(r["never_heard_ticks"] for r in co) / pt, 4) if pt else None,
            "lane_wrong_frac": round(sum(r["lane_wrong_ticks"] for r in co) / sw, 4) if sw else None,
            "lane_changes_mean": mean([r["lane_changes"] for r in co], 1),
            "slot": {
                "missions_with_two_claims": sum(r["slot"]["claim_pair_s"] > 0 for r in co),
                "missions_with_conflict": sum(r["slot"]["conflict_s"] > 0 for r in co),
                "missions_with_conflict_over_1s": sum(r["slot"]["conflict_longest_s"] >= 1.0 for r in co),
                "missions_with_same_slot": sum(r["slot"]["conflict_same_s"] > 0 for r in co),
                "missions_with_parked_conflict": sum(r["slot"]["conflict_parked_s"] > 0 for r in co),
                "missions_with_conflict_at_end": sum(bool(r["slot"]["conflict_at_end"]) for r in co),
                "conflict_mean_s": mean([r["slot"]["conflict_s"] for r in co], 1),
                "conflict_longest_s": max(r["slot"]["conflict_longest_s"] for r in co),
                "seeds_with_conflict_over_1s": sorted(r["seed"] for r in co if r["slot"]["conflict_longest_s"] >= 1.0),
                "stale_finder_claim_missions": sum(r["slot"]["stale_finder_claim_s"] > 0 for r in co),
            },
        })  # fmt: skip
    shots = [r["one_shots"] for r in rows if "one_shots" in r]
    if shots:
        s["one_shots"] = {k: {f: int(sum(x[k][f] for x in shots)) for f in ("sent", "missed_by_someone", "missed_by_all")} for k in ONE_SHOT}
    for label, ref_rows in (refs or {}).items():
        # a = the reference mission, b = this cell's mission on the same seed
        ref = {r["seed"]: r for r in ref_rows}
        both = [(ref[r["seed"]], r) for r in rows if r["seed"] in ref]
        if not both:
            continue
        gained = sum(b["assembled"] and not a["assembled"] for a, b in both)
        lost = sum(a["assembled"] and not b["assembled"] for a, b in both)
        more = sum(b["robots_lost"] > a["robots_lost"] for a, b in both)
        fewer = sum(b["robots_lost"] < a["robots_lost"] for a, b in both)
        dt = [b["t_assembled_s"] - a["t_assembled_s"] for a, b in both if a["assembled"] and b["assembled"]]
        s.setdefault("paired", {})[label] = {
            "pairs": len(both),
            "assembly_gained": gained,
            "assembly_lost": lost,
            "assembly_sign_p": sign_p(gained, lost),
            "locate_gained": sum(b["located"] and not a["located"] for a, b in both),
            "locate_lost": sum(a["located"] and not b["located"] for a, b in both),
            "robots_lost_diff_mean": mean([b["robots_lost"] - a["robots_lost"] for a, b in both]),
            "seeds_more_lost": more,
            "seeds_fewer_lost": fewer,
            "robots_lost_sign_p": sign_p(more, fewer),
            # time to assemble, on the seeds where both sides assembled
            "t_assembled_both": len(dt),
            "t_assembled_diff_median_s": med(dt),
            "t_assembled_slower": sum(d > 0 for d in dt),
            "t_assembled_faster": sum(d < 0 for d in dt),
        }
    return s


def table(summary: dict) -> str:
    def f(v, w, nd=1):
        return f"{'-':>{w}}" if v is None else (f"{v:>{w}.{nd}f}" if isinstance(v, float) else f"{v:>{w}}")

    head = (f"{'cell':<28}{'n':>3}{'located':>8}{'assembl':>8}{'lost':>6}{'t_loc':>7}{'t_asm':>7}{'t/out':>6}"
            f"{'metres':>8}{'cover':>6}{'ice':>5}{'bounce':>7}{'unheard':>8}{'laneBad':>8}{'slotClash':>10}")  # fmt: skip
    lines = [head, "-" * len(head)]
    for cell, s in summary.items():
        n = s["missions"]
        slot = s.get("slot")
        clash = "-" if slot is None else f"{slot['missions_with_conflict_over_1s']}/{slot['missions_with_two_claims']}"
        lines.append(
            f"{cell:<28}{n:>3}{str(s['located']) + '/' + str(n):>8}{str(s['assembled']) + '/' + str(n):>8}"
            f"{f(s['robots_lost_mean'], 6, 2)}{f(s['t_located_median_s'], 7)}{f(s['t_assembled_median_s'], 7)}{s['timed_out']:>6}"
            f"{f(s['robot_metres_mean'], 8)}{f(s['coverage_mean'], 6, 2)}{f(s['ice_events_mean'], 5, 1)}{s['missions_with_bounce']:>7}"
            f"{f(s.get('false_lost_frac'), 8, 3)}{f(s.get('lane_wrong_frac'), 8, 3)}{clash:>10}"
        )
    lines += [
        "lost: robots down per mission, mean. t_loc / t_asm: median seconds over the missions that located / assembled",
        "only, so a cell that only finishes its easy seeds looks fast. t/out: missions that ran to the 100 s limit.",
        "metres: robot-metres walked per mission, from the 5 Hz track. ice: slip-reflex detections per mission.",
        "bounce: missions where one robot set off the reflex three or more times on the same patch.",
        "unheard: fraction of (standing agent, standing teammate) tick-pairs where the agent had dropped the teammate.",
        "laneBad: fraction of sweep ticks on a lane other than the one the true line gives.",
        "slotClash: missions where two helpers claimed the same or neighbouring dig position for over 1 s, out of the",
        "missions where two helpers held claims at all.",
    ]
    return "\n".join(lines)


def code_key(header: dict) -> str:
    """The physics and the agents a job ran: hashes of the modules it imported."""
    sha = header.get("sha256", {})
    return "/".join(sha.get(f, "?") for f in ("fleet_search.py", "fleet_sim.py", "fleet_reflex.py"))


def load_rows(path: pathlib.Path) -> tuple[list, list]:
    """Missions and job headers from a log. Each mission is tagged with the code of the job that ran it."""
    rows, headers = [], []
    code = "?"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off when a job died
            if d.get("type") == "header":
                code = d.get("commit") or code_key(d)
                headers.append({**d, "log": path.name})
            else:
                rows.append({**d, "code": code})
    return rows, headers


def write_summary(study: str, log: pathlib.Path, out: pathlib.Path, extra_logs: list | None = None, borrow: list | None = None) -> dict:
    """Summaries are per commit: a cell is only ever paired with missions that ran the same code.

    `borrow` is a list of "LOG:CELL=NEWCELL": take CELL's missions from another study's log under a new name
    (the team-size study's four-robot independent cell is the radio study's independent cell)."""
    rows, headers = load_rows(log)
    order = [c["cell"] for c in cells_for(study)]
    for extra in extra_logs or []:
        r2, h2 = load_rows(pathlib.Path(extra))
        rows += r2
        headers += h2
        order += [c for c in dict.fromkeys(r["cell"] for r in r2) if c not in order]
    borrowed = []
    for spec in borrow or []:
        path, _, names = spec.partition(":")
        old, _, new = names.partition("=")
        r2, h2 = load_rows(pathlib.Path(path))
        took = [{**r, "cell": new, "borrowed_from": f"{pathlib.Path(path).name}:{old}"} for r in r2 if r["cell"] == old]
        rows += took
        headers += [h for h in h2 if h not in headers]
        borrowed.append({"cell": new, "from": f"{pathlib.Path(path).name}:{old}", "missions": len(took)})
    side = log.with_suffix(".notes.json")
    side = json.loads(side.read_text()) if side.exists() else {}
    # The same (code, cell, seed) run by two jobs is a free determinism check: keep the first, count the rest.
    seen, kept, repeats = {}, [], {"missions_run_twice": 0, "identical": 0}
    same = ("located", "t_located_s", "assembled", "t_assembled_s", "robots_lost", "coverage", "messages", "duration_s", "robot_metres", "falls", "ends")
    for r in rows:
        key = (r["code"], r["cell"], r["seed"])
        if key in seen:
            repeats["missions_run_twice"] += 1
            repeats["identical"] += all(r[k] == seen[key][k] for k in same)
            continue
        seen[key] = r
        kept.append(r)
    rows = kept
    summary = {}
    for code in sorted({r["code"] for r in rows}):
        here = [r for r in rows if r["code"] == code]
        summary[code] = {}
        for cell in order:
            mine = [r for r in here if r["cell"] == cell]
            if mine:
                refs = {label: [r for r in here if pred(r)] for label, pred in references(mine[0]).items()}
                summary[code][cell] = summarise(mine, refs)
    doc = {
        "study": study,
        "what": __doc__.split("\n\n")[3 if study == "comms" else 4].strip() if study in ("comms", "scale") else study,
        "commits": sorted({r["code"] for r in rows}),
        "notes": side.get("notes", []),
        "field": FIELD, "team_size": TEAM_SIZE, "lost_after_ticks": LOST_AFTER, "heartbeat_ticks": HEARTBEAT,
        "hazard_radius_m": HAZARD_RADIUS, "n_slots": N_SLOTS, "straggler_gap_m": STRAGGLER_GAP,
        "borrowed_cells": borrowed,
        "repeated_missions": repeats,
        "jobs": headers,
        "summary": summary,
        "rows": rows,
    }  # fmt: skip
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    for code, cells in summary.items():
        print(f"\ncommit {code}")
        print(table(cells))
    print(f"\nwrote {out}  ({len(rows)} missions)")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, choices=("comms", "scale", "fix"))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=FIRST_SEED)
    ap.add_argument("--seed-list", default=None, help="comma-separated seeds, instead of --seed-start and --seeds")
    ap.add_argument("--cells", default=None, help="semicolon-separated cell names to run (default: all of the study)")
    ap.add_argument("--log", default=None, help="mission log, .jsonl, appended to and resumed from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--also", action="append", default=[], help="extra mission logs to fold into the summary")
    ap.add_argument("--borrow", action="append", default=[], help="LOG:CELL=NEWCELL, a cell taken from another study's log")
    ap.add_argument("--summarise", action="store_true", help="do not run anything, rebuild the summary from the log")
    args = ap.parse_args()

    results = HERE / "results"
    name = {"comms": "fleet_comms", "scale": "fleet_scale", "fix": "fleet_comms_fix"}[args.study]
    log = pathlib.Path(args.log) if args.log else results / f"{name}.jsonl"
    out = pathlib.Path(args.out) if args.out else results / f"{name}.json"
    if args.summarise:
        write_summary(args.study, log, out, args.also, args.borrow)
        return 0

    cells = cells_for(args.study)
    if args.cells:
        want = [c.strip() for c in args.cells.split(";")]
        unknown = [w for w in want if w not in {c["cell"] for c in cells}]
        if unknown:
            raise SystemExit(f"unknown cells {unknown}; have {[c['cell'] for c in cells]}")
        cells = [c for w in want for c in cells if c["cell"] == w]
    seeds = [int(s) for s in args.seed_list.split(",")] if args.seed_list else list(range(args.seed_start, args.seed_start + args.seeds))
    done_rows, _ = load_rows(log)
    done = {(r["cell"], r["seed"]) for r in done_rows}
    log.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "header", "study": args.study, "cells": [c["cell"] for c in cells], "seeds": seeds, **code_state()}
    with log.open("a") as fh:
        fh.write(json.dumps(header) + "\n")
    print(json.dumps(header), flush=True)
    t0 = time.time()
    todo = [(s, c) for s in seeds for c in cells if (c["cell"], s) not in done]  # seed-major: a killed job leaves whole seeds
    for k, (seed, c) in enumerate(todo):
        kw = {key: v for key, v in c.items() if key != "cell"}
        r = {"cell": c["cell"], **run_mission(seed, **kw)}
        line = json.dumps(r)
        with log.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        print(f"# {k + 1}/{len(todo)}  {time.time() - t0:.0f} s elapsed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
