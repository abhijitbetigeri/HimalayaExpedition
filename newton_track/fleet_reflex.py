"""Ice reflex for the fleet: feel the ice, stop, back out, hand control back.

Sits between the mission layer (`fleet_search.py`) and `FleetSim.step`:

    reflex = IceReflex(sim)
    cmd_out, events = reflex.filter(cmd_in)   # once per tick, BEFORE sim.step(cmd_out)
    sim.step(cmd_out)

Detection is proprioceptive. It reads body poses and velocities from
`sim.state_0` and body masses from `sim.model`, which a real robot gets from leg
kinematics, its IMU and its own CAD. It never reads `sim.on_ice`, `sim.patches`
or contact forces. Two measured facts make it work:

1. Ice cannot push sideways. The net ground reaction over one tick is the rate
   of change of whole-body momentum plus weight. On snow (mu 0.4) this policy's
   gait (four steps a second per foot) asks the ground for 0.14 to 0.3 of the
   vertical load sideways on most ticks. On ice (mu 0.08) the ratio sits at 0.06
   to 0.10.
2. Some foot should be still. In a walking gait the slower of the two feet is
   the planted one, so its horizontal speed is the support-foot slip with no
   contact detector. On its own this false-triggers in turns (the policy pivots
   and scuffs on snow); gated by fact 1 it does not.

Evidence = distance the slower foot slid, summed over the ticks of the last
0.2 s on which the robot was supported and the traction ratio was under
`u_max`. Ice is confirmed when that exceeds `slip_trigger`.

Once confirmed, the reflex takes the command: zero for 0.9 s, back at 0.25 m/s
until the pelvis is 0.6 m on the snow side of the estimated edge, zero for 0.5 s,
then it releases. About 4.6 s in all. If the foot that slid was the left one, it
also steps right at 0.15 m/s while it backs (why only that side: see below).

Measured, this version, on harness seeds nothing was chosen on (300 and up; 200
to 231 were a first fresh set, used up, see `--first-seed`). "Before" is the
previous version (0.3 s pause, straight back) on the same seeds. The committed
results/fleet_reflex.json is that previous version on seeds 10 to 25: regenerate.

- Snow only, 79.6 robot-minutes of mission-like commands (seeds 200-204 and
  300-304): no alarm, tick for tick the same as before, as it must be: nothing
  changed ahead of the first alarm. Before, on seeds 10-14: 1 alarm in 39.8
  robot-minutes, 0.28 s before a fall on plain snow that happens on the same tick
  with the reflex disabled, so a fall misread as ice.
- The four published approaches (0.5 and 0.25 m/s, square-on and at 45 degrees),
  128 paired encounters with a mu 0.08 patch, seeds 300-315: 126 fall within 18 s
  without the reflex, 2 with it (before: 3); 126 end upright with both feet on
  snow and control released. All 128 are detected before the pelvis reaches the
  edge (a toe on the ice is enough), and the reported edge is within 0.2 m of
  the true one (median within 0.03 m).
- Square-on at 0.5 m/s only (`--cells headon`), which is the fleet's sweep and
  where it loses robots: 256 encounters, seeds 300-331: 6 fall (before: 24; 21
  against 3 discordant pairs). The published mix hid this: three of its four
  approaches almost never fail.
- Along the edge with one foot on the ice (`--cells graze`), 64 a side, seeds
  300-315. Left foot on the ice: released with a foot still on the ice 2 of 64
  (before: 16). Right foot: 0 (before: 0). Nobody falls either way in this
  harness, because its stand-in mission stands still after an alarm; in the
  fleet a robot released on the ice walks on and went down twice in 120 missions.

Other ice (previous version, seeds 10-25, 128 per cell, falls without / with):

    friction   2 mm proud (as committed)   flush (`--flush`)
    0.05       128 / 15                    128 / 96
    0.08       116 / 1                     102 / 62
    0.12        23 / 0                       1 / 0
    0.15         0 / 0                       0 / 0
    0.40         0 / 0, no alarm             0 / 0, no alarm

- It does not work on flush ice at sweep speed: at 0.08, 62 of the 64 encounters
  at 0.5 m/s fall and none of the 64 at 0.25 m/s. Flush, the ice is felt about
  12 cm later (pelvis 6 cm past the edge instead of 6 cm short of it): a sole
  across a flush edge still grips with the part on snow, where a toe on the
  proud lip carries the foot's load on ice at once. This version on the same
  flush cells loses 36 of 128 at 0.08 and 81 at 0.05 (the longer pause saves
  most of the 45 degree ones), but square-on at 0.5 m/s it is still 32 of 32.
- On the proud patch at 0.05 this version loses 19 of 128 against 15, the
  difference at 0.5 m/s and 45 degrees (9 of 32 against 3; same seeds, so a
  hint, not a result): standing longer may cost something on ice that slippery.
- Every encounter at 0.12 and at 0.15 raises an alarm too (128 of 128), though
  nobody falls at 0.15 without the reflex and at 0.12 only on the proud patch,
  at 45 degrees. A patch at snow friction raises none.
- The proud patch is less slippery than its number: the sole sinks up to 7 mm,
  further than the 2 mm lip, so about 6 % of the load lands on the snow plane
  underneath. Measured friction under an upright foot: 0.099 for a nominal 0.08.

What it does not cover. The mission must act on the event: a stand-in mission
that ignored it and kept walking at the patch for 30 s bounced off 4 times and
lost 5 of 16 robots. A robot can still be released with a foot on the ice after
an oblique or along-the-edge entry (backing 0.6 m along its own path is not
0.6 m from the ice), and for a second after a release the detector is not armed
(`arm_ticks`): one fleet robot fell in that second. The longer pause made this
WORSE at 0.5 m/s and 45 degrees: released with a foot on the ice 5 of 32, against
1 of 32 before (seeds 300-315; not traced, the suspicion is drift while it
stands on the ice). All five ended upright and off the ice. Not tested: a robot that is
standing still or turning when it meets the ice (detection needs a foot sliding
faster than `slide_min`), a gradual edge, slopes.

Why the pause is 0.9 s. Stopping from 0.43 m/s takes this policy about 0.5 s and
9 cm, ice or not, so a square-on robot ends with both soles on the ice whatever
the reflex does. Asking it to walk backwards while it is still sliding forwards
is what dropped it. Square-on at 0.5 m/s, seeds 10-25, 128 encounters each,
falls by pause length: 0.1 s 20, 0.3 s 11, 0.6 s 7, 0.9 s 5, 1.2 s 6, 1.8 s 7
(with the long pauses a few go down while standing).

Why the sidestep is one-sided. This policy turns clockwise when it walks
backwards (20 to 30 degrees over one back-out, more with a foot slipping). With
the ice under its right foot that swings it away from the ice; with the ice
under its left foot it backs into it. Seeds 10-17, 32 a side, released with a
foot on the ice: straight back 0 right / 7 left; stepping right for a left foot
0 / 0. Stepping left for a right foot as well fixed nothing and fought the turn:
back-outs of 7.5 s median, 46 of 54 over 6 s, against 4.5 s (seeds 200-215).

Tried and dropped, with the measurement:
- Slower-foot speed with no traction gate: zero alarms walking straight, but
  roughly 0.6 to 1.2 per robot-minute overall, all of them in turns and in-place
  turns, at thresholds that still caught the ice (offline, 15 robot-minutes).
- Which foot is loaded from knee torque: the torque distributions with the foot
  loaded and unloaded overlap completely. From leg statics (J^T): 5700 N rms
  error; at this step rate the leg's own inertia swamps the load.
- Confirming at 0.025 m instead of 0.015 m: about 5 ticks later, and 10 of 48
  robots at 0.5 m/s fell during the back-out against 3 of 144. The price is
  margin: the highest evidence seen on snow in episodes with no fall is 0.0094 m.
- Stopping without backing out (at the 0.025 m trigger): 3 of 32 fell, 23 of 32
  were left standing with a foot on the ice.
- Square-on at 0.5 m/s, seeds 10-25, 128 each, against 11 falls for the 0.3 s
  pause: backing at 0.15 m/s 11, at 0.35 m/s 15, a 1 s ramp 11 (the back-out
  speed is not the lever); a backward command during the pause, -0.15 m/s 19,
  -0.3 m/s 30; easing the forward command down over 0.3 s instead of dropping it
  8 with a 0.9 s pause, over 0.6 s 29 (deeper onto the ice is worse than a hard
  stop).
- Heading hold while backing (`steer`), with the retreat leaned away from the
  sliding foot: no robot released on the ice on either side, but 4 of 32 fell
  with the ice under the right foot (0 or 1 without). Yaw commands with a foot
  on ice cost more than the drift they remove.
- Telling a one-foot contact from a two-foot one by which foot's sliding made
  the evidence: the two overlap (ratio of the lesser to the greater foot 0.0 to
  0.9 for one foot on the ice, 0.3 to 1.0 for two).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np

from g1_newton import POLICY_HZ

DT = 1.0 / POLICY_HZ
G = 9.81

WALK, BRAKE, BACK, SETTLE = "walk", "brake", "back", "settle"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class IceReflex:
    """Per-robot ice detector and back-out reflex. See the module docstring."""

    def __init__(
        self,
        sim,
        *,
        u_max: float = 0.15,  # traction ratio |F_horizontal| / F_vertical below which the ground "is not pushing back"
        slide_min: float = 0.10,  # m/s: slower-foot speed that counts as the support foot sliding
        pivot_radius: float = 0.10,  # m: a foot turning about a point inside its own sole is not sliding
        window: int = 10,  # ticks of evidence kept (0.2 s)
        slip_trigger: float = 0.015,  # m of traction-less sliding inside the window that confirms ice
        support_min: float = 0.3,  # fraction of body weight on the ground for a tick to count
        arm_ticks: int = 50,  # ignore the first second after start (the spawn drop) and after a release
        brake_ticks: int = 45,  # zero command before backing: long enough for the slide to end. 15 cost falls, see docstring
        back_speed: float = 0.25,  # m/s backwards
        back_ramp: int = 25,  # ticks over which the backward command ramps up
        steer: bool = False,  # while backing, hold the heading and correct sideways drift. No measured gain
        clear: float = 0.6,  # m: back until the pelvis is this far on the snow side of the estimated edge
        back_timeout: int = 500,  # ticks: give up backing after 10 s and release
        settle_ticks: int = 25,  # zero command before handing control back
        edge_offset: float = 0.08,  # m from the sliding foot's ankle body to the ice edge, along the way it was going
        sidestep: float = 0.15,  # m/s to the right while backing, when the foot that slid is the left one (not used with steer)
    ):
        self.sim = sim
        self.n = n = sim.n
        self.B = B = sim.bodies_per_robot
        self.u_max, self.slide_min, self.pivot_radius = u_max, slide_min, pivot_radius
        self.window, self.slip_trigger, self.support_min = window, slip_trigger, support_min
        self.arm_ticks, self.brake_ticks, self.back_speed = arm_ticks, brake_ticks, back_speed
        self.clear, self.back_timeout, self.settle_ticks, self.edge_offset = clear, back_timeout, settle_ticks, edge_offset
        self.back_ramp, self.steer = back_ramp, steer
        self.sidestep = sidestep
        self._lat = np.zeros(sim.n)  # sidestep while backing: -1 to the right, 0 none (body frame, +y is left)

        self._mass = sim.model.body_mass.numpy()[:B].astype(np.float64)
        self._weight = float(self._mass.sum() * G)
        self._feet = list(sim.foot_local)

        self.active = np.zeros(n, dtype=bool)
        self.phase = [WALK] * n
        # Telemetry, for the film and for whoever wants to plot it. Not ground truth.
        self.traction = np.zeros(n)  # last traction ratio
        self.evidence = np.zeros(n)  # m of traction-less sliding in the window

        self._p = np.zeros((n, 3))  # whole-body linear momentum at the previous call
        self._p_t = -10  # sim.t of that momentum
        self._armed_in = np.full(n, arm_ticks)
        self._ev = np.zeros((n, window))  # ring of per-tick evidence
        self._k = 0
        self._trail = [collections.deque(maxlen=150) for _ in range(n)]  # pelvis x, y of the last 3 s
        self._phase_t = np.zeros(n, dtype=int)
        self._edge = np.zeros((n, 2))  # estimated ice edge
        self._retreat = np.zeros((n, 2))  # unit vector, the way back onto snow
        self._yaw_hold = np.zeros(n)

    # -- detection ----------------------------------------------------------

    def _sense(self):
        """Per robot: traction ratio, supported flag, slower-foot slide speed, and that foot's ankle x, y."""
        sim, n, B = self.sim, self.n, self.B
        bq = sim.state_0.body_q.numpy().reshape(n, B, 7)
        bqd = sim.state_0.body_qd.numpy().reshape(n, B, 6).astype(np.float64)
        p = np.einsum("b,nbk->nk", self._mass, bqd[:, :, :3])
        fresh = sim.t == self._p_t + 1  # momentum rate needs consecutive ticks
        force = (p - self._p) / DT
        force[:, 2] += self._weight
        self._p, self._p_t = p, sim.t
        fz = np.maximum(force[:, 2], 1.0)
        u = np.hypot(force[:, 0], force[:, 1]) / fz
        supported = (force[:, 2] > self.support_min * self._weight) & fresh
        fv = bqd[:, self._feet, :]
        speed = np.hypot(fv[:, :, 0], fv[:, :, 1]) - self.pivot_radius * np.abs(fv[:, :, 5])
        slow = np.argmin(speed, axis=1)
        idx = np.arange(n)
        slide = np.maximum(speed[idx, slow], 0.0)
        return u, supported, slide, bq[idx, [self._feet[f] for f in slow], :2]

    # -- the filter ---------------------------------------------------------

    def filter(self, cmd_in: np.ndarray):
        sim = self.sim
        cmd = np.array(cmd_in, dtype=np.float32, copy=True).reshape(self.n, 3)
        events = []
        u, supported, slide, slider_xy = self._sense()
        k = self._k = (self._k + 1) % self.window
        for r in range(self.n):
            if sim.down[r]:
                self.active[r] = False
                self.phase[r] = WALK
                self.evidence[r] = 0.0
                continue
            x, y, yaw = sim.pose(r)
            self.traction[r] = u[r]
            self._phase_t[r] += 1

            if self.phase[r] == WALK:
                self._trail[r].append((x, y))
                hit = supported[r] and slide[r] > self.slide_min and u[r] < self.u_max
                self._ev[r, k] = slide[r] * DT if hit else 0.0
                self.evidence[r] = self._ev[r].sum()
                if self._armed_in[r] > 0:
                    self._armed_in[r] -= 1
                elif self.evidence[r] > self.slip_trigger:
                    self._confirm(r, x, y, yaw, slider_xy[r])
                    events.append({"robot": r, "type": "ice", "x": float(self._edge[r, 0]), "y": float(self._edge[r, 1]), "t": int(sim.t)})

            if self.phase[r] == BRAKE:
                cmd[r] = 0.0
                if self._phase_t[r] >= self.brake_ticks:
                    self._enter(r, BACK)
            elif self.phase[r] == BACK:
                d = self._retreat[r]
                cleared = float((np.array([x, y]) - self._edge[r]) @ d)
                if cleared >= self.clear or self._phase_t[r] >= self.back_timeout:
                    self._enter(r, SETTLE)
                    cmd[r] = 0.0
                else:
                    v = self.back_speed * min(1.0, self._phase_t[r] / self.back_ramp) if self.back_ramp else self.back_speed
                    if self.steer:
                        h = np.array([math.cos(yaw), math.sin(yaw)])
                        left = np.array([-h[1], h[0]])
                        cmd[r] = (
                            v * float(np.clip(d @ h, -1.0, 0.2)),
                            float(np.clip(v * (d @ left), -0.1, 0.1)),
                            float(np.clip(2.0 * _wrap(self._yaw_hold[r] - yaw), -0.5, 0.5)),
                        )
                    else:
                        # Back and to the side, the sideways part ramped with the backward part.
                        cmd[r] = (-v, self._lat[r] * self.sidestep * v / self.back_speed, 0.0)
            elif self.phase[r] == SETTLE:
                cmd[r] = 0.0
                if self._phase_t[r] >= self.settle_ticks:
                    self._enter(r, WALK)
                    self.active[r] = False
                    self._armed_in[r] = self.arm_ticks
                    self._ev[r] = 0.0
                    self._trail[r].clear()
        return cmd, events

    def _enter(self, r: int, phase: str):
        self.phase[r] = phase
        self._phase_t[r] = 0

    def _confirm(self, r: int, x: float, y: float, yaw: float, slider_xy: np.ndarray):
        """Ice confirmed for robot r: estimate the edge, pick the way back, take the command."""
        h = np.array([math.cos(yaw), math.sin(yaw)])
        # The way it came: towards where the pelvis was about 0.4 m of path ago.
        here = np.array([x, y])
        d = -h
        for px, py in reversed(self._trail[r]):
            v = np.array([px, py]) - here
            if np.hypot(*v) >= 0.4:
                d = v / np.hypot(*v)
                break
        # The edge: just ahead of the foot that is sliding right now. The tick that confirms ice
        # always adds evidence, so the slower foot on this tick is the one standing on it, and a
        # foot starts to slide once its toe is on the ice. Taking the foot furthest along the
        # path instead put the edge 0.2 m short on 45 degree approaches (the other foot).
        self._edge[r] = slider_xy - self.edge_offset * d
        self._retreat[r] = d
        # This policy turns clockwise as it walks backwards (about 20 to 30 degrees over one back-out,
        # more with a foot slipping). With the ice on its right that swings it away. With the ice on its
        # left it backs INTO the ice, so when the sliding foot is the left one, step right as well.
        # Stepping left for a right foot was tried and dropped: see the docstring.
        left_of_pelvis = (slider_xy - here) @ np.array([-h[1], h[0]]) > 0.0
        self._lat[r] = -1.0 if left_of_pelvis else 0.0
        self._yaw_hold[r] = yaw
        self.active[r] = True
        self._enter(r, BRAKE)


# ===========================================================================
# Measurement. Everything below may read ground truth; nothing above does.
# ===========================================================================

RESULTS = Path(__file__).resolve().parent / "results" / "fleet_reflex.json"
PARAMS = ("u_max", "slide_min", "pivot_radius", "window", "slip_trigger", "support_min", "arm_ticks", "brake_ticks", "back_speed", "back_ramp", "steer", "clear", "back_timeout", "settle_ticks", "edge_offset", "sidestep")  # fmt: skip
SPEEDS = (0.5, 0.25)
ANGLES = (0.0, 45.0)  # degrees between the robot's path and the edge normal
GRAZE_RIGHT, GRAZE_LEFT = 90.0, -90.0  # walking along the edge with only the right / only the left foot on the ice
APPROACH = {GRAZE_RIGHT: "along the edge, right foot on ice", GRAZE_LEFT: "along the edge, left foot on ice"}
CELL_SETS = {
    "standard": [(v, a) for v in SPEEDS for a in ANGLES],  # the published table
    "headon": [(0.5, 0.0)] * 4,  # the encounter the fleet actually loses robots to: sweep speed, square-on
    "graze": [(0.5, GRAZE_RIGHT), (0.5, GRAZE_LEFT)] * 2,
}
ROBOTS = 8
FIRST_SEED = 300
ICE_SPACING = 12.0
SNOW_SPACING = 30.0
ICE_TICKS = 900  # 18 s
SNOW_TICKS = 3000  # 60 s
TOE = 0.12  # m, ankle body to toe. A foot counts as off the ice when its ankle is this far outside
SNOW_BEHAVIOURS = (
    "straight 0.5, heading hold",
    "straight 0.25, heading hold",
    "0.5 with yaw +1.0 / -1.0, swapped every 6 s",
    "0.25 with yaw -1.0 / +1.0, swapped every 6 s",
    "stop-start: 0.5 for 2 s, 0 for 2 s",
    "arc: 0.5 with yaw +0.3 / -0.3",
    "waypoints: walk 0.5, stop and turn in place to a new heading",
    "random segments: 0 / 0.25 / 0.5, yaw in [-1, 1], 0.5 to 3 s each",
)


def ice_layout():
    """One patch per robot, entered through its west edge (x = patch.x - patch.hx)."""
    from fleet_sim import IcePatch  # noqa: PLC0415

    return [IcePatch((r % 4) * ICE_SPACING + 1.5, (r // 4) * ICE_SPACING, 1.5, 2.5) for r in range(ROBOTS)]


def ice_cells(kind: str = "standard"):
    return CELL_SETS[kind] * (ROBOTS // 4)


def ice_starts(patches, cells, rng):
    starts = []
    for p, (_, ang) in zip(patches, cells):
        a = math.radians(ang)
        d = rng.uniform(1.0, 1.6)  # varies the gait phase at the edge
        yoff = rng.uniform(-0.3, 0.3)
        if abs(ang) >= 80.0:
            # Along the west edge, pelvis 5 to 17 cm outside it: northbound puts the right foot on the
            # ice, southbound the left. The two are not mirror images for this policy (see docstring).
            x = p.x - p.hx - (0.11 + 0.2 * yoff)
            starts.append((x, p.y - p.hy - d, math.pi / 2) if ang > 0 else (x, p.y + p.hy + d, -math.pi / 2))
        else:
            starts.append((p.x - p.hx - d * math.cos(a), p.y + yoff - d * math.sin(a), a))
    return starts


def flush_snowfield(builder, patches, snow_mu: float, ice_mu: float):
    """Stand-in for fleet_sim.add_snowfield with the ice FLUSH: coplanar boxes, ice tiles and the snow
    tiles round them, every top at z = 0. A flush box over the stock snow plane would not do: the sole
    would rest on both. Here a foot over the ice touches only ice (checked: the friction MuJoCo applies
    under the feet is the nominal one), and a foot across the edge shares its load between the two."""
    import newton  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    hz = 0.02
    xs = sorted({-25.0, 65.0} | {p.x - p.hx for p in patches} | {p.x + p.hx for p in patches})
    ys = sorted({-25.0, 40.0} | {p.y - p.hy for p in patches} | {p.y + p.hy for p in patches})
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
            ice = any(p.contains(cx, cy) for p in patches)
            cfg = newton.ModelBuilder.ShapeConfig(mu=ice_mu if ice else snow_mu, ke=5.0e4, kd=5.0e2, kf=1.0e3)
            builder.add_shape_box(
                body=-1, xform=wp.transform(wp.vec3(cx, cy, -hz)), hx=0.5 * (xs[i + 1] - xs[i]), hy=0.5 * (ys[j + 1] - ys[j]), hz=hz,
                cfg=cfg, label=f"{'ice' if ice else 'snow'}_{i}_{j}", custom_attributes={"mujoco:geom_priority": 2 if ice else 1},
            )  # fmt: skip


def snow_starts():
    return [((r % 4) * SNOW_SPACING, (r // 4) * SNOW_SPACING, 0.0) for r in range(ROBOTS)]


def reset(sim, starts, seed: int):
    """Put a built FleetSim back to t = 0 with new start poses. A build costs minutes on this
    machine, an episode costs seconds. Draws joint noise the way FleetSim.__init__ does.

    Checked: in a fresh process the same seed gives the same eight fall ticks whether it runs
    first, after another seed, or after a reflex episode, and two fresh processes agree."""
    import mujoco  # noqa: PLC0415
    import newton  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    q0 = np.zeros(sim.n * sim.nq, dtype=np.float32)
    for r, (x, y, yaw) in enumerate(starts):
        o = r * sim.nq
        q0[o : o + 3] = (x, y, 0.76)
        q0[o + 3 : o + 7] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        q0[o + 7 : o + sim.nq] = sim.default_q + rng.uniform(-0.03, 0.03, sim.nj)
    mujoco.mj_resetData(sim.solver.mj_model, sim.solver.mj_data)
    for s in (sim.state_0, sim.state_1):
        s.joint_q.assign(q0)
        s.joint_qd.zero_()
        newton.eval_fk(sim.model, s.joint_q, s.joint_qd, s)
    sim.control.joint_target_q.zero_()
    sim.prev_act[:] = 0.0
    sim.down[:] = False
    sim.t = 0
    sim._q = q0.copy()
    sim._qd = np.zeros(sim.n * sim.nv, dtype=np.float32)
    sim.slip[:] = 0.0


def foot_xy(sim, r: int):
    bq = sim.state_0.body_q.numpy()
    return [bq[r * sim.bodies_per_robot + i][:2].copy() for i in sim.foot_local]


def snow_commands(sim, t: int, rng, st):
    """What the mission layer asks for: forward 0 to 0.5, lateral 0, yaw rate within +-1."""
    cmd = np.zeros((sim.n, 3), dtype=np.float32)
    for r in range(sim.n):
        yaw = sim.pose(r)[2]
        k = r % 8
        if k == 0:
            cmd[r] = (0.5, 0, np.clip(2 * _wrap(st["h"][r] - yaw), -1, 1))
        elif k == 1:
            cmd[r] = (0.25, 0, np.clip(2 * _wrap(st["h"][r] - yaw), -1, 1))
        elif k == 2:
            cmd[r] = (0.5, 0, 1.0 if (t // 300) % 2 == 0 else -1.0)
        elif k == 3:
            cmd[r] = (0.25, 0, -1.0 if (t // 300) % 2 == 0 else 1.0)
        elif k == 4:
            cmd[r] = (0.5 if (t // 100) % 2 == 0 else 0.0, 0, 0)
        elif k == 5:
            cmd[r] = (0.5, 0, 0.3 if (t // 500) % 2 == 0 else -0.3)
        elif k == 6:
            if t >= st["until"][r]:
                st["h"][r] = _wrap(st["h"][r] + rng.uniform(-2.5, 2.5))
                st["until"][r] = t + 250
            err = _wrap(st["h"][r] - yaw)
            cmd[r] = (0.5 if abs(err) < 0.3 else 0.0, 0, np.clip(2 * err, -1, 1))
        else:
            if t >= st["until"][r]:
                st["seg"][r] = (rng.choice([0, 0.25, 0.5, 0.5]), 0, rng.uniform(-1, 1))
                st["until"][r] = t + rng.integers(25, 150)
            cmd[r] = st["seg"][r]
    return cmd


def run_snow(sim, seed: int, ticks: int = SNOW_TICKS, reflex_kwargs: dict | None = None) -> list[dict]:
    """One snow-only episode with the reflex in the loop. The mission ignores the alarms."""
    starts = snow_starts()
    reset(sim, starts, seed)
    rng = np.random.default_rng(1000 + seed)
    st = {"h": np.array([s[2] for s in starts]), "until": np.zeros(sim.n, int), "seg": np.zeros((sim.n, 3), np.float32)}
    reflex = IceReflex(sim, **(reflex_kwargs or {}))
    rec = [
        {"seed": seed, "behaviour": SNOW_BEHAVIOURS[r % 8], "alarm_ticks": [], "upright_ticks": 0, "fell_at": None, "max_evidence": 0.0}
        for r in range(sim.n)
    ]
    for t in range(ticks):
        cmd, events = reflex.filter(snow_commands(sim, t, rng, st))
        for e in events:
            rec[e["robot"]]["alarm_ticks"].append(e["t"])
        sim.step(cmd)
        for r in range(sim.n):
            if sim.down[r]:
                if rec[r]["fell_at"] is None:
                    rec[r]["fell_at"] = sim.t
            else:
                rec[r]["upright_ticks"] += 1
                rec[r]["max_evidence"] = max(rec[r]["max_evidence"], float(reflex.evidence[r]))
    return rec


def run_ice(sim, patches, seed: int, use_reflex: bool, ticks: int = ICE_TICKS, reflex_kwargs: dict | None = None, cells=None) -> list[dict]:
    """One batch of encounters: every robot walks at its patch under a heading hold.

    With the reflex, the stand-in mission stops asking a robot to advance once that
    robot has reported ice (the real mission replans around the hazard)."""
    cells = cells or ice_cells()
    starts = ice_starts(patches, cells, np.random.default_rng(1000 + seed))
    reset(sim, starts, seed)
    reflex = IceReflex(sim, **(reflex_kwargs or {})) if use_reflex else None
    n = sim.n
    edge_x = [p.x - p.hx for p in patches]
    rec = [
        {
            "seed": seed, "robot": r, "speed": cells[r][0], "angle_deg": cells[r][1], "reflex": use_reflex,
            "fell_at": None, "detected_at": None, "pelvis_past_edge_at_detection": None, "lead_ankle_past_edge_at_detection": None,
            "edge_estimate_error": None, "released_at": None, "max_pelvis_past_edge": -9.0, "touched_ice": False,
            "released_foot_on_ice": None,
        }
        for r in range(n)
    ]  # fmt: skip
    told = np.zeros(n, dtype=bool)
    for _ in range(ticks):
        cmd = np.zeros((n, 3), dtype=np.float32)
        for r in range(n):
            if not told[r]:
                cmd[r] = (cells[r][0], 0.0, np.clip(2.0 * _wrap(starts[r][2] - sim.pose(r)[2]), -1, 1))
        if reflex is not None:
            was_active = reflex.active.copy()
            cmd, events = reflex.filter(cmd)
            for e in events:
                r = e["robot"]
                told[r] = True
                if rec[r]["detected_at"] is None:
                    rec[r]["detected_at"] = e["t"]
                    rec[r]["pelvis_past_edge_at_detection"] = float(sim.pose(r)[0] - edge_x[r])
                    rec[r]["lead_ankle_past_edge_at_detection"] = float(max(f[0] for f in foot_xy(sim, r)) - edge_x[r])
                    rec[r]["edge_estimate_error"] = float(e["x"] - edge_x[r])  # + = estimate lies inside the ice
            for r in range(n):
                if was_active[r] and not reflex.active[r] and not sim.down[r] and rec[r]["released_at"] is None:
                    rec[r]["released_at"] = sim.t
                    rec[r]["released_foot_on_ice"] = bool(any(patches[r].contains(f[0], f[1]) for f in foot_xy(sim, r)))
        sim.step(cmd)
        for r in range(n):
            if sim.down[r]:
                if rec[r]["fell_at"] is None:
                    rec[r]["fell_at"] = sim.t
                continue
            x, y, _ = sim.pose(r)
            rec[r]["max_pelvis_past_edge"] = float(max(rec[r]["max_pelvis_past_edge"], x - edge_x[r]))
            if any(patches[r].contains(f[0], f[1], TOE) for f in foot_xy(sim, r)):
                rec[r]["touched_ice"] = True
    for r in range(n):
        x, y, _ = sim.pose(r)
        feet_off = not any(patches[r].contains(f[0], f[1], TOE) for f in foot_xy(sim, r))
        rec[r]["fell"] = bool(sim.down[r])
        rec[r]["end_pelvis_on_ice"] = bool(sim.on_ice(r))
        rec[r]["end_feet_off_ice"] = bool(feet_off)
        rec[r]["end_reflex_active"] = bool(reflex.active[r]) if reflex is not None else False
        # Saved: upright, both feet (to the toes) back on snow, and control handed back.
        rec[r]["saved"] = bool(use_reflex and not sim.down[r] and feet_off and not rec[r]["end_reflex_active"])
    return rec


def summarise_ice(records: list[dict]) -> list[dict]:
    rows = []
    cells = sorted({(x["speed"], x["angle_deg"]) for x in records}, key=lambda c: (-c[0], abs(c[1]), -c[1]))
    for v, a in cells:
        base = [x for x in records if x["speed"] == v and x["angle_deg"] == a and not x["reflex"]]
        refl = [x for x in records if x["speed"] == v and x["angle_deg"] == a and x["reflex"]]
        det = [x for x in refl if x["detected_at"] is not None]
        pel = [x["pelvis_past_edge_at_detection"] for x in det]
        ank = [x["lead_ankle_past_edge_at_detection"] for x in det]
        err = [x["edge_estimate_error"] for x in det]
        rel = [(x["released_at"] - x["detected_at"]) * DT for x in det if x["released_at"] is not None]
        rows.append({
            "speed": v, "angle_deg": a, "encounters": len(base),
            "falls_without": sum(x["fell"] for x in base),
            "falls_with": sum(x["fell"] for x in refl),
            "detected": len(det),
            "saved": sum(x["saved"] for x in refl),
            "upright_but_not_clear": sum((not x["fell"]) and (not x["saved"]) for x in refl),
            "released_with_a_foot_on_ice": sum(bool(x.get("released_foot_on_ice")) for x in refl),
            "pelvis_past_edge_at_detection_m": {"median": float(np.median(pel)), "max": float(np.max(pel))} if pel else None,
            "lead_ankle_past_edge_at_detection_m": {"median": float(np.median(ank)), "max": float(np.max(ank))} if ank else None,
            "edge_estimate_error_m": {"median": float(np.median(err)), "min": float(np.min(err)), "max": float(np.max(err))} if err else None,
            "max_pelvis_past_edge_with_m": float(np.max([x["max_pelvis_past_edge"] for x in refl])) if refl else None,
            "seconds_detection_to_release": {"median": float(np.median(rel)), "max": float(np.max(rel))} if rel else None,
        })  # fmt: skip
    return rows


def summarise_snow(records: list[dict]) -> dict:
    rows = []
    for b in SNOW_BEHAVIOURS:
        rs = [x for x in records if x["behaviour"] == b]
        mins = sum(x["upright_ticks"] for x in rs) * DT / 60.0
        rows.append({
            "behaviour": b, "robot_minutes": mins, "alarms": sum(len(x["alarm_ticks"]) for x in rs),
            "falls_on_snow": sum(x["fell_at"] is not None for x in rs),
            "max_evidence_m": max(x["max_evidence"] for x in rs),
        })  # fmt: skip
    mins = sum(x["robot_minutes"] for x in rows)
    alarms = sum(x["alarms"] for x in rows)
    return {"rows": rows, "robot_minutes": mins, "alarms": alarms, "alarms_per_robot_minute": alarms / mins if mins else None}


def print_tables(out: dict):
    s = out.get("snow")
    if s:
        print(f"\nFalse alarms on snow only ({s['robot_minutes']:.1f} robot-minutes upright, seeds {out['snow_seeds']})")
        print("| what the mission asks for | robot-min | alarms | falls on snow | peak evidence / trigger |")
        print("|---|---|---|---|---|")
        trig = out["params"]["slip_trigger"]
        for r in s["rows"]:
            print(f"| {r['behaviour']} | {r['robot_minutes']:.1f} | {r['alarms']} | {r['falls_on_snow']} | {r['max_evidence_m'] / trig:.2f} |")
        print(f"| **all** | {s['robot_minutes']:.1f} | {s['alarms']} | {sum(r['falls_on_snow'] for r in s['rows'])} | |")
        print(f"alarms per robot-minute: {s['alarms_per_robot_minute']:.3f}")
    rows = out.get("ice")
    if rows:
        print(f"\nPaired ice encounters (mu {out['ice_mu']} {out.get('ground', 'raised')} patch in mu {out['snow_mu']} snow, seeds {out['ice_seeds']}, {out['ice_ticks']} ticks)")
        print("| approach | n | falls without | falls with | detected | saved | upright, not clear | released with a foot on the ice | pelvis past edge at detection, median / max m | edge estimate error, median (min..max) m |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            n = r["encounters"]
            pel, err = r["pelvis_past_edge_at_detection_m"], r["edge_estimate_error_m"]
            pel_s = f"{pel['median']:+.2f} / {pel['max']:+.2f}" if pel else "-"
            err_s = f"{err['median']:+.2f} ({err['min']:+.2f}..{err['max']:+.2f})" if err else "-"
            print(
                f"| {r['speed']} m/s, {APPROACH.get(r['angle_deg'], '%.0f deg' % r['angle_deg'])} | {n} | {r['falls_without']}/{n} | {r['falls_with']}/{n} | "
                f"{r['detected']}/{n} | {r['saved']}/{n} | {r['upright_but_not_clear']}/{n} | {r.get('released_with_a_foot_on_ice', 0)}/{n} | {pel_s} | {err_s} |"
            )
        tot = sum(r["encounters"] for r in rows)
        print(
            f"| **all** | {tot} | {sum(r['falls_without'] for r in rows)}/{tot} | {sum(r['falls_with'] for r in rows)}/{tot} | "
            f"{sum(r['detected'] for r in rows)}/{tot} | {sum(r['saved'] for r in rows)}/{tot} | "
            f"{sum(r['upright_but_not_clear'] for r in rows)}/{tot} | {sum(r.get('released_with_a_foot_on_ice', 0) for r in rows)}/{tot} | | |"
        )
        print("saved = upright at the end, both feet (to the toes) on snow, reflex released. Negative distances are before the edge.")
        print("The distance columns measure across the west edge and mean nothing for the two along-the-edge rows.")


def measure(snow_sim, ice_sim, patches, snow_seeds, ice_seeds, reflex_kwargs: dict | None = None, log=print,
            cells: str = "standard", ice_mu: float | None = None, ground: str = "raised") -> dict:  # fmt: skip
    """Both tables. Either sim may be None to skip its half. `ice_mu` and `ground` only label the
    output: they must be what `ice_sim` was built with."""
    import fleet_sim  # noqa: PLC0415

    out = {
        "params": {k: getattr(IceReflex(ice_sim or snow_sim, **(reflex_kwargs or {})), k) for k in PARAMS},
        "snow_mu": fleet_sim.SNOW_MU, "ice_mu": fleet_sim.ICE_MU if ice_mu is None else ice_mu, "ground": ground, "cells": cells,
        "snow_seeds": list(snow_seeds), "ice_seeds": list(ice_seeds), "snow_ticks": SNOW_TICKS, "ice_ticks": ICE_TICKS,
    }  # fmt: skip
    if snow_sim is not None:
        recs = []
        for seed in snow_seeds:
            t0 = time.time()
            recs += run_snow(snow_sim, seed, reflex_kwargs=reflex_kwargs)
            log(f"snow seed {seed}: {sum(len(x['alarm_ticks']) for x in recs[-ROBOTS:])} alarms, {time.time() - t0:.0f} s")
        out["snow"] = summarise_snow(recs)
        out["snow_records"] = recs
    if ice_sim is not None:
        recs = []
        for seed in ice_seeds:
            t0 = time.time()
            a = run_ice(ice_sim, patches, seed, use_reflex=False, cells=ice_cells(cells))
            b = run_ice(ice_sim, patches, seed, use_reflex=True, reflex_kwargs=reflex_kwargs, cells=ice_cells(cells))
            recs += a + b
            log(f"ice seed {seed}: falls without {sum(x['fell'] for x in a)}/{ROBOTS}, with {sum(x['fell'] for x in b)}/{ROBOTS}, saved {sum(x['saved'] for x in b)}/{ROBOTS}, {time.time() - t0:.0f} s")
        out["ice"] = summarise_ice(recs)
        out["ice_records"] = recs
    return out


def write_results(out: dict, path: Path = RESULTS, only: str | None = None) -> dict:
    """Write the JSON. With `only`, keep the other half of an existing file from the same parameters."""
    if only and path.exists():
        old = json.loads(path.read_text())
        if old.get("params") == json.loads(json.dumps(out["params"])):
            other = "ice" if only == "snow" else "snow"
            for k in (other, f"{other}_records", f"{other}_seeds"):
                if k in old:
                    out[k] = old[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snow-seeds", type=int, default=5, help="60 s episodes of 8 robots: 8 robot-minutes each")
    ap.add_argument("--ice-seeds", type=int, default=16, help="2 encounters per cell per seed, each run with and without the reflex")
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED, help="used to choose the parameters, do not score on them: 0 to 41 and 100 to 103. 200 to 231 scored a discarded variant. This version was scored on 300 to 331")
    ap.add_argument("--only", choices=("snow", "ice"), default=None)
    ap.add_argument("--cells", choices=sorted(CELL_SETS), default="standard", help="which encounters: the published four, square-on at sweep speed only, or along the edge")
    ap.add_argument("--ice-mu", type=float, default=None, help="ice friction (default: fleet_sim.ICE_MU)")
    ap.add_argument("--flush", action="store_true", help="ice flush with the snow instead of 2 mm proud (see flush_snowfield)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import fleet_sim  # noqa: PLC0415
    from fleet_sim import FleetSim  # noqa: PLC0415

    ice_mu = fleet_sim.ICE_MU if args.ice_mu is None else args.ice_mu
    variant = (args.cells != "standard") or args.flush or (args.ice_mu is not None)
    if args.out is None:
        tag = (f"_{args.cells}" if args.cells != "standard" else "") + ("_flush" if args.flush else "") + (f"_mu{ice_mu:g}" if args.ice_mu is not None else "")
        args.out = RESULTS.with_name(RESULTS.stem + tag + ".json")
    if variant and args.only is None:
        args.only = "ice"  # the snow half does not depend on any of these
    if args.flush:
        fleet_sim.add_snowfield = flush_snowfield  # FleetSim looks the name up when it builds the ground

    snow_sim = ice_sim = patches = None
    if args.only != "ice":
        t0 = time.time()
        snow_sim = FleetSim(snow_starts(), [], seed=0)
        print(f"built the snow scene in {time.time() - t0:.0f} s", flush=True)
    if args.only != "snow":
        t0 = time.time()
        patches = ice_layout()
        ice_sim = FleetSim(ice_starts(patches, ice_cells(), np.random.default_rng(0)), patches, ice_mu=ice_mu, seed=0)
        print(f"built the ice scene in {time.time() - t0:.0f} s", flush=True)
    s0 = args.first_seed
    out = measure(
        snow_sim, ice_sim, patches, range(s0, s0 + args.snow_seeds), range(s0, s0 + args.ice_seeds), log=lambda s: print(s, flush=True),
        cells=args.cells, ice_mu=ice_mu, ground="flush" if args.flush else "raised",
    )  # fmt: skip
    out = write_results(out, args.out, args.only)
    print_tables(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
