"""Flush ice in the fleet's mission: what it costs, and whether sweeping slower buys it back.

    npy newton_track/fleet_flush.py --shard 0/2        # run half the seeds (resumable; two shards at once)
    npy newton_track/fleet_flush.py --reference        # the default cell again with nothing patched
    npy newton_track/fleet_flush.py --report           # tables, paired comparisons, results/fleet_flush.json

The reflex study (FLEET.md, "The reflex, looked at harder") found the slip reflex leans on the
2 mm lip of the ice patches: on flush ice at friction 0.08 it loses every square-on approach at
0.5 m/s and none at 0.25 m/s, in its own harness. This asks what that does to a whole mission.

Design, fixed before any mission was run. Full missions, four robots, coordinated, clean
radio, slip reflex on. Two grounds (ice raised 2 mm as committed, or flush: `FleetSim(flush=True)`)
by three sweep speeds (0.5 as committed, 0.35, 0.25 m/s): six cells, the same twenty seeds in each,
180 to 199, which no fleet study had used. Two more cells are controls: both grounds with the
"ice" at snow friction (0.4), at 0.5 m/s, same seeds. If the flush ground is harmless apart from
its ice, the flush control loses nobody and raises no alarm, like the raised one.

`fleet_search.py` is not edited. The sweep speed is the module constant `fleet_search.V_SWEEP`,
read when an agent decides, and the ground is built by `fleet_search.FleetSim`, looked up when a
`Mission` is constructed; `patched()` assigns both and puts them back. Every row checks that the
patch took: the ground the sim was built with, the largest forward speed any sweeping robot
was commanded, and the speed the robots actually made over the first ice-free metres. And
`--reference` runs the default cell (raised, 0.5) in a process that patches nothing, to be
compared state for state with the patched one (`--report` does that).

What else is tied to the sweep speed in `fleet_search.py`, and was left alone:
  - `V_SWEEP` is also the speed of a helper walking to its dig slot (role "converge").
  - The pinpointing robot walks at a fixed 0.35 m/s (`_go(me, goal, 0.35, 0.3)`), so at a
    sweep of 0.25 it is the fastest robot on the field, and at 0.35 it is no slower than the sweep.
  - Pacing: `v = clip(V_SWEEP - 0.6 * ahead, 0.15, V_SWEEP)`. The gain and the 0.15 floor are
    absolute. At 0.5 a robot ahead of the line can drop to 30 % of sweep speed and is at the
    floor 0.58 m ahead; at 0.25 it can only drop to 60 % and is at the floor 0.17 m ahead. The
    line has less authority to stay abreast when it sweeps slowly.
  - `steer` ramps speed down as 0.15 + 0.8 * distance near a goal; holding steps back at 0.2.
  - Times that do not scale with speed: the mission limit `T_MAX` (100 s), `LOST_AFTER` 1.5 s,
    `ROSTER_AFTER` 10 s, the 2 s yield, the 1 s stuck timer, 3 s all-parked. `STRAGGLER_GAP` is
    3 m, which is 6 s of walking at 0.5 and 12 s at 0.25.
  - The field is 14 m and robots start 1 m short of it, so the top is 14.7 m of walking: 29 s
    at a true 0.5 m/s and 59 s at a true 0.25 m/s, before any stop. The 100 s limit was set for
    0.5. To tell "ran out of clock" from "could not do it", a mission that ends on the limit
    is scored there, and then ticked on to at most 200 s with the result of that kept apart
    (`overtime`). Nothing in the mission is changed to do so.
  - In `fleet_reflex.py` the back-out is 0.25 m/s whatever the sweep speed, and the detector
    needs the support foot sliding faster than 0.10 m/s.

Each finished mission is one JSON line appended to results/fleet_flush_rows.jsonl, so a run can
be stopped and started again. Rows carry the hashes of the three files they depend on.

Results, seeds 180 to 199, n = 20 missions a cell, every cell the same seeds (`--report`
prints the tables, results/fleet_flush.json has every row). Run once, on fleet_sim.py with the
flush option added to commit 7246561; nothing was tuned and no mission was rerun.

    ice      sweep   located  assembled  robots lost  median s to    hit the      assembled
             m/s                         (total)      assemble       100 s limit  by 200 s
    raised   0.5     20/20    20/20       0           36.2           0            20/20
    raised   0.35    20/20    19/20       1           56.9           1            20/20
    raised   0.25    19/20    15/20       0           61.3           5            20/20
    flush    0.5     20/20    14/20      23           33.5           0            14/20
    flush    0.35    20/20    20/20       0           50.4           0            20/20
    flush    0.25    18/20    16/20       0           65.1           4            20/20
    controls, ice at snow friction, 0.5 m/s: raised 20/20, flush 20/20, nobody lost, no alarm,
    time to assemble within 0.2 s of each other on every seed.

- Flush ice at today's speed breaks the mission. 23 robots lost in 20 missions against 0, more
  on 16 seeds and fewer on none (sign test p = 0.00003); a dig team on 14 seeds against 20
  (6 against 0, exact McNemar p = 0.03). Every one of the six failures had lost two or three
  robots. The beacon is still located 20 times in 20. All 31 alarms fired with a foot on the
  ice at 0.46 to 0.50 m/s asked, and 23 were followed by that robot going down: 22 in the
  back-out, a median 1.4 s after the alarm, one after release with a foot still on the ice.
- Slowing down buys all of it back. At 0.35 and at 0.25 m/s nobody was lost on flush ice in
  20 missions each (33 and 31 alarms, no fall). At 0.35 the team assembled 20 times in 20.
- What it costs is time, and at 0.25 the clock. Commanded 0.5 / 0.35 / 0.25 the line makes
  0.41 / 0.28 / 0.18 m/s over the first metres. On flush ice at 0.35 the team assembles a
  median 14.5 s later than today's mission on raised ice (seed by seed +2.9 to +29.5 s; the
  slowest took 93.7 s of the 100). At 0.25, four missions in 20 on flush ice and five on raised
  end on the 100 s limit with nobody lost; ticked on, all nine assemble by 127 s. So 0.25 fails
  on the clock and only on the clock, and 0.35 fits with little to spare.
- The one robot lost on raised ice was at 0.35: converging, 8.5 s after its own alarm, a foot
  back on the patch, no second alarm.

What this does not show. Twenty seeds: no loss in 33 flush encounters at 0.35 m/s still allows
a true rate of several per cent, and the reflex harness never ran 0.35 (it has 0.5 and 0.25).
One ice friction, one field, sharp patch edges. The time limit was left at 100 s.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEEDS = tuple(range(180, 200))
SPEEDS = (0.5, 0.35, 0.25)
GROUNDS = ("raised", "flush")
ROWS = HERE / "results" / "fleet_flush_rows.jsonl"
SUMMARY = HERE / "results" / "fleet_flush.json"
OVERTIME_S = 200.0  # a mission that ends on the 100 s limit is ticked on to here, scored apart
TOE = 0.12  # m, ankle body to toe (as in fleet_reflex.py)
CODE_FILES = ("fleet_sim.py", "fleet_search.py", "fleet_reflex.py")


def cell_name(ground: str, v: float, ice_mu: float | None = None) -> str:
    return f"{ground}@{v:g}" + ("" if ice_mu is None else f"/mu{ice_mu:g}")


# The 2 x 3 grid, then the two controls (ice at snow friction). (name, ground, sweep speed, ice_mu or None for the default)
CELLS = [(cell_name(g, v), g, v, None) for g in GROUNDS for v in SPEEDS]
CONTROLS = [(cell_name(g, 0.5, 0.4), g, 0.5, 0.4) for g in GROUNDS]
DEFAULT_CELL = cell_name("raised", 0.5)
REFERENCE = "unpatched"


def code_hashes() -> dict:
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:12] for f in CODE_FILES}


@contextlib.contextmanager
def patched(v_sweep: float, flush: bool):
    """Set the sweep speed and the ground for missions built inside the block. fleet_search.py is not edited."""
    import fleet_search  # noqa: PLC0415
    import fleet_sim  # noqa: PLC0415

    old_v, old_sim = fleet_search.V_SWEEP, fleet_search.FleetSim

    def make_sim(*args, **kwargs):
        return fleet_sim.FleetSim(*args, flush=flush, **kwargs)

    fleet_search.V_SWEEP, fleet_search.FleetSim = v_sweep, make_sim
    try:
        yield
    finally:
        fleet_search.V_SWEEP, fleet_search.FleetSim = old_v, old_sim


def _rect_dist(p, x: float, y: float) -> float:
    """Distance from a point to an ice patch (0 inside)."""
    return math.hypot(max(abs(x - p.x) - p.hx, 0.0), max(abs(y - p.y) - p.hy, 0.0))


def fly(mission, cell: str, extra: dict | None = None) -> dict:
    """Tick a mission to its end and return its result with the study's bookkeeping.

    Watches, never steers: it wraps the reflex filter to count alarms and reads the sim after
    each tick. Everything recorded past `mission.result()` may use ground truth."""
    import mujoco  # noqa: PLC0415
    import fleet_search as fs  # noqa: PLC0415
    from g1_newton import POLICY_HZ  # noqa: PLC0415

    sim, n = mission.sim, mission.n
    patches = mission.scenario.patches
    dt = 1.0 / POLICY_HZ
    t_wall = time.time()

    alarms, falls = [], []
    last_cmd = np.zeros((n, 3), dtype=np.float32)
    if mission.reflex is not None:
        inner = mission.reflex.filter

        def filter_and_count(cmd_in):
            cmd, events = inner(cmd_in)
            for e in events:
                r = e["robot"]
                x, y, _ = sim.pose(r)
                feet = [f[:2] for f in (sim.state_0.body_q.numpy()[r * sim.bodies_per_robot + i] for i in sim.foot_local)]
                alarms.append({
                    "t_s": round(e["t"] * dt, 2), "robot": r, "x": round(x, 2), "y": round(y, 2), "role": mission.agents[r].role,
                    "asked_vx": round(float(cmd_in[r][0]), 3),  # what the agent wanted on the tick the alarm fired
                    "foot_to_ice_m": round(min((_rect_dist(p, float(f[0]), float(f[1])) for p in patches for f in feet), default=99.0), 3),
                })  # fmt: skip
            return cmd, events

        mission.reflex.filter = filter_and_count

    inner_step = sim.step
    phase_at_step = ["-"] * n  # reflex phase after this tick's filter, before the physics

    def step_and_keep(cmds):
        last_cmd[:] = cmds
        if mission.reflex is not None:
            phase_at_step[:] = list(mission.reflex.phase)
        return inner_step(cmds)

    sim.step = step_and_keep

    # Speed check: the opening stretch, from 2 s until the first robot is 1.2 m into the field.
    # No patch starts below y = 1.5 (Scenario.sample), so this is plain snow in every cell.
    open_cmd, open_y0, open_t0, open_y1, open_t1 = [], None, None, None, None
    vx_max_sweeping = 0.0  # largest forward command to a robot that was sweeping or converging
    vx_max_any = 0.0
    ice_ticks = 0  # robot-ticks upright with the pelvis over ice

    def one_tick():
        nonlocal open_y0, open_t0, open_y1, open_t1, vx_max_sweeping, vx_max_any, ice_ticks
        was_down = sim.down.copy()
        mission.tick()
        roles = [a.role for a in mission.agents]
        ys = [sim.pose(r)[1] for r in range(n)]
        for r in range(n):
            if not sim.down[r]:
                vx_max_any = max(vx_max_any, float(last_cmd[r][0]))
                if roles[r] in ("sweep", "converge"):
                    vx_max_sweeping = max(vx_max_sweeping, float(last_cmd[r][0]))
                ice_ticks += int(sim.on_ice(r))
        if open_t1 is None:
            quiet = all(role == "sweep" for role in roles) and not sim.down.any()
            if mission.t >= 2 * POLICY_HZ and open_t0 is None and quiet:
                open_t0, open_y0 = mission.t, list(ys)
            if open_t0 is not None:
                if quiet and max(ys) < 1.2:
                    open_cmd.append(float(np.mean(last_cmd[:, 0])))
                    open_y1 = list(ys)
                else:
                    open_t1 = mission.t - 1
        for r in range(n):
            if sim.down[r] and not was_down[r]:
                x, y, _ = sim.pose(r)
                feet = [f[:2] for f in (sim.state_0.body_q.numpy()[r * sim.bodies_per_robot + i] for i in sim.foot_local)]
                mine = [a["t_s"] for a in alarms if a["robot"] == r]
                falls.append({
                    "t_s": round(mission.t * dt, 2), "robot": r, "x": round(x, 2), "y": round(y, 2), "role": roles[r],
                    "reflex_phase": phase_at_step[r], "pelvis_on_ice": bool(sim.on_ice(r)),
                    "foot_on_ice": bool(any(p.contains(float(f[0]), float(f[1]), TOE) for p in patches for f in feet)),
                    "s_since_own_alarm": None if not mine else round(mission.t * dt - mine[-1], 2), "own_alarms_before": len(mine),
                })  # fmt: skip

    while not mission.done:
        one_tick()

    row = mission.result()
    standing = int((~sim.down).sum())
    if mission.t_assembled is not None:
        end = "assembled"
    elif standing == 0:
        end = "none_standing"
    elif mission.t >= fs.T_MAX:
        end = "time_limit"
    else:
        end = "all_parked"
    mj = sim.solver.mj_model
    ground_geoms = [g for g in range(mj.ngeom) if mj.geom_bodyid[g] == 0]
    span = None if open_t1 is None or open_y1 is None or open_t1 <= open_t0 else (open_t1 - open_t0) * dt
    row.update({
        "cell": cell, "end": end, "alarms": len(alarms), "alarm_log": list(alarms), "fall_log": list(falls),
        "alarms_clear_of_ice": sum(a["foot_to_ice_m"] > 0.3 for a in alarms),
        "ice_robot_seconds": round(ice_ticks * dt, 2),
        "v_sweep": fs.V_SWEEP, "sim_flush": bool(getattr(sim, "flush", False)),
        "ground_planes": int(sum(mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in ground_geoms)), "ground_geoms": len(ground_geoms),
        "ice_mu_model": sorted({round(float(mj.geom_friction[g][0]), 4) for g in ground_geoms if mj.geom_priority[g] == 2}),
        "cmd_vx_max_sweeping": round(vx_max_sweeping, 4), "cmd_vx_max_any": round(vx_max_any, 4),
        "opening": None if span is None else {
            "seconds": round(span, 2), "asked_vx_mean": round(float(np.mean(open_cmd)), 4),
            "made_vy_mean": round(float(np.mean([(b - a) / span for a, b in zip(open_y0, open_y1)])), 4),
        },
        "q_sha": hashlib.sha256(sim._q.tobytes() + sim._qd.tobytes()).hexdigest()[:16], "ticks": mission.t,
    })  # fmt: skip

    row["overtime"] = None
    if end == "time_limit":
        cap = int(OVERTIME_S * POLICY_HZ)
        while mission.t < cap and mission.t_assembled is None and (~sim.down).any() and mission._parked_ticks <= 3 * POLICY_HZ:
            one_tick()
        late = mission.result()
        row["overtime"] = {k: late[k] for k in ("located", "t_located_s", "assembled", "t_assembled_s", "robots_lost", "duration_s")}
        row["overtime"]["alarms"] = len(alarms)
        row["overtime"]["falls_after_limit"] = [f for f in falls if f["t_s"] > fs.T_MAX * dt]
    row["wall_s"] = round(time.time() - t_wall, 1)
    row["code"] = code_hashes()
    row.update(extra or {})
    return row


def run_cell(seed: int, ground: str, v: float, ice_mu: float | None, name: str) -> dict:
    import fleet_search as fs  # noqa: PLC0415

    t0 = time.time()
    with patched(v, ground == "flush"):
        kw = {} if ice_mu is None else {"ice_mu": ice_mu}
        mission = fs.Mission(seed, "coordinated", 4, True, **kw)
        build = round(time.time() - t0, 1)
        row = fly(mission, name, {"ground": ground, "ice_mu": fs.ICE_MU if ice_mu is None else ice_mu, "build_s": build})
    if row["sim_flush"] != (ground == "flush") or row["v_sweep"] != v or row["ground_planes"] != (0 if ground == "flush" else 1):
        raise RuntimeError(f"the patch did not take: {name} seed {seed}: {row['sim_flush']}, {row['v_sweep']}, {row['ground_planes']} planes")
    if row["cmd_vx_max_sweeping"] > v + 1e-6:
        raise RuntimeError(f"a sweeping robot was commanded {row['cmd_vx_max_sweeping']} m/s in a cell that sweeps at {v}")
    return row


def run_reference(seed: int) -> dict:
    """The default mission with nothing patched: fleet_search as it is on disk."""
    import fleet_search as fs  # noqa: PLC0415
    import fleet_sim  # noqa: PLC0415

    if fs.FleetSim is not fleet_sim.FleetSim or fs.V_SWEEP != 0.5:
        raise RuntimeError("fleet_search is patched in this process")
    t0 = time.time()
    mission = fs.Mission(seed, "coordinated", 4, True)
    return fly(mission, REFERENCE, {"ground": "raised", "ice_mu": fs.ICE_MU, "build_s": round(time.time() - t0, 1)})


def read_rows(path: pathlib.Path = ROWS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_row(row: dict, path: pathlib.Path = ROWS):
    """One whole line in one write on an O_APPEND descriptor, so two shards can share the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(row, default=float) + "\n").encode())
    finally:
        os.close(fd)


# ===========================================================================
# Report
# ===========================================================================


def _median(xs):
    return None if not xs else round(float(np.median(xs)), 1)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    late = [r for r in rows if r["end"] == "time_limit"]
    return {
        "n": n,
        "located": sum(r["located"] for r in rows),
        "assembled": sum(r["assembled"] for r in rows),
        "robots_lost_total": sum(r["robots_lost"] for r in rows),
        "robots_lost_per_mission": round(sum(r["robots_lost"] for r in rows) / n, 2),
        "missions_with_a_loss": sum(r["robots_lost"] > 0 for r in rows),
        "t_located_median_s": _median([r["t_located_s"] for r in rows if r["located"]]),
        "t_assembled_median_s": _median([r["t_assembled_s"] for r in rows if r["assembled"]]),
        "duration_mean_s": round(float(np.mean([r["duration_s"] for r in rows])), 1),
        "alarms_per_mission": round(sum(r["alarms"] for r in rows) / n, 2),
        "alarms_clear_of_ice": sum(r["alarms_clear_of_ice"] for r in rows),
        "hit_time_limit": len(late),
        "ended": {k: sum(r["end"] == k for r in rows) for k in ("assembled", "time_limit", "all_parked", "none_standing")},
        "of_those_on_the_limit_assembled_by_200s": sum(bool(r["overtime"] and r["overtime"]["assembled"]) for r in late),
        "assembled_by_200s": sum(r["assembled"] or bool(r["overtime"] and r["overtime"]["assembled"]) for r in rows),
        "robots_lost_by_200s_total": sum(r["overtime"]["robots_lost"] if r["overtime"] else r["robots_lost"] for r in rows),
        "latest_assembly_s": max((r["overtime"]["t_assembled_s"] if r["overtime"] else r["t_assembled_s"]) or 0.0 for r in rows),
        "alarms_total": sum(r["alarms"] for r in rows),
        "alarms_then_that_robot_fell": sum(
            any(f["robot"] == a["robot"] and 0.0 <= f["t_s"] - a["t_s"] <= 8.0 for f in r["fall_log"]) for r in rows for a in r["alarm_log"]
        ),
        "falls_by_reflex_phase": {
            k: sum(f["reflex_phase"] == k for r in rows for f in r["fall_log"] if f["t_s"] <= 100.0) for k in ("walk", "brake", "back", "settle")
        },
        "falls_with_no_alarm_of_its_own_before": sum(f["own_alarms_before"] == 0 for r in rows for f in r["fall_log"] if f["t_s"] <= 100.0),
        "opening_made_speed_mean": round(float(np.mean([r["opening"]["made_vy_mean"] for r in rows if r["opening"]])), 3),
        "opening_asked_speed_mean": round(float(np.mean([r["opening"]["asked_vx_mean"] for r in rows if r["opening"]])), 3),
        "cmd_vx_max_sweeping": max(r["cmd_vx_max_sweeping"] for r in rows),
        "cmd_vx_max_any": max(r["cmd_vx_max_any"] for r in rows),
    }


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign test: probability of a split at least as lopsided as k of n at p = 1/2."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / 2.0**n)


def paired(a_rows: dict, b_rows: dict, a: str, b: str) -> dict:
    """Per seed, cell b against cell a. `*_rows` map seed -> row."""
    seeds = sorted(set(a_rows) & set(b_rows))
    table = []
    for s in seeds:
        ra, rb = a_rows[s], b_rows[s]
        table.append({
            "seed": s,
            "lost": [ra["robots_lost"], rb["robots_lost"]], "located": [ra["located"], rb["located"]], "assembled": [ra["assembled"], rb["assembled"]],
            "t_assembled_s": [ra["t_assembled_s"], rb["t_assembled_s"]], "alarms": [ra["alarms"], rb["alarms"]],
            "duration_s": [ra["duration_s"], rb["duration_s"]], "end": [ra["end"], rb["end"]],
        })  # fmt: skip
    more = sum(t["lost"][1] > t["lost"][0] for t in table)
    fewer = sum(t["lost"][1] < t["lost"][0] for t in table)
    a_only = sum(t["assembled"][0] and not t["assembled"][1] for t in table)
    b_only = sum(t["assembled"][1] and not t["assembled"][0] for t in table)
    both = [t for t in table if t["assembled"][0] and t["assembled"][1]]
    dt = [t["t_assembled_s"][1] - t["t_assembled_s"][0] for t in both]
    return {
        "a": a, "b": b, "n": len(seeds),
        "lost_total": [sum(t["lost"][0] for t in table), sum(t["lost"][1] for t in table)],
        "seeds_b_lost_more": more, "seeds_b_lost_fewer": fewer, "seeds_lost_the_same": len(table) - more - fewer,
        "lost_sign_test_p": float(f"{_binom_two_sided(more, more + fewer):.2g}"),
        "assembled": [sum(t["assembled"][0] for t in table), sum(t["assembled"][1] for t in table)],
        "assembled_only_in_a": a_only, "assembled_only_in_b": b_only,
        "assembled_mcnemar_exact_p": float(f"{_binom_two_sided(a_only, a_only + b_only):.2g}"),
        "both_assembled": len(both),
        "t_assembled_b_minus_a_median_s": _median(dt), "t_assembled_b_minus_a_range_s": None if not dt else [round(min(dt), 1), round(max(dt), 1)],
        "per_seed": table,
    }  # fmt: skip


def report(path: pathlib.Path = ROWS, out_path: pathlib.Path = SUMMARY) -> dict:
    rows = read_rows(path)
    codes = {json.dumps(r["code"], sort_keys=True) for r in rows}
    if len(codes) > 1:
        raise RuntimeError(f"rows from {len(codes)} different versions of the code in {path}: rerun everything")
    by = {}
    for r in rows:
        if r["seed"] in by.setdefault(r["cell"], {}):
            raise RuntimeError(f"two rows for {r['cell']} seed {r['seed']}")
        by[r["cell"]][r["seed"]] = r
    out = {
        "what": "Flush ice in full fleet missions. See newton_track/fleet_flush.py.",
        "design": "4 robots, coordinated, clean radio, reflex on, snow 0.4, ice 0.08 (controls: 0.4). Same seeds in every cell. Fixed before any mission was run.",
        "seeds": list(SEEDS), "code": rows[0]["code"] if rows else None, "cells": {}, "controls": {}, "paired": {}, "checks": {},
    }  # fmt: skip

    def have(name):
        return name in by and len(by[name]) > 0

    print(f"rows: {len(rows)} in {path}")
    hdr = "| ice | sweep m/s | n | located | assembled | lost / mission | median s to locate | median s to assemble | mean duration s | alarms / mission | hit 100 s limit | assembled by 200 s |"
    print("\n" + hdr + "\n" + "|---" * 12 + "|")
    for name, ground, v, mu in CELLS + CONTROLS:
        if not have(name):
            continue
        s = summarise([by[name][k] for k in sorted(by[name])])
        (out["controls"] if mu is not None else out["cells"])[name] = s
        n = s["n"]
        label = ground + ("" if mu is None else f", ice at mu {mu:g} (control)")
        print(
            f"| {label} | {v:g} | {n} | {s['located']}/{n} | {s['assembled']}/{n} | {s['robots_lost_per_mission']:.2f} ({s['robots_lost_total']}) | "
            f"{s['t_located_median_s'] or '-'} | {s['t_assembled_median_s'] or '-'} | {s['duration_mean_s']} | {s['alarms_per_mission']:.2f} | "
            f"{s['hit_time_limit']}/{n} | {s['assembled_by_200s']}/{n} |"
        )
    print("lost / mission: mean, with the total over the cell in brackets. Medians are over the missions that located / assembled.")
    print("assembled by 200 s: missions that ended on the 100 s limit were ticked on, and scored apart; the mission was not changed.")

    print("\nHow the sweep speed took (first ice-free metres, all robots sweeping) and where robots fell:")
    print("| cell | asked m/s | made m/s | largest sweep command | largest command | alarms, and that robot down within 8 s | falls in walk / brake / back / settle | falls with no alarm first | alarms clear of ice | last assembly s |")
    print("|---" * 10 + "|")
    for name, *_ in CELLS + CONTROLS:
        s = out["cells"].get(name) or out["controls"].get(name)
        if s:
            f = s["falls_by_reflex_phase"]
            print(
                f"| {name} | {s['opening_asked_speed_mean']} | {s['opening_made_speed_mean']} | {s['cmd_vx_max_sweeping']} | {s['cmd_vx_max_any']} | "
                f"{s['alarms_total']}, {s['alarms_then_that_robot_fell']} | "
                f"{f['walk']} / {f['brake']} / {f['back']} / {f['settle']} | {s['falls_with_no_alarm_of_its_own_before']} | {s['alarms_clear_of_ice']} | {s['latest_assembly_s']} |"
            )

    pairs = [
        (cell_name("raised", 0.5), cell_name("flush", 0.5)),
        (cell_name("flush", 0.5), cell_name("flush", 0.25)),
        (cell_name("flush", 0.5), cell_name("flush", 0.35)),
        (cell_name("raised", 0.5), cell_name("raised", 0.25)),
        (cell_name("raised", 0.25), cell_name("flush", 0.25)),
        (cell_name("raised", 0.5, 0.4), cell_name("flush", 0.5, 0.4)),
        (cell_name("raised", 0.5), cell_name("flush", 0.35)),  # added after the results were in: what the slower sweep costs against today's mission
    ]
    for a, b in pairs:
        if not (have(a) and have(b)):
            continue
        p = paired(by[a], by[b], a, b)
        out["paired"][f"{a} -> {b}"] = p
        print(f"\nPaired, {p['n']} seeds: {a}  ->  {b}")
        print(
            f"  robots lost {p['lost_total'][0]} -> {p['lost_total'][1]}: more on {p['seeds_b_lost_more']} seeds, fewer on {p['seeds_b_lost_fewer']}, "
            f"same on {p['seeds_lost_the_same']} (sign test p = {p['lost_sign_test_p']})"
        )
        print(
            f"  assembled {p['assembled'][0]} -> {p['assembled'][1]}: only before on {p['assembled_only_in_a']} seeds, only after on {p['assembled_only_in_b']} "
            f"(exact McNemar p = {p['assembled_mcnemar_exact_p']})"
        )
        if p["both_assembled"]:
            lo, hi = p["t_assembled_b_minus_a_range_s"]
            print(f"  where both assembled ({p['both_assembled']} seeds): time to assemble changes by a median {p['t_assembled_b_minus_a_median_s']:+} s ({lo:+} to {hi:+})")
        if (a, b) in pairs[:2]:
            print("  | seed | lost | assembled | s to assemble | alarms | ended |")
            print("  |---|---|---|---|---|---|")
            for t in p["per_seed"]:
                ta = ["-" if x is None else x for x in t["t_assembled_s"]]
                print(
                    f"  | {t['seed']} | {t['lost'][0]} -> {t['lost'][1]} | {'yes' if t['assembled'][0] else 'no'} -> {'yes' if t['assembled'][1] else 'no'} | "
                    f"{ta[0]} -> {ta[1]} | {t['alarms'][0]} -> {t['alarms'][1]} | {t['end'][0]} -> {t['end'][1]} |"
                )

    # The default cell against the same mission with nothing patched: must be the same mission, state for state.
    if have(REFERENCE) and have(DEFAULT_CELL):
        keys = ("located", "t_located_s", "locate_error_m", "assembled", "t_assembled_s", "robots_lost", "coverage", "messages", "duration_s", "q_sha", "ticks", "alarms")
        seeds = sorted(set(by[REFERENCE]) & set(by[DEFAULT_CELL]))
        bad = [s for s in seeds if any(by[REFERENCE][s][k] != by[DEFAULT_CELL][s][k] for k in keys)]
        out["checks"]["default_cell_equals_unpatched"] = {"seeds_compared": seeds, "seeds_that_differ": bad, "compared": list(keys)}
        print(f"\nDefault cell ({DEFAULT_CELL}) against the unpatched mission: {len(seeds) - len(bad)} of {len(seeds)} seeds identical (result, final state hash, tick count, alarms)")
        if bad:
            print(f"  DIFFER: {bad}")
    out["rows"] = rows
    if rows:
        out_path.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", default="0/1", help="i/n: run the seeds whose index is i modulo n (two processes: 0/2 and 1/2)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="run only these seeds (default: 180 to 199)")
    ap.add_argument("--cells", nargs="*", default=None, help="run only these cells, by name (default: the six, then the two controls)")
    ap.add_argument("--reference", action="store_true", help="run the default mission with nothing patched, for the identity check")
    ap.add_argument("--report", action="store_true", help="print the tables from the rows file and write the summary")
    ap.add_argument("--rows", type=pathlib.Path, default=ROWS)
    args = ap.parse_args()

    if args.report:
        report(args.rows, args.rows.with_name(SUMMARY.name) if args.rows != ROWS else SUMMARY)
        return 0

    i, n = (int(v) for v in args.shard.split("/"))
    seeds = [s for k, s in enumerate(args.seeds if args.seeds is not None else SEEDS) if k % n == i]
    cells = [(REFERENCE, "raised", 0.5, None)] if args.reference else [c for c in CELLS + CONTROLS if args.cells is None or c[0] in args.cells]
    code = code_hashes()
    done = set()
    for r in read_rows(args.rows):
        if r["code"] != code:
            raise RuntimeError(f"{args.rows} holds rows from another version of the code ({r['code']} against {code}): move it away and rerun everything")
        done.add((r["cell"], r["seed"]))
    t0 = time.time()
    for seed in seeds:  # seed-major, so a run that stops early has whole seeds, paired
        for name, ground, v, mu in cells:
            if (name, seed) in done:
                continue
            row = run_reference(seed) if args.reference else run_cell(seed, ground, v, mu, name)
            append_row(row, args.rows)
            print(
                f"[{time.time() - t0:6.0f} s] {name:<16} seed {seed}: located {row['located']!s:<5} assembled {row['assembled']!s:<5} lost {row['robots_lost']} "
                f"alarms {row['alarms']} {row['duration_s']:5.1f} s {row['end']}" + (f" (by 200 s: assembled {row['overtime']['assembled']})" if row["overtime"] else ""),
                flush=True,
            )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
