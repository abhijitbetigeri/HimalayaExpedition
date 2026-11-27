# Fleet track

A team of Unitree G1s runs an avalanche search across a snowfield with hidden
ice. Built on the Newton track (read `NEWTON_TRACK.md` first for the engine, the
policy and the friction results this depends on). Added 2026-11-26/27. Code is in
`newton_track/fleet_*.py`.

## The task

Someone is buried in a 12 x 14 m debris field with a transceiver beacon. Four
robots line up along the bottom edge and must:

1. sweep the field so no strip is left unsearched (transceiver range 2 m);
2. pinpoint the beacon once any of them picks up its signal;
3. assemble a dig team of three at the burial point.

Three patches of ice lie under the snow, about 5 % of the field, and nobody can
see them. This policy cannot walk on ice: the single-robot sweep puts its limit
at friction 0.15 to 0.2, the snow here is 0.4 and the ice 0.08.

## How the team is built

**Each robot is its own agent.** It knows its own pose, its own transceiver
reading and what arrived in its inbox, and it decides its own walking command.
There is no central planner. Agents agree on lanes and dig positions because
they all apply the same deterministic rule to the same shared roster.

**They share a broadcast bus** with five message types:

| message | meaning |
|---|---|
| `POSE` | heartbeat at 5 Hz: who I am, where, what I am doing, which dig slot I claim |
| `DOWN` | I fell, here |
| `HAZARD` | there is ice here |
| `SIGNAL` | my transceiver reads the beacon: estimated position and range |
| `FOUND` | pinpointed: dig here |

**What coordination consists of:**

- Lanes are an equal split of the field among robots level with the front, left
  to right. When one goes down, or falls more than 3 m behind, the rest close up.
- The line stays abreast: whoever is ahead of the pack slows down.
- A hazard one robot finds is routed around by all of them.
- On a signal, the robot with the shortest range pinpoints. The others each claim
  one of twelve dig slots around it and keep the claim; a taken slot blocks its
  two neighbours.

**Each robot also has a slip reflex** (`fleet_reflex.py`). It estimates how much
sideways grip the ground is giving from its own motion, confirms ice when a
planted foot slides while that grip is low, stops, backs out onto snow, and
reports the ice edge. It reads nothing about the ground. Measured alone on 128
paired encounters: 116 falls without it, 1 with it, and one false alarm in 39.8
robot-minutes on snow.

**The ablation.** `--mode independent` is the same robots, sensors, controller,
reflex and ice with the bus switched off. Each robot sweeps the lane it started
in and reacts only to what it senses itself. Both modes run on the same seeds.

## Running it

`npy` is `~/projects/newton-sandbox/.venv/bin/python`, as in `NEWTON_TRACK.md`.

```bash
npy newton_track/test_fleet_route.py                       # planner geometry, 1 s
npy newton_track/fleet_sim.py                              # measured snow and ice friction
npy newton_track/fleet_reflex.py                           # reflex tables (slow)
npy newton_track/fleet_search.py --seeds 20 --seed-start 60 --reflex   # 25 min
npy newton_track/fleet_search.py --film 25 --reflex        # one mission to demo/, needs a display
npy newton_track/report.py                                 # regenerate the tables below
```

| File | What it is |
|---|---|
| `fleet_sim.py` | N policy-driven G1s in one MuJoCo-CPU scene on snow with ice patches. About 230 control ticks per second for four robots. No decisions. |
| `g1_collision.py` | 29 collision-only primitives per robot, so a fallen robot lies on the snow. Upright walking is bit-identical with and without them. |
| `fleet_reflex.py` | The slip reflex and its own evaluation. |
| `fleet_search.py` | The mission: agents, bus, path planner, scoring, paired evaluation. |
| `test_fleet_route.py` | Ten geometry cases for the planner, each a shape that broke an earlier version in a real mission. |
| `film_fleet.py` | Wide-shot renderer: snow, footprints, the beacon, mapped hazards, radio links, role markers. Rendering only. |

## Results

Scoring. **Located**: a robot declared the burial point within 1 m of the truth.
**Assembled**: three robots standing, holding, within 1.7 m of the true beacon
for a full second. Three is fixed, so a team that has lost two of four cannot
assemble. A mission has 100 simulated seconds.

Which seeds count. The planner was debugged on seeds 0-39 (and looked at on 40
and 42), so those cannot score it honestly. **Seeds 60-79 are the result.** The
other two sets are shown so the gap between tuned-on and untouched is visible.

<!-- BEGIN GENERATED (newton_track/report.py, fleet section) -->

### Fleet search, seeds 60-79 (held out: nothing was tuned on these)

4 robots, snow mu 0.4, ice mu 0.08, three ice patches. Every row is the same seeds.

| slip reflex | radio | beacon located | dig team assembled | robots lost per mission | median s to locate | median s to assemble |
|---|---|---|---|---|---|---|
| off | on | 19/20 | 11/20 | 1.45 | 25.7 | 28.9 |
| off | off | 18/20 | 0/20 | 1.55 | 24.9 | - |
| on | on | 20/20 | 20/20 | 0.05 | 37.0 | 47.3 |
| on | off | 20/20 | 0/20 | 0.10 | 25.7 | - |

- Reflex off, radio on, did not assemble: seed 64 (lost 2), seed 65 (lost 2), seed 66 (lost 2), seed 68 (lost 2), seed 75 (lost 3), seed 76 (lost 2), seed 77 (lost 2), seed 78 (lost 2), seed 79 (lost 2).

### Fleet search, seeds 0-19 (used to debug the planner)

4 robots, snow mu 0.4, ice mu 0.08, three ice patches. Every row is the same seeds.

| slip reflex | radio | beacon located | dig team assembled | robots lost per mission | median s to locate | median s to assemble |
|---|---|---|---|---|---|---|
| on | on | 20/20 | 20/20 | 0.05 | 38.8 | 45.3 |
| on | off | 18/20 | 0/20 | 0.25 | 28.1 | - |


### Fleet search, seeds 20-39 (used to debug the planner)

4 robots, snow mu 0.4, ice mu 0.08, three ice patches. Every row is the same seeds.

| slip reflex | radio | beacon located | dig team assembled | robots lost per mission | median s to locate | median s to assemble |
|---|---|---|---|---|---|---|
| on | on | 19/20 | 19/20 | 0.15 | 36.6 | 45.6 |
| on | off | 20/20 | 1/20 | 0.20 | 30.1 | 59.6 |

- Reflex on, radio on, did not assemble: seed 30 (not located).

<!-- END GENERATED -->

### Reading the numbers

- **With the reflex and the radio, the team did the whole job on all twenty
  held-out seeds**, losing one robot in twenty missions.
- **The reflex is what makes the task possible.** Without it the team loses 1.45
  robots a mission and assembles 11 times in 20. Every one of the nine failures
  had lost two or more robots, and three are needed to dig. Coordination does
  not protect against ice on its own (1.45 lost with the radio, 1.55 without):
  a team with no reflex learns about a patch by losing a robot to it.
- **"Assembled 0/20 with the radio off" is true almost by construction, and
  should not be quoted as the size of the coordination effect.** Without a radio
  a robot only walks to the beacon if its own transceiver hears it. The range is
  2 m and the lanes are 3 m apart, so three robots gathering needs luck the
  layout does not offer (it happened once in 80 missions). The row shows that
  the task needs communication. It does not measure how good the coordination is.
- **"Located" is the comparison a lone robot can win**, and with the reflex on
  it is 20/20 either way on the held-out seeds. Without the reflex the radio is
  worth one beacon in twenty (19 vs 18): a robot fell and the line closed up
  over its lane.
- **Coordination costs time.** Median time to locate is 37 s with the radio
  against 26 s without, because the line waits for its slowest member and backs
  off ice together.
- **The debug sets score about the same as the held-out set** (20/20 and 19/20
  assembled), which is some evidence the final planner is not fitted to them.
  It is not proof: three planner versions were thrown away on these seeds.
- **One unexplained failure**: seed 30, radio and reflex on, nobody lost, beacon
  never located in 72.5 s. Not traced.

## What went wrong on the way, and what it taught

The first planner stepped to one side of the first hazard in its way. It failed
three separate ways, each found by tracing one mission:

| seed | what happened | cause |
|---|---|---|
| 8 | A robot facing two hazard discs side by side replanned left, right, left for 36 s and walked back onto the ice | no commitment between ticks |
| 21, 25 | A robot backed off a patch and walked straight back in, every six seconds; six reflex events on one patch, two robots lost | a rule that skipped obstacles beside the target, meant for parked teammates, also skipped ice |
| 40 | A second hazard found during a detour inherited "pass on the right" from the first; thirteen reflex events on one patch | remembered side, and no check that the leg to the waypoint was clear |

The third was found by the agent running the radio study, with the mechanism.
After that the heuristic was replaced by a small visibility-graph planner
(`plan_path`): nodes around each inflated hazard disc, an edge only where the
whole leg is clear, shortest path, a committed first waypoint. "The leg is
clear" became a property of every edge. Two bugs in the replacement were caught
before it was committed: a 5 cm tolerance that let a robot ratchet into a
keep-out zone one replan at a time, and waypoints that landed inside the walking
controller's 25 cm "arrived" distance, on which two robots stood for 70 s.

Other things the missions exposed:

- **A scoring bug hid a success.** "Assembled" first required every team
  member's command to be exactly zero on the same tick. Three robots stood round
  the beacon and the mission was scored a failure.
- **A queue at the dig site.** The first helper to park blocked the two behind
  it, which waited 80 s for a teammate that was never going to move. Parked
  teammates are now obstacles to walk round, not something to wait for.
- **Standing is not a zero command for this policy.** A robot told to hold
  drifted half a metre in six seconds. Holding now steps back past 25 cm.
- **Hard yaw reversals drop the robot on plain snow.** Three of nine one-minute
  runs flipping the yaw rate at 0.5 m/s. The mission slew-limits it.
- **Held-out seeds earned their keep.** The skip-the-hazard regression raised
  robots lost per mission from 0.10 to 0.40 on the seeds being debugged and
  showed as 0.65 on fresh ones. Each round of debugging used up a seed set, and
  the evaluation moved to a new one three times.

## What is NOT claimed

- **This is not particle snow.** The ground is a friction surface with ice
  patches. A field this size in MPM is a few hundred thousand particles, and the
  single-robot MPM scene does not yet pass its own sanity check
  (`NEWTON_TRACK.md`, "Snow"). The footprints and drifts in the film are drawn by
  the renderer.
- **The ice is easy to detect.** Patches stand 2 mm proud of the snow with a
  sharp edge and one friction value. A flush or gradual edge, other frictions,
  and a robot that meets ice while turning or standing were not tested.
- **The world is flat and the sensor is simple.** No slope, no wind, a
  transceiver with clean range and bearing inside 2 m and nothing outside it.
- **The radio is perfect** in these tables. Packet loss and range limits are a
  separate study, in progress.
- **No robot gets up.** Every loss is permanent.
- **Four robots, one field size, one ice density.** Team-size effects are a
  separate study, in progress.
