# Newton track

A third stack for the project, next to MuJoCo Playground (`CLAUDE.md`) and Isaac
Lab (`ISAAC_TRACK.md`). Added 2026-11-26. Everything is in `newton_track/`.

[Newton](https://github.com/newton-physics/newton) is the Warp-based physics
engine from Disney Research, Google DeepMind and NVIDIA. One scene description
runs under interchangeable solvers. Three of them matter here: `SolverMuJoCo`,
`SolverSteel` (Skild AI's rigid solver, [skild-ai/steel](https://github.com/skild-ai/steel))
and `SolverImplicitMPM` for granular and snow-like material.

## What it adds

1. **A control that walks on every seed.** `CLAUDE.md` says no baseline here is a
   known-good control, and that GPU rollouts move by +-35 steps run to run. The
   Newton track uses the G1 29-DoF walking policy that ships with Newton and runs
   MuJoCo's C engine on CPU. It is deterministic (every MuJoCo friction, wind and
   slope cell was run twice with identical survival counts; the Steel, pyramidal
   and speed runs were run once) and the control survives 32/32 seeds for 20 s.
   `ice_sweep.py` refuses to print a comparison if the control falls.
2. **A policy we did not train.** The friction thesis was only ever tested on
   policies trained in this repo. This one was trained by the Newton developers in
   Isaac Lab on MuJoCo-Warp. It walks unchanged down to mu 0.2 and falls on every
   seed at mu 0.1, on calibrated friction. Slope and speed shift that cliff in a
   predictable way (see "Reading the numbers").
3. **A second engine on the same seeds.** `--rigid steel` reruns the friction
   sweep under Skild's Steel solver. The two engines agree on where the policy
   stops walking.
4. **Measured friction, not configured friction.** `calibrate_friction.py` slides
   a block and reports the coefficient the ground actually delivers. Both engines
   get it wrong by default, in different directions.
5. **Snow as material.** `snow_mpm.py` puts the robot on an MPM particle bed with
   two-way coupling. This part runs but is NOT a result yet; see "Snow" below.

## Setup

Newton has its own environment. It does not go in the repo's `.venv` (different
MuJoCo and Warp pins; see the hard convention in `CLAUDE.md`).

```bash
uv venv --python 3.12 ~/projects/newton-sandbox/.venv
uv pip install --python ~/projects/newton-sandbox/.venv/bin/python "newton[examples]==1.8.0"
```

Below, `npy` means `~/projects/newton-sandbox/.venv/bin/python`. The first run of
each solver compiles Warp kernels for one to three minutes; later runs use the
cache. The G1 asset and policy download once to `~/Library/Caches/newton`.

```bash
npy newton_track/calibrate_friction.py                  # 30 s. Run this first.
npy newton_track/ice_sweep.py --check-determinism       # ~50 min, 37 cells, each twice
npy newton_track/ice_sweep.py --cone pyramidal --only friction  # 4 min
npy newton_track/ice_sweep.py --rigid steel --only friction   # 15 min
npy newton_track/ice_sweep.py --only speed              # does slowing down help on ice
npy newton_track/snow_block_test.py                     # 2 min snow-coupling preflight
npy newton_track/snow_mpm.py --smoke                    # 1 min
npy newton_track/report.py                              # regenerate the tables below
npy newton_track/film.py --mu 0.1 --out demo/newton_ice.mp4   # one seed, an illustration
```

## Module map

| File | What it is |
|---|---|
| `g1_newton.py` | `G1Himalaya`: the Newton G1 policy under one `Conditions` (ground friction, slope, wind, command). `rollout(seed, steps)` returns survival, body-frame speed and heading drift. `rigid="mujoco"` or `"steel"`. |
| `calibrate_friction.py` | Sliding-block measurement of effective friction in four configurations. Non-zero exit if a configuration in use is off by more than 10 %. |
| `ice_sweep.py` | Paired sweep over friction, wind and slope. Same seeds in every cell. Aborts if the control falls. |
| `snow_mpm.py` | `G1Snow`: the same robot on an MPM snow bed through `SolverCoupledProxy`. Reports sinkage and the share of the robot's weight the snow carries. |
| `snow_block_test.py` | A block on the same bed. Separates coupling faults from snow rheology. |
| `snow_foot_test.py` | A sole-sized plate under the robot's weight, tipped by a growing torque. Asks the bed for the moment support rigid ground gives a stance foot. Exits 1 at the defaults: that is the finding. |
| `snow_friction_test.py` | Drags the foot collider across the bed under load. Checks the snow is not ice to the foot. |
| `report.py` | Builds the tables in this file from `results/*.json`. |
| `film.py` | Renders one rollout to mp4 through Newton's GL viewer and system ffmpeg. Needs a display. Clips go to `demo/newton_*.mp4`, which `.gitignore` excludes. |

## Results

Policy: `newton-assets/unitree_g1/rl_policies/mjw_g1_29DOF.onnx`, commanded
0.5 m/s forward, zero yaw rate. 50 Hz control, 200 Hz physics. A fall is the
pelvis below 0.40 m or tilted past 60 degrees. Survival is censored at 1000
steps: "0 falls" means no falls in 20 s, not "never falls".

<!-- BEGIN GENERATED (newton_track/report.py) -->

### Friction calibration (sliding block, effective mu)

| nominal ground mu | MuJoCo + ground priority | MuJoCo default (max) | Steel, both sides mu | Steel, 0.75 block (average) |
|---|---|---|---|---|
| 0.05 | 0.052 | 0.751 | 0.052 | 0.402 |
| 0.1 | 0.101 | 0.751 | 0.102 | 0.427 |
| 0.2 | 0.200 | 0.751 | 0.202 | 0.477 |
| 0.4 | 0.400 | 0.751 | 0.402 | 0.577 |
| 0.75 | 0.751 | 0.751 | 0.752 | 0.752 |

### Friction (32 paired seeds x 1000 steps = 20 s, flat, no wind)

| ground mu | falls, MuJoCo | falls, Steel | falls, MuJoCo, pyramidal cone | median steps (MuJoCo / Steel / MuJoCo, pyramidal cone) | body speed m/s, MuJoCo |
|---|---|---|---|---|---|
| 0.75 | 0/32 | 0/32 | 0/32 | 1000 / 1000 / 1000 | 0.43 |
| 0.4 | 0/32 | 0/32 | 0/32 | 1000 / 1000 / 1000 | 0.43 |
| 0.3 | 0/32 | 0/32 | 0/32 | 1000 / 1000 / 1000 | 0.42 |
| 0.2 | 0/32 | 0/32 | 5/32 | 1000 / 1000 / 1000 | 0.42 |
| 0.15 | 20/32 | 27/32 | 30/32 | 66 / 142 / 54 | 0.44 |
| 0.1 | 32/32 | 32/32 | 32/32 | 42 / 46 / 38 | 0.37 |
| 0.05 | 32/32 | 32/32 | 32/32 | 34 / 35 / 33 | 0.29 |

- MuJoCo, mu=0.15: the 20 falls happen at steps 49-77 (median 59); the other 12 seeds are still walking at step 1000.
- Steel, mu=0.15: the 27 falls happen at steps 50-189 (median 140); the other 5 seeds are still walking at step 1000.
- MuJoCo, pyramidal cone, mu=0.2: the 5 falls happen at steps 61-107 (median 73); the other 27 seeds are still walking at step 1000.
- MuJoCo, pyramidal cone, mu=0.15: the 30 falls happen at steps 42-100 (median 52); the other 2 seeds are still walking at step 1000.

### Wind (MuJoCo, same seeds)

| condition | falls | of which before step 150 | after | median steps | body speed m/s | heading drift deg |
|---|---|---|---|---|---|---|
| mu=0.75 wind=20N | 0/32 | 0 | 0 | 1000 | +0.24 | 7 |
| mu=0.75 wind=40N | 0/32 | 0 | 0 | 1000 | -0.54 | 54 |
| mu=0.75 wind=60N | 30/32 | 20 | 10 | 138 | -1.46 | 33 |
| mu=0.75 wind=80N | 32/32 | 32 | 0 | 88 | -1.65 | 33 |
| mu=0.75 wind=20N cross | 0/32 | 0 | 0 | 1000 | +0.33 | 90 |
| mu=0.75 wind=40N cross | 0/32 | 0 | 0 | 1000 | -0.14 | 141 |
| mu=0.75 wind=60N cross | 32/32 | 28 | 4 | 64 | +0.48 | 30 |
| mu=0.75 wind=80N cross | 32/32 | 32 | 0 | 39 | +0.12 | 33 |
| mu=0.2 wind=20N | 0/32 | 0 | 0 | 1000 | +0.23 | 1 |
| mu=0.2 wind=40N | 6/32 | 6 | 0 | 1000 | -0.10 | 11 |
| mu=0.2 wind=60N | 32/32 | 32 | 0 | 96 | -0.67 | 86 |
| mu=0.2 wind=80N | 32/32 | 32 | 0 | 74 | -1.37 | 32 |
| mu=0.2 wind=20N cross | 0/32 | 0 | 0 | 1000 | +0.37 | 73 |
| mu=0.2 wind=40N cross | 31/32 | 14 | 17 | 340 | +0.73 | 50 |
| mu=0.2 wind=60N cross | 32/32 | 32 | 0 | 49 | +0.45 | 28 |
| mu=0.2 wind=80N cross | 32/32 | 32 | 0 | 36 | +0.17 | 27 |

### Slope at mu 0.75 (MuJoCo, same seeds, uphill)

| condition | falls | of which before step 150 | after | median steps | body speed m/s | heading drift deg |
|---|---|---|---|---|---|---|
| mu=0.75 slope=5deg | 0/32 | 0 | 0 | 1000 | +0.35 | 1 |
| mu=0.75 slope=10deg | 0/32 | 0 | 0 | 1000 | +0.29 | 11 |
| mu=0.75 slope=15deg | 17/32 | 0 | 17 | 986 | +0.14 | 88 |
| mu=0.75 slope=20deg | 31/32 | 7 | 24 | 268 | -0.01 | 74 |

### Friction threshold on a slope (MuJoCo, same seeds)

Margin = mu - tan(slope): the friction left over after holding position on the slope.

| slope | mu | margin | falls | median steps |
|---|---|---|---|---|
| 0 deg | 0.4 | 0.400 | 0/32 | 1000 |
| 0 deg | 0.3 | 0.300 | 0/32 | 1000 |
| 0 deg | 0.2 | 0.200 | 0/32 | 1000 |
| 0 deg | 0.15 | 0.150 | 20/32 | 66 |
| 0 deg | 0.1 | 0.100 | 32/32 | 42 |
| 0 deg | 0.05 | 0.050 | 32/32 | 34 |
| 5 deg | 0.4 | 0.313 | 0/32 | 1000 |
| 5 deg | 0.35 | 0.263 | 0/32 | 1000 |
| 5 deg | 0.3 | 0.213 | 0/32 | 1000 |
| 5 deg | 0.25 | 0.163 | 8/32 | 1000 |
| 5 deg | 0.2 | 0.113 | 32/32 | 47 |
| 10 deg | 0.4 | 0.224 | 2/32 | 1000 |
| 10 deg | 0.35 | 0.174 | 28/32 | 78 |
| 10 deg | 0.3 | 0.124 | 32/32 | 47 |
| 10 deg | 0.25 | 0.074 | 32/32 | 37 |
| 10 deg | 0.2 | 0.024 | 32/32 | 32 |

### Commanded speed on ice (MuJoCo, same seeds)

| condition | falls | of which before step 150 | after | median steps | body speed m/s | heading drift deg |
|---|---|---|---|---|---|---|
| mu=0.15 cmd=0m/s | 0/32 | 0 | 0 | 1000 | -0.01 | 101 |
| mu=0.15 cmd=0.25m/s | 0/32 | 0 | 0 | 1000 | +0.18 | 10 |
| mu=0.15 cmd=0.5m/s | 20/32 | 20 | 0 | 66 | +0.44 | 35 |
| mu=0.1 cmd=0m/s | 0/32 | 0 | 0 | 1000 | -0.01 | 122 |
| mu=0.1 cmd=0.25m/s | 23/32 | 23 | 0 | 68 | +0.30 | 28 |
| mu=0.1 cmd=0.5m/s | 32/32 | 32 | 0 | 42 | +0.37 | 24 |
| mu=0.05 cmd=0m/s | 12/32 | 12 | 0 | 1000 | +0.04 | 102 |
| mu=0.05 cmd=0.25m/s | 32/32 | 32 | 0 | 42 | +0.28 | 18 |
| mu=0.05 cmd=0.5m/s | 32/32 | 32 | 0 | 34 | +0.29 | 12 |

### Snow (depth 0.15 m, voxel 0.05 m, 4 coupled steps per tick, 1 seeds x 120 steps, device cpu)

| ground | yield pressure / cohesion | falls | fell at step | sinkage cm | body speed m/s | snow share of weight |
|---|---|---|---|---|---|---|
| bare | - | 0/1 | - | -0.4 | 0.39 | - |
| powder | 20 kPa / 0 kPa | 0/1 | - | 14.4 | 0.45 | -5% |
| packed | 100 kPa / 20 kPa | 1/1 | 72 | 10.7 | 0.53 | 33% |
| hard | 1000 kPa / 100 kPa | 1/1 | 61 | 1.9 | 0.61 | 76% |

<!-- END GENERATED -->

### Reading the numbers

- **The cliff is between mu 0.2 and 0.15, in both engines.** Above it nothing
  changes: same survival, same speed. Below it everything falls inside a second.
  Real ice is 0.05 to 0.15, so the whole of it is on the wrong side. Under the
  pyramidal cone the cliff starts slightly earlier (5/32 fall at 0.2), which is
  the direction its lower off-axis friction predicts.
- **At mu 0.15 the outcome is decided in the first seconds.** Every fall is before
  step 200 and the seeds that get through the start are still walking at 20 s. So
  that cell measures whether the robot survives the reset perturbation on ice,
  not a steady-state fall rate.
- **Slope spends friction.** Holding position on a slope already takes
  tan(slope) of the available friction. Define margin = mu - tan(slope). The
  policy stops walking at a margin of about 0.15 to 0.2 on the flat, on 5 degrees
  and on 10 degrees. So mu 0.2 is walkable flat and falls 32/32 on a 5 degree
  slope, and a 10 degree slope needs mu 0.4. The threshold creeps up a little
  with slope (2/32 still fall at margin 0.224 on 10 degrees).
- **Slowing down buys about one step of friction, no retraining.** At mu 0.15,
  commanding 0.25 m/s instead of 0.5 takes falls from 20/32 to 0/32. At mu 0.1 it
  takes 32/32 to 23/32, and standing still is 0/32. At mu 0.05 even standing
  loses 12/32. A speed limit keyed to estimated friction (`estimate_mu.py`) is
  therefore worth something on its own, down to about 0.15.
- **On grippy ground the slope limit is not friction.** At mu 0.75 the policy
  holds 10 degrees with no falls, loses half the seeds at 15 and nearly all at 20.
  Those falls mostly come late (after step 150), unlike the friction falls.
- **Wind pushes this policy around long before it knocks it over.** At 40 N head
  on (about 56 km/h at 5500 m by the drag model in `wind.py`) nobody falls on
  grippy ground, but the robot is walking backwards relative to its own body.
  Wind direction is fixed in the world while the robot's heading drifts, so a
  "headwind" row is a headwind only at the start.
- **Heading drifts by tens of degrees in 20 s.** The command is zero yaw rate and
  the policy has no heading reference. Any metric measured along a fixed world
  axis mixes slowing down with turning away.

## Non-obvious findings

**Both engines deliver the wrong friction by default, and in opposite ways.**
`SIM_ACCURACY.md` item 1 found Isaac's `average` combine mode. Here:

- `SolverMuJoCo` uses MuJoCo's rule, the **max** of the two geoms. A 0.75 boot on
  0.05 ice measures 0.751. The ice does not exist. Fix: `geom_priority=1` on the
  ground, which makes the ground's value win. Measured 0.052.
- `SolverSteel` takes contacts from Newton's own pipeline, which **averages**. The
  same pair measures 0.402. Newton has no combine-mode or priority setting for
  it. Fix: give the robot's shapes the ground's coefficient, so the pair averages
  to the value wanted. Measured 0.052.

Any new ground, robot or solver in this track goes through
`calibrate_friction.py` before its numbers mean anything.

**MuJoCo's default friction cone is anisotropic.** With the pyramidal cone,
nominal 0.20 delivers 0.187 along a contact axis and 0.143 on the diagonal. The
elliptic cone delivers 0.200 and 0.202. The track uses elliptic; the policy was
presumably trained on pyramidal, so the friction sweep is repeated under
`--cone pyramidal` as a check.

**Slope is tilted gravity, not tilted geometry.** Same physics for an infinite
plane, and it cannot repeat the radians bug that built an 81 degree wall in
`fixed_line_scene.py`. The projected-gravity observation uses the tilted vector.
The catch: the scene's z axis is then the ground normal, so the nominal pose
stands perpendicular to the slope, leaning back from vertical by the slope
angle. `_slope_stance` pitches the base to true vertical and takes the angle
back out at the ankles (feet within 0.5 degrees of flat, checked at build). The
first slope sweep did not do this. Fixing it changed no conclusion (5 degrees at
mu 0.2 is 32/32 falls either way) but the earlier slope rows were discarded.

**`use_mujoco_cpu=True` is the fast path off-GPU.** It runs MuJoCo's C engine
instead of mujoco_warp: 1000 policy steps in about 2.5 s on this Mac, roughly
eight times faster than mujoco_warp on the same CPU. It needs MuJoCo's own
contacts, so it is not available to the coupled snow scene.

**World-frame progress is not speed.** The first Steel sweep reported 0.04 m/s at
mu 0.3 and looked like a solver disagreement. The robot was walking at 0.43 m/s in
a slow arc. `mean_speed` is now the body-frame forward velocity.

**`ViewerGL(headless=True)` still needs a display on macOS**, and returns frames
at twice the requested size on a Retina screen. With no screen pyglet dies with
"list index out of range". With the wrong frame size ffmpeg writes a clip of
sky-coloured noise and reports success. `film.py` sizes the encoder from the
first frame.

**The policy's USD carries no materials.** Every part imports at one grey and the
viewer's default half-gloss renders a chrome mannequin. `paint_g1` reads the
per-part finish from Unitree's own MJCF in the same asset folder (dark pelvis,
hip housings, feet and head; light shells elsewhere) and makes the meshes matte.
Rendering only: stored rollouts reproduce exactly after the change.

## Snow

Status: **the scene runs and the plumbing is checked. The physics is not yet
trustworthy enough to quote.** The snow table above is one seed per row on 5 cm
voxels, and it fails its own sanity check: on `hard` snow the robot sinks under
2 cm and still falls at step 61, while the same seed walks the bare-ground
control. A surface firm enough to stand on should reproduce the control. Until
it does, a fall on softer snow cannot be blamed on the snow.

What the table does show, qualitatively: sinkage and the share of weight the
snow carries both move the right way with snow strength (14 cm and none for
powder, 11 cm and a third for packed, 2 cm and three quarters for hard), and
cohesionless powder lets the foot through to the base whatever its yield
pressure.

What is established, in the order it was learned:

1. **The robot must be spawned on the snow.** The first version left it at ground
   level, buried to the shins, and reported the full depth as sinkage.
2. **Count snow force on the robot only.** `collect_collider_impulses` includes
   the static floor, which carries the snow's own weight. For the smoke bed that
   is about 330 N, the robot's weight by coincidence. The first version reported
   "snow carries 98 % of the robot" for a robot standing on the floor in two
   post-holes.
3. **One coupled step per 20 ms tick is too coarse for the robot.** The feet reach
   the floor in 0.2 s, close to free fall, whatever the snow strength (yield
   pressure 5 kPa to 500 kPa, cohesion 0 to 50 kPa). At four coupled steps per
   tick (5 ms) a standing robot on packed snow gets about 80 % of its weight from
   the snow. `staggered` coupling mode and a proxy `mass_scale` of 50 or 300 did
   not help at 20 ms.
4. **A free block does not show this.** `snow_block_test.py`: a 34 kg block with
   proper inertia is held at both step sizes and only sinks 1.9 cm further at
   20 ms. So the failure in 3 is specific to an articulated robot loading the snow
   through 0.6 kg foot links. That is not explained.
5. **A body with mass but no inertia falls through at any step size.** A trap for
   whoever writes the next test, not a property of snow.

### The hard-snow failure, diagnosed and not fixed (2026-11-27)

A debugging agent spent a day on the sanity check. It did not make the robot
stand. Everything here is seed 0, one run, 5 cm voxels unless stated, so each
row is an illustration. The weight behind the conclusion is that more than
thirty variants fail the same way. Every variant is an argument of `G1Snow`,
defaulting to the old behaviour.

What it found:

- **It is not a walking problem.** Told to stand still, the robot on bare
  ground is at rest within a second. On hard snow it never settles: it marches
  in place, the loaded sole ends up rolled and pitched 10 to 20 degrees, and it
  falls or is about to by 2 s.
- **It needs the policy in the loop.** With joint targets held and no policy,
  the robot topples at tick 54 on bare ground and at ticks 53 and 51 on snow.
  The bed carries a passive robot as well as the ground does.
- **The bed gives a stance foot a fraction of the moment support rigid ground
  does.** `snow_foot_test.py`: a sole-sized plate under the robot's weight has
  tilted 2 degrees at 21 to 55 % of the rigid tipping torque in roll, in every
  variant, including snow that cannot deform at all and voxels down to 2.5 cm.
  So it is not snow stiffness, strength or resolution.
- **The force is not one-sided.** Under the policy the snow pulls a lifting foot
  down in about a fifth of foot-ticks, by up to 70 N with the stock collider,
  and sideways by about 100 N.
- **The mean vertical force is right**: 0.98 to 1.00 of the weight while the
  robot is up. The 76 % in the table is averaged over the fall.

Which of the weak moment support and the grabbing is the trigger was not
separated.

Two things wrong in the snow table above:

- **The 1.9 cm "sinkage" on hard snow is the spawn pose.** The nominal pose
  puts the soles 2.4 cm below the surface. MuJoCo's ground pushes them out. The
  MPM contact is velocity-level and never does. `--spawn-clearance 0.024`
  starts them on the surface.
- **The stock foot collider is a box 2 cm thick.** Grid nodes above its top
  face push the foot down. `--foot-collider-height 0.12` removes that and does
  not make the robot stand.

The five suspects from the brief:

| suspect | verdict | evidence |
|---|---|---|
| unconfined bed | no | with walls it falls at step 58 (stock 61) |
| coupling force quality | not the cause alone | every coupling variant below still falls |
| foot slides on snow | no, the opposite | the stock foot box does not slide under a pull of 1.5 x its load |
| voxel size | no | at 3.5 cm it falls at tick 82 standing; the plate is no better at 3.5 or 2.5 cm |
| spawn transient | no | stand 50 ticks, then walk: falls 24 ticks after the walk command |

Walking on hard snow, 120 steps:

| variant | survived | sink cm | snow share of weight |
|---|---|---|---|
| bare control | 120 | -0.4 | - |
| stock | 61 | 1.9 | 0.76 |
| walls | 58 | 1.0 | 0.73 |
| 8 coupled steps per tick | 64 | 8.4 | 0.65 |
| relaxation 0.5 | 66 | 4.0 | 0.81 |
| Aitken, 3 passes | 57 | 3.8 | 0.72 |
| 12 cm foot collider | 63 | -0.5 | 0.81 |
| stand 50 ticks, then walk | 74 of 170 | 1.6 | 0.82 |

Standing at zero command on hard snow, 100 ticks. "clr" is the 2.4 cm spawn
clearance, "E x" multiplies the Young's modulus, "proj" projects particles out
of the feet each step. Speed is the mean horizontal body speed after tick 25: a
robot at rest reads about 0.03.

| variant | survived | speed m/s | max foot tilt deg | snow share |
|---|---|---|---|---|
| bare control | 100 | 0.03 | 10 | - |
| stock | 87 | 0.70 | 122 | 0.79 |
| E x100 | 99 | 0.64 | 120 | 0.82 |
| E x100 + relaxation 0.5 | 100 | 0.10 | 32 | 0.96 |
| 12 cm collider | 100 | 0.62 | 131 | 0.68 |
| 12 cm + clr | 100 | 0.09 | 48 | 0.98 |
| 12 cm + clr + proj + Aitken 3 | 100 | 0.27 | 47 | 0.99 |
| 12 cm + clr, 3.5 cm voxels | 82 | 0.45 | 116 | 0.86 |
| 12 cm + clr + E x100 + relaxation 0.5 | 82 | 0.51 | 105 | 0.78 |

Thirteen more combinations sit between these. The two best rows (speed about
0.1 m/s) are still marching and tilting at 2 s. Neither settles, and combining
them is worse than either.

Open before any snow number is quoted:

- The hard-snow sanity check, still. A fix probably needs a contact with
  positional non-penetration that feeds back as force. The proxy coupling in
  Newton 1.8.0 does not have one, and its ADMM coupled solver does not appear
  to cover MPM.
- The default foot proxy has about 3x the true articulated inertia in pitch and
  yaw and a third of it in roll. Correcting it did not help on its own. Its
  interaction with relaxation was not explored.
- Everything above is one seed.
- Snow parameters are order of magnitude: 300 kg/m^3, three named types in
  `snow_mpm.SNOW`. Nothing is fitted to a snow pit.
- The policy has never felt sinkage. Even with the physics fixed, a fall in snow
  says the flat-ground policy trips while wading, not that snow is unwalkable.

**Do not spend GPU money on finer voxels expecting this to pass.** The plate
test shows no improvement from 5 cm to 2.5 cm, and it runs free on CPU. Run it
first.

GPU run, if the plate test ever passes (this costs money: ask first, as for every job here):

```bash
hf cp newton_track/g1_newton.py hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/
hf jobs uv run --namespace iteratehack --flavor l4x1 --timeout 40m \
  -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
  --label name=himalaya-traction --label task=newton-snow \
  newton_track/snow_mpm.py --voxel 0.025 --seeds 8 --steps 500
```

That line has not been run. `snow_mpm.py` carries inline `uv` metadata and looks
for `g1_newton.py` in `/mnt/himalaya-g1`, following the bucket convention in
`CLAUDE.md`; do the free `cpu-basic` pass with `--smoke` first.

## What is NOT claimed

- That this policy represents the repo's own. It is a different network with a
  different observation layout (141 inputs, 43 joints including hands).
- That step counts compare across tracks. `STATE.md` section 5 stands: fall-step
  counts are not comparable across physics backends.
- Anything about snow beyond the five numbered points and the one-seed
  diagnosis above. The robot does not yet stand on particle snow.
- That the Steel result is independent of friction modelling choices. Steel ran
  with default settings. Two others (50 solver iterations; 4 velocity iterations)
  were tried on 4 seeds x 300 steps at mu 0.3 and 0.75 and changed speed by under
  0.01 m/s. Its other friction modes were not evaluated.
- Recovery. Every number here is time to first fall.
