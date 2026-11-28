"""Can the snow bed hold one G1 foot the way rigid ground does? A preflight for snow_mpm.py.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_foot_test.py                # 5 cm voxels
    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_foot_test.py --voxel 0.025  # slow on CPU

`snow_mpm.py` fails its own sanity check: on HARD snow the robot barely sinks
and still falls where the bare-ground control walks. `snow_block_test.py` only
asks whether the bed carries a weight. A walking robot needs more than that from
the ground under a stance foot: the foot must stay LEVEL, and the ground must
return the ankle's roll and pitch torque. Rigid ground does both until the centre
of pressure reaches the edge of the sole, i.e. up to W x 3.5 cm in roll and
W x 10 cm in pitch for the 0.20 x 0.07 m sole under load W.

This file puts a plate with the sole's footprint on hard snow, loads it with the
robot's full weight (single stance), and then ramps a torque about one axis up
to 1.2 x the rigid-ground tipping torque. It reports

  sink      how far the plate settles under the load alone, mm
  self tilt how far it rolls or pitches under the load alone, degrees (rigid: 0)
  M(1 deg), M(2 deg), M(5 deg)
            the torque at which the plate has tilted that far, as a fraction of
            the rigid-ground tipping torque. Rigid ground: 1.0 at every angle.

The plate is one free body, so two things are set to match the robot rather than
a loose plate. Its mass is the effective mass MuJoCo reports for the G1 foot
(2.6 kg, what the proxy collider is given), with the rest of the weight as an
external load. Its inertia is the articulated inertia measured at the foot of the
floating robot (0.10 kg m^2 in roll, 0.08 in pitch). With a loose plate's own
inertia (0.003 in roll) the same test mostly measures the one-step lag of the
"lagged" coupling, which is 30 x weaker at the robot's inertia.

Exit status 0 only if the bed holds the plate level (self tilt under 1 degree)
and returns at least 70 % of the rigid tipping torque by 2 degrees on both axes.
It does not. Measured 2026-11-27 with the defaults (5 cm voxels, stock 2 cm
collider, one coupling pass), one run per row:

      axis  rigid tip Nm |  sink mm  self tilt |  M(1 deg)  M(2 deg)  M(5 deg)
      roll          11.9 |     21.3      +1.22 |      0.23      0.48      0.74
     pitch          34.0 |     21.3      -0.14 |      0.44      0.64      0.70

and M(2 deg) in roll stayed between 0.21 and 0.55 through every variant tried
with an earlier copy of this test (same plate, bed and inertia, a 12 cm tall
collider, one run each): as is 0.28, three Aitken passes 0.55, Q1 collider
basis 0.41, 3.5 cm voxels 0.24, 2.5 cm voxels 0.21, 300 rheology iterations at
tolerance 1e-7 0.34, and snow that cannot deform at all (rigid elasticity,
100 x the strength) 0.23. So the missing support is not the snow's stiffness or strength and it is
not cured by resolution: it comes from the collider contact and its coupling.
Some of the tilt stays when the torque is removed: taken to 0.4 of the tipping
torque and back, the plate keeps 0.5 to 1.2 of its 1.7 to 3.9 degrees.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy

wp.config.quiet = True

DEPTH = 0.15
HX, HY = 0.10, 0.035  # the G1 foot's collision box
PROXY_MASS = 2.6  # MuJoCo's effective mass for the ankle-roll link
WEIGHT = 340.0  # the whole robot on one foot
FOOT_INERTIA = {"roll": 0.098, "pitch": 0.077}  # articulated, measured on the floating robot
DT = 0.005


def tip(axis: str, *, voxel: float, height: float, iterations: int, young: float, yield_pressure: float,
        cohesion: float, bed: tuple[float, float], settle_s: float = 0.2, ramp_s: float = 0.6) -> dict:
    hz = 0.5 * height
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    # x is the sole's long axis: a torque about x rolls it, a torque about y pitches it.
    inertia = np.array([FOOT_INERTIA["roll"], FOOT_INERTIA["pitch"], FOOT_INERTIA["pitch"]])
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, DEPTH + hz + 0.001)), mass=PROXY_MASS, inertia=wp.mat33(np.diag(inertia))
    )
    builder.add_shape_box(body, hx=HX, hy=HY, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.75))
    builder.add_ground_plane()

    lo, hi = np.array([-bed[0], -bed[1], 0.0]), np.array([bed[0], bed[1], DEPTH])
    res = np.ceil(2 * (hi - lo) / voxel).astype(int)
    cell = (hi - lo) / res
    radius = float(cell.max() * 0.5)
    builder.add_particle_grid(
        pos=wp.vec3(*lo), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1, dim_y=int(res[1]) + 1, dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
        mass=float(np.prod(cell) * 300.0), jitter=0.5 * radius, radius_mean=radius,
        custom_attributes={
            "mpm:young_modulus": young, "mpm:poisson_ratio": 0.3, "mpm:friction": 0.5, "mpm:damping": 0.01,
            "mpm:yield_pressure": yield_pressure, "mpm:tensile_yield_ratio": 0.2, "mpm:yield_stress": cohesion,
            "mpm:hardening": 5.0,
        },
    )  # fmt: skip
    model = builder.finalize()

    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = voxel
    cfg.strain_basis = "P0"
    cfg.max_iterations = 50
    solver = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=100),
                bodies=[body],
                joints=list(range(model.joint_count)),
                substeps=1,
            ),
            SolverCoupledProxy.Entry(
                name="mpm",
                solver=lambda v: SolverImplicitMPM(model=v, config=cfg),
                particles=list(range(model.particle_count)),
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjc", destination="mpm", bodies=[body], mode="lagged",
                    proxy_relaxation_mode="aitken" if iterations > 1 else "fixed",
                    collision_pipeline=lambda _m: None,
                )
            ],
            iterations=iterations,
        ),
    )
    state = model.state()
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    rigid_tip = WEIGHT * (HY if axis == "roll" else HX)
    torque_index = 3 if axis == "roll" else 4
    n_settle, n_ramp = round(settle_s / DT), round(ramp_s / DT)
    marks: dict[float, float | None] = {1.0: None, 2.0: None, 5.0: None}
    out = {"axis": axis, "rigid_tip_nm": rigid_tip}
    angle0 = 0.0
    for i in range(n_settle + n_ramp):
        state.clear_forces()
        torque = 1.2 * rigid_tip * max(0, i - n_settle + 1) / n_ramp
        f = state.body_f.numpy()
        # The load comes on over the first half of the settle, not as a step.
        f[body, 2] = -(WEIGHT - PROXY_MASS * 9.81) * min(1.0, (i + 1) / (0.5 * n_settle))
        f[body, torque_index] = torque
        state.body_f.assign(f)
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, DT)
        q = state.body_q.numpy()[body]
        rot = wp.quat(*[float(v) for v in q[3:7]])
        if axis == "roll":
            angle = float(np.degrees(np.arcsin(wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))[2])))
        else:
            angle = float(np.degrees(np.arcsin(-wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))[2])))
        if i == n_settle - 1:
            angle0 = angle
            out["sink_mm"] = 1000.0 * (DEPTH - (float(q[2]) - hz))
            out["self_tilt_deg"] = angle
        if i >= n_settle:
            for deg in marks:
                if marks[deg] is None and abs(angle - angle0) > deg:
                    marks[deg] = torque / rigid_tip
            if abs(angle - angle0) > 20.0:
                break
    out["fraction_at_deg"] = marks
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--height", type=float, default=0.02, help="collider height, m; 0.02 is the stock G1 foot box")
    ap.add_argument("--iterations", type=int, default=1, help="proxy coupling passes per step (Aitken above 1)")
    ap.add_argument("--young", type=float, default=1.4e6)
    ap.add_argument("--yield-pressure", type=float, default=1.0e6, help="default: hard snow")
    ap.add_argument("--cohesion", type=float, default=1.0e5, help="default: hard snow")
    ap.add_argument("--bed", type=float, nargs=2, default=(0.2, 0.15), help="bed half extents x y, m")
    args = ap.parse_args()

    print(f"plate {2 * HX:.2f} x {2 * HY:.2f} m, {args.height * 100:g} cm tall, {WEIGHT:g} N, on {DEPTH} m of snow "
          f"(yield {args.yield_pressure / 1e3:g} kPa, cohesion {args.cohesion / 1e3:g} kPa, E {args.young / 1e6:g} MPa), "
          f"voxel {args.voxel} m, {args.iterations} coupling pass(es)\n")
    print(f"{'axis':>6} {'rigid tip Nm':>13} | {'sink mm':>8} {'self tilt':>10} | {'M(1 deg)':>9} {'M(2 deg)':>9} {'M(5 deg)':>9}")
    ok = True
    for axis in ("roll", "pitch"):
        r = tip(axis, voxel=args.voxel, height=args.height, iterations=args.iterations, young=args.young,
                yield_pressure=args.yield_pressure, cohesion=args.cohesion, bed=tuple(args.bed))
        m = r["fraction_at_deg"]
        cells = "".join(f"{'held' if m[d] is None else format(m[d], '.2f'):>10}" for d in (1.0, 2.0, 5.0))
        print(f"{axis:>6} {r['rigid_tip_nm']:>13.1f} | {r['sink_mm']:>8.1f} {r['self_tilt_deg']:>+10.2f} |{cells}", flush=True)
        ok &= abs(r["self_tilt_deg"]) < 1.0 and (m[2.0] is None or m[2.0] >= 0.7)
    print("\nM = torque at that tilt over the rigid-ground tipping torque; 'held' = never tilted that far up to 1.2.")
    print("OK: the bed supports the foot like firm ground." if ok else
          "FAIL: the bed lets a loaded foot tilt well below the torque rigid ground would return.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
