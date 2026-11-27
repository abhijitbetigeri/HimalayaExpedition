"""Does the snow bed carry a plain block? A preflight for snow_mpm.py.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_block_test.py

`snow_mpm.py` couples MuJoCo to implicit MPM through Newton's experimental
`SolverCoupledProxy`. While building it the robot repeatedly fell straight
through 15 cm of snow in 0.2 s, i.e. free fall, while the script went on printing
plausible numbers. This file puts a single block on the same bed, so a coupling
fault can be told apart from snow rheology before a robot is involved.

Each case drops a block on the bed and reports where its underside ends up
after 1 s. `held` means it stayed in the upper two thirds of the bed.

Expected, and asserted (exit status 1 otherwise):

  good           held        proper inertia, 5 ms coupled step
  coarse_dt      held, lower same block, one coupled step per 20 ms policy tick
  zero_inertia   sinks       mass set on the body, shape density 0, no inertia

What the cases say about the robot. A FREE block survives the 20 ms step and
only sinks further. The ROBOT does not: at one coupled step per tick its feet
reach the floor in 0.2 s, at four per tick the snow carries most of its weight
(numbers in NEWTON_TRACK.md). The difference is not explained. The working
guess is that a foot is a 0.6 kg link whose load arrives through joints, so the
one-step lag in "lagged" feedback matters far more than for a 34 kg free body;
raising the proxy `mass_scale` to 50 and 300 did NOT fix the 20 ms case, which
counts against the simplest version of that guess.

`zero_inertia` is a trap for anyone writing the next test: a body given a mass
but no inertia, with a density-0 shape, falls through even at 5 ms.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy

wp.config.quiet = True

DEPTH = 0.15
VOXEL = 0.05
# A block the size of two G1 feet side by side, carrying the robot's weight.
HX, HY, HZ, MASS = 0.10, 0.07, 0.05, 34.0


def drop(*, dt: float, give_inertia: bool, seconds: float = 1.0) -> float:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    kw = {}
    if give_inertia:
        ix = MASS / 3.0 * np.array([HY**2 + HZ**2, HX**2 + HZ**2, HX**2 + HY**2])
        kw["inertia"] = wp.mat33(np.diag(ix))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, DEPTH + HZ + 0.002)), mass=MASS, **kw)
    builder.add_shape_box(body, hx=HX, hy=HY, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.add_ground_plane()

    lo, hi = np.array([-0.3, -0.3, 0.0]), np.array([0.3, 0.3, DEPTH])
    res = np.ceil(2 * (hi - lo) / VOXEL).astype(int)
    cell = (hi - lo) / res
    radius = float(cell.max() * 0.5)
    builder.add_particle_grid(
        pos=wp.vec3(*lo), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1, dim_y=int(res[1]) + 1, dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
        mass=float(np.prod(cell) * 300.0), jitter=0.5 * radius, radius_mean=radius,
        custom_attributes={  # "packed" snow from snow_mpm.SNOW
            "mpm:young_modulus": 1.4e6, "mpm:poisson_ratio": 0.3, "mpm:friction": 0.5, "mpm:damping": 0.01,
            "mpm:yield_pressure": 1.0e5, "mpm:tensile_yield_ratio": 0.2, "mpm:yield_stress": 2.0e4,
            "mpm:hardening": 5.0,
        },
    )  # fmt: skip
    model = builder.finalize()

    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = VOXEL
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
                substeps=max(1, round(dt * 200)),
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
                    source="mjc", destination="mpm", bodies=[body], mode="lagged", collision_pipeline=lambda _m: None
                )
            ]
        ),
    )
    state = model.state()
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    for _ in range(round(seconds / dt)):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, dt)
    return float(state.body_q.numpy()[body][2]) - HZ


CASES = {
    "good": (dict(dt=0.005, give_inertia=True), True),
    "coarse_dt": (dict(dt=0.02, give_inertia=True), True),
    "zero_inertia": (dict(dt=0.005, give_inertia=False), False),
}


def main() -> int:
    ok = True
    print(f"block {2 * HX:.2f} x {2 * HY:.2f} m, {MASS:g} kg, on {DEPTH} m of packed snow, voxel {VOXEL} m\n")
    zs = {}
    for name, (kw, expect_held) in CASES.items():
        z = zs[name] = drop(**kw)
        held = z > DEPTH / 3
        verdict = "held" if held else "sinks"
        match = held == expect_held
        ok &= match
        print(f"{name:<13} underside at {z:+.3f} m  {verdict:<5}  {'as expected' if match else 'UNEXPECTED'}")
    deeper = zs["coarse_dt"] < zs["good"]
    print(f"\ncoarse_dt sinks {100 * (zs['good'] - zs['coarse_dt']):.1f} cm further than good"
          f"  {'as expected' if deeper else 'UNEXPECTED'}")
    return 0 if ok and deeper else 1


if __name__ == "__main__":
    sys.exit(main())
