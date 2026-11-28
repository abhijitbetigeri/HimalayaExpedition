"""What friction does the snow bed give a foot? The MPM analogue of calibrate_friction.py.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_friction_test.py

On bare ground the robot's foot gets a Coulomb coefficient of 0.75 against the
plane, and `calibrate_friction.py` measures that it is delivered. On snow there
is no such contact: the foot is an MPM collider, and the tangential force is
whatever the grid contact (Coulomb, coefficient = the shape's `mu`) and the snow
under it (internal friction + cohesion) can pass on. This file measures it.

A block with the footprint of one G1 foot (0.20 x 0.07 m) and half the robot's
weight is set on HARD snow, left to settle, and then pulled sideways by a force
that ramps from zero to 1.5 x its weight. Reported per case:

  mu_break   pull / weight at the moment the block has slid 5 mm; "> 1.5" if it
             never does
  T/N        tangential over normal snow force on the block at that moment,
             as harvested by the coupling (what MuJoCo is actually handed)
  sink       how far the underside ends up below the snow surface before the pull

The cases differ only in the collider's height: 2 cm is the stock G1 foot box,
12 cm is the `--foot-collider-height 0.12` variant of `snow_mpm.py`.

Measured 2026-11-27, 5 cm voxels, one run per cell:

    collider   pull along   mu_break    T/N   sink cm
      2 cm     long axis     > 1.5       -      0.31
      2 cm     short axis    > 1.5       -      0.31
     12 cm     long axis      0.91     0.92     0.18
     12 cm     short axis     0.44     0.48     0.18

So the snow is not ice to the foot. The stock box sits 3 mm into the bed and is
keyed there: on a 5 cm grid a 3 mm lip resists like a 5 cm one. The tall box
slides, at more than twice the 0.15 to 0.2 below which this policy falls on bare
ground, but not at the 0.75 the shape asks for and not equally along both axes.
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
HX, HY, MASS = 0.10, 0.035, 17.0
MU = 0.75  # the robot's shape coefficient
DT = 0.005


def pull(*, height: float, direction: tuple[float, float] = (1.0, 0.0), mass_scale: float = 1.0,
         settle_s: float = 0.4, ramp_s: float = 1.2, peak: float = 1.5) -> dict:
    hz = 0.5 * height
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    # Inertia of a 10 cm tall block in every case, so the cases differ in collider geometry only.
    ih = 0.05
    ix = MASS / 3.0 * np.array([HY**2 + ih**2, HX**2 + ih**2, HX**2 + HY**2])
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, DEPTH + hz + 0.002)), mass=MASS, inertia=wp.mat33(np.diag(ix))
    )
    builder.add_shape_box(body, hx=HX, hy=HY, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU))
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
        custom_attributes={  # "hard" snow from snow_mpm.SNOW
            "mpm:young_modulus": 1.4e6, "mpm:poisson_ratio": 0.3, "mpm:friction": 0.5, "mpm:damping": 0.01,
            "mpm:yield_pressure": 1.0e6, "mpm:tensile_yield_ratio": 0.2, "mpm:yield_stress": 1.0e5,
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
                    source="mjc", destination="mpm", bodies=[body], mode="lagged", mass_scale=mass_scale,
                    collision_pipeline=lambda _m: None,
                )
            ]
        ),
    )
    feedback = solver._proxy_mappings[0].coupling_forces  # wrench handed to MuJoCo on the next step
    state = model.state()
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    weight = MASS * 9.81
    d = np.array([*direction, 0.0]) / np.linalg.norm(direction)
    start = None
    out = {"height": height, "mu_break": float("nan"), "t_over_n": float("nan"), "sink_cm": float("nan")}
    n_settle, n_ramp = round(settle_s / DT), round(ramp_s / DT)
    for i in range(n_settle + n_ramp):
        state.clear_forces()
        frac = 0.0
        if i >= n_settle:
            frac = peak * (i - n_settle + 1) / n_ramp
            f = state.body_f.numpy()
            f[body, :3] = frac * weight * d
            state.body_f.assign(f)
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, DT)
        p = state.body_q.numpy()[body][:3]
        if i == n_settle - 1:
            start = p.copy()
            out["sink_cm"] = round(100 * float(DEPTH - (p[2] - hz)), 2)
        if start is not None and float((p - start) @ d) > 0.005:
            w = feedback.numpy()[body]
            out["mu_break"] = round(frac, 3)
            out["t_over_n"] = round(float(np.hypot(w[0], w[1]) / max(w[2], 1e-6)), 3)
            break
    return out


def main() -> int:
    print(f"block {2 * HX:.2f} x {2 * HY:.2f} m footprint, {MASS:g} kg, shape mu {MU}, on {DEPTH} m of HARD snow, "
          f"voxel {VOXEL} m, {DT * 1e3:g} ms coupled step\n")
    print(f"{'collider height':>16} {'pull along':>11} | {'mu_break':>8} {'T/N':>6} {'sink cm':>8}")
    rows = []
    for height in (0.02, 0.12):
        for name, direction in (("long axis", (1.0, 0.0)), ("short axis", (0.0, 1.0))):
            r = pull(height=height, direction=direction)
            rows.append(r)
            held = np.isnan(r["mu_break"])
            mu = f"{'> ' + format(1.5, 'g'):>8}" if held else f"{r['mu_break']:>8.2f}"
            tn = f"{'-':>6}" if held else f"{r['t_over_n']:>6.2f}"
            print(f"{height:>14.2f} m {name:>11} | {mu} {tn} {r['sink_cm']:>8.2f}", flush=True)
    # A foot that slides below the ice threshold of the friction sweep (0.15 to 0.2) would explain a fall.
    slippery = [r for r in rows if r["mu_break"] < 0.3]  # NaN (never slid) compares False
    if slippery:
        print("\nFAIL: a collider slides at a pull below 0.3 x weight; the snow surface is ice to the foot.")
        return 1
    print("\nOK: no collider slides below a pull of 0.3 x weight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
