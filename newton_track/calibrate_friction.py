"""Measure the friction the ground actually delivers, instead of trusting config.

SIM_ACCURACY.md item 1 found that Isaac's default `average` combine mode turned
"ice at 0.06" into ground at 0.43. MuJoCo's default is `max`, which is worse: a
0.75 boot on 0.05 ice is simulated as 0.75. `g1_newton.add_ground` gives the
ground `geom_priority=1` so its coefficient wins. This script checks that claim
with a block: it rests on the ground under tilted gravity, and it slides if and
only if tan(tilt) exceeds the effective coefficient.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/calibrate_friction.py

For each nominal ground mu it bisects the tilt at which a block starts to slide
and reports mu_eff = tan(critical tilt), in four configurations: the two
`g1_newton` uses (MuJoCo with ground priority; Steel with both sides at mu) and
the two it avoids (MuJoCo's default max rule; Steel with a 0.75 block, which
Newton's contact pipeline averages). Exit status is non-zero if either
configuration in use is off by more than 10 %.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g1_newton import CONE, add_ground, gravity_vector  # noqa: E402

BLOCK_MU = 0.75  # same as the robot's shapes


def slides(mu_ground: float, tilt_deg: float, *, mode: str, seconds: float = 1.5) -> bool:
    """mode: 'priority' (what the MuJoCo path uses), 'max' (MuJoCo default),
    'steel' (SolverSteel, both sides at mu, what g1_newton does for Steel) or
    'steel_avg' (SolverSteel with a 0.75 block: Newton's pipeline averages)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    if mode == "max":
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu_ground))
    else:
        add_ground(builder, mu_ground)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05)), mass=5.0)
    block_mu = mu_ground if mode == "steel" else BLOCK_MU
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=block_mu))
    model = builder.finalize()
    model.set_gravity(tuple(float(g) for g in gravity_vector(tilt_deg)))
    s0, s1 = model.state(), model.state()
    control = model.control()
    dt = 1.0 / 200
    if mode.startswith("steel"):
        solver = newton.solvers.SolverSteel(model)
        pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
        contacts = pipeline.contacts()
    else:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, solver="newton", cone=CONE)
        pipeline, contacts = None, None
    for _ in range(int(seconds / dt)):
        s0.clear_forces()
        if pipeline is not None:
            pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    # Gravity pulls toward -y when the slope rises along +y.
    y = float(s0.body_q.numpy()[body][1])
    return y < -0.02


def critical_mu(mu_ground: float, mode: str) -> float:
    lo, hi = 0.0, 60.0  # degrees: holds at lo, slides at hi
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if slides(mu_ground, mid, mode=mode):
            hi = mid
        else:
            lo = mid
    return math.tan(math.radians(0.5 * (lo + hi)))


def main() -> int:
    rows = []
    worst = 0.0
    print(f"block mu = {BLOCK_MU}; effective mu = tan(tilt at which it starts to slide); cone = {CONE}\n")
    print(f"{'nominal':>8} | {'MuJoCo+priority':>15} {'MuJoCo default':>15} | {'Steel, both=mu':>15} {'Steel, 0.75 block':>18}")
    for mu in (0.05, 0.1, 0.2, 0.4, 0.75):
        eff = {m: critical_mu(mu, m) for m in ("priority", "max", "steel", "steel_avg")}
        for m in ("priority", "steel"):  # the two configurations g1_newton actually uses
            worst = max(worst, abs(eff[m] - mu) / mu)
        rows.append({"nominal": mu, **{k: round(v, 4) for k, v in eff.items()}})
        print(f"{mu:>8g} | {eff['priority']:>15.3f} {eff['max']:>15.3f} | {eff['steel']:>15.3f} {eff['steel_avg']:>18.3f}")
    out = pathlib.Path(__file__).resolve().parent / "results" / "friction_calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"block_mu": BLOCK_MU, "cone": CONE, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    if worst > 0.10 or not np.isfinite(worst):
        print(f"FAIL: worst error in a configuration we use is {worst:.1%} (limit 10 %)")
        return 1
    print(f"OK: worst error in the two configurations we use is {worst:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
