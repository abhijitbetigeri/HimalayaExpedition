"""Paired sweep of the Newton-shipped G1 policy over friction, wind and slope.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/ice_sweep.py
    ~/projects/newton-sandbox/.venv/bin/python newton_track/ice_sweep.py --seeds 64 --steps 1000

What this is for. The MuJoCo track's survival numbers suffer from two problems
CLAUDE.md documents at length: GPU rollouts are not reproducible run to run
(+-35 steps in the mean), and the baseline policy only walks on a minority of
seeds, so there is no known-good control. This sweep has neither problem:

- it runs MuJoCo's C engine on CPU, which is deterministic. `--check-determinism`
  runs every cell twice and fails if a single survival count differs;
- the control cell (mu 0.75, no wind, flat) must survive on EVERY seed, or the
  script exits non-zero before printing any comparison.

Every condition sees the same seeds, so each row is paired with the control.
Reported per cell: falls out of n, median and mean survival, mean body-frame
forward speed while upright, and median absolute heading drift. Survival is censored at --steps; a cell where nobody falls says
"no falls in N steps", not "never falls".

The policy is not ours and was never trained on ice. The sweep measures where a
competent flat-ground humanoid policy stops working, on calibrated friction.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g1_newton import CONE, POLICY_HZ, Conditions, G1Himalaya  # noqa: E402

MU_GRID = (0.75, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05)
WIND_GRID = (0.0, 20.0, 40.0, 60.0, 80.0)  # newtons on the torso
SLOPE_GRID = (0.0, 5.0, 10.0, 15.0, 20.0)  # degrees uphill
MARGIN_SLOPES = (5.0, 10.0)
MARGIN_MUS = (0.2, 0.25, 0.3, 0.35, 0.4)
# Friction levels at which wind and slope are swept: home turf, and the lowest
# friction the policy still walks on.
SWEEP_MUS = (0.75, 0.2)


def run_cell(cond: Conditions, seeds: range, steps: int, cone: str, rigid: str = "mujoco") -> dict:
    env = G1Himalaya(cond, cone=cone, rigid=rigid)
    rolls = [env.rollout(s, steps) for s in seeds]
    surv = np.array([r["survived"] for r in rolls])
    speeds = np.array([r["mean_speed"] for r in rolls])
    return {
        "label": cond.label(),
        "mu_ground": cond.mu_ground,
        "slope_deg": cond.slope_deg,
        "wind_n": cond.wind_n,
        "wind_dir_deg": cond.wind_dir_deg,
        "n": len(rolls),
        "falls": int(sum(r["fell"] for r in rolls)),
        "survived_median": float(np.median(surv)),
        "survived_mean": round(float(surv.mean()), 1),
        "survived_min": int(surv.min()),
        "speed_mean": round(float(speeds.mean()), 3),
        "abs_yaw_deg_median": round(float(np.median([abs(r["yaw_deg"]) for r in rolls])), 1),
        "survived": surv.tolist(),
    }


def fmt(cell: dict, steps: int) -> str:
    return (
        f"{cell['label']:<28} falls {cell['falls']:>3}/{cell['n']:<3} "
        f"median {cell['survived_median']:>6.0f}/{steps}  mean {cell['survived_mean']:>6.1f}  "
        f"speed {cell['speed_mean']:>5.2f} m/s  |yaw| {cell['abs_yaw_deg_median']:>5.1f} deg"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000, help=f"policy steps per rollout ({POLICY_HZ} Hz)")
    ap.add_argument("--cone", default=CONE, choices=("elliptic", "pyramidal"))
    ap.add_argument("--speed", type=float, default=0.5, help="commanded forward speed, m/s")
    ap.add_argument(
        "--rigid",
        default="mujoco",
        choices=("mujoco", "steel"),
        help="rigid solver. steel = Skild's SolverSteel, a second engine on the same policy and seeds: "
        "a result that only one solver shows is a solver artefact (SIM_ACCURACY.md item 3 makes the same argument for dt)",
    )
    ap.add_argument("--only", default=None, choices=("friction", "wind", "slope", "speed"), help="run one section")
    ap.add_argument("--check-determinism", action="store_true", help="run every cell twice and compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = range(args.seeds)
    cmd = (args.speed, 0.0, 0.0)
    t0 = time.time()

    def cell(**kw) -> dict:
        kw.setdefault("command", cmd)
        c = run_cell(Conditions(**kw), seeds, args.steps, args.cone, args.rigid)
        if args.check_determinism:
            again = run_cell(Conditions(**kw), seeds, args.steps, args.cone, args.rigid)
            if again["survived"] != c["survived"]:
                raise SystemExit(f"NOT DETERMINISTIC: {c['label']}\n  {c['survived']}\n  {again['survived']}")
        print(fmt(c, args.steps), flush=True)
        return c

    print(f"G1 29-DoF (Newton-shipped policy), {args.seeds} paired seeds x {args.steps} steps, "
          f"rigid={args.rigid}, cone={args.cone}\n")
    print("-- control")
    control = cell(mu_ground=0.75)
    if control["falls"]:
        print(
            f"\nABORT: the control fell on {control['falls']}/{control['n']} seeds. Without a control that "
            "walks on every seed no comparison below would mean anything."
        )
        return 2

    results = {"friction": [control], "wind": [], "slope": [], "speed": []}
    do = lambda section: args.only in (None, section)  # noqa: E731
    if do("friction"):
        print("\n-- friction (flat, no wind)")
        for mu in MU_GRID[1:]:
            results["friction"].append(cell(mu_ground=mu))

    for mu in SWEEP_MUS if do("wind") else ():
        print(f"\n-- headwind and crosswind at mu={mu:g}")
        for w in WIND_GRID[1:]:
            results["wind"].append(cell(mu_ground=mu, wind_n=w, wind_dir_deg=180.0))
        for w in WIND_GRID[1:]:
            c = cell(mu_ground=mu, wind_n=w, wind_dir_deg=90.0)
            c["label"] += " cross"
            results["wind"].append(c)

    if do("slope"):
        print("\n-- uphill slope at mu=0.75")
        for sl in SLOPE_GRID[1:]:
            results["slope"].append(cell(mu_ground=0.75, slope_deg=sl))
        # Slope spends friction: holding position on a slope already takes
        # tan(slope) of it. This grid locates the fall threshold in mu at two
        # slopes, to compare with the flat threshold from the friction section.
        print("\n-- friction threshold on a slope  (margin = mu - tan(slope))")
        for sl in MARGIN_SLOPES:
            for mu in MARGIN_MUS:
                c = cell(mu_ground=mu, slope_deg=sl)
                c["margin"] = round(mu - math.tan(math.radians(sl)), 3)
                results["slope"].append(c)

    # Does simply asking for less speed buy traction? No retraining, same policy.
    # Opt-in (--only speed): it is a follow-up question, not part of the main sweep.
    if args.only == "speed":
        print("\n-- commanded speed on ice")
        for mu in (0.15, 0.1, 0.05):
            for v in (0.0, 0.25, 0.5):
                c = cell(mu_ground=mu, command=(v, 0.0, 0.0))
                c["label"] += f" cmd={v:g}m/s"
                c["command_vx"] = v
                results["speed"].append(c)

    payload = {
        "policy": "newton-assets unitree_g1/rl_policies/mjw_g1_29DOF.onnx",
        "engine": "newton.solvers.SolverMuJoCo(use_mujoco_cpu=True)" if args.rigid == "mujoco" else "newton.solvers.SolverSteel",
        "cone": args.cone,
        "seeds": args.seeds,
        "steps": args.steps,
        "command": cmd,
        "deterministic_checked": bool(args.check_determinism),
        "results": results,
    }
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parent / "results" / (
        f"ice_sweep_{args.rigid}_{args.cone}" + (f"_{args.only}" if args.only else "") + ".json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
