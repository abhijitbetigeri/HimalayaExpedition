"""Inference-time causal ablation for the ice thesis.

Answers "what is the survival difference actually CAUSED by" without training
anything. Every number here comes from replaying already-trained checkpoints
across physics conditions, so the whole table costs minutes, not GPU-hours.

Why this exists
---------------
CLAUDE.md already names the problem: "Four differences, never tested apart."
`ice_randomize` bundles wide friction WITH compliant ground, and the eval env
stacks wind and patches on top. A single survival number over that stack cannot
attribute anything.

The design is lifted from npow/icefall-g1, which established a G1 ice-axe
self-arrest by publishing a 4-condition ablation table rather than a success
rate. Its strongest row was a NEGATIVE control: disable the axe pick and every
rollout runs away to 13.5 m/s. That is what makes the positive result mean
something.

Two ablation kinds, and they are not interchangeable
----------------------------------------------------
  INFERENCE-TIME (this file): one trained policy, N physics conditions. Cheap.
      Answers "what does this policy depend on?"
  TRAINING-TIME (a GPU run per cell): N policies, one condition. Expensive.
      Answers "what does the policy need to have SEEN to work?"

The open question in CLAUDE.md -- distribution vs step count -- is a
training-time question and this file cannot settle it. What it CAN do is stop
you spending GPU budget on training-time cells that the cheap table already
rules out.

Reading the table
-----------------
C0 is the validity floor. If an arm cannot walk at C0 it has not learned to
walk at all and every other cell is measuring noise -- that was ice-v2, which
failed at 26-47/300 on BENIGN ground. C1..C4 add one factor each. MU is a
dose-response sweep, which is strictly more informative than a binary ablation:
the claim worth making is not "the ice arm survives ice" but "the ice arm's
survival degrades more gracefully as mu falls."

The two controls are what keep the harness honest:
  MU=1.00 (dry rock)   -- both arms MUST walk. If not, the harness is broken.
  MU=0.01 (near-zero)  -- both arms MUST fail fast. If something "survives"
                          here it is standing still, not walking, and the
                          survival metric is measuring the wrong thing.

Usage
-----
    python ablate_eval.py \
        --arm base-v2=/mnt/himalaya-g1/base-v2/checkpoints/000050000000 \
        --arm ice-v4=/mnt/himalaya-g1/ice-v4/checkpoints/000050000000 \
        --rollouts 16 --out ablation.json
"""

import argparse
import copy
import functools
import json
import pathlib

import jax
import jax.numpy as jp
import numpy as np

import ice_patch

TASK = "flat_terrain"
# Default matches train_ice.py's eval so numbers are comparable to eval.json.
# Shorten with --steps for a smoke run; CPU is ~50x slower than the L4.
EPISODE_STEPS = 500

# Contact softness used for the "compliant" cells. Midpoint of the ranges
# ice_randomize samples over, so this cell represents that randomizer's median
# draw rather than an arbitrary new setting.
COMPLIANT_TIMECONST = 0.07
COMPLIANT_DMAX = 0.85
COMPLIANT_DMIN_OFFSET = 0.05

# Stock friction is U(0.4, 1.0); 0.7 is its midpoint. Used wherever a cell is
# meant to represent "stock ground" with no ice.
STOCK_MU = 0.7


def _conditions(mu_sweep):
    """(key, label, mu, compliant, wind, patches).

    mu=None means "leave the model's own pair_friction alone", which only
    happens when patches are on and supply friction themselves.
    """
    rows = [
        ("C0", "stock ground, hard, no wind, no patch", STOCK_MU, False, False, False),
        ("C1", "+ ice friction (mu=0.05)",              0.05,     False, False, False),
        ("C2", "+ compliant ground",                    0.05,     True,  False, False),
        ("C3", "+ wind",                                0.05,     True,  True,  False),
        ("C4", "+ patches (full Himalayan stack)",      None,     True,  True,  True),
    ]
    for mu in mu_sweep:
        rows.append((f"MU={mu:.2f}", f"friction sweep, mu={mu:.2f}",
                     mu, False, False, False))
    return rows


def build_env(mu, compliant, wind_on, patches_on):
    """Raw (un-brax-wrapped) eval env with exactly the requested physics.

    The randomizer is deliberately NOT involved. `domain_randomize` is a
    brax `randomization_fn`; driving the raw env means pair_friction is
    whatever we set here, which is the entire point -- one factor at a time,
    no sampling variance smeared across the cell.
    """
    cfg = copy.deepcopy(ice_patch.default_config())
    cfg.wind_config.enable = wind_on
    cfg.patch_config.enable = patches_on
    env = ice_patch.load(TASK, config=cfg)

    pf = env.mjx_model.pair_friction
    sr = env.mjx_model.pair_solref
    si = env.mjx_model.pair_solimp
    if mu is not None:
        # Rows 0:2 are left_foot_floor / right_foot_floor. Rows 2-4 are the
        # condim=1 self-collision pairs -- touching those changes the robot,
        # not the terrain. See ice_randomize.FEET.
        pf = pf.at[0:2, 0:2].set(mu)
    if compliant:
        sr = sr.at[0:2, 0].set(COMPLIANT_TIMECONST)
        si = si.at[0:2, 0].set(COMPLIANT_DMAX - COMPLIANT_DMIN_OFFSET)
        si = si.at[0:2, 1].set(COMPLIANT_DMAX)
    env._mjx_model = env.mjx_model.tree_replace(
        {"pair_friction": pf, "pair_solref": sr, "pair_solimp": si}
    )
    return env


def run_arm(policy, env, n_rollouts, seed, episode_steps=EPISODE_STEPS):
    """Survival steps, DISTANCE TRAVELLED, and the min foot friction seen.

    Survival alone cannot tell walking apart from not-walking: a policy that
    freezes or shuffles in place never trips the fall termination and scores a
    perfect episode. Measured here on base-v2 at mu=1.0, one rollout survived
    100/100 steps having moved 4.6 cm net -- while covering 0.59 m of path at
    0.30 m/s. So it was neither fallen nor frozen; it was circling.

    That rules out both obvious progress metrics. Net displacement scores
    circling the same as standing. Path length scores circling the same as
    walking. The metric that separates all three is body-frame forward
    velocity against the command, which is what Playground's own
    `tracking_lin_vel` reward uses. `track_vx` below is the mean of that over
    the alive portion of the episode; compare it to the commanded 1.0 m/s.

    GOTCHA: jit AFTER the model swap, never before. `env.step` closes over
    `self._mjx_model`, which jax bakes in as a constant at trace time. A jitted
    step captured before build_env's tree_replace would silently keep running
    the OLD friction and every cell in the table would report C0.
    """
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    survived, travelled, tracked, mus = [], [], [], []
    for ep in range(n_rollouts):
        # Shared seeds across every arm and every condition. This is what lets
        # cells be compared to each other at all; icefall-g1's strongest claim
        # was that two conditions produced numerically identical trajectories,
        # which is only checkable on matched seeds.
        rng = jax.random.PRNGKey(seed * 1000 + ep)
        st = reset(rng)
        st.info["command"] = jp.array([1.0, 0.0, 0.0])
        # Commanded direction is +x, so along-track displacement of the base
        # is the progress the command actually asked for.
        x0 = float(st.data.qpos[0])
        alive, fell = 1, False
        x_at_fall = x0
        vx_body = []
        for _ in range(episode_steps):
            rng, ar = jax.random.split(rng)
            act, _ = policy(st.obs, ar)
            st = step(st, act)
            # Re-pin every step: Joystick resamples the command on its own
            # schedule, and a policy that gets told to stand still will
            # "survive" trivially.
            st.info["command"] = jp.array([1.0, 0.0, 0.0])
            mus.append(float(jp.min(st.info.get("foot_mu", jp.ones(2)))))
            if not fell:
                # Freeze both odometers at the fall: metres and velocities
                # accumulated while tumbling are not locomotion.
                x_at_fall = float(st.data.qpos[0])
                # World velocity rotated into the base frame. qpos[3:7] is the
                # base quaternion (w, x, y, z); the command is body-frame, so
                # world-x displacement is not the thing being commanded.
                q = np.asarray(st.data.qpos[3:7], dtype=np.float64)
                w, x, y, z = q
                fwd = np.array([1 - 2 * (y * y + z * z),
                                2 * (x * y + w * z),
                                2 * (x * z - w * y)])
                vx_body.append(float(np.dot(np.asarray(st.data.qvel[:3]), fwd)))
            fell = fell or float(st.done) > 0.0
            alive += not fell
        survived.append(alive)
        travelled.append(x_at_fall - x0)
        tracked.append(float(np.mean(vx_body)) if vx_body else 0.0)
    return (np.array(survived), np.array(travelled), np.array(tracked),
            (min(mus) if mus else float("nan")))


def load_arm(path):
    from brax.training.agents.ppo import checkpoint as ck

    # Deliberately NOT passing network_factory. brax's `get_network` already
    # splats the checkpoint's own `network_factory_kwargs` (stored in
    # ppo_network_config.json) into the factory, so binding train_ice.py's
    # kwargs via functools.partial raises a duplicate-keyword TypeError on
    # every overlapping key -- policy_hidden_layer_sizes, activation, and the
    # rest. Reading them off the checkpoint is also strictly more robust: an
    # arm stays loadable even if train_ice.py's config later drifts.
    # Orbax hard-errors on a relative path ('Checkpoint path should be
    # absolute'), which is easy to hit since every other path in this repo
    # is relative. Resolve rather than making the caller remember.
    path = pathlib.Path(path).expanduser().resolve()
    return jax.jit(ck.load_policy(path, deterministic=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True,
                   metavar="NAME=CKPT_PATH",
                   help="Repeatable. e.g. --arm ice-v4=/mnt/.../checkpoints/000050000000")
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--steps", type=int, default=EPISODE_STEPS,
                   help="Episode length. Lower it for a CPU smoke run.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mu-sweep", type=float, nargs="*",
                   default=[1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])
    p.add_argument("--sweep-only", action="store_true",
                   help="Run only the mu sweep, skipping the C0..C4 stack.")
    p.add_argument("--out", default="ablation.json")
    args = p.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--arm needs NAME=PATH, got {spec!r}")
        print(f"loading {name} <- {path}", flush=True)
        arms[name] = load_arm(path)

    rows = _conditions(args.mu_sweep)
    if args.sweep_only:
        rows = [r for r in rows if r[0].startswith("MU=")]
    results = {}
    for key, label, mu, compliant, wind_on, patches_on in rows:
        env = build_env(mu, compliant, wind_on, patches_on)
        results[key] = {"label": label, "mu": mu, "compliant": compliant,
                        "wind": wind_on, "patches": patches_on, "arms": {}}
        for name, policy in arms.items():
            surv, dist, trk, min_mu = run_arm(policy, env, args.rollouts,
                                              args.seed, args.steps)
            results[key]["arms"][name] = {
                "mean": float(surv.mean()), "std": float(surv.std()),
                "median": float(np.median(surv)), "max": int(surv.max()),
                "all": surv.tolist(), "min_foot_mu": min_mu,
                "distance_mean": float(dist.mean()),
                "distance_std": float(dist.std()),
                "distance_all": dist.tolist(),
                "track_vx_mean": float(trk.mean()),
                "track_vx_std": float(trk.std()),
                "track_vx_all": trk.tolist(),
            }
            print(f"  [{key}] {name}: {surv.mean():6.1f} +/- {surv.std():5.1f} "
                  f"of {args.steps + 1} | net {dist.mean():+.2f} m "
                  f"| track_vx {trk.mean():+.2f} m/s (cmd 1.00)", flush=True)

    names = list(arms)
    lines = ["| Condition | " + " | ".join(f"{n} (steps, track_vx)"
                                            for n in names) + " |",
             "|---|" + "---:|" * len(names)]
    for key, _, _, _, _, _ in rows:
        cells = [f"{results[key]['arms'][n]['mean']:.0f} ± "
                 f"{results[key]['arms'][n]['std']:.0f} "
                 f"({results[key]['arms'][n]['track_vx_mean']:+.2f} m/s)"
                 for n in names]
        lines.append(f"| {key} {results[key]['label']} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    # Numbers before artefacts, per CLAUDE.md: write the json first so a
    # formatting bug cannot destroy the run.
    pathlib.Path(args.out).write_text(json.dumps(
        {"episode_steps": args.steps + 1, "n_rollouts": args.rollouts,
         "seed": args.seed, "conditions": results,
         "markdown_table": table}, indent=2) + "\n")
    print(f"\nwrote {args.out}\n\n{table}", flush=True)

    # CENSORING GUARD. Learned the hard way on this repo: a 100-step smoke run
    # of base-v2 reported 101 +/- 0 at mu=1.0 and looked like a clean ceiling.
    # Extending the same rollout to 400 steps showed it actually falls at step
    # ~250. The "perfect" score was the episode ending before the failure, not
    # the policy surviving -- and every cell compared against that reference
    # inherits the error. Short episodes also depress track_vx, because the
    # first ~2 s is gait startup (measured: 0.02-0.36 m/s in window 1, rising
    # to 0.41-0.43 by window 2).
    for name in names:
        saturated = [k for k, *_ in rows
                     if results[k]["arms"][name]["mean"] >= args.steps + 1
                     and results[k]["arms"][name]["std"] == 0.0]
        if saturated:
            print(f"\nWARNING [{name}]: {len(saturated)} cell(s) saturated at "
                  f"the {args.steps}-step episode limit ({', '.join(saturated)}). "
                  f"Those are CENSORED, not survival times -- the episode ended "
                  f"before anything happened. Every comparison against them is "
                  f"a lower bound. Re-run with --steps long enough for the "
                  f"easiest cell to actually fail.", flush=True)

    # Harness validity, checked explicitly rather than left to the reader.
    for name in names:
        top = results.get(f"MU={max(args.mu_sweep):.2f}", {}).get("arms", {}).get(name)
        bot = results.get(f"MU={min(args.mu_sweep):.2f}", {}).get("arms", {}).get(name)
        if args.steps < 200:
            print(f"NOTE [{name}]: at --steps {args.steps} the first ~100 "
                  f"steps are gait startup, which depresses track_vx well "
                  f"below steady state. Do not read tracking off a short run.",
                  flush=True)
        if top and top["track_vx_mean"] < 0.5 and args.steps >= 200:
            print(f"WARNING [{name}]: on DRY ROCK tracks only "
                  f"{top['track_vx_mean']:+.2f} m/s of a commanded 1.00 while "
                  f"surviving {top['mean']:.0f} steps. This arm is upright but "
                  f"not locomoting even on the easy control, so every harder "
                  f"cell below is measuring balance, not walking.", flush=True)
        if top and top["mean"] < 0.8 * args.steps:
            print(f"WARNING [{name}]: fails the dry-rock control "
                  f"({top['mean']:.0f}). Nothing else in this table is "
                  f"interpretable -- this arm cannot walk.", flush=True)
        if bot and bot["mean"] > 0.5 * args.steps:
            if bot["track_vx_mean"] < 0.3:
                print(f"WARNING [{name}]: 'survives' near-frictionless ground "
                      f"({bot['mean']:.0f} steps) while tracking only "
                      f"{bot['track_vx_mean']:+.2f} m/s of a commanded 1.00 "
                      f"-- it is staying upright, not walking. Survival-steps "
                      f"is the wrong headline for this arm.", flush=True)
            else:
                print(f"WARNING [{name}]: survives near-frictionless ground "
                      f"({bot['mean']:.0f} steps) AND tracks "
                      f"{bot['track_vx_mean']:+.2f} m/s. Not an artefact -- "
                      f"check mu is really being applied.", flush=True)


if __name__ == "__main__":
    main()
