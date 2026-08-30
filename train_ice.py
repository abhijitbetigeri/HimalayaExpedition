# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda12]", "mujoco", "playground", "brax", "mediapy", "wandb"]
# ///
"""Train the G1 on ice, snow and wind. Runs as a Hugging Face job.

  hf jobs uv run --namespace iteratehack --flavor l40sx1 --timeout 6h -d \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=train \
      train_ice.py -- --run-name ice-v1

PREREQUISITE: `hf jobs uv run` uploads ONLY this file. The env modules must be
in the bucket first:

  for f in ice_randomize.py wind.py ice_patch.py; do
    hf cp $f hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/$f
  done

`hf cp`, not `hf upload` -- upload targets repos, buckets need cp (or
`hf buckets sync`). Re-run it after ANY edit to those three files, or the job
silently trains against a stale copy. Verify with
`--dry-run`, which builds everything and steps the env but trains nothing --
run that on cpu-basic before spending GPU time.

Baseline for the A/B
--------------------
`--baseline` trains the stock env with the stock U(0.4, 1.0) randomizer and no
wind. That is the comparison the whole demo rests on, so it is a first-class
flag rather than something to reconstruct by hand later. Both arms MUST use the
same --seed and --num-timesteps or the comparison means nothing.

Note on evaluation -- read before quoting any number
---------------------------------------------------
brax applies the SAME `randomization_fn` to `eval_env` as to the training env
(train.py:759-769); there is no separate eval randomizer. So the reward curves
printed DURING training are not comparable across arms: the ice arm is evaluated
with friction drawn from 0.05-1.0 on compliant ground, the baseline from
0.4-1.0 on hard ground. Same env class, different terrain distribution.

The apples-to-apples comparison is the FINAL eval below, which drives the raw
`eval_env` object directly rather than brax's wrapped copy. That one is
identical for both arms: XML default friction, patches and wind on, same seed.
`eval.json` and `eval.mp4` are therefore fair; the training curves are not.
"""

import argparse
import functools
import json
import os
import pathlib
import subprocess
import sys
import time

# The uv image ships no GL stack and `import mujoco` walks the EGL path eagerly,
# so this has to happen before any mujoco import. Same dance as smoke_test.py.
# ffmpeg is a SYSTEM binary, not a pip dep: mediapy shells out to it, so
# `mediapy` in the dependency list is not enough. Without it a run trains all the
# way through, renders every frame, and then dies on write_video with the whole
# GPU spend already sunk (this happened to ice-v1 at 52.9M steps).
subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 ffmpeg",
    shell=True, check=False,
)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Overridable so --dry-run can be exercised off-box, where /mnt does not exist.
BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
if str(BUCKET) not in sys.path:
    sys.path.insert(0, str(BUCKET))  # where the env modules live on the box

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402


def _patch_brax_jax_compat():
    """brax 0.14.2 calls `jax.device_put_replicated`; jax >= 0.10 removed it.

    brax declares `jax>=0.4.6` with no upper bound, so uv resolves the latest
    jax and PPO dies at train.py:756 the moment training starts -- after env
    construction, which is why a dry run that stops short of ppo.train sails
    straight past it. Last jax with the symbol is 0.9.2.

    Shimming beats pinning jax down: mujoco 3.12 and mujoco_warp are working on
    the current jax and dragging the whole stack back three minor versions to
    satisfy one deleted helper risks far more than it fixes.

    Single-device only, on purpose. Replicating correctly across several devices
    means building a properly sharded global array, and getting that subtly
    wrong would corrupt training silently. We run on l40sx1. On a multi-GPU
    flavor this raises instead of guessing.
    """
    if hasattr(jax, "device_put_replicated"):
        return "native"

    def device_put_replicated(x, devices):
        if len(devices) != 1:
            raise RuntimeError(
                f"device_put_replicated shim is single-device only, got "
                f"{len(devices)} devices. Use a single-GPU flavor, or pin "
                f"jax==0.9.2 which still ships the real function."
            )
        return jax.tree.map(lambda leaf: jp.asarray(leaf)[None], x)

    jax.device_put_replicated = device_put_replicated
    return "shimmed"




def robust_write(path: pathlib.Path, text: str, attempts: int = 5) -> bool:
    """Write to the bucket mount, retrying transient FUSE failures.

    The HF bucket is a network mount: `mkdir(parents=True, exist_ok=True)`
    immediately followed by a write can raise FileNotFoundError because the new
    directory is not visible yet. It is racy, not deterministic -- two jobs
    launched two seconds apart, one died here and the other did not. Losing a
    finished GPU run to that is unacceptable, so retry and, failing everything,
    keep going rather than take the run down.
    """
    for i in range(attempts):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return True
        except (FileNotFoundError, OSError) as e:
            if i == attempts - 1:
                print(f"WARN: could not write {path} after {attempts} tries: {e}",
                      flush=True)
                print("----- content follows so it is at least in the log -----",
                      flush=True)
                print(text[:4000], flush=True)
                return False
            time.sleep(1.0 + i)
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None, help="defaults to a timestamp")
    p.add_argument("--num-timesteps", type=int, default=200_000_000)
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="flat_terrain",
                   choices=["flat_terrain", "rough_terrain"])
    p.add_argument("--ascent", action="store_true",
                   help="train fixed-line ascent on a slope instead of "
                        "locomotion. Different task, much harder.")
    p.add_argument("--slope-deg", type=float, default=30.0)
    p.add_argument("--baseline", action="store_true",
                   help="stock env + stock randomizer, no wind/patches")
    p.add_argument("--no-wind", action="store_true")
    p.add_argument("--no-patches", action="store_true")
    p.add_argument("--wandb-project", default=None, help="off unless set")
    p.add_argument("--dry-run", action="store_true",
                   help="build the env AND run a few real PPO steps. Must reach "
                        "ppo.train: an env-only check misses version breaks "
                        "inside brax, which is exactly how the "
                        "device_put_replicated failure reached a GPU job.")
    return p.parse_args()


def build_envs(args):
    """Returns (train_env, eval_env, randomization_fn).

    eval_env always has the full Himalayan stack on, even for --baseline.
    """
    import ice_randomize as ir

    if args.ascent:
        import fixed_line
        # Same randomizer -- an icy fixed line is the whole point. Both envs
        # here are the ascent env; there is no stock baseline to compare
        # against because Playground ships no ascent task.
        mk = lambda: fixed_line.load(config_overrides={
            "line_config.slope_deg": args.slope_deg})
        return mk(), mk(), ir.domain_randomize

    import ice_patch

    # Playground penalises feet_slip at -0.25. On ice slipping is unavoidable, so
    # that term punishes the SURFACE rather than the gait, and the cheapest way to
    # stop slipping is to stop walking -- with stand_still at -1.0 and termination
    # at -100, a degenerate crouch is a rational local optimum. The one ice policy
    # that does work (Isaac) had its equivalent term relaxed, so this is a live
    # suspect. Overridable for the ablation; unset leaves stock behaviour.
    _slip = os.environ.get("HIMALAYA_FEET_SLIP")

    # Playground rewards feet_phase at +1.0 -- a gait-phase tracking term that
    # enforces a fixed stepping rhythm. Isaac's G1 reward set has NO phase term
    # (it uses feet_air_time), and Isaac is the only stack where ice training
    # works. On ice a fixed cadence is exactly wrong: adapting rhythm is how you
    # stay upright. Prime suspect for why the same experiment succeeds in Isaac
    # (98%) and fails here (10.7%) at a matched 150M steps.
    _phase = os.environ.get("HIMALAYA_FEET_PHASE")

    def himalaya_env():
        overrides = {
            "wind_config.enable": not args.no_wind,
            "patch_config.enable": not args.no_patches,
        }
        if _slip is not None:
            overrides["reward_config.scales.feet_slip"] = float(_slip)
        if _phase is not None:
            overrides["reward_config.scales.feet_phase"] = float(_phase)
        return ice_patch.load(args.task, config_overrides=overrides)

    eval_env = ice_patch.load(args.task)  # everything on, always

    if args.baseline:
        from mujoco_playground._src.locomotion.g1 import randomize as stock
        # NOT locomotion.load(). brax builds the value network from the TRAIN
        # env, so a 216-dim stock env against a 219-dim eval env crashes at the
        # first eval. Instead use our env with wind and patches switched off:
        # disabled wind consumes no randomness and is bitwise identical to stock
        # on the same seed (verified), and disabled patches skip the model swap
        # entirely. So this IS the stock baseline -- it just carries three
        # constant zeros on the critic obs to keep the shapes comparable.
        train_env = ice_patch.load(args.task, config_overrides={
            "wind_config.enable": False,
            "patch_config.enable": False,
        })
        return train_env, eval_env, stock.domain_randomize

    return himalaya_env(), eval_env, ir.domain_randomize


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    if args.baseline:
        run_name += "-baseline"
    out = BUCKET / "runs" / run_name
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"WARN: mkdir {out} failed ({e}); robust_write will retry",
              flush=True)

    print("jax:", jax.__version__, "devices:", jax.devices(), flush=True)
    print("brax/jax compat:", _patch_brax_jax_compat(), flush=True)
    if not args.dry_run:
        assert jax.devices()[0].platform == "gpu", \
            "no GPU visible -- do not burn a job slot training on CPU"

    train_env, eval_env, randomization_fn = build_envs(args)
    print(f"train env : {type(train_env).__name__}", flush=True)
    print(f"eval env  : {type(eval_env).__name__} (full stack, always)", flush=True)
    print(f"randomizer: {randomization_fn.__module__}", flush=True)

    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params

    ppo_params = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    ppo_params.num_timesteps = args.num_timesteps
    ppo_params.num_envs = args.num_envs

    robust_write(out / "config.json", json.dumps({
        "run_name": run_name,
        "args": vars(args),
        "ppo": {k: str(v) for k, v in ppo_params.items()},
    }, indent=2))

    if args.dry_run:
        print("\n=== dry run: env + a real PPO pass ===", flush=True)
        wenv = wrapper.wrap_for_brax_training(
            train_env, episode_length=ppo_params.episode_length,
            randomization_fn=functools.partial(
                randomization_fn, rng=jax.random.split(jax.random.PRNGKey(0), 4)))
        state = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
        step = jax.jit(wenv.step)
        for _ in range(5):
            state = step(state, jp.zeros((4, train_env.action_size)))
        assert not np.isnan(np.asarray(state.data.qpos)).any(), "NaN in dry run"
        print("env builds, randomizes and steps cleanly", flush=True)

        # Shrink to the smallest run that still executes the real code path.
        ppo_params.num_timesteps = 8192
        ppo_params.num_envs = 32
        ppo_params.num_evals = 1
        ppo_params.batch_size = 16
        ppo_params.num_minibatches = 2
        ppo_params.unroll_length = 5
        ppo_params.episode_length = 100

    run = None
    if args.wandb_project and not args.dry_run:
        import wandb
        run = wandb.init(project=args.wandb_project, name=run_name,
                         config=vars(args))

    t0 = time.time()

    def progress(step, metrics):
        rew = metrics.get("eval/episode_reward", float("nan"))
        print(f"[{time.time()-t0:7.0f}s] step {step:>12,}  reward {rew:8.2f}",
              flush=True)
        if run:
            run.log({"step": step, **{k: float(v) for k, v in metrics.items()}})

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in ppo_params.items()
           if k not in ("network_factory", "num_timesteps", "num_envs")},
        num_timesteps=ppo_params.num_timesteps,
        num_envs=ppo_params.num_envs,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=args.seed,
        save_checkpoint_path=str(out / "checkpoints"),
    )

    make_inference_fn, params, _ = train_fn(
        environment=train_env, eval_env=eval_env)
    print(f"training done in {(time.time()-t0)/60:.1f} min", flush=True)

    if args.dry_run:
        print("DRY RUN OK -- ppo.train completed end to end", flush=True)
        print("wrote", out / "config.json", flush=True)
        return

    # Eval rollout on the full Himalayan stack, rendered for the demo.
    print("rendering eval rollout...", flush=True)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    reset, step = jax.jit(eval_env.reset), jax.jit(eval_env.step)

    # N_EVAL rollouts, not one. The first version of this reported a single
    # rollout per arm, which produced a 37-vs-64 "difference" between two
    # policies that both fall over in about a second -- pure noise dressed up
    # as a result. Survival time on ice is high variance; one sample cannot
    # distinguish two policies.
    N_EVAL = 16

    def run_eval(e, label):
        r_, s_ = jax.jit(e.reset), jax.jit(e.step)
        surv, mus_, first = [], [], None
        for ep in range(N_EVAL):
            rng = jax.random.PRNGKey(args.seed * 1000 + ep)
            st = r_(rng)
            st.info["command"] = jp.array([1.0, 0.0, 0.0])
            tj, alive, fell = [st], 1, False
            for _ in range(500):
                rng, ar = jax.random.split(rng)
                a_, _ = inference_fn(st.obs, ar)
                st = s_(st, a_)
                st.info["command"] = jp.array([1.0, 0.0, 0.0])
                tj.append(st)
                mus_.append(float(jp.min(st.info.get("foot_mu", jp.ones(2)))))
                if not fell and float(st.done) > 0.0:
                    fell = True
                if not fell:
                    alive += 1
            surv.append(alive)
            if ep == 0:
                first = tj
        surv = np.array(surv)
        print(f"  [{label}] survived {surv.mean():.1f} +/- {surv.std():.1f} "
              f"of 501 (median {np.median(surv):.0f}, best {surv.max()})",
              flush=True)
        return surv, mus_, first

    print(f"eval, {N_EVAL} rollouts each:", flush=True)
    survivals, mus, traj = run_eval(eval_env, "himalayan: ice+wind+patches")

    # Diagnostic, not the headline. If a policy cannot walk on plain flat
    # ground either, the problem is that training never gave it enough easy
    # experience -- not that the eval terrain is hard.
    try:
        from mujoco_playground import locomotion as _loco
        easy = _loco.load("G1JoystickFlatTerrain")
        easy_surv, _, _ = run_eval(easy, "control: stock flat terrain")
    except Exception as e:
        print(f"  [control] skipped: {e}", flush=True)
        easy_surv = None


    # Numbers first. These ARE the result; the video only illustrates it, and
    # letting a missing codec throw away ten minutes of GPU is unacceptable.
    print(f"eval over {N_EVAL} rollouts: survived "
          f"{survivals.mean():.1f} +/- {survivals.std():.1f} of 501 steps "
          f"(median {np.median(survivals):.0f}, best {survivals.max()}), "
          f"min foot friction {min(mus):.3f}", flush=True)
    robust_write(out / "eval.json", json.dumps(
        {"n_rollouts": int(N_EVAL),
         "survived_mean": float(survivals.mean()),
         "survived_std": float(survivals.std()),
         "survived_median": float(np.median(survivals)),
         "survived_max": int(survivals.max()),
         "survived_all": survivals.tolist(),
         "total_steps": 501,
         "min_foot_mu": min(mus),
         "control_flat_mean": (float(easy_surv.mean())
                               if easy_surv is not None else None),
         "control_flat_all": (easy_surv.tolist()
                              if easy_surv is not None else None)}, indent=2))
    print("wrote", out / "eval.json", flush=True)

    try:
        import mediapy
        frames = eval_env.render(traj[::2], height=480, width=640)
        try:
            mediapy.write_video(out / "eval.mp4", frames,
                                fps=1.0 / (2 * eval_env.dt))
            print("wrote", out / "eval.mp4", flush=True)
        except Exception as e:  # no ffmpeg, bad codec, etc.
            print(f"mp4 failed ({e}); falling back to frames", flush=True)
            fr = out / "frames"; fr.mkdir(exist_ok=True)
            for i, f in enumerate(frames[::5]):
                mediapy.write_image(fr / f"{i:03d}.png", f)
            print(f"wrote {len(frames[::5])} pngs to {fr}", flush=True)
    except Exception as e:
        print(f"rendering failed entirely: {e}. Numbers above still stand.",
              flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
