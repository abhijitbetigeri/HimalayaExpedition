# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda12]", "mujoco", "playground", "brax", "mediapy", "imageio-ffmpeg"]
# ///
"""Roll out a trained policy and render it, on a GPU, many seeds at once.

The GPU counterpart to `local_view.py`. Same job -- pull a checkpoint, drive the
G1, write an mp4 -- but it answers a question the Mac cannot afford to ask.

`local_view.py` runs ONE seed because a CPU rollout costs ~3 minutes. One seed is
exactly the sample size `diagnose_eval2.py` was written to escape: "FELL at step
415" is a draw from a distribution, not a property of the policy. On a GPU the
seeds are vmapped, so N of them cost about what one costs, and you get the spread
for free alongside the video.

Three things make this fast, and all three matter:

  1. warp, not jax. Playground's G1 config defaults to impl="warp", which is the
     right backend HERE and the wrong one on a Mac (see FAST_CPU in local_view.py).
     This script leaves the defaults alone -- do not copy the local overrides in.
  2. lax.scan, not a Python loop. 600 python-level dispatches per seed would leave
     the GPU idle between kernels.
  3. No per-step `float(state.done)`. That is a blocking device->host sync every
     single step; it is invisible on CPU and ruinous on a GPU. Done flags come back
     once, as an array, after the whole scan.

Run (ask before launching -- this costs money):
  hf jobs uv run --namespace iteratehack --flavor l4x1 --timeout 25m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=gpu-view \
      gpu_view.py

Validate the whole path first, for about a cent, no GPU:
  hf jobs uv run --namespace iteratehack --flavor cpu-basic --timeout 20m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=gpu-view-dry \
      gpu_view.py --dry-run

Outputs land in the bucket and come down with:
  hf buckets sync hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/videos ./videos
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

# The uv bookworm image ships no GL stack, and `import mujoco` walks the EGL path
# eagerly -- so without this the import itself dies, not just the render. Same
# preamble as smoke_test.py; we are root in the job.
# ffmpeg is a SYSTEM binary, not a pip dep -- mediapy shells out to it. Same
# install line train_ice.py uses, for the same reason it uses it.
subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 ffmpeg",
    shell=True, check=False,
)


def ensure_ffmpeg() -> None:
    """Last-resort `ffmpeg` on PATH for platforms apt cannot reach.

    The apt line above covers the job image. This covers a Mac, where there is no
    apt and no system ffmpeg, so that `--allow-cpu` can be exercised end to end
    before spending anything on a GPU. The imageio-ffmpeg wheel bundles a static
    build; its file is named ffmpeg-<platform>-<version> and mediapy looks up the
    literal name "ffmpeg", so it has to be copied under that name.
    """
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        sys.exit("no ffmpeg and no imageio-ffmpeg -- cannot write the video")
    src = pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe())
    bindir = pathlib.Path("/tmp/ffmpeg-bin")
    bindir.mkdir(exist_ok=True)
    dst = bindir / "ffmpeg"
    if not dst.exists():
        shutil.copy(src, dst)
        dst.chmod(0o755)
    os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"


def pick_gl_backend() -> str:
    """EGL renders on the GPU; OSMesa is the CPU fallback. Probe, don't assume."""
    if ctypes.util.find_library("EGL"):
        try:
            ctypes.CDLL("libEGL.so.1")
            return "egl"
        except OSError:
            pass
    return "osmesa" if ctypes.util.find_library("OSMesa") else "glfw"


# Respect an explicit MUJOCO_GL (a Mac needs "cgl" offscreen, which no probe here
# would ever guess) and only fall back to probing.
GL = os.environ.get("MUJOCO_GL") or pick_gl_backend()
os.environ["MUJOCO_GL"] = GL
os.environ["PYOPENGL_PLATFORM"] = GL

BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
# `hf jobs uv run` ships THIS FILE ONLY, so any future helper import has to come
# from the mounted bucket. Same fallback smoke_test.py uses.
if str(BUCKET) not in sys.path:
    sys.path.insert(0, str(BUCKET))

# Rows 0:2 of pair_friction are left/right foot vs floor; columns 0:2 are the two
# TANGENTIAL components, the ones that govern slipping. Rows 2+ are the condim=1
# self-collision pairs and must not be touched.
FEET = slice(0, 2)
TANGENTIAL = slice(0, 2)


def robust_write(path: pathlib.Path, text: str, attempts: int = 5) -> bool:
    """Write to the bucket mount, retrying transient FUSE failures.

    Same hazard train_ice.py documents: the bucket is a network mount, and a
    mkdir immediately followed by a write can raise FileNotFoundError because the
    directory is not visible yet. Racy, not deterministic. Never take a finished
    run down over it -- retry, then dump to the log so the numbers survive.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base-v2-baseline",
                    help="bucket run name (default: the best we have; see docstring)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=16,
                    help="rolled out simultaneously; on a GPU 16 costs about what 1 does")
    ap.add_argument("--mu", type=float, default=None,
                    help="pin both feet to this friction (0.05 = ice, 1.0 = rock)")
    # NOT --env: `hf jobs uv run` has its own -e/--env for environment variables
    # and swallows it even when it appears after the script name, so the script
    # silently runs with the default. Cost one GPU job to find.
    ap.add_argument("--arena", default="stock",
                    choices=["stock", "patchy-benign", "both"],
                    help="stock = locomotion.load; patchy-benign = ice_patch with "
                         "wind AND patches off, which is exactly what "
                         "diagnose_eval2.py calls 'benign'. 'both' runs the pair on "
                         "identical seeds -- the only way to tell a config "
                         "difference apart from seed variance.")
    ap.add_argument("--video-seed", default="median",
                    help="which seed to render: median | best | worst | <int>")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny rollout on cpu-basic to validate the path, no GPU")
    ap.add_argument("--no-cpu-tuning", action="store_true",
                    help="off GPU, use the SHIPPED config (impl=warp) instead of the "
                         "tuned one. ~3x slower, and the only way to reproduce the "
                         "historical cpu-basic runs bit for bit.")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="skip the GPU assertion (implied by --dry-run)")
    return ap.parse_args()


def load_policy(run: str):
    """Load a run's newest checkpoint, patching brax's null-initializer bug.

    brax serialises unset optional initializers as null and then refuses to read
    them back (KERNEL_INITIALIZER[None] -> KeyError). Safe to patch: initializers
    only produce INITIAL values and these are trained weights.
    """
    import jax
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    root = BUCKET / "runs" / run / "checkpoints"
    if not root.exists():
        sys.exit(f"no checkpoints at {root} -- is --run right, and is the bucket mounted?")
    ckpts = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not ckpts:
        sys.exit(f"no checkpoint dirs under {root}")

    # Copy out of the mount before rewriting the config -- never mutate the bucket.
    local = pathlib.Path(f"/tmp/ckpt_{run}")
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(ckpts[-1], local)

    cfg_path = local / "ppo_network_config.json"
    cfg = json.loads(cfg_path.read_text())
    kw = cfg.get("network_factory_kwargs", {})
    fixed = [k for k, v in kw.items() if k.endswith("_kernel_init_fn") and v is None]
    for k in fixed:
        kw[k] = "lecun_uniform"
    if fixed:
        cfg_path.write_text(json.dumps(cfg))
    print(f"policy: {run} @ {ckpts[-1].name}", flush=True)
    return ppo_checkpoint.load_policy(str(local), deterministic=True)


def main():
    args = parse_args()

    import jax
    import jax.numpy as jp
    import numpy as np
    import mujoco

    backend = jax.default_backend()
    print(f"jax backend: {backend}  devices: {jax.devices()}  GL: {GL}", flush=True)
    if backend != "gpu" and not (args.allow_cpu or args.dry_run):
        sys.exit(
            "refusing to run: jax.default_backend() is "
            f"'{backend}', not 'gpu'.\n"
            "This script exists to use the GPU -- on CPU it is strictly slower "
            "than local_view.py, which is tuned for it (impl=jax, naconmax=128).\n"
            "Use a GPU flavor, or pass --allow-cpu if you meant it."
        )

    if args.dry_run:
        args.steps, args.seeds = min(args.steps, 20), min(args.seeds, 2)
        print(f"dry run: {args.steps} steps x {args.seeds} seeds", flush=True)

    # On GPU: stock config, on purpose. impl="warp" and naconmax=8*8192 are the
    # GPU-correct defaults and local_view.py's CPU overrides would make this slower.
    #
    # Off GPU (--dry-run on cpu-basic, or --allow-cpu): take those overrides. warp
    # has no cpu fast path -- measured 972 ms/step vs 300 for impl="jax" -- so a dry
    # run on the stock config burns three times the wall clock proving nothing extra.
    tuned = backend != "gpu" and not args.no_cpu_tuning
    overrides = {"impl": "jax", "naconmax": 128} if tuned else {}
    impl = overrides.get("impl", "warp")

    def build_env(kind: str):
        """Construct one env. `overrides` is copied per call, never shared.

        CLAUDE.md's config trap: playground declares `config` as a DEFAULT
        ARGUMENT, so one ConfigDict is shared by every env built in the process and
        config_overrides mutates it permanently. Building two envs here is exactly
        the situation that bites, so each gets its own dict.
        """
        if kind == "stock":
            from mujoco_playground import locomotion
            env = locomotion.load("G1JoystickFlatTerrain",
                                  config_overrides=dict(overrides))
        elif kind == "patchy-benign":
            # Imported from the bucket on the job (see the sys.path insert above).
            # wind AND patches off reproduces diagnose_eval2.py's "benign" cell.
            import ice_patch
            env = ice_patch.load("flat_terrain", config_overrides={
                "wind_config.enable": False, "patch_config.enable": False,
                **overrides})
        else:
            sys.exit(f"unknown env kind {kind}")
        print(f"env: {kind} (impl={impl})", flush=True)

        if args.mu is not None:
            # ice_randomize.load() does NOT apply friction -- it only bumps njmax;
            # the randomizer is a separate function brax calls during training. Set
            # the model field directly, as local_view.py does.
            m = env.mjx_model
            env._mjx_model = m.tree_replace(
                {"pair_friction": m.pair_friction.at[FEET, TANGENTIAL].set(args.mu)})
            env.mj_model.pair_friction[FEET, TANGENTIAL] = args.mu  # renderer in sync
            print(f"  friction pinned to mu={args.mu} on both feet "
                  f"({'ice' if args.mu <= 0.15 else 'rock'})", flush=True)
        return env

    inference_fn = load_policy(args.run)
    cmd = jp.array([1.0, 0.0, 0.0])   # walk forward

    def measure(env):
        """Roll every seed through `env` and return survival + trajectories.

        The rng path here is deliberately identical to diagnose_eval2.py's --
        PRNGKey(seed), reset, then split once per step -- so that with the same env
        the two scripts draw the same initial state and the same actions. That is
        what makes an env-vs-env comparison on fixed seeds meaningful: only the env
        differs.
        """
        def rollout(seed):
            rng = jax.random.PRNGKey(seed)
            state = env.reset(rng)
            state.info["command"] = cmd

            def body(carry, _):
                state, rng = carry
                rng, act_rng = jax.random.split(rng)
                action, _ = inference_fn(state.obs, act_rng)
                state = env.step(state, action)
                # env.step() resamples the joystick command internally, so re-pin it
                # every step -- same as local_view.py.
                state.info["command"] = cmd
                return (state, rng), (state.data.qpos, state.data.qvel, state.done)

            _, out = jax.lax.scan(body, (state, rng), None, length=args.steps)
            return state.data.qpos, out   # initial qpos + the scanned trajectory

        t0 = time.time()
        qpos0, (qpos, qvel, done) = jax.jit(jax.vmap(rollout))(jp.arange(args.seeds))
        jax.block_until_ready(done)
        dt = time.time() - t0
        total = args.steps * args.seeds
        print(f"  done in {dt:.1f}s  ({dt / total * 1000:.2f} ms per env-step, "
              f"{total / dt:.0f} env-steps/s)", flush=True)

        # done has shape (seeds, steps). A seed that never falls has no True at all,
        # so argmax would return 0 -- guard on .any() rather than trusting the index.
        #
        # The +2 makes these numbers directly comparable to local_view.py's, which
        # counts trajectory STATES: the initial one plus every step up to and
        # including the terminal one. done[i] means the state after step i+1 is
        # terminal, so that is i+2 states. Without it the two scripts disagree by
        # one on the same fall.
        d = np.asarray(done) != 0.0
        fell = d.any(axis=1)
        survived = np.where(fell, d.argmax(axis=1) + 2, args.steps + 1)
        return survived, fell, qpos0, qpos, qvel, round(total / dt)

    def pick_seed(survived):
        order = np.argsort(survived)
        if args.video_seed == "median":
            return int(order[len(order) // 2])
        if args.video_seed == "best":
            return int(order[-1])
        if args.video_seed == "worst":
            return int(order[0])
        return int(args.video_seed)

    def render(env, kind, survived, fell, qpos0, qpos, qvel, pick):
        """Write the mp4 for one seed, truncated at its fall."""
        n = int(survived[pick]) - 1 if fell[pick] else args.steps
        qp = np.concatenate([np.asarray(qpos0[pick])[None], np.asarray(qpos[pick])[:n]])
        qv = np.asarray(qvel[pick])[:n]
        qv = np.concatenate([np.zeros_like(qv[:1]), qv])

        # Render straight from qpos with a plain MuJoCo Renderer rather than
        # env.render(): the scan hands back arrays, not a list of State objects, and
        # rebuilding States just to satisfy that signature is pure overhead.
        #
        # The "track" camera follows the robot. Without it the view is static and
        # the G1 walks out of frame after a few seconds -- which made the first
        # batch of clips useless as footage.
        n_frames = len(range(0, len(qp), args.stride))
        print(f"rendering {kind} seed {pick}, {n_frames} frames ...", flush=True)
        model = env.mj_model
        data = mujoco.MjData(model)
        frames = []
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for i in range(0, len(qp), args.stride):
                data.qpos[:] = qp[i]
                data.qvel[:] = qv[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render())

        import mediapy
        ensure_ffmpeg()
        try:
            out_mp4 = OUT_DIR / f"{tag(kind)}.mp4"
            mediapy.write_video(out_mp4, frames, fps=1.0 / (args.stride * env.dt))
            print(f"wrote {out_mp4}", flush=True)
        except Exception as e:   # no ffmpeg, bad codec, flaky mount
            print(f"WARN: video write failed ({e}) -- falling back to pngs", flush=True)
            fr = OUT_DIR / f"{tag(kind)}-frames"
            fr.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames[::5]):
                mediapy.write_image(fr / f"{i:03d}.png", f)
            print(f"wrote {len(frames[::5])} pngs to {fr}", flush=True)

    def tag(kind):
        t = f"{args.run}-{kind}"
        return t if args.mu is None else f"{t}-mu{args.mu}"

    OUT_DIR = BUCKET / "videos"
    kinds = ["stock", "patchy-benign"] if args.arena == "both" else [args.arena]
    results = {}

    for kind in kinds:
        print(f"\n=== {kind}: {args.steps} steps x {args.seeds} seeds ===", flush=True)
        env = build_env(kind)
        survived, fell, qpos0, qpos, qvel, rate = measure(env)
        pick = pick_seed(survived)

        for sd in range(args.seeds):
            mark = "  <- rendered" if sd == pick else ""
            print(f"  seed {sd:2d}: {survived[sd]:4d}/{args.steps + 1} steps  "
                  f"{'FELL' if fell[sd] else 'upright'}{mark}")
        # Report survival at 300 too: every earlier number in this repo was measured
        # over 300 steps, and a 600-step run is not comparable to them without it.
        alive300 = int((survived > 300).sum()) if args.steps >= 300 else None
        line = (f"  mean {survived.mean():.1f}  median {np.median(survived):.1f}  "
                f"min {survived.min()}  max {survived.max()}  "
                f"fell {int(fell.sum())}/{args.seeds}")
        if alive300 is not None:
            line += f"  still up at 300: {alive300}/{args.seeds}"
        print(line, flush=True)

        results[kind] = {
            "survived": survived.tolist(), "fell": fell.tolist(),
            "mean": float(survived.mean()), "median": float(np.median(survived)),
            "min": int(survived.min()), "max": int(survived.max()),
            "alive_at_300": alive300, "rendered_seed": pick,
            "env_steps_per_s": rate,
        }

        # Numbers BEFORE artefacts. CLAUDE.md's rule, learned the expensive way in
        # train_ice.py: a render failure must never destroy a finished run.
        robust_write(OUT_DIR / f"{tag(kind)}.json", json.dumps(
            {"run": args.run, "env": kind, "mu": args.mu, "steps": args.steps,
             "seeds": args.seeds, "backend": backend, **results[kind]}, indent=2))
        print(f"wrote {OUT_DIR / f'{tag(kind)}.json'}", flush=True)

        render(env, kind, survived, fell, qpos0, qpos, qvel, pick)

    if len(kinds) > 1:
        a, b = (np.array(results[k]["survived"]) for k in kinds)
        print("\n" + "=" * 62, flush=True)
        print(f"{'seed':>5}  {kinds[0]:>14}  {kinds[1]:>14}  {'delta':>8}")
        print("-" * 62)
        for sd in range(args.seeds):
            print(f"{sd:>5}  {a[sd]:>14}  {b[sd]:>14}  {b[sd] - a[sd]:>+8}")
        print("-" * 62)
        print(f"{'mean':>5}  {a.mean():>14.1f}  {b.mean():>14.1f}  "
              f"{b.mean() - a.mean():>+8.1f}")
        print("=" * 62, flush=True)
        # This line used to claim the gap WAS the env config. It is not: identical
        # reruns of one arm move the mean by ~35 and single seeds by >200 (see the
        # nondeterminism finding in CLAUDE.md). Pairing inside one job removes the
        # between-job drift, not the within-job noise.
        print("\nSame seeds, same rng path, same backend. GPU rollouts are NOT "
              "reproducible, so treat a delta smaller than the run-to-run spread "
              "(~35 in the mean at 16 seeds x 600 steps) as no difference.",
              flush=True)
        robust_write(OUT_DIR / f"{args.run}-env-compare.json",
                     json.dumps({"run": args.run, "steps": args.steps,
                                 "seeds": args.seeds, "backend": backend,
                                 "results": results}, indent=2))

    print(f"\npull it down with:\n  hf buckets sync "
          f"hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/videos ./videos",
          flush=True)


if __name__ == "__main__":
    main()
