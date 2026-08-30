"""Watch a trained MuJoCo policy walk, interactively, on your own Mac. No GPU.

This is the smallest end-to-end check that the whole pipeline is real: it pulls a
trained checkpoint out of the bucket, loads it, and drives the G1 in a live MuJoCo
window you can orbit and pause.

Default policy is `base-v2-baseline`, the best MuJoCo policy we have. It is NOT a
known-good control: it walks on a minority of seeds (0, 1 and 5 of the first eight
reach 300 steps; the rest fall at 200, 148, 125, 68, 48) and survives 600 steps on
none. The "301/300 on every seed" this docstring used to claim was a misreading of
`diagnostics/eval_control.json`, which records [301, 301, 196]. See CLAUDE.md.

Seed 0 is one of the ones that walks, so it is still the right default for checking
that the PLUMBING works. Just do not read a single rollout as a measurement.

    python local_view.py                      # watch the working baseline
    python local_view.py --run ice-v2         # watch a failing ice policy
    python local_view.py --headless -o out.mp4  # write video instead of a window

Notes
-----
CPU only, so expect a few steps per second rather than real time. That is fine for
watching gait; it is useless for training.

Downloads the checkpoint to ./checkpoints/<run>/ on first use and reuses it after.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")   # native window on macOS

BUCKET = "hf://buckets/iteratehack/jobs-artifacts/himalaya-g1"
HERE = pathlib.Path(__file__).parent

# The G1 joystick default config hardcodes impl="warp" (playground
# locomotion/g1/joystick.py) and naconmax=8*8192. Both are sized for thousands of
# parallel envs on a CUDA box. mujoco_warp has NO cpu fast path -- it emits
# GPU-shaped kernels and runs them serially -- so on a Mac it is the worst of the
# three backends. Measured here, stock G1JoystickFlatTerrain, 50 steps:
#
#     impl=warp (default)          972 ms/step   600 steps ~= 10 min
#     impl=jax                     394 ms/step              ~=  4 min
#     impl=jax, naconmax=128       300 ms/step              ~=  3 min
#
# Same final qpos to 3dp, so this is a pure speedup, not a physics change.
# impl="cpp" is NOT an option: playground's reset() calls mjx.forward, which
# raises "forward requires JAX backend implementation".
#
# On a GPU box leave the defaults alone -- warp is much faster there, and
# naconmax=128 would overflow once you vmap thousands of envs.
FAST_CPU = {"impl": "jax", "naconmax": 128}


def fetch_checkpoint(run: str) -> pathlib.Path:
    """Pull the newest checkpoint for `run` out of the bucket (cached locally)."""
    dest = HERE / "checkpoints" / run
    if dest.exists() and any(dest.rglob("ppo_network_config.json")):
        print(f"using cached {dest}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {run} ...")
        subprocess.run(
            ["hf", "buckets", "sync", f"{BUCKET}/runs/{run}/checkpoints", str(dest)],
            check=True,
        )

    steps = sorted([p for p in dest.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not steps:
        sys.exit(f"no checkpoint dirs under {dest} -- is the run name right?")
    ckpt = steps[-1]

    # brax serialises unset optional initializers as null and then refuses to read
    # them back (KERNEL_INITIALIZER[None] -> KeyError). Patch in place; safe because
    # initializers only produce INITIAL values and we are loading trained weights.
    cfg_path = ckpt / "ppo_network_config.json"
    cfg = json.loads(cfg_path.read_text())
    kw = cfg.get("network_factory_kwargs", {})
    fixed = [k for k, v in kw.items() if k.endswith("_kernel_init_fn") and v is None]
    for k in fixed:
        kw[k] = "lecun_uniform"
    if fixed:
        cfg_path.write_text(json.dumps(cfg))
        print(f"patched brax null initializers: {fixed}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base-v2-baseline",
                    help="bucket run name (default: the best we have; see docstring)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--headless", action="store_true", help="render to file, no window")
    ap.add_argument("-o", "--out", default="local_view.mp4")
    ap.add_argument("--backend", choices=["jax", "warp"], default="jax",
                    help="jax is ~3x faster on CPU but is NOT the backend the "
                         "policies were trained under. Measured on base-v2 at "
                         "mu=0.8: warp survives 251/251, jax FALLS at 112 -- same "
                         "policy, same seed. Use warp for anything you will quote "
                         "or film; jax only for a quick look.")
    ap.add_argument("--mu", type=float, default=None,
                    help="pin foot-floor friction (0.05 = bare ice, 0.8 = dry rock). "
                         "Stock XML value if omitted.")
    args = ap.parse_args()

    # Check this BEFORE the rollout, not after. launch_passive needs mjpython on
    # macOS, and finding that out at the end throws away the whole rollout.
    # mjpython leaves sys.executable pointing at plain python3, so ask the viewer
    # module the same question it asks itself rather than sniffing argv.
    if not args.headless and sys.platform == "darwin":
        import mujoco.viewer as _v
        if not isinstance(getattr(_v, "_MJPYTHON", None), _v._MjPythonBase):
            sys.exit("on macOS the interactive viewer needs mjpython:\n"
                     f"    .venv/bin/mjpython {' '.join(sys.argv)}\n"
                     "or pass --headless -o out.mp4 to write a video instead.")

    if args.headless:
        # macOS has no osmesa; offscreen there is CGL. Linux boxes use osmesa/egl.
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "osmesa"

    import jax
    import jax.numpy as jp
    import mujoco
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    ckpt = fetch_checkpoint(args.run)
    print("loading policy ...")
    inference_fn = jax.jit(ppo_checkpoint.load_policy(str(ckpt), deterministic=True))

    # Plain stock env by default: verifying the pipeline should not also involve
    # wind, patches and compliant ground. --ice adds the friction change only.
    sys.path.insert(0, str(HERE))
    from mujoco_playground import locomotion
    over = dict(FAST_CPU) if args.backend == "jax" else {}
    env = locomotion.load("G1JoystickFlatTerrain", config_overrides=over)
    print(f"backend: impl={over.get('impl', 'warp (stock, as trained)')}")
    print("env: stock G1JoystickFlatTerrain")

    if args.mu is not None:
        # IMPORTANT: ice_randomize.load() does NOT apply friction -- it only bumps
        # njmax. The randomizer is a separate function that brax calls during
        # training, so loading that module here would silently leave the robot on
        # stock ground. Set the model field directly instead.
        #
        # Rows 0:2 of pair_friction are left/right foot vs floor; columns 0:2 are
        # the two TANGENTIAL components (the ones that govern slipping). Rows 2+
        # are self-collision pairs and must not be touched.
        m = env.mjx_model
        m = m.tree_replace({"pair_friction":
                            m.pair_friction.at[0:2, 0:2].set(args.mu)})
        env._mjx_model = m
        env.mj_model.pair_friction[0:2, 0:2] = args.mu   # keep the renderer in sync
        print(f"friction pinned to mu={args.mu} on both feet "
              f"({'ice' if args.mu <= 0.15 else 'rock'})")

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    state.info["command"] = jp.array([1.0, 0.0, 0.0])   # walk forward

    print(f"rolling out {args.steps} steps on {jax.default_backend()} "
          f"(impl={FAST_CPU['impl']}, ~0.3 s/step on CPU) ...")
    traj = [state]
    for i in range(args.steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step(state, action)
        state.info["command"] = jp.array([1.0, 0.0, 0.0])
        traj.append(state)
        if float(state.done) != 0.0:
            print(f"FELL at step {i+1}")
            break
        if (i + 1) % 100 == 0:
            print(f"  {i+1} steps, still upright")
    print(f"survived {len(traj)}/{args.steps + 1} steps")

    if args.headless:
        import mediapy
        # mediapy shells out to the ffmpeg BINARY, which macOS does not ship. The
        # imageio-ffmpeg wheel bundles one, so borrow it rather than making the
        # user brew install anything.
        if shutil.which("ffmpeg") is None:
            try:
                import imageio_ffmpeg
                os.environ["PATH"] = (os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
                                      + os.pathsep + os.environ["PATH"])
                shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(),
                            pathlib.Path(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))
                            / "ffmpeg")
            except Exception as e:
                sys.exit(f"need ffmpeg: pip install imageio-ffmpeg  ({e})")
        # The G1 model ships a "track" camera that follows the robot. Without it
        # the default view is static and the robot simply walks out of frame after
        # a few seconds -- which made the first batch of clips useless as footage.
        frames = env.render(traj[::2], height=480, width=640, camera="track")
        mediapy.write_video(args.out, frames, fps=1.0 / (2 * env.dt))
        print("wrote", args.out)
        return

    # Interactive: replay the trajectory in a real MuJoCo window.
    import time
    import mujoco.viewer
    model, data = env.mj_model, mujoco.MjData(env.mj_model)
    print("\nopening viewer -- drag to orbit, space to pause, ESC to quit")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for s in traj:
                if not viewer.is_running():
                    break
                data.qpos[:] = s.data.qpos
                data.qvel[:] = s.data.qvel
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(env.dt)


if __name__ == "__main__":
    main()
