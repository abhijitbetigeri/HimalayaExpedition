"""Render ablation cells side by side so the numbers have a picture.

Same policy, same seed, same action stream -- only `pair_friction` differs
between panels. So any visible difference in behaviour is caused by the one
variable the ablation claims to isolate, which is the whole point of the
table in ablate_eval.py.

    MUJOCO_GL=glfw python record_ablation.py \
        --model checkpoints/base-v2-baseline/000028016640 \
        --steps 300 --out videos/ablation_mu.mp4
"""

import argparse
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "glfw")  # macOS: egl/osmesa are unavailable

import jax
import jax.numpy as jp
import numpy as np

import ablate_eval as A


def rollout_frames(policy, mu, steps, seed, height, width):
    """Trajectory + the step the robot fell, for one friction setting."""
    env = A.build_env(mu, False, False, False)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    st = reset(rng)
    st.info["command"] = jp.array([1.0, 0.0, 0.0])
    traj, fell_at = [st], None
    for i in range(steps):
        rng, ar = jax.random.split(rng)
        act, _ = policy(st.obs, ar)
        st = step(st, act)
        st.info["command"] = jp.array([1.0, 0.0, 0.0])
        traj.append(st)
        if fell_at is None and float(st.done) > 0.0:
            fell_at = i
    print(f"  mu={mu:.2f}: rolled {steps} steps, fell at "
          f"{fell_at if fell_at is not None else 'never'}", flush=True)
    frames = env.render(traj[::2], height=height, width=width)
    return np.asarray(frames), fell_at


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mus", type=float, nargs="+", default=[1.0, 0.05])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--out", default="videos/ablation_mu.mp4")
    a = p.parse_args()

    policy = A.load_arm(a.model)
    panels, falls = [], []
    for mu in a.mus:
        f, fell = rollout_frames(policy, mu, a.steps, a.seed, a.height, a.width)
        panels.append(f)
        falls.append(fell)

    # Pad short panels by holding their last frame, so the join stays aligned
    # in time rather than silently truncating the longer rollout.
    n = max(len(f) for f in panels)
    panels = [np.concatenate([f, np.repeat(f[-1:], n - len(f), axis=0)])
              if len(f) < n else f for f in panels]
    # A visible seam, so nobody mistakes one panel for the other.
    seam = np.zeros((n, a.height, 4, 3), dtype=panels[0].dtype)
    joined = np.concatenate(
        [x for pair in zip(panels, [seam] * len(panels)) for x in pair][:-1],
        axis=2)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for mu, fell in zip(a.mus, falls):
        print(f"panel mu={mu:.2f}: fell at step "
              f"{fell if fell is not None else 'never'}", flush=True)
    try:
        # imageio drives imageio_ffmpeg's BUNDLED binary, so this needs no
        # system ffmpeg. mediapy.write_video shells out to an `ffmpeg` on PATH
        # instead and dies with "Program 'ffmpeg' is not found" -- which is the
        # same failure CLAUDE.md records killing a finished GPU run, and the
        # reason train_ice.py apt-installs ffmpeg. It does not have to: this
        # path removes that dependency entirely.
        import imageio.v2 as imageio
        imageio.mimwrite(out, joined, fps=25, codec="libx264",
                         quality=8, macro_block_size=1)
        print(f"wrote {out} ({joined.shape[0]} frames, {joined.shape[2]}x"
              f"{joined.shape[1]})", flush=True)
    except Exception as e:
        # Numbers/frames before artefacts: a codec failure must not lose the run.
        print(f"mp4 failed ({e}); writing PNGs instead", flush=True)
        d = out.with_suffix("")
        d.mkdir(parents=True, exist_ok=True)
        import mediapy
        for i, f in enumerate(joined[::5]):
            mediapy.write_image(d / f"{i:03d}.png", f)
        print(f"wrote {len(joined[::5])} PNGs to {d}", flush=True)


if __name__ == "__main__":
    main()
