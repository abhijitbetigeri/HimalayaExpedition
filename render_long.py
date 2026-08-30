# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda12]", "mujoco", "playground", "brax", "mediapy"]
# ///
"""One continuous take: walk on rock, ground turns to ice, go down.

Why one shot instead of two clips
---------------------------------
Two 6-second clips side by side ask the viewer to compare thumbnails, and the
first frame of each is the reset crouch, so they look alike. A single unbroken
take with the surface changing underneath the robot needs no comparison at all --
the same gait that was working stops working, on camera.

Mechanically: pair_friction lives on the MODEL, and a jitted step bakes the model
in as a constant. So build TWO jitted steps over two models and switch which one
is called mid-rollout. State carries across untouched, so the robot does not
notice anything except the physics.

Renders under impl="warp" -- the backend these policies were TRAINED under.
impl="jax" is ~3x faster on CPU but degrades the policy badly (same policy at
mu=0.8: 301/301 under warp, falls at 112 under jax), so it must not be used for
anything filmed or quoted.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 40m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=render-long \
      render_long.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 ffmpeg",
    shell=True, check=False,
)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
OUT = BUCKET / "videos"
RUN = os.environ.get("RENDER_RUN", "base-v2-baseline")

ROCK_MU = 0.8
ICE_MU = 0.06
ROCK_STEPS = int(os.environ.get("ROCK_STEPS", "1400"))   # ~28 s at 50 Hz
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "2600"))  # ~52 s if it survives

import jax                     # noqa: E402
import jax.numpy as jp         # noqa: E402
from brax.training.agents.ppo import checkpoint as ppo_checkpoint  # noqa: E402
from mujoco_playground import locomotion  # noqa: E402


def load_policy(run):
    root = BUCKET / "runs" / run / "checkpoints"
    ckpts = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    local = pathlib.Path(f"/tmp/ck_{run}")
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(ckpts[-1], local)
    # brax serialises unset optional initializers as null then cannot read them back.
    cfg_p = local / "ppo_network_config.json"
    cfg = json.loads(cfg_p.read_text())
    kw = cfg.get("network_factory_kwargs", {})
    for k, v in list(kw.items()):
        if k.endswith("_kernel_init_fn") and v is None:
            kw[k] = "lecun_uniform"
    cfg_p.write_text(json.dumps(cfg))
    print("checkpoint:", ckpts[-1].name, flush=True)
    return jax.jit(ppo_checkpoint.load_policy(str(local), deterministic=True))


def with_mu(env, mu):
    """A copy of the env's model with foot-floor friction pinned to mu.

    Rows 0:2 of pair_friction are the two foot-floor pairs; columns 0:2 are the
    tangential components that govern slipping. Rows 2+ are self-collision pairs
    and are left alone.
    """
    m = env.mjx_model
    return m.tree_replace({"pair_friction": m.pair_friction.at[0:2, 0:2].set(mu)})


inference_fn = load_policy(RUN)
env = locomotion.load("G1JoystickFlatTerrain")   # stock impl=warp, as trained
print("env built (impl=warp, as trained)", flush=True)

rock_model, ice_model = with_mu(env, ROCK_MU), with_mu(env, ICE_MU)


def make_step(model):
    def _step(state, action):
        env._mjx_model = model
        return env.step(state, action)
    return jax.jit(_step)


step_rock, step_ice = make_step(rock_model), make_step(ice_model)

env._mjx_model = rock_model
reset = jax.jit(env.reset)
rng = jax.random.PRNGKey(0)
state = reset(rng)
state.info["command"] = jp.array([1.0, 0.0, 0.0])

traj, switched_at, fell_at = [state], None, None
for i in range(TOTAL_STEPS):
    on_ice = i >= ROCK_STEPS
    if on_ice and switched_at is None:
        switched_at = i
        print(f"--- mu {ROCK_MU} -> {ICE_MU} at step {i} "
              f"({i * env.dt:.1f}s) ---", flush=True)
    rng, act_rng = jax.random.split(rng)
    action, _ = inference_fn(state.obs, act_rng)
    state = (step_ice if on_ice else step_rock)(state, action)
    state.info["command"] = jp.array([1.0, 0.0, 0.0])
    traj.append(state)
    if float(state.done) != 0.0:
        fell_at = i + 1
        print(f"FELL at step {fell_at} ({fell_at * env.dt:.1f}s)", flush=True)
        # Keep filming briefly so the fall is visible rather than a hard cut.
        break
    if (i + 1) % 200 == 0:
        print(f"  {i+1} steps ({(i+1)*env.dt:.0f}s), upright, "
              f"{'ICE' if on_ice else 'rock'}", flush=True)

print(f"total {len(traj)} steps = {len(traj) * env.dt:.1f}s of footage", flush=True)

# The G1 model ships a "track" camera; without it the robot walks out of frame.
frames = env.render(traj[::2], height=480, width=854, camera="track")
OUT.mkdir(parents=True, exist_ok=True)
import mediapy  # noqa: E402
name = f"transition_{RUN}.mp4"
mediapy.write_video(OUT / name, frames, fps=1.0 / (2 * env.dt))

# ---------------------------------------------------------------- seed sweep
# The single clip above is ONE seed. base-v2 is not a reliable walker -- only 3 of
# the first 8 seeds reach 300 steps even on stock ground -- so a single paired
# rollout cannot show that ice is what caused a fall. Run the SAME seeds on both
# surfaces and report the paired difference, which is the only honest number here.
print("\n=== paired seed sweep: same seeds, rock vs ice ===", flush=True)
SWEEP_SEEDS = list(range(int(os.environ.get("SWEEP_N", "12"))))
SWEEP_STEPS = int(os.environ.get("SWEEP_STEPS", "600"))
sweep = {"rock": [], "ice": []}
for surface, stepfn, mu in (("rock", step_rock, ROCK_MU), ("ice", step_ice, ICE_MU)):
    env._mjx_model = rock_model if surface == "rock" else ice_model
    for sd in SWEEP_SEEDS:
        r = jax.random.PRNGKey(sd)
        st = reset(r)
        st.info["command"] = jp.array([1.0, 0.0, 0.0])
        n = 0
        for _ in range(SWEEP_STEPS):
            r, ar = jax.random.split(r)
            a, _ = inference_fn(st.obs, ar)
            st = stepfn(st, a)
            st.info["command"] = jp.array([1.0, 0.0, 0.0])
            n += 1
            if float(st.done) != 0.0:
                break
        sweep[surface].append(n)
    print(f"  {surface:5s} mu={mu}: {sweep[surface]}", flush=True)

import statistics  # noqa: E402
paired = [i - r for r, i in zip(sweep["rock"], sweep["ice"])]
sweep_stats = {
    "seeds": SWEEP_SEEDS, "steps_cap": SWEEP_STEPS,
    "rock": sweep["rock"], "ice": sweep["ice"],
    "rock_mean": round(statistics.mean(sweep["rock"]), 1),
    "ice_mean": round(statistics.mean(sweep["ice"]), 1),
    "paired_delta_mean": round(statistics.mean(paired), 1),
    "ice_worse_on_n_seeds": sum(1 for d in paired if d < 0),
    "of_seeds": len(SWEEP_SEEDS),
}
print(json.dumps(sweep_stats, indent=2), flush=True)
(OUT / f"seed_sweep_{RUN}.json").write_text(json.dumps(sweep_stats, indent=2))

meta = {
    "run": RUN, "rock_mu": ROCK_MU, "ice_mu": ICE_MU, "sweep": sweep_stats,
    "dt": env.dt, "fps": 1.0 / (2 * env.dt),
    "switch_step": switched_at, "switch_seconds": (switched_at or 0) * env.dt,
    "fell_step": fell_at, "fell_seconds": (fell_at * env.dt) if fell_at else None,
    "total_steps": len(traj), "total_seconds": len(traj) * env.dt,
}
(OUT / f"transition_{RUN}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
print("wrote", OUT / name, flush=True)
