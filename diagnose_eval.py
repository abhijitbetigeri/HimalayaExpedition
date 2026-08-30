# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "mujoco", "playground", "brax", "mediapy"]
# ///
"""Why does a trained policy fall after 0.76 s? Factorial eval, no GPU needed.

The problem this isolates
-------------------------
ice-v2 and base-v2 both die at ~step 38-47 of 501 in the full-stack eval, despite
training rewards differing 4x (-2.49 vs -8.12). Two independently trained policies
failing near-identically is a weak signal for "both policies are bad" and a strong
one for "the eval defeats both". Watching the frames confirms the robot collapses
and then sinks through the floor -- the no-reset artifact CLAUDE.md documents.

So: hold the policy fixed and vary the environment. 2x2 over the two things
eval_env turns on that training's benign case does not.

    A  wind off, patches off   <- benign. If THIS fails, the eval path is broken
                                  and both training runs are probably fine.
    B  wind off, patches on    <- ice alone
    C  wind on,  patches off   <- wind alone
    D  wind on,  patches on    <- the full stack that produced 38/501

Reading the result:
    A survives, D fails      -> policy is fine, eval is just brutally hard.
                                Fix by curriculum, not by debugging code.
    A also fails             -> eval path bug. Do not train anything else until
                                it is found; every A/B so far is meaningless.
    A,B survive, C fails     -> wind is the killer, not friction. That would mean
                                the project's headline is aimed at the wrong thing.

Runs on cpu-basic: single-env rollouts, no training. Costs about a cent.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 45m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=diagnose-eval \
      diagnose_eval.py
"""

import json
import os
import pathlib
import subprocess
import sys
import time

# `import mujoco` walks the EGL path eagerly even when nothing is rendered.
subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0",
    shell=True, check=False,
)
os.environ.setdefault("MUJOCO_GL", "osmesa")     # CPU box: no GPU for EGL
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
if str(BUCKET) not in sys.path:
    sys.path.insert(0, str(BUCKET))

import jax                      # noqa: E402
import jax.numpy as jp          # noqa: E402

RUNS = os.environ.get("DIAG_RUNS", "ice-v2,base-v2-baseline").split(",")
SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "0,1,2").split(",")]
STEPS = int(os.environ.get("DIAG_STEPS", "300"))
RUN = RUNS[0]
CKPT_ROOT = BUCKET / "runs" / RUN / "checkpoints"

print(f"jax {jax.__version__} devices={jax.devices()}", flush=True)
print(f"diagnosing run={RUN} steps={STEPS}", flush=True)

# Latest checkpoint = highest step number.
ckpts = sorted([p for p in CKPT_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
assert ckpts, f"no checkpoints under {CKPT_ROOT}"
CKPT = ckpts[-1]
print("checkpoint:", CKPT, flush=True)

from brax.training.agents.ppo import checkpoint as ppo_checkpoint  # noqa: E402

# brax writes a checkpoint it cannot read back: unset optional initializers are
# serialized as null, and load_config does KERNEL_INITIALIZER[None] -> KeyError.
# Every checkpoint this project has produced is affected, so this is not a
# diagnostic-only workaround -- rendering, resuming and deployment all hit it.
#
# Safe because initializers only produce INITIAL values; loading trained weights
# never calls them. Patch a copy, never the shared bucket artifact.
import shutil  # noqa: E402

LOCAL_CKPT = pathlib.Path("/tmp/ckpt")
if LOCAL_CKPT.exists():
    shutil.rmtree(LOCAL_CKPT)
shutil.copytree(CKPT, LOCAL_CKPT)

cfg_path = LOCAL_CKPT / "ppo_network_config.json"
cfg = json.loads(cfg_path.read_text())
kw = cfg.get("network_factory_kwargs", {})
patched = [k for k, v in kw.items() if k.endswith("_kernel_init_fn") and v is None]
for k in patched:
    kw[k] = "lecun_uniform"
if patched:
    cfg_path.write_text(json.dumps(cfg))
    print(f"patched null initializers: {patched}", flush=True)

policy_fn = ppo_checkpoint.load_policy(str(LOCAL_CKPT), deterministic=True)
inference_fn = jax.jit(policy_fn)
print("policy loaded", flush=True)

import ice_patch  # noqa: E402

CONDITIONS = [
    ("A benign      (wind off, patches off)", False, False),
    ("B ice only    (wind off, patches ON )", False, True),
    ("C wind only   (wind ON,  patches off)", True, False),
    ("D full stack  (wind ON,  patches ON )", True, True),
]

results = {}
for label, wind, patches in CONDITIONS:
    t0 = time.time()
    env = ice_patch.load("flat_terrain", config_overrides={
        "wind_config.enable": wind,
        "patch_config.enable": patches,
    })
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    state.info["command"] = jp.array([1.0, 0.0, 0.0])

    survived, min_mu, fell_at = 1, 1.0, None
    for i in range(STEPS):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step(state, action)
        state.info["command"] = jp.array([1.0, 0.0, 0.0])
        min_mu = min(min_mu, float(jp.min(state.info.get("foot_mu", jp.ones(2)))))
        if float(state.done) == 0.0:
            survived += 1
        elif fell_at is None:
            fell_at = i + 1
            # Keep stepping would just sink through the floor; the number we want
            # is time-to-first-fall, so stop here.
            break

    results[label] = {
        "survived": survived, "of": STEPS, "fell_at": fell_at,
        "min_foot_mu": round(min_mu, 4),
        "pct": round(100.0 * survived / STEPS, 1),
        "secs": round(time.time() - t0, 1),
    }
    print(f"{label}: survived {survived}/{STEPS} "
          f"({results[label]['pct']}%)  min_mu={min_mu:.3f}  "
          f"[{results[label]['secs']}s]", flush=True)

out = BUCKET / "diagnostics"
out.mkdir(parents=True, exist_ok=True)
(out / f"eval_factorial_{RUN}.json").write_text(json.dumps(results, indent=2))

print("\n" + "=" * 62, flush=True)
for k, v in results.items():
    print(f"{v['survived']:4d}/{v['of']}  ({v['pct']:5.1f}%)  mu>={v['min_foot_mu']:.3f}  {k}")
print("=" * 62, flush=True)

a = results["A benign      (wind off, patches off)"]
d = results["D full stack  (wind ON,  patches ON )"]
if a["pct"] < 50:
    print("\nVERDICT: benign case ALSO fails -> the EVAL PATH is broken, not the "
          "policy. Every A/B number so far is meaningless until this is fixed.")
elif d["pct"] < 50:
    print("\nVERDICT: policy is fine; the full stack is simply too hard. This is a "
          "CURRICULUM problem, not a bug. Anneal difficulty during training.")
else:
    print("\nVERDICT: policy survives even the full stack here -- so the original "
          "38/501 came from something else in train_ice.py's eval loop.")
