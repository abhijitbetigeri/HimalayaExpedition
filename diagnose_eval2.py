# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "mujoco", "playground", "brax", "mediapy"]
# ///
"""Does ANY trained policy walk on easy ground? The control the first run lacked.

What the first factorial showed (single rollout per cell, ice-v2 only):
    A benign      33/300 (11.0%)   <- WORST
    B ice only    94/300 (31.3%)
    C wind only   86/300 (28.7%)
    D full stack  35/300 (11.7%)

Failing faster on flat mu=0.6 ground than on ice is physically incoherent, so
something beyond "ice is hard" is wrong. But that run could not distinguish
between two very different explanations, because it tested one policy at one seed:

    (i)  the EVAL PATH is broken   -> then the baseline policy also collapses on
                                      easy ground, and every A/B so far is void
    (ii) ice-v2 is simply a BAD policy -> then the baseline walks fine and the
                                      training, not the harness, is what to fix

This run adds the two things that separate them:
  - the BASELINE policy as a control (it was trained WITHOUT wind or patches, so
    benign ground is exactly its training distribution -- it has no excuse to fall)
  - three seeds per cell, because CLAUDE.md notes that toggling wind changes RNG
    consumption, so each condition draws a different initial state and n=1 cannot
    separate physics from initial-condition variance

Only benign and full-stack are run: those are the two cells that decide it, and
CPU rollouts are slow enough that the middle cells are not worth the wall clock.

Run (cpu-basic, no GPU, ~a cent):
  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 55m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=diagnose-control \
      diagnose_eval2.py
"""

import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0",
    shell=True, check=False,
)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
if str(BUCKET) not in sys.path:
    sys.path.insert(0, str(BUCKET))

import jax             # noqa: E402
import jax.numpy as jp  # noqa: E402
from brax.training.agents.ppo import checkpoint as ppo_checkpoint  # noqa: E402

RUNS = os.environ.get("DIAG_RUNS", "ice-v2,base-v2-baseline").split(",")
SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "0,1,2").split(",")]
STEPS = int(os.environ.get("DIAG_STEPS", "300"))

CONDITIONS = [("benign", False, False), ("full", True, True)]


def load_policy(run):
    """Load a run's latest checkpoint, patching brax's null-initializer bug."""
    root = BUCKET / "runs" / run / "checkpoints"
    ckpts = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    assert ckpts, f"no checkpoints under {root}"
    local = pathlib.Path(f"/tmp/ckpt_{run}")
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(ckpts[-1], local)

    cfg_path = local / "ppo_network_config.json"
    cfg = json.loads(cfg_path.read_text())
    kw = cfg.get("network_factory_kwargs", {})
    for k, v in list(kw.items()):
        if k.endswith("_kernel_init_fn") and v is None:
            kw[k] = "lecun_uniform"
    cfg_path.write_text(json.dumps(cfg))
    print(f"  {run}: {ckpts[-1].name}", flush=True)
    return jax.jit(ppo_checkpoint.load_policy(str(local), deterministic=True))


import ice_patch  # noqa: E402

# Build each env once and reuse across policies/seeds -- JIT compile dominates
# runtime on CPU (~30-60 s a piece), so rebuilding per cell would blow the budget.
envs = {}
for name, wind, patches in CONDITIONS:
    envs[name] = ice_patch.load("flat_terrain", config_overrides={
        "wind_config.enable": wind, "patch_config.enable": patches})
    print(f"built env: {name}", flush=True)

results = {}
for run in RUNS:
    inference_fn = load_policy(run)
    for cname, _, _ in CONDITIONS:
        env = envs[cname]
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        cell = []
        for seed in SEEDS:
            t0 = time.time()
            rng = jax.random.PRNGKey(seed)
            state = reset(rng)
            state.info["command"] = jp.array([1.0, 0.0, 0.0])
            survived = 1
            for _ in range(STEPS):
                rng, act_rng = jax.random.split(rng)
                action, _ = inference_fn(state.obs, act_rng)
                state = step(state, action)
                state.info["command"] = jp.array([1.0, 0.0, 0.0])
                if float(state.done) == 0.0:
                    survived += 1
                else:
                    break
            cell.append(survived)
            print(f"  {run:18s} {cname:7s} seed={seed}: {survived:3d}/{STEPS} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        results[f"{run}|{cname}"] = {
            "seeds": cell,
            "mean": round(statistics.mean(cell), 1),
            "max": max(cell),
            "pct_mean": round(100.0 * statistics.mean(cell) / STEPS, 1),
        }

out = BUCKET / "diagnostics"
out.mkdir(parents=True, exist_ok=True)
(out / "eval_control.json").write_text(json.dumps(results, indent=2))

print("\n" + "=" * 68, flush=True)
print(f"{'policy':20s} {'condition':10s} {'seeds':>18s} {'mean':>8s} {'%':>7s}")
print("-" * 68)
for k, v in results.items():
    run, cond = k.split("|")
    print(f"{run:20s} {cond:10s} {str(v['seeds']):>18s} {v['mean']:>8.1f} {v['pct_mean']:>6.1f}%")
print("=" * 68, flush=True)

base_benign = results.get("base-v2-baseline|benign", {}).get("pct_mean", 0)
ice_benign = results.get("ice-v2|benign", {}).get("pct_mean", 0)

print()
if base_benign < 50 and ice_benign < 50:
    print("VERDICT: BOTH policies collapse on easy ground -> the EVAL PATH is at "
          "fault, not the training. Fix the harness before running any more "
          "training; every A/B number so far is void.")
elif base_benign >= 50 and ice_benign < 50:
    print("VERDICT: the baseline WALKS on easy ground but ice-v2 does not -> the "
          "eval harness is sound and ice training genuinely damaged the policy. "
          "Suspect the reward shaping / friction range, not the code.")
else:
    print("VERDICT: both walk on easy ground -> the harness is fine and the "
          "original 38/501 was the full stack being genuinely too hard. "
          "Curriculum, not debugging.")
