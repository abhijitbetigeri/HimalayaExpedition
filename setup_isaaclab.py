# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
# prerelease: isaacsim-core==6.0.1.0 hard-depends on tinyobjloader==2.0.0rc13, a
# pre-release. pip accepts pinned pre-releases silently; uv refuses unless told.
# This is why the original bare-pip install "just worked" and every uv attempt did not.
# NOTE: Python 3.12, not the 3.11 the earlier probes used. Isaac Lab main declares
# requires-python >=3.12, and pypi.nvidia.com publishes cp310/cp311/cp312 wheels for
# isaacsim, so 3.12 satisfies both -- and matches the job image's native interpreter.
#
# index-strategy: on 3.12 the isaacsim[all] closure spans BOTH indexes (e.g.
# mujoco-usd-converter is on PyPI at a version NVIDIA's index does not carry), and
# uv's default refuses to mix them as a dependency-confusion guard. Both indexes are
# first-party here (PyPI + NVIDIA), so relaxing it is fine -- but it is a real
# security relaxation, not a formality, so it stays scoped to this script.
#
# The ==6.0.1.0 pin is load-bearing, not cosmetic. 6.0.1.0 publishes ONLY cp312
# wheels; older lines are cp310/cp311. Left unpinned under unsafe-best-match, uv
# walks down to isaacsim 5.0.0.0 on PyPI, whose packages are wheel_stub placeholders
# that then fail to fetch the real wheels. Pinning keeps resolution on NVIDIA's real
# 6.0.1.0 wheels while still allowing PyPI to satisfy the stragglers.
#
# Corollary: probe v5 booted Kit on Python 3.11, so what it validated was Isaac Sim
# 5.x, NOT 6.0.1. The boot recipe (multi_gpu=False etc.) still needs confirming on
# 6.0.1 -- treat a crash here as "unverified on 6.0", not a regression.
"""Install Isaac Lab from source, instantiate the G1 env, measure throughput.

Everything below is built on what the probes established:
  - isaacsim 6.0.1.0 via uv, OMNI_KIT_ACCEPT_EULA=YES, ~5 min
  - Kit boots headless only with multi_gpu=False / active_gpu=0 / physics_gpu=0
  - PhysX uses the GPU only when the World/env is given device="cuda:0"
  - apt: libgl1 & friends, or the MDL material stack fails on libGL.so.1
  - `pip` is declared as a dep because uv's venv ships none and Isaac Lab's
    installer shells out to `python -m pip`

Answers:
  1. does Isaac Lab install from source here at all
  2. does Isaac-Velocity-Flat-G1-v0 build and step on GPU
  3. what are the friction event's ACTUAL runtime values (source said (0.8,0.8))
  4. env-steps/sec at scale -> how much training $30 actually buys

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 50m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=isaaclab-setup \
      setup_isaaclab.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"
OUT = pathlib.Path("/mnt/himalaya-g1/isaaclab")
results = {}


def sh(cmd, timeout, label, tail=25):
    print(f"\n$ {cmd}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
        results[label] = "PASS" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
        return p.stdout
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT", flush=True)
        results[label] = "TIMEOUT"
        return ""


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git && echo apt-ok", 600, "1. apt", tail=2)

sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "2. clone IsaacLab", tail=3)

# Install the core packages directly rather than via ./isaaclab.sh, which pulls in
# every RL framework and doubles the install time. rsl_rl is the one COLA-style
# PPO work needs.
# Isaac Lab now splits its physics backends into separate packages; isaaclab_tasks
# imports isaaclab_ovphysx at registration time, so the backends are NOT optional.
# isaaclab_newton is included because Isaac Sim 6.0 ships Newton and it is the
# fallback if PhysX ever misbehaves.
for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -3",
       1200, f"3. install {pkg}", tail=4)
sh(f"{PY} -m pip install --no-cache-dir rsl-rl-lib 2>&1 | tail -2", 600, "4. rsl_rl", tail=3)

# Subprocess: a Kit crash must not take the whole job down.
ENVTEST = textwrap.dedent("""
    import os, time
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, device="cuda:0")
    simulation_app = app_launcher.app
    print("APP_OK", flush=True)

    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401  (registers the envs)
    from isaaclab_tasks.utils import parse_env_cfg

    g1 = sorted(k for k in gym.registry.keys() if "G1" in k)
    print("G1 TASKS:", g1, flush=True)

    TASK = "Isaac-Velocity-Flat-G1-v0"
    NUM = 1024
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM)

    # The claim, read off the live config rather than the source file.
    ev = cfg.events.physics_material
    print("FRICTION EVENT params:", ev.params, flush=True)

    env = gym.make(TASK, cfg=cfg)
    print("ENV_OK obs:", env.observation_space, flush=True)
    print("        act:", env.action_space, flush=True)

    obs, _ = env.reset()
    act = torch.zeros(env.unwrapped.num_envs,
                      env.unwrapped.action_space.shape[-1], device="cuda:0")
    for _ in range(20):            # warm up jit/kernels before timing
        env.step(act)
    torch.cuda.synchronize()

    N = 200
    t0 = time.time()
    for _ in range(N):
        env.step(act)
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = NUM * N / dt
    print(f"THROUGHPUT {sps:,.0f} env-steps/sec with {NUM} envs", flush=True)
    print(f"BUDGET at $0.80/h: {sps*3600/1e6:,.1f} M steps per GPU-hour", flush=True)

    env.close()
    simulation_app.close()
    print("DONE", flush=True)
""")
pathlib.Path("env_test.py").write_text(ENVTEST)
out = sh(f"{PY} env_test.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -40",
         1800, "5. G1 env + throughput", tail=40)

results["6. VERDICT isaaclab-g1"] = "PASS" if "THROUGHPUT" in out else "FAIL"

try:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "setup_summary.txt").write_text(
        "\n".join(f"{v:12}  {k}" for k, v in results.items()) + "\n")
except Exception as e:
    print("bucket write failed:", e, flush=True)

print("\n" + "=" * 56, flush=True)
for k, v in results.items():
    print(f"{v:12}  {k}")
print("=" * 56, flush=True)
