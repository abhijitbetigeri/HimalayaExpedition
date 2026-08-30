# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Train the BASELINE G1 flat-terrain locomotion policy on stock Isaac Lab.

Deliberately unmodified: this policy only ever sees the shipped friction event,
    velocity_env_cfg.py: static (0.8, 0.8), dynamic (0.6, 0.6)
a POINT value with no randomization (verified: neither G1 config overrides it).
That is precisely what makes it the "before" asset -- a policy that has never once
encountered a slippery surface, which we then film falling on ice.

It is also the base that COLA's residual layer sits on top of:
    A_collab = A_wbc + A_residual(mu_hat)

The four uv settings in the header are all load-bearing; see setup_isaaclab.py for
why each one is needed. Env facts measured on this stack: obs (123,), act (37,),
23.7k env-steps/s at 1024 envs on an L4.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor a100-large --timeout 90m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=train-baseline \
      train_baseline.py
"""

import os
import pathlib
import subprocess
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"
OUT = pathlib.Path("/mnt/himalaya-g1/baseline")
TASK = "Isaac-Velocity-Flat-G1-v0"
NUM_ENVS = 4096
MAX_ITER = 1500


def sh(cmd, timeout, label, tail=15):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
    print(f"[{label}] rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    return p.returncode == 0


print("=" * 60 + f"\nBASELINE TRAINING  {TASK}\n"
      f"envs={NUM_ENVS}  iters={MAX_ITER}\n" + "=" * 60, flush=True)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv", 120, "gpu")
sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git && echo apt-ok", 900, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)

for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -2",
       1200, f"install {pkg}", tail=2)
sh(f"{PY} -m pip install --no-cache-dir rsl-rl-lib 2>&1 | tail -1", 600, "rsl_rl", tail=2)

train_py = subprocess.run(
    f"find {LAB} -path '*rsl_rl*' -name train.py | head -1",
    shell=True, capture_output=True, text=True).stdout.strip()
print("train script:", train_py, flush=True)
assert train_py, "could not locate rsl_rl train.py in the Isaac Lab clone"

OUT.mkdir(parents=True, exist_ok=True)

# Checkpoints must survive a timeout: the job dies at --timeout with no warning and
# an ephemeral filesystem, so mirror the log dir to the bucket every 3 minutes
# rather than copying only at the end.
sync = subprocess.Popen(
    f"while true; do sleep 180; cp -r {LAB}/logs/rsl_rl {OUT}/ 2>/dev/null; done",
    shell=True)

print("\n" + "=" * 60 + "\nTRAINING\n" + "=" * 60, flush=True)
ok = sh(f"cd {LAB} && {PY} {train_py} --task {TASK} --headless "
        f"--num_envs {NUM_ENVS} --max_iterations {MAX_ITER} 2>&1 "
        "| grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -120",
        5400, "train", tail=120)

sync.terminate()
sh(f"cp -r {LAB}/logs/rsl_rl {OUT}/ 2>/dev/null; find {OUT} -name '*.pt' | tail -10",
   600, "final sync", tail=12)

print("\nTRAINING", "COMPLETE" if ok else "FAILED", flush=True)
print("checkpoints ->", OUT, flush=True)
