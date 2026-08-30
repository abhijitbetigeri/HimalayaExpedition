# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["isaacsim[all,extscache]"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# ///
"""Feasibility probe v5: Isaac Sim on HF Jobs.

Probe history (so the same ground isn't re-lost):
  v1  bare pip/python -> installed into a different interpreter than the script's.
  v2  sys.executable -m pip -> uv's ephemeral venv ships no pip.
  v3  PEP 723 deps worked; Kit blocked on an interactive EULA prompt.
  v4  OMNI_KIT_ACCEPT_EULA=YES cleared that; Kit then CRASHED inside the RTX Hydra
      renderer (librtx.hydra.so / fabric_scene_delegate) at startup.
  v5  this file.

Established so far: isaacsim 6.0.1.0 installs (~24 GB, ~5 min), Vulkan 1.3.239 with
a real NVIDIA ICD, L4 driver 580.178.04, ~360 GiB disk, ~28.6 GB RAM.

v5 changes:
  - Isaac Lab clone + friction-range grep run FIRST. That is the project's critical
    path (the Isaac equivalent of Playground's U(0.4, 1.0)) and needs no working Kit.
  - Kit boot attempts run in isolated subprocesses, so a native crash records a
    result instead of killing the probe.
  - Prime suspect: vulkaninfo enumerates GPU0 AND GPU1 while nvidia-smi shows one
    L4, so multi-GPU init is tried first.
  - Newton is tested standalone as a fallback: it is a real physics engine that
    needs no Kit at all, so RL training could proceed even if the renderer cannot.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 35m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=isaac-probe5 \
      probe_isaac.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")          # Kit writes caches under $HOME
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

PY = sys.executable
OUT = pathlib.Path("/mnt/himalaya-g1/isaac-probe")
results = {}


def sh(cmd, timeout, label, tail=30, show=True):
    print(f"\n$ {cmd}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if show:
            print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
        results[label] = "PASS" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
        return p.stdout
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT", flush=True)
        results[label] = "TIMEOUT"
        return ""


# ---------------------------------------------------------------- critical path
print("#" * 70, flush=True)
print("# PART A - Isaac Lab friction ranges (the project's actual blocker)", flush=True)
print("#" * 70, flush=True)

sh("git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab",
   900, "A1. clone IsaacLab", tail=4)
sh("ls /tmp/IsaacLab/source", 60, "A2. IsaacLab packages")
sh("find /tmp/IsaacLab -ipath '*locomotion*' -ipath '*g1*' | head -20",
   120, "A3. G1 locomotion configs")
sh("grep -rn -A6 'randomize_rigid_body_material' /tmp/IsaacLab/source "
   "--include=*.py | grep -iE 'friction_range|func=|static|dynamic' | head -40",
   240, "A4. friction randomization ranges")
sh("grep -rn 'static_friction_range\\|dynamic_friction_range' /tmp/IsaacLab/source "
   "--include=*.py | head -30", 240, "A5. explicit friction ranges")

# ------------------------------------------------------------------ kit boot
print("\n" + "#" * 70, flush=True)
print("# PART B - can Kit boot headless at all?", flush=True)
print("#" * 70, flush=True)

BOOT = textwrap.dedent("""
    import os, sys, json
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    cfg = json.loads(sys.argv[1])
    from isaacsim import SimulationApp
    app = SimulationApp(cfg)
    print("BOOT_OK", flush=True)
    from isaacsim.core.api import World
    w = World(stage_units_in_meters=1.0)
    w.scene.add_default_ground_plane()
    w.reset()
    for _ in range(20):
        w.step(render=False)
    print("STEP_OK", flush=True)
    app.close()
    print("CLOSE_OK", flush=True)
""")
pathlib.Path("boot_try.py").write_text(BOOT)

CONFIGS = [
    ("B1. single-GPU", {"headless": True, "multi_gpu": False,
                        "active_gpu": 0, "physics_gpu": 0}),
    ("B2. single-GPU + no fabric delegate", {"headless": True, "multi_gpu": False,
                                             "active_gpu": 0, "physics_gpu": 0,
                                             "fabric_scene_delegate": False}),
    ("B3. plain headless (v4 baseline)", {"headless": True}),
]

booted = None
for label, cfg in CONFIGS:
    print(f"\n--- {label}: {cfg} ---", flush=True)
    out = sh(f"{PY} boot_try.py '{json.dumps(cfg)}' 2>&1 | tail -25", 900, label, tail=25)
    if "STEP_OK" in out:
        results[label] = "PASS"
        booted = (label, cfg)
        print(f"*** {label} BOOTED AND STEPPED PHYSICS ***", flush=True)
        break
    results[label] = "CRASH" if "BOOT_OK" not in out else "BOOT_ONLY"

# --------------------------------------------------------------- newton fallback
print("\n" + "#" * 70, flush=True)
print("# PART C - Newton standalone (no Kit) - the fallback if the renderer is dead",
      flush=True)
print("#" * 70, flush=True)
sh(f"{PY} -c \"import newton, mujoco, warp; warp.init(); "
   "print('newton', newton.__version__, '| mujoco', mujoco.__version__, "
   "'| warp', warp.__version__); print('warp devices:', warp.get_devices())\"",
   600, "C1. newton/warp standalone")

# ------------------------------------------------------------------- summary
try:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe5_summary.txt").write_text(
        "\n".join(f"{v:12}  {k}" for k, v in results.items()) + "\n")
except Exception as e:
    print("bucket write failed:", e, flush=True)

print("\n" + "=" * 56, flush=True)
for k, v in results.items():
    print(f"{v:12}  {k}")
print("=" * 56, flush=True)
print(f"\nVERDICT: Kit boots with {booted[0]}" if booted
      else "\nVERDICT: Kit does NOT boot headless here - Newton/Isaac Lab-on-Newton "
           "is the remaining Isaac path")
