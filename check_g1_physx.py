# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["isaacsim[all,extscache]"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# ///
"""Two blockers, one job.

Established by probe v5:
  - isaacsim 6.0.1.0 installs via uv (~24 GB, ~5 min)
  - Kit boots headless ONLY with {"multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
  - Isaac Lab's shared locomotion config pins friction to a POINT, not a range:
        velocity_env_cfg.py: static (0.8, 0.8), dynamic (0.6, 0.6)
  - PhysX reported "no suitable CUDA GPU was found" -> physics ran on CPU
  - libGL.so.1 was missing (my omission), breaking the iray/MDL material stack

This job answers:
  A. does G1's own config OVERRIDE that friction event? (decides the whole thesis)
  B. can PhysX use the GPU once libGL and friends are installed? (decides whether
     Isaac Lab RL training is feasible here at all)

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 30m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=g1-physx \
      check_g1_physx.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
OUT = pathlib.Path("/mnt/himalaya-g1/isaac-probe")
results = {}


def sh(cmd, timeout, label, tail=40):
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


# =============================== PART A: does G1 override the friction event? ===
print("#" * 70 + "\n# PART A - G1 friction: inherited point value, or overridden?\n" + "#" * 70,
      flush=True)

sh("git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab && echo cloned",
   900, "A1. clone", tail=3)

G1DIR = "/tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1"
print("\n--- G1 rough_env_cfg.py: any physics_material / friction override? ---", flush=True)
sh(f"grep -n -i -B2 -A8 'physics_material\\|friction' {G1DIR}/rough_env_cfg.py || "
   "echo 'NO friction/physics_material override in G1 rough_env_cfg.py'",
   120, "A2. G1 rough override")
print("\n--- G1 flat_env_cfg.py ---", flush=True)
sh(f"grep -n -i -B2 -A8 'physics_material\\|friction' {G1DIR}/flat_env_cfg.py || "
   "echo 'NO friction/physics_material override in G1 flat_env_cfg.py'",
   120, "A3. G1 flat override")
print("\n--- the inherited event, in full ---", flush=True)
sh("grep -n -A12 'physics_material = EventTerm' "
   "/tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
   120, "A4. inherited physics_material event")
print("\n--- what does G1 rough_env_cfg actually inherit from? ---", flush=True)
sh(f"head -40 {G1DIR}/rough_env_cfg.py", 120, "A5. G1 class header")

# ================================= PART B: is PhysX able to use the GPU? =======
print("\n" + "#" * 70 + "\n# PART B - PhysX on GPU (decides if RL training is viable here)\n"
      + "#" * 70, flush=True)

sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 && echo apt-ok",
   600, "B1. apt GL libs", tail=3)

GPU_TEST = textwrap.dedent("""
    import os
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False,
                         "active_gpu": 0, "physics_gpu": 0})
    print("BOOT_OK", flush=True)

    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("torch device:", torch.cuda.get_device_name(0), flush=True)

    from isaacsim.core.api import World
    w = World(stage_units_in_meters=1.0, device="cuda:0", backend="torch")
    w.scene.add_default_ground_plane()
    w.reset()

    pc = w.get_physics_context()
    print("physics device :", pc.device, flush=True)
    try:
        print("use_gpu_pipeline:", pc.use_gpu_pipeline, flush=True)
    except Exception as e:
        print("use_gpu_pipeline unavailable:", e, flush=True)
    try:
        print("use_gpu_sim     :", pc.use_gpu_sim, flush=True)
    except Exception as e:
        print("use_gpu_sim unavailable:", e, flush=True)

    for _ in range(50):
        w.step(render=False)
    print("STEPPED_50_ON", pc.device, flush=True)
    app.close()
    print("DONE", flush=True)
""")
pathlib.Path("gpu_test.py").write_text(GPU_TEST)
out = sh(f"{PY} gpu_test.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -30",
         1200, "B2. PhysX GPU test")

gpu_ok = "STEPPED_50_ON cuda" in out
results["B3. VERDICT physx-on-gpu"] = "PASS" if gpu_ok else "FAIL"

try:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g1_physx_summary.txt").write_text(
        "\n".join(f"{v:12}  {k}" for k, v in results.items()) + "\n")
except Exception as e:
    print("bucket write failed:", e, flush=True)

print("\n" + "=" * 56, flush=True)
for k, v in results.items():
    print(f"{v:12}  {k}")
print("=" * 56, flush=True)
print("\nPhysX RUNS ON GPU - Isaac Lab RL training is viable here"
      if gpu_ok else
      "\nPhysX still CPU-only - large-scale Isaac Lab training is NOT viable on this flavor")
