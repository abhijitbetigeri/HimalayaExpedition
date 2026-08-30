# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Dump the Isaac Lab events API so the ice EventTerm is written against reality.

Needed because the ice design in ice_randomize.py (log-uniform mu over [0.05, 1.0],
sampled INDEPENDENTLY PER FOOT) cannot be expressed with the stock event:
`randomize_rigid_body_material` samples UNIFORM, and uniform over [0.05, 1.0] puts
~95% of its mass above 0.15 -- i.e. you would barely train on ice at all.

So we need a custom event func. This job prints the exact source of the stock one
plus the physx view accessors it uses, so the replacement matches the real API
(signature, env_ids semantics, material tensor shape, bucket behaviour) instead of
my recollection of it.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 25m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=events-api \
      inspect_events_api.py
"""

import os
import pathlib
import subprocess
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"
OUT = pathlib.Path("/mnt/himalaya-g1/api")


def sh(cmd, timeout, label, tail=200):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
    return p.stdout


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends git && echo ok",
   600, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)

EV = f"{LAB}/source/isaaclab/isaaclab/envs/mdp/events.py"

print("\n" + "#" * 70 + "\n# 1. randomize_rigid_body_material - full source\n" + "#" * 70,
      flush=True)
sh(f"grep -n -A80 'def randomize_rigid_body_material' {EV}", 120, "stock func")

print("\n" + "#" * 70 + "\n# 2. what other friction/material events exist\n" + "#" * 70,
      flush=True)
sh(f"grep -n '^def ' {EV}", 120, "event catalogue")

print("\n" + "#" * 70 + "\n# 3. the G1 flat env cfg we will subclass\n" + "#" * 70,
      flush=True)
G1 = f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1"
sh(f"cat {G1}/flat_env_cfg.py", 120, "g1 flat cfg")

print("\n" + "#" * 70 + "\n# 4. foot body names on the Isaac G1 asset (37 DOF)\n" + "#" * 70,
      flush=True)
sh(f"grep -rn -i 'ankle\\|foot\\|feet' {G1}/rough_env_cfg.py | head -30", 120, "g1 feet")

print("\n" + "#" * 70 + "\n# 5. how the ground plane's material is configured\n" + "#" * 70,
      flush=True)
sh(f"grep -rn -B3 -A12 'physics_material' "
   f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
   120, "terrain material")
sh(f"grep -rn -A15 'class TerrainImporterCfg' {LAB}/source/isaaclab/isaaclab/terrains/terrain_importer_cfg.py "
   "| head -40", 120, "terrain importer cfg")

try:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "api_dumped.txt").write_text("see job logs\n")
except Exception as e:
    print("bucket write failed:", e, flush=True)
print("\nAPI DUMP COMPLETE", flush=True)
