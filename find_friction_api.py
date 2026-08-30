# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Locate the real friction-randomization API in Isaac Lab.

`randomize_rigid_body_material` is referenced by velocity_env_cfg.py as
`mdp.randomize_rigid_body_material`, but it is NOT defined in
isaaclab/envs/mdp/events.py in this revision -- Isaac Lab split physics into
backend packages (isaaclab_physx / isaaclab_ovphysx / isaaclab_newton) and the
material events appear to have moved with it.

NO dependencies on purpose: this only greps a git clone, so it runs in seconds on
cpu-basic instead of installing 24 GB of isaacsim first.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 10m \
      --label name=himalaya-traction --label task=find-friction-api \
      find_friction_api.py
"""

import subprocess

LAB = "/tmp/IsaacLab"


def sh(cmd, timeout=300, tail=120):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
    return p.stdout


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends git && echo ok", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned", 600, 2)

print("\n" + "#" * 70 + "\n# 1. WHERE is randomize_rigid_body_material defined?\n" + "#" * 70)
sh(f"grep -rn 'def randomize_rigid_body_material' {LAB}/source")

print("\n" + "#" * 70 + "\n# 2. its full source\n" + "#" * 70)
sh(f"F=$(grep -rl 'def randomize_rigid_body_material' {LAB}/source --include=*.py | head -1); "
   f"echo \"FILE: $F\"; grep -n -A75 'def randomize_rigid_body_material' $F")

print("\n" + "#" * 70 + "\n# 3. every material/friction event available\n" + "#" * 70)
sh(f"grep -rn 'def .*material\\|def .*friction' {LAB}/source --include=*.py | head -25")

print("\n" + "#" * 70 + "\n# 4. what does the mdp namespace re-export?\n" + "#" * 70)
sh(f"find {LAB}/source -path '*envs/mdp/__init__.py' | head -5 | xargs -I{{}} sh -c "
   "'echo \"--- {}\"; cat {}'", 300, 60)

print("\n" + "#" * 70 + "\n# 5. G1 foot / ankle body names (needed for per-foot events)\n" + "#" * 70)
sh(f"grep -rn -i 'ankle_roll\\|foot_link\\|\\.\\*_ankle' {LAB}/source/isaaclab_tasks "
   "--include=*.py | head -20")
sh(f"grep -rn -i 'G1_MINIMAL_CFG\\|G1_CFG' {LAB}/source/isaaclab_assets --include=*.py | head -10")

print("\n" + "#" * 70 + "\n# 6. MJWarp / Newton solver cfg - solimp/solref reachable?\n" + "#" * 70)
sh(f"grep -rn -A30 'class MJWarpSolverCfg' {LAB}/source/isaaclab_newton --include=*.py | head -50")

print("\n" + "#" * 70 + "\n# 7. terrain physics material (the GROUND's friction)\n" + "#" * 70)
sh(f"grep -rn -B2 -A15 'physics_material' {LAB}/source/isaaclab/isaaclab/terrains/terrain_importer_cfg.py "
   "| head -40")

print("\nDONE", flush=True)
