# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Higher-fidelity retrain: ice on slopes with a CORRECTED contact model.

Why this run exists
-------------------
The flat-ice policy works (97.8% survival across mu 0.05-1.0) but it has only ever
seen level ground. Filming it on a 14-19 degree slope would show it sliding, which
demonstrates a gap rather than a capability. A mountain policy has to be trained on
mountains.

Isaac Lab's ROUGH velocity task already generates pyramid slopes, stairs and
discrete obstacles, and G1RoughEnvCfg carries the height-scanner observation the
policy needs to see terrain ahead. So the recipe is the rough task plus the same
friction widening that made the flat version work -- one change from a known-good
baseline, which is the discipline that has separated the runs that worked from the
ones that did not.

Deltas from stock G1RoughEnvCfg:
  friction     (0.8, 0.8)/(0.6, 0.6) point value  ->  (0.05, 1.0)/(0.04, 0.9)
  feet_slide   -0.1 -> -0.01
  friction_combine_mode  average -> MIN
  solver iterations      4/0 -> 8/2

The combine mode is a correction, not a tweak. PhysX combines the two contacting
materials, and Isaac Lab defaults to `average` -- so a foot at mu=0.8 on ice at
mu=0.06 was actually experiencing mu=0.43. Physically the slipperier surface
dominates: a grippy boot does not give you purchase on ice. Every ice number
produced before this ran on ground roughly 7x grippier than intended.

Expect results to get WORSE. That is the point: a policy that only worked because
the contact model was generous is not a result. See SIM_ACCURACY.md.

Longer than the flat run: rough terrain plus a 20x friction range is a much harder
problem than either alone, and the flat run's 1500 iterations were already only just
enough.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor a100-large --timeout 3h \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=train-ice-slope \
      train_ice_slope.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"
OUT = pathlib.Path("/mnt/himalaya-g1/ice-slope-hifi")
TASK = "Isaac-Velocity-Rough-G1-IceHifi-v0"
NUM_ENVS = 4096
MAX_ITER = int(os.environ.get("MAX_ITER", "3000"))

ICE_STATIC = (0.05, 1.0)
ICE_DYNAMIC = (0.04, 0.9)


def sh(cmd, timeout, label, tail=15):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
    print(f"[{label}] rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    return p.returncode == 0


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git ffmpeg && echo apt-ok", 900, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)
for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks",
            "isaaclab_visualizers"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -1",
       1200, f"install {pkg}", tail=1)
sh(f"{PY} -m pip install --no-cache-dir rsl-rl-lib 2>&1 | tail -1", 600, "rsl_rl", tail=1)

G1DIR = pathlib.Path(f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                     f"locomotion/velocity/config/g1")

(G1DIR / "ice_slope_env_cfg.py").write_text(textwrap.dedent(f'''
    """G1 on icy slopes, corrected contact model. Generated."""
    from isaaclab.utils.configclass import configclass
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab.sim import SimulationCfg
    from .rough_env_cfg import G1RoughEnvCfg


    @configclass
    class G1IceSlopeEnvCfg(G1RoughEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            # Stock is a POINT value, (0.8, 0.8)/(0.6, 0.6) -- the humanoid task
            # does not randomise ground friction at all. Span ice to dry rock.
            self.events.physics_material.params["static_friction_range"] = {ICE_STATIC}
            self.events.physics_material.params["dynamic_friction_range"] = {ICE_DYNAMIC}
            # Sliding is unavoidable on ice; at -0.1 the term punishes the surface
            # rather than the gait and the policy learns to stand still instead.
            self.rewards.feet_slide.weight = -0.01

            # THE correction: PhysX defaults to averaging the two materials, so a
            # grippy foot on ice yielded ~mu=0.43. The slipperier surface should
            # dominate -- that is why ice is dangerous regardless of your boots.
            self.scene.terrain.physics_material.friction_combine_mode = "min"
            self.scene.terrain.physics_material.restitution_combine_mode = "min"

            # Grazing low-friction contacts are the worst-conditioned case for an
            # iterative solver: stick-vs-slip has to be decided with very little
            # normal force. The locomotion defaults (4/0) are tuned for grippy
            # ground. Costs ~20-35% throughput.
            self.sim = SimulationCfg(physics=PhysxCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                gpu_max_rigid_patch_count=10 * 2**15,
            ))


    @configclass
    class G1IceSlopeEnvCfg_PLAY(G1IceSlopeEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 16
            self.observations.policy.enable_corruption = False
            self.events.push_robot = None
            self.events.base_external_force_torque = None
'''))

with open(G1DIR / "__init__.py", "a") as f:
    f.write(textwrap.dedent('''

        import gymnasium as gym  # noqa: E402
        from . import agents  # noqa: E402

        for _id, _cls in [("Isaac-Velocity-Rough-G1-IceHifi-v0", "G1IceSlopeEnvCfg"),
                          ("Isaac-Velocity-Rough-G1-IceHifi-Play-v0", "G1IceSlopeEnvCfg_PLAY")]:
            gym.register(
                id=_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.ice_slope_env_cfg:{_cls}",
                    "rsl_rl_cfg_entry_point":
                        f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
                },
            )
    '''))

# Never spend an hour of GPU on a config that did not take effect.
VERIFY = textwrap.dedent(f'''
    import os
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    import gymnasium as gym
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg("{TASK}", device="cuda:0", num_envs=4)
    p = cfg.events.physics_material.params
    print("static :", p["static_friction_range"], flush=True)
    print("dynamic:", p["dynamic_friction_range"], flush=True)
    print("terrain:", cfg.scene.terrain.terrain_type, flush=True)
    print("combine:", cfg.scene.terrain.physics_material.friction_combine_mode, flush=True)
    assert cfg.scene.terrain.physics_material.friction_combine_mode == "min", \
        "friction combine mode did not apply -- ice would be 7x too grippy"
    print("feet_slide:", cfg.rewards.feet_slide.weight, flush=True)
    assert p["static_friction_range"] == {ICE_STATIC}, "friction override did not apply"
    assert cfg.scene.terrain.terrain_type == "generator", "not on generated terrain!"
    print("VERIFY_OK", flush=True)
    app.close()
''')
pathlib.Path("verify.py").write_text(VERIFY)
if not sh(f"{PY} verify.py 2>&1 | grep -E 'static|dynamic|terrain|feet_slide|VERIFY_OK|Error|assert' "
          "| tail -10", 900, "verify", tail=10):
    sys.exit("verification failed - not spending GPU time")

train_py = subprocess.run(f"find {LAB} -path '*rsl_rl*' -name train.py | head -1",
                          shell=True, capture_output=True, text=True).stdout.strip()
OUT.mkdir(parents=True, exist_ok=True)
sync = subprocess.Popen(
    f"while true; do sleep 180; cp -r {LAB}/logs/rsl_rl {OUT}/ 2>/dev/null; done", shell=True)

print("\n" + "=" * 60 + f"\nTRAINING ICE + SLOPES  {MAX_ITER} iters\n" + "=" * 60, flush=True)
ok = sh(f"cd {LAB} && {PY} {train_py} --task {TASK} --headless "
        f"--num_envs {NUM_ENVS} --max_iterations {MAX_ITER} 2>&1 "
        "| grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -60",
        10800, "train", tail=60)

sync.terminate()
sh(f"cp -r {LAB}/logs/rsl_rl {OUT}/ 2>/dev/null; find {OUT} -name '*.pt' | tail -5",
   600, "final sync", tail=8)
print("\nICE+SLOPE TRAINING", "COMPLETE" if ok else "FAILED", flush=True)
