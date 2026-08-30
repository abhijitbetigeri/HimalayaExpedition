# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Film the ice policies on ground that LOOKS like snow and ice, not a grey plane.

The gap this closes: every clip so far set friction to 0.20 and called it snow.
Friction is a number. The ground still rendered as the same grey checkerboard, so a
"snow" clip and a "rock" clip were visually identical and the footage proved
nothing to a viewer. Physics and appearance are configured separately in Isaac Lab
and only the physics had been touched.

  snow    near-white, high roughness, no metallic -- diffuse like packed neve
  ice     pale blue, LOW roughness -- specular, so it reads as wet/glazed
  rock    dark grey, high roughness

Each surface pairs the right look with the right friction, so what is on screen
matches what the solver is doing.

Runs on h200 for speed; this is rendering-bound, not physics-bound.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor h200 --timeout 50m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=film-snowscape \
      film_snowscape.py
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"
BUCKET = pathlib.Path("/mnt/himalaya-g1")
POLICIES = {
    "baseline": BUCKET / "baseline/rsl_rl/g1_flat/2026-08-29_20-54-12/exported/policy.pt",
    "ice": BUCKET / "ice-isaac/rsl_rl/g1_flat/2026-08-29_21-48-51/exported/policy.pt",
}


def sh(cmd, timeout, label, tail=10):
    print(f"\n$ {cmd}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT -- continuing", flush=True)
        return False


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git ffmpeg && echo apt-ok", 900, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)
for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks",
            "isaaclab_visualizers", "isaaclab_contrib"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -1",
       1200, f"install {pkg}", tail=1)
sh(f"{PY} -m pip install --no-cache-dir rsl-rl-lib imageio imageio-ffmpeg 2>&1 | tail -1",
   600, "deps", tail=1)

G1DIR = pathlib.Path(f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                     f"locomotion/velocity/config/g1")
(G1DIR / "snowscape_env_cfg.py").write_text(textwrap.dedent('''
    """Surfaces that LOOK like what they physically are. Generated."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils.configclass import configclass
    from .flat_env_cfg import G1FlatEnvCfg

    # Appearance and physics are set separately in Isaac Lab, and until now only
    # the physics was touched -- which is why "snow" rendered as a grey plane.
    SNOW = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.95, 0.98),
                                       roughness=0.9, metallic=0.0)
    ICE  = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.74, 0.85, 0.93),
                                       roughness=0.12, metallic=0.0)
    ROCK = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.29, 0.28),
                                       roughness=0.95, metallic=0.0)


    class _Base(G1FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.scene.env_spacing = 8.0
            self.observations.policy.enable_corruption = False
            # The green blob in every previous clip was this: the velocity command
            # marker, rendered larger than the robot.
            self.commands.base_velocity.debug_vis = False
            self.events.push_robot = None
            self.events.base_external_force_torque = None
            self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.resampling_time_range = (1e6, 1e6)


    @configclass
    class G1SnowCfg(_Base):
        """Packed snow / neve: mu 0.20, and it finally looks like snow."""
        def __post_init__(self):
            super().__post_init__()
            self.scene.terrain.visual_material = SNOW
            self.events.physics_material.params["static_friction_range"] = (0.20, 0.20)
            self.events.physics_material.params["dynamic_friction_range"] = (0.16, 0.16)


    @configclass
    class G1GlacierCfg(_Base):
        """Hard glacial ice: mu 0.12, pale blue and specular."""
        def __post_init__(self):
            super().__post_init__()
            self.scene.terrain.visual_material = ICE
            self.events.physics_material.params["static_friction_range"] = (0.12, 0.12)
            self.events.physics_material.params["dynamic_friction_range"] = (0.10, 0.10)


    @configclass
    class G1RockCfg(_Base):
        def __post_init__(self):
            super().__post_init__()
            self.scene.terrain.visual_material = ROCK
            self.events.physics_material.params["static_friction_range"] = (0.80, 0.80)
            self.events.physics_material.params["dynamic_friction_range"] = (0.60, 0.60)
'''))
with open(G1DIR / "__init__.py", "a") as f:
    f.write(textwrap.dedent('''

        import gymnasium as gym  # noqa: E402
        from . import agents  # noqa: E402
        for _id, _cls in [("Isaac-Snowscape-G1-Snow-v0", "G1SnowCfg"),
                          ("Isaac-Snowscape-G1-Glacier-v0", "G1GlacierCfg"),
                          ("Isaac-Snowscape-G1-Rock-v0", "G1RockCfg")]:
            gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                         disable_env_checker=True,
                         kwargs={"env_cfg_entry_point":
                                     f"{__name__}.snowscape_env_cfg:{_cls}",
                                 "rsl_rl_cfg_entry_point":
                                     f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"})
    '''))

for name, jit in POLICIES.items():
    if pathlib.Path(jit).exists():
        subprocess.run(f"cp {jit} /tmp/policy_{name}.pt", shell=True)
        print(f"policy {name} staged", flush=True)
    else:
        print(f"!! missing {jit}", flush=True)

FILM = textwrap.dedent('''
    import json, os, pathlib
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

    import gymnasium as gym, torch, imageio, numpy as np
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg

    OUT = pathlib.Path("/mnt/himalaya-g1/videos"); OUT.mkdir(parents=True, exist_ok=True)
    STEPS = 1500
    SHOTS = [("ice", "snow", "Isaac-Snowscape-G1-Snow-v0"),
             ("baseline", "snow", "Isaac-Snowscape-G1-Snow-v0"),
             ("ice", "glacier", "Isaac-Snowscape-G1-Glacier-v0"),
             ("baseline", "glacier", "Isaac-Snowscape-G1-Glacier-v0"),
             ("ice", "rock", "Isaac-Snowscape-G1-Rock-v0")]
    report = {}
    for pol, surf, task in SHOTS:
        pth = f"/tmp/policy_{pol}.pt"
        if not pathlib.Path(pth).exists():
            print(f"SKIP {pol}/{surf}", flush=True); continue
        tag = f"{pol}_on_{surf}"
        print(f"=== {tag} ===", flush=True)
        try:
            policy = torch.jit.load(pth).to("cuda:0").eval()
            cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
            cfg.viewer.origin_type = "asset_root"
            cfg.viewer.asset_name = "robot"
            cfg.viewer.env_index = 0
            cfg.viewer.eye = (1.9, 1.9, 1.0)      # close enough to read the gait
            cfg.viewer.lookat = (0.0, 0.0, 0.55)  # pelvis height
            cfg.viewer.resolution = (1280, 720)
            env = gym.make(task, cfg=cfg, render_mode="rgb_array")
            obs_d, _ = env.reset(); obs = obs_d["policy"]
            frames, falls = [], torch.zeros(1, device="cuda:0")
            with torch.inference_mode():
                for i in range(STEPS):
                    obs_d, _, term, trunc, _ = env.step(policy(obs))
                    obs = obs_d["policy"]
                    falls += term.float()
                    if i % 2 == 0:
                        f = env.unwrapped.render()
                        if f is not None:
                            frames.append(np.asarray(f))
            env.close()
            if not frames:
                report[tag] = {"error": "no frames"}; continue
            mid = np.asarray(frames[len(frames)//2], dtype=float)
            if float(mid.std()) < 12.0:
                print(f"!! {tag}: frame std {mid.std():.1f} -- subject may be "
                      f"out of shot", flush=True)
            p = OUT / f"snowscape_{tag}.mp4"
            imageio.mimwrite(p, frames, fps=25, quality=9, macro_block_size=1)
            fpe = falls.mean().item()
            report[tag] = {"falls_per_env": round(fpe, 2), "steps": STEPS,
                           "frame_std": round(float(mid.std()), 1), "file": str(p)}
            print(f"WROTE {p}  falls/env {fpe:.2f}  std {mid.std():.1f}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); report[tag] = {"error": repr(e)}
    (OUT / "snowscape_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    app.close()
''')
pathlib.Path("f.py").write_text(FILM)
sh(f"{PY} f.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -35",
   2400, "film", tail=35)
