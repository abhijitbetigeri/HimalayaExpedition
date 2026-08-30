# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Film the Isaac ice policy actually walking on ice -- flat, and up a steep slope.

Two earlier attempts at this failed and both failures inform the design:

  1. `play.py --video --enable_cameras` ran clean and recorded NOTHING. No
     RecordVideo output anywhere in the log. So do not rely on it for capture.
  2. Hand-rolling `OnPolicyRunner` died with
     `MLPModel.__init__() got an unexpected keyword argument 'stochastic'` --
     an rsl-rl config/version mismatch that train.py avoids via its Hydra decorator.

The way around BOTH: `play.py` exports a TorchScript policy to
`logs/.../exported/policy.pt` on startup. That is a plain `torch.jit` module with
no rsl-rl dependency at all, so stage 1 runs play.py purely to obtain it and
stage 2 drives an explicit render loop with `env.unwrapped.render()`.

Surfaces filmed, friction pinned to a POINT so clips are comparable:
    rock   mu = 0.80   what the baseline was trained for
    ice    mu = 0.08   bare ice, flat
    slope  mu = 0.08   bare ice on a pyramid slope -- the mountain case

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 60m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=isaac-film \
      isaac_film.py
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
    "baseline": BUCKET / "baseline/rsl_rl/g1_flat/2026-08-29_20-54-12/model_1499.pt",
    "ice": BUCKET / "ice-isaac/rsl_rl/g1_flat/2026-08-29_21-48-51/model_1499.pt",
}


def sh(cmd, timeout, label, tail=12):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
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
sh(f"{PY} -m pip install --no-cache-dir rsl-rl-lib imageio imageio-ffmpeg 2>&1 | tail -1",
   600, "rsl_rl", tail=1)

G1DIR = pathlib.Path(f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                     f"locomotion/velocity/config/g1")

(G1DIR / "film_env_cfg.py").write_text(textwrap.dedent('''
    """Filming variants: flat rock, flat ice, and ICE ON A STEEP SLOPE."""
    from isaaclab.utils.configclass import configclass
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from .flat_env_cfg import G1FlatEnvCfg


    class _FilmBase(G1FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            # A handful of robots so the camera frames the action, and no noise or
            # shoves -- what is on screen should be the policy, not the disturbance
            # generator.
            self.scene.num_envs = 6
            self.scene.env_spacing = 3.0
            self.observations.policy.enable_corruption = False
            self.events.push_robot = None
            self.events.base_external_force_torque = None
            # Drive them forward at a steady clip rather than sampling commands, so
            # the clip shows sustained locomotion instead of milling about.
            self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.resampling_time_range = (1e6, 1e6)


    @configclass
    class G1FilmRockCfg(_FilmBase):
        def __post_init__(self):
            super().__post_init__()
            self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
            self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)


    @configclass
    class G1FilmIceCfg(_FilmBase):
        def __post_init__(self):
            super().__post_init__()
            self.events.physics_material.params["static_friction_range"] = (0.08, 0.08)
            self.events.physics_material.params["dynamic_friction_range"] = (0.06, 0.06)


    # ---- the mountain case: bare ice on a steep pyramid slope --------------------
    SLOPE_TERRAIN = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=3,
        num_cols=3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.25, 0.35),
                platform_width=2.0, border_width=0.25,
            ),
        },
    )


    @configclass
    class G1FilmSlopeIceCfg(_FilmBase):
        def __post_init__(self):
            super().__post_init__()
            self.scene.terrain.terrain_type = "generator"
            self.scene.terrain.terrain_generator = SLOPE_TERRAIN
            self.events.physics_material.params["static_friction_range"] = (0.08, 0.08)
            self.events.physics_material.params["dynamic_friction_range"] = (0.06, 0.06)
'''))

with open(G1DIR / "__init__.py", "a") as f:
    f.write(textwrap.dedent('''

        import gymnasium as gym  # noqa: E402
        from . import agents  # noqa: E402

        for _id, _cls in [("Isaac-Film-G1-Rock-v0", "G1FilmRockCfg"),
                          ("Isaac-Film-G1-Ice-v0", "G1FilmIceCfg"),
                          ("Isaac-Film-G1-SlopeIce-v0", "G1FilmSlopeIceCfg")]:
            gym.register(
                id=_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.film_env_cfg:{_cls}",
                    "rsl_rl_cfg_entry_point":
                        f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
                },
            )
    '''))

# ---- stage 1: let play.py export a TorchScript policy we can drive ourselves ----
play_py = subprocess.run(f"find {LAB} -path '*rsl_rl*' -name play.py | head -1",
                         shell=True, capture_output=True, text=True).stdout.strip()
for name, ckpt in POLICIES.items():
    print(f"\n=== exporting {name} ===", flush=True)
    sh(f"cd {LAB} && timeout 900 {PY} {play_py} --task Isaac-Film-G1-Rock-v0 "
       f"--checkpoint {ckpt} --num_envs 2 --headless 2>&1 "
       "| grep -viE 'neuraylib|material_library|warning' | tail -8",
       1000, f"export {name}", tail=8)
    got = subprocess.run(f"find {LAB}/logs -name 'policy.pt' -newermt '-20 minutes' | head -1",
                         shell=True, capture_output=True, text=True).stdout.strip()
    if got:
        dst = f"/tmp/policy_{name}.pt"
        subprocess.run(f"cp {got} {dst}", shell=True)
        print(f"exported -> {dst}", flush=True)
    else:
        print(f"!! no policy.pt exported for {name}", flush=True)
    subprocess.run(f"rm -rf {LAB}/logs", shell=True)

# ---- stage 2: explicit capture loop ---------------------------------------------
FILM = textwrap.dedent('''
    import json, os, pathlib, sys
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    # enable_cameras is what actually makes render() return pixels. play.py's
    # --video path silently produced nothing without it taking effect.
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

    import gymnasium as gym, torch, imageio, numpy as np
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg

    OUT = pathlib.Path("/mnt/himalaya-g1/videos"); OUT.mkdir(parents=True, exist_ok=True)
    STEPS = int(os.environ.get("FILM_STEPS", "1500"))   # 1500 * 0.02s = 30 s

    SHOTS = [
        ("ice",      "ice",      "Isaac-Film-G1-Ice-v0"),
        ("baseline", "ice",      "Isaac-Film-G1-Ice-v0"),
        ("ice",      "slopeice", "Isaac-Film-G1-SlopeIce-v0"),
        ("ice",      "rock",     "Isaac-Film-G1-Rock-v0"),
    ]
    report = {}
    for pol_name, surf, task in SHOTS:
        pol_path = f"/tmp/policy_{pol_name}.pt"
        if not pathlib.Path(pol_path).exists():
            print(f"SKIP {pol_name}/{surf}: no exported policy", flush=True); continue
        tag = f"{pol_name}_on_{surf}"
        print(f"\\n=== filming {tag} ===", flush=True)
        try:
            policy = torch.jit.load(pol_path).to("cuda:0").eval()
            cfg = parse_env_cfg(task, device="cuda:0", num_envs=6)
            env = gym.make(task, cfg=cfg, render_mode="rgb_array")
            obs_d, _ = env.reset()
            obs = obs_d["policy"] if isinstance(obs_d, dict) else obs_d

            frames, alive = [], 0
            with torch.inference_mode():
                for i in range(STEPS):
                    act = policy(obs)
                    obs_d, _, term, trunc, _ = env.step(act)
                    obs = obs_d["policy"] if isinstance(obs_d, dict) else obs_d
                    alive += float((~term.bool()).float().mean().item())
                    if i % 2 == 0:                      # 25 fps from a 50 Hz sim
                        f = env.unwrapped.render()
                        if f is not None:
                            frames.append(np.asarray(f))
            env.close()

            if frames:
                path = OUT / f"isaac_{tag}.mp4"
                imageio.mimwrite(path, frames, fps=25, quality=8,
                                 macro_block_size=1)
                report[tag] = {"frames": len(frames), "steps": STEPS,
                               "mean_alive_frac": round(alive / STEPS, 3),
                               "file": str(path)}
                print(f"WROTE {path}  {len(frames)} frames  "
                      f"alive {alive/STEPS:.1%}", flush=True)
            else:
                report[tag] = {"error": "render() returned no frames"}
                print(f"!! {tag}: render() returned nothing", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            report[tag] = {"error": repr(e)}

    (OUT / "isaac_film_report.json").write_text(json.dumps(report, indent=2))
    print("\\n" + json.dumps(report, indent=2), flush=True)
    app.close()
''')
pathlib.Path("film.py").write_text(FILM)
sh(f"{PY} film.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -60",
   3000, "film", tail=60)
