# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Self-correcting filming loop: shoot, SCORE, adjust, re-shoot until it is good.

Every previous filming attempt shipped whatever came out. Four batches were bad in
four different ways -- subject out of frame, six robots reading as fragments, a
debug marker larger than the robot, ground that looked nothing like snow -- and
each time a human had to notice. This scores its own output and fixes itself.

The score is geometric, not aesthetic, so it is decidable:

    coverage  fraction of pixels that are not background. Too low = the robot is a
              speck or absent; too high = the camera is inside it.
    centring  centroid of those pixels. Off-centre means the tracking camera is
              not actually framing the subject.

On a bad score it moves the camera (closer/further, up/down) and re-shoots, keeping
the best attempt. Contrast matters too: a WHITE robot on WHITE snow is nearly
invisible, so the snow is tinted blue-grey -- still unmistakably snow, but the robot
reads against it.

Two deliverables:
  1. walk on snow          -- both policies, same surface
  2. walk / slip / recover -- waits for the recovery policy, then composes one
                              continuous take with a policy handoff

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor h200 --timeout 3h \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=autofilm \
      autofilm.py
"""

import glob
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
BUCKET = pathlib.Path("/mnt/himalaya-g1")
WALK = BUCKET / "ice-isaac/rsl_rl/g1_flat/2026-08-29_21-48-51/exported/policy.pt"
BASE = BUCKET / "baseline/rsl_rl/g1_flat/2026-08-29_20-54-12/exported/policy.pt"


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
(G1DIR / "auto_env_cfg.py").write_text(textwrap.dedent('''
    """Snow that looks like snow, against which a white robot is visible."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils.configclass import configclass
    from .flat_env_cfg import G1FlatEnvCfg

    # Blue-grey rather than white: the G1 is white, so near-white snow left almost
    # no contrast and the robot vanished into the ground.
    SNOW = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.86, 0.94),
                                       roughness=0.9, metallic=0.0)


    class _Auto(G1FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1              # one subject, always
            self.scene.env_spacing = 8.0
            self.scene.terrain.visual_material = SNOW
            self.observations.policy.enable_corruption = False
            self.commands.base_velocity.debug_vis = False   # the green blob
            self.events.push_robot = None
            self.events.base_external_force_torque = None
            self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.resampling_time_range = (1e6, 1e6)
            self.events.physics_material.params["static_friction_range"] = (0.20, 0.20)
            self.events.physics_material.params["dynamic_friction_range"] = (0.16, 0.16)


    @configclass
    class G1AutoSnowCfg(_Auto):
        pass


    @configclass
    class G1AutoSnowRecoverCfg(_Auto):
        """Same snow, but the robot may start down and contact must not end it."""
        def __post_init__(self):
            super().__post_init__()
            self.terminations.base_contact = None
            self.episode_length_s = 60.0
'''))
with open(G1DIR / "__init__.py", "a") as f:
    f.write(textwrap.dedent('''

        import gymnasium as gym  # noqa: E402
        from . import agents  # noqa: E402
        for _id, _cls in [("Isaac-Auto-G1-Snow-v0", "G1AutoSnowCfg"),
                          ("Isaac-Auto-G1-SnowRecover-v0", "G1AutoSnowRecoverCfg")]:
            gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                         disable_env_checker=True,
                         kwargs={"env_cfg_entry_point": f"{__name__}.auto_env_cfg:{_cls}",
                                 "rsl_rl_cfg_entry_point":
                                     f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"})
    '''))

for n, p_ in (("walk", WALK), ("base", BASE)):
    if pathlib.Path(p_).exists():
        subprocess.run(f"cp {p_} /tmp/policy_{n}.pt", shell=True)
        print(f"staged {n}", flush=True)

MAIN = textwrap.dedent('''
    import json, os, pathlib
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

    import gymnasium as gym, torch, imageio, numpy as np
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg

    OUT = pathlib.Path("/mnt/himalaya-g1/videos"); OUT.mkdir(parents=True, exist_ok=True)
    DEV = "cuda:0"


    def score(frames):
        """Geometric quality: is the subject visibly IN the shot, and centred?

        Background is a smooth gradient (sky + flat ground), so the robot is the
        set of pixels departing from the local median. Coverage says how big it
        reads; the centroid says whether the camera is actually pointing at it.
        """
        f = np.asarray(frames[len(frames) // 2], dtype=float)
        g = f.mean(axis=2)
        mask = np.abs(g - np.median(g)) > 22
        cov = float(mask.mean())
        ys, xs = np.nonzero(mask)
        if xs.size < 50:
            return dict(cov=cov, cx=0.5, cy=0.5, ok=False, why="subject not found")
        cx, cy = float(xs.mean() / g.shape[1]), float(ys.mean() / g.shape[0])
        ok = (0.015 <= cov <= 0.30) and (0.25 <= cx <= 0.75) and (0.25 <= cy <= 0.85)
        why = ("too small/absent" if cov < 0.015 else
               "fills the frame" if cov > 0.30 else
               "off-centre" if not (0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.85) else "ok")
        return dict(cov=round(cov, 4), cx=round(cx, 3), cy=round(cy, 3), ok=ok, why=why)


    def shoot(task, policy_path, steps, eye, lookat, recover_path=None):
        policy = torch.jit.load(policy_path).to(DEV).eval()
        rec = torch.jit.load(recover_path).to(DEV).eval() if recover_path else None
        cfg = parse_env_cfg(task, device=DEV, num_envs=1)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = eye
        cfg.viewer.lookat = lookat
        cfg.viewer.resolution = (1280, 720)
        env = gym.make(task, cfg=cfg, render_mode="rgb_array")
        obs_d, _ = env.reset(); obs = obs_d["policy"]
        robot = env.unwrapped.scene["robot"]
        frames, events, mode = [], [], "walk"
        with torch.inference_mode():
            for i in range(steps):
                # Narrative: shove it over at 1/3, hand to recovery while down,
                # hand back once upright. The handoff IS the system working.
                if rec is not None:
                    h = float(robot.data.root_pos_w[0, 2])
                    up = float(-robot.data.projected_gravity_b[0, 2])
                    if i == steps // 3 and mode == "walk":
                        robot.write_root_velocity_to_sim(
                            torch.tensor([[0., 3.2, 0.9, 0., 0., 0.]], device=DEV))
                        events.append((i, "SHOVE")); mode = "falling"
                    elif mode == "falling" and (h < 0.45 or up < 0.5):
                        events.append((i, "RECOVER")); mode = "recover"
                    elif mode == "recover" and h > 0.65 and up > 0.85:
                        events.append((i, "WALK ON")); mode = "walk"
                act = rec(obs) if (rec is not None and mode == "recover") else policy(obs)
                obs_d, _, term, trunc, _ = env.step(act)
                obs = obs_d["policy"]
                if i % 2 == 0:
                    fr = env.unwrapped.render()
                    if fr is not None:
                        frames.append(np.asarray(fr))
        env.close()
        return frames, events


    def autoshoot(name, task, policy_path, steps, recover_path=None, tries=4):
        """Shoot, score, adjust the camera, repeat. Keep the best."""
        eye, lookat = (2.0, 2.0, 1.05), (0.0, 0.0, 0.55)
        best = None
        for attempt in range(1, tries + 1):
            print(f"[{name}] attempt {attempt}: eye={eye}", flush=True)
            frames, events = shoot(task, policy_path, steps, eye, lookat, recover_path)
            if not frames:
                print(f"[{name}]   no frames", flush=True); continue
            s = score(frames)
            print(f"[{name}]   cov={s['cov']} centre=({s['cx']},{s['cy']}) "
                  f"-> {s['why']}", flush=True)
            if best is None or s["cov"] > best[1]["cov"]:
                best = (frames, s, events, eye)
            if s["ok"]:
                print(f"[{name}]   ACCEPTED", flush=True)
                break
            # Adjust and retry rather than shipping it.
            d = (eye[0] ** 2 + eye[1] ** 2) ** 0.5
            if s["cov"] < 0.015:
                d *= 0.6                      # far too small: come in
            elif s["cov"] > 0.30:
                d *= 1.6                      # too close: pull back
            if s["cy"] > 0.85:
                lookat = (lookat[0], lookat[1], max(0.2, lookat[2] - 0.2))
            elif s["cy"] < 0.25:
                lookat = (lookat[0], lookat[1], lookat[2] + 0.2)
            k = d / (2 ** 0.5)
            eye = (k, k, max(0.6, eye[2] * (d / max(1e-6, (eye[0]**2+eye[1]**2)**0.5))))
        if best is None:
            return {"error": "no frames at all"}
        frames, s, events, eye = best
        p = OUT / f"auto_{name}.mp4"
        imageio.mimwrite(p, frames, fps=25, quality=9, macro_block_size=1)
        print(f"[{name}] WROTE {p}  {len(frames)} frames  score={s}", flush=True)
        return {"file": str(p), "score": s, "eye": list(eye),
                "events": [[int(i), e] for i, e in events]}


    report = {}
    report["walk_on_snow_ICE"] = autoshoot(
        "walk_on_snow_ICE", "Isaac-Auto-G1-Snow-v0", "/tmp/policy_walk.pt", 1200)
    report["walk_on_snow_BASELINE"] = autoshoot(
        "walk_on_snow_BASELINE", "Isaac-Auto-G1-Snow-v0", "/tmp/policy_base.pt", 1200)

    # Walk clips are written above and are now safe. Only NOW wait on recovery.
    import glob as _glob, time as _time, shutil as _shutil
    _dl = _time.time() + 70 * 60
    while not pathlib.Path("/tmp/policy_recover.pt").exists() and _time.time() < _dl:
        _h = sorted(_glob.glob("/mnt/himalaya-g1/getup-v3/rsl_rl/*/*/exported/policy.pt"))
        if _h:
            _shutil.copy(_h[-1], "/tmp/policy_recover.pt")
            print("recovery policy found:", _h[-1], flush=True)
            break
        print("waiting for the recovery policy...", flush=True)
        _time.sleep(180)

    if pathlib.Path("/tmp/policy_recover.pt").exists():
        report["snow_slip_recover_walk"] = autoshoot(
            "snow_slip_recover_walk", "Isaac-Auto-G1-SnowRecover-v0",
            "/tmp/policy_walk.pt", 2400, recover_path="/tmp/policy_recover.pt")
    else:
        report["snow_slip_recover_walk"] = {"error": "no recovery policy available"}

    (OUT / "autofilm_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    app.close()
''')
pathlib.Path("m.py").write_text(MAIN)
sh(f"{PY} m.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -45",
   7200, "autofilm", tail=45)
