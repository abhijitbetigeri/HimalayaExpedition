# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Is the robot missing limbs because of RENDERING or because of the POLICY?

The clips show a torso, a head and two foot stubs -- no arms, no legs. Earlier
clips rendered a full humanoid, so something regressed. Two very different causes:

  rendering   the limb meshes are not being drawn (asset/material/USD problem)
  policy      the robot is folded into itself (control problem)

This settles it by rendering the DEFAULT POSE with ZERO actions -- no policy in the
loop at all. If the default pose shows a full robot, rendering is fine and the
policy is folding it. If the default pose is also limbless, the asset is at fault
and no amount of camera or policy work will help.

Also dumps body count and per-body world positions, which show directly whether the
limbs exist in the scene and where they are.
"""
import os, pathlib, subprocess, sys, textwrap
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")
PY, LAB = sys.executable, "/tmp/IsaacLab"

def sh(c, t, label, tail=8):
    print(f"\n$ {c}", flush=True)
    try:
        p = subprocess.run(c, shell=True, timeout=t, executable="/bin/bash",
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT", flush=True); return False

sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git ffmpeg && echo ok", 900, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)
for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks",
            "isaaclab_visualizers", "isaaclab_contrib"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -1",
       1200, f"i {pkg}", tail=1)
sh(f"{PY} -m pip install --no-cache-dir 'rsl-rl-lib==5.0.1' onnxscript imageio "
   "imageio-ffmpeg 2>&1 | tail -1", 600, "deps", tail=1)

D = textwrap.dedent('''
    import os, pathlib
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"; os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    import gymnasium as gym, torch, imageio, numpy as np
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg
    OUT = pathlib.Path("/mnt/himalaya-g1/diag"); OUT.mkdir(parents=True, exist_ok=True)

    # The STOCK task -- no snow, no HDRI, no custom terrain. If the robot is whole
    # here, the asset is fine and the regression is in one of those changes.
    cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device="cuda:0", num_envs=1)
    cfg.viewer.origin_type = "asset_root"; cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (2.2, 2.2, 1.2); cfg.viewer.lookat = (0.0, 0.0, 0.6)
    cfg.viewer.resolution = (1280, 720)
    env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg, render_mode="rgb_array")
    env.reset()
    robot = env.unwrapped.scene["robot"]
    print("num bodies:", len(robot.body_names), flush=True)
    pos = robot.data.body_pos_w[0]
    for i, n in enumerate(robot.body_names):
        if any(k in n for k in ("pelvis", "knee", "ankle_roll", "elbow", "shoulder_pitch", "torso")):
            print(f"  {n:28s} z={float(pos[i,2]):.3f}", flush=True)
    # Settle a few steps with ZERO actions, then render. No policy at all.
    act = torch.zeros(1, env.unwrapped.action_space.shape[-1], device="cuda:0")
    for _ in range(10):
        env.step(act)
    fr = env.unwrapped.render()
    if fr is not None:
        imageio.imwrite(OUT / "default_pose.png", np.asarray(fr))
        print("WROTE default_pose.png", flush=True)
    else:
        print("render() returned None", flush=True)
    env.close(); app.close()
''')
pathlib.Path("/tmp/d.py").write_text(D)
r = subprocess.run([PY, "/tmp/d.py"], capture_output=True, text=True, timeout=2400)
for l in r.stdout.splitlines():
    if not any(k in l for k in ("neuraylib", "material_library", "[Warning]")):
        print(l, flush=True)
