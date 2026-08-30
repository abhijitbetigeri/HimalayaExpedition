# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Runtime introspection of Isaac Lab's mdp namespace.

Static grep cannot find these terms: isaaclab/envs/mdp/__init__.py is just
`lazy_export()`, so `mdp.randomize_rigid_body_material` is resolved dynamically on
attribute access and has no greppable `def` in the tree. So ask the live object.

What this pins down before the ice EventTerm gets written:
  - is randomize_rigid_body_material a function or a ManagerTermBase class
  - its real signature and source (env_ids semantics, num_buckets, tensor shapes)
  - the G1 robot's actual body names, so per-foot events target the right ones
  - the ground/terrain material config, since the FLOOR's friction is what matters
    for ice, not just the robot's feet

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor l4x1 --timeout 30m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=introspect-mdp \
      introspect_mdp.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")

PY = sys.executable
LAB = "/tmp/IsaacLab"


def sh(cmd, timeout, label, tail=20):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
    return p.returncode == 0


sh("apt-get update -qq && apt-get install -y -qq --no-install-recommends "
   "libgl1 libglu1-mesa libegl1 libvulkan1 libxrandr2 libxinerama1 libxcursor1 "
   "libxi6 libsm6 libice6 libxt6 libgomp1 git && echo apt-ok", 900, "apt", tail=2)
sh(f"git clone --depth 1 -q https://github.com/isaac-sim/IsaacLab.git {LAB} && echo cloned",
   900, "clone", tail=2)
for pkg in ["isaaclab", "isaaclab_ov", "isaaclab_physx", "isaaclab_ovphysx",
            "isaaclab_newton", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"]:
    sh(f"{PY} -m pip install --no-cache-dir -e {LAB}/source/{pkg} 2>&1 | tail -1",
       1200, f"install {pkg}", tail=1)

SCRIPT = textwrap.dedent('''
    import os, inspect, sys, pathlib
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")

    # Tee everything to the bucket. Relying on a stdout tail loses the early output
    # behind Kit's very chatty startup -- learned the hard way twice.
    REPORT = pathlib.Path("/mnt/himalaya-g1/api/mdp_introspection.txt")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _log = open(REPORT, "w")
    class Tee:
        def write(self, s):
            sys.__stdout__.write(s); _log.write(s); _log.flush()
        def flush(self):
            sys.__stdout__.flush(); _log.flush()
    sys.stdout = Tee()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    print("APP_OK", flush=True)

    import isaaclab.envs.mdp as mdp

    # How does one actually WRITE material properties? That is the API the custom
    # ice event needs, and it is the thing worth getting right first.
    print("\\n=== physx view material accessors ===", flush=True)
    try:
        from isaaclab.assets import Articulation
        meths = [m for m in dir(Articulation) if "material" in m.lower()]
        print("Articulation material methods:", meths, flush=True)
    except Exception as e:
        print("articulation introspect failed:", e, flush=True)

    term = mdp.randomize_rigid_body_material
    print("\\n=== randomize_rigid_body_material ===", flush=True)
    print("type   :", type(term), flush=True)
    print("module :", getattr(term, "__module__", "?"), flush=True)
    print("isclass:", inspect.isclass(term), flush=True)
    try:
        print("file   :", inspect.getfile(term), flush=True)
    except Exception as e:
        print("file   : ?", e, flush=True)
    print("\\n--- SOURCE ---", flush=True)
    try:
        print(inspect.getsource(term), flush=True)
    except Exception as e:
        print("no source:", e, flush=True)

    print("\\n=== other material/friction terms in mdp ===", flush=True)
    for n in sorted(dir(mdp)):
        if "material" in n.lower() or "friction" in n.lower():
            print("  ", n, flush=True)

    print("\\n=== G1 env: bodies, feet, terrain material ===", flush=True)
    import gymnasium as gym
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device="cuda:0", num_envs=4)
    print("physics_material params:", cfg.events.physics_material.params, flush=True)
    print("physics_material func  :", cfg.events.physics_material.func, flush=True)
    print("terrain physics_material:", getattr(cfg.scene.terrain, "physics_material", None),
          flush=True)

    env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg)
    robot = env.unwrapped.scene["robot"]

    print("\\n=== live material tensor ===", flush=True)
    view = robot.root_physx_view
    print("view type:", type(view), flush=True)
    mats = view.get_material_properties()
    print("material tensor shape:", mats.shape, flush=True)
    print("  (expect [num_envs, num_shapes, 3] = static, dynamic, restitution)",
          flush=True)
    print("sample row 0:", mats[0][:4], flush=True)
    print("has set_material_properties:", hasattr(view, "set_material_properties"),
          flush=True)

    # Which shape indices belong to the feet -- needed to write per-foot friction.
    feet = [i for i, n in enumerate(robot.body_names) if "ankle_roll" in n]
    print("foot body indices:", feet,
          [robot.body_names[i] for i in feet], flush=True)
    env.close()
    app.close()
    print("DONE", flush=True)
''')
pathlib.Path("introspect.py").write_text(SCRIPT)
# No tail: the report file in the bucket is the artifact, stdout is just a mirror.
sh(f"{PY} introspect.py 2>&1 | grep -vE 'neuraylib|material_library|\\[Warning\\]' | tail -60",
   1500, "introspect", tail=60)
