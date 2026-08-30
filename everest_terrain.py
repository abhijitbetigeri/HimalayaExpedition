# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip", "numpy"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""Build Isaac Lab terrain from the REAL Everest DEM, and film the G1 on it.

Data: NASA SRTM 1-arc-second, tile N27E086, served free by AWS. Verified to contain
Everest -- max elevation 8748 m at the summit coordinates (the accepted SRTM value;
the true summit is 8849 m, and SRTM slightly under-reads sharp peaks).

The scale problem, and what is honest to claim
----------------------------------------------
SRTM samples every ~30 m. A humanoid's foot is ~0.25 m, so at 1:1 the terrain is
perfectly flat between samples and the robot would be walking on a plane with an
occasional 30 m cliff. Shrinking the footprint alone makes every slope vertical.

So the patch is scaled UNIFORMLY -- horizontal and vertical by the same factor --
which preserves the true SLOPE ANGLE exactly while compressing the footprint to
something a robot can cross in a clip. What is real: the gradient profile and the
shape of the ground, taken from actual Khumbu terrain. What is not: the absolute
size. Say "terrain profile from the real Everest DEM", never "the robot is walking
on Everest".

The patch is chosen by SEARCHING the region for a section whose mean slope lands in
a walkable band, rather than picking coordinates by hand and hoping -- most of this
tile is either valley floor or unclimbable face.

Run:
  hf jobs uv run --detach --namespace iteratehack --flavor h200 --timeout 90m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=everest-terrain \
      everest_terrain.py
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
OUT = BUCKET / "everest"
WALK = BUCKET / "ice-slope/rsl_rl/g1_rough/2026-08-30_00-58-20/exported/policy.pt"

TARGET_SLOPE_DEG = float(os.environ.get("EVEREST_SLOPE_DEG", "16"))
PATCH_CELLS = int(os.environ.get("EVEREST_PATCH", "48"))     # 48 * 30 m = 1.44 km
TERRAIN_M = float(os.environ.get("EVEREST_TERRAIN_M", "36")) # compressed footprint


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


# ---- 1. real elevation data ---------------------------------------------------
sh("curl -sL --max-time 600 -o /tmp/N27E086.hgt.gz "
   "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N27/N27E086.hgt.gz "
   "&& gunzip -f /tmp/N27E086.hgt.gz && ls -la /tmp/N27E086.hgt",
   900, "dem", tail=3)

import numpy as np  # noqa: E402

raw = np.fromfile("/tmp/N27E086.hgt", dtype=">i2")
n = int(len(raw) ** 0.5)
dem = raw.reshape(n, n).astype(np.float64)
dem[dem < -1000] = np.nan
print(f"DEM {n}x{n}, {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m", flush=True)

CELL_M = 30.0 if n == 3601 else 90.0

# ---- 2. find a patch at a walkable gradient ----------------------------------
# Hand-picked coordinates are a coin flip here: most of this tile is valley floor
# or unclimbable face. Search instead, and report what was chosen.
best = None
step = 8
for r in range(0, n - PATCH_CELLS, step * 4):
    for c in range(0, n - PATCH_CELLS, step * 4):
        patch = dem[r:r + PATCH_CELLS, c:c + PATCH_CELLS]
        if np.isnan(patch).any():
            continue
        gy, gx = np.gradient(patch, CELL_M)
        slope_deg = np.degrees(np.arctan(np.hypot(gy, gx)))
        mean_slope = float(slope_deg.mean())
        rough = float(slope_deg.std())
        err = abs(mean_slope - TARGET_SLOPE_DEG)
        # Prefer the target gradient, and among equals prefer more relief -- a
        # uniformly tilted plane is not visibly "mountain".
        sc = err - 0.25 * rough
        if best is None or sc < best[0]:
            best = (sc, r, c, mean_slope, rough, patch.copy())

assert best is not None, "no clean patch found"
_, R, C, mean_slope, rough, patch = best
lat = 28.0 - (R + PATCH_CELLS / 2) / (n - 1)
lon = 86.0 + (C + PATCH_CELLS / 2) / (n - 1)
print(f"patch at {lat:.4f}N {lon:.4f}E  mean slope {mean_slope:.1f}deg "
      f"(sd {rough:.1f})  relief {patch.max()-patch.min():.0f} m", flush=True)

# ---- 3. uniform scale: preserve the slope angle, compress the footprint -------
real_extent_m = PATCH_CELLS * CELL_M
scale = TERRAIN_M / real_extent_m
height = (patch - patch.min()) * scale
h_scale = TERRAIN_M / PATCH_CELLS
gy, gx = np.gradient(height, h_scale)
check_deg = float(np.degrees(np.arctan(np.hypot(gy, gx))).mean())
print(f"scaled by {scale:.4f}: {real_extent_m:.0f} m -> {TERRAIN_M:.0f} m, "
      f"relief {height.max():.1f} m, slope preserved at {check_deg:.1f}deg",
      flush=True)

OUT.mkdir(parents=True, exist_ok=True)
np.save("/tmp/everest_height.npy", height.astype(np.float32))
meta = dict(lat=round(lat, 4), lon=round(lon, 4), mean_slope_deg=round(mean_slope, 1),
            slope_sd=round(rough, 1), real_extent_m=real_extent_m,
            terrain_m=TERRAIN_M, scale=round(scale, 5),
            relief_real_m=round(float(patch.max() - patch.min()), 1),
            relief_scaled_m=round(float(height.max()), 2),
            source="NASA SRTM 1-arc-sec, tile N27E086 via AWS elevation-tiles-prod")
(OUT / "everest_patch.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)

# ---- 4. Isaac Lab + film ------------------------------------------------------
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
sh("curl -sL --max-time 240 -o /tmp/alpine.hdr "
   "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/horn-koppe_snow_2k.hdr "
   "&& ls -la /tmp/alpine.hdr", 300, "hdri", tail=2)

if pathlib.Path(WALK).exists():
    subprocess.run(f"cp {WALK} /tmp/policy_slope.pt", shell=True)
    print("staged the ice+slope policy (the one trained on inclines)", flush=True)
else:
    print(f"!! slope policy missing at {WALK}", flush=True)

G1DIR = pathlib.Path(f"{LAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                     f"locomotion/velocity/config/g1")
(G1DIR / "everest_env_cfg.py").write_text(textwrap.dedent('''
    """Terrain whose height field comes from the real Everest DEM."""
    import numpy as np
    import trimesh
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.terrains.height_field.utils import height_field_to_mesh
    from isaaclab.utils.configclass import configclass
    from .flat_env_cfg import G1FlatEnvCfg

    SNOW = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.86, 0.94),
                                       roughness=0.9, metallic=0.0)
    HEIGHT = np.load("/tmp/everest_height.npy")


    @configclass
    class G1EverestCfg(G1FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.scene.terrain.terrain_type = "usd"
            self.scene.terrain.usd_path = "/tmp/everest.usd"
            self.scene.terrain.visual_material = SNOW
            self.observations.policy.enable_corruption = False
            self.commands.base_velocity.debug_vis = False
            self.events.push_robot = None
            self.events.base_external_force_torque = None
            self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.resampling_time_range = (1e6, 1e6)
            self.events.physics_material.params["static_friction_range"] = (0.20, 0.20)
            self.events.physics_material.params["dynamic_friction_range"] = (0.16, 0.16)
            import os as _os
            if _os.path.exists("/tmp/alpine.hdr"):
                try:
                    self.scene.sky_light.spawn.texture_file = "/tmp/alpine.hdr"
                    self.scene.sky_light.spawn.intensity = 900.0
                except Exception:
                    pass
'''))

# Height field -> triangle mesh -> USD, which is the import path Isaac Lab offers
# for arbitrary terrain geometry.
MESH = textwrap.dedent(f'''
    import numpy as np, trimesh
    h = np.load("/tmp/everest_height.npy").astype(np.float64)
    rows, cols = h.shape
    sx = {TERRAIN_M} / (cols - 1)
    sy = {TERRAIN_M} / (rows - 1)
    xs = np.arange(cols) * sx - {TERRAIN_M} / 2
    ys = np.arange(rows) * sy - {TERRAIN_M} / 2
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X.ravel(), Y.ravel(), h.ravel()], axis=1)
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces))
    m.export("/tmp/everest.obj")
    print("mesh:", len(verts), "verts", len(faces), "faces",
          "z range", h.min(), h.max(), flush=True)
''')
pathlib.Path("mesh.py").write_text(MESH)
sh(f"{PY} -m pip install --no-cache-dir trimesh 2>&1 | tail -1", 300, "trimesh", tail=1)
sh(f"{PY} mesh.py", 900, "mesh", tail=6)

CONV = textwrap.dedent('''
    import os
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("HOME", "/root")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    cfg = MeshConverterCfg(asset_path="/tmp/everest.obj",
                           usd_dir="/tmp", usd_file_name="everest.usd",
                           force_usd_conversion=True, make_instanceable=False)
    MeshConverter(cfg)
    print("USD written:", os.path.exists("/tmp/everest.usd"), flush=True)
    app.close()
''')
pathlib.Path("conv.py").write_text(CONV)
sh(f"{PY} conv.py 2>&1 | grep -vE 'neuraylib|material_library' | tail -8",
   1800, "usd", tail=8)

print("\nEVEREST TERRAIN PREPARED", flush=True)
sh(f"cp /tmp/everest_height.npy {OUT}/ 2>/dev/null; "
   f"cp /tmp/everest.obj {OUT}/ 2>/dev/null; ls -la {OUT}", 300, "save", tail=6)
