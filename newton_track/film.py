"""Render one Newton-track rollout to mp4 (headless OpenGL, system ffmpeg).

    ~/projects/newton-sandbox/.venv/bin/python newton_track/film.py --mu 0.75 --out demo/newton_rock.mp4
    ~/projects/newton-sandbox/.venv/bin/python newton_track/film.py --mu 0.1  --out demo/newton_ice.mp4
    ~/projects/newton-sandbox/.venv/bin/python newton_track/film.py --snow 0.15 --snow-type powder \\
        --steps 100 --out demo/newton_snow.mp4

Needs a display: `ViewerGL(headless=True)` still opens a pyglet context, which
fails with "list index out of range" in a session with no screen (an agent
shell on a locked Mac, a bare ssh). On a Retina display the frame comes back at
twice the requested size, so the encoder is opened from the first frame's
actual shape, not from `--size`.

A clip is an illustration of one seed, never a measurement (STATE.md section 5).
The numbers are in results/*.json. Frames are piped straight to ffmpeg, so
nothing is lost if the rollout dies late (CLAUDE.md: "results before
artefacts"), and the rollout summary is printed BEFORE the encoder is closed.
`*.mp4` is gitignored in this repo; clips belong in the HF dataset.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import warp as wp

import newton.viewer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g1_newton import POLICY_HZ, Conditions, G1Himalaya  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mu", type=float, default=0.75)
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--slope", type=float, default=0.0)
    ap.add_argument("--snow", type=float, default=0.0, help="snow depth in m; > 0 uses the coupled MPM scene")
    ap.add_argument("--snow-type", default="powder", choices=("powder", "packed", "hard"))
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--rigid", default="mujoco", choices=("mujoco", "steel"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--after-fall", type=int, default=40, help="frames to keep filming after a fall")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH")
        return 2
    w, h = (int(v) for v in args.size.split("x"))
    cond = Conditions(mu_ground=args.mu, wind_n=args.wind, slope_deg=args.slope)
    if args.snow > 0:
        from snow_mpm import SNOW, G1Snow  # noqa: PLC0415

        yp, coh = SNOW[args.snow_type]
        env = G1Snow(
            cond, depth=args.snow, yield_pressure=yp, cohesion=coh, voxel=args.voxel,
            bed_length=args.steps / POLICY_HZ * 0.5 + 0.8, bed_half_width=0.35,
        )  # fmt: skip
    else:
        env = G1Himalaya(cond, rigid=args.rigid)

    viewer = newton.viewer.ViewerGL(width=w, height=h, headless=True)
    viewer.set_model(env.model)
    # High-altitude daylight instead of the viewer's dusk gradient, and less
    # exposure so the white shells keep their shading.
    r = viewer.renderer
    r.sky_upper = (0.30, 0.52, 0.86)
    r.sky_lower = (0.80, 0.88, 0.96)
    r.exposure = 0.85 * float(getattr(r, "exposure", 1.0))
    # Draw the particles ourselves: the viewer's default particle colour is sand.
    viewer.show_particles = False
    n_particles = env.model.particle_count
    snow_colors = wp.full(n_particles, value=wp.vec3(0.93, 0.95, 0.98), dtype=wp.vec3) if n_particles else None
    env.reset(args.seed)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = None

    fell_at = None
    frames = 0
    for t in range(args.steps):
        env.step()
        pos, _ = env.base()
        # three-quarter view that follows the pelvis; the robot walks along +y.
        viewer.set_camera(wp.vec3(float(pos[0]) + 2.0, float(pos[1]) - 1.35, 1.05), pitch=-9.0, yaw=146.0)
        viewer.begin_frame(t / POLICY_HZ)
        viewer.log_state(env.state_0)
        if n_particles:
            viewer.log_points("/snow", points=env.state_0.particle_q, radii=env.model.particle_radius, colors=snow_colors)
        viewer.end_frame()
        frame = viewer.get_frame().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        if enc is None:
            fh, fw = frame.shape[:2]
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
                 "-r", str(POLICY_HZ), "-i", "-", "-vf", f"scale={w}:{h}", "-pix_fmt", "yuv420p", "-crf", "20",
                 str(out)],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        enc.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes())
        frames += 1
        if fell_at is None and env.fallen():
            fell_at = t + 1
        if fell_at is not None and t + 1 - fell_at >= args.after_fall:
            break

    print(f"{cond.label()} seed {args.seed}: " + (f"fell at step {fell_at}" if fell_at else f"upright for {frames} steps"))
    enc.stdin.close()
    rc = enc.wait()
    viewer.close()
    print(f"wrote {out} ({frames} frames)" if rc == 0 else f"ffmpeg failed with {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
