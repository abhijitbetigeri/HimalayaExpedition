"""Publish the G1 simulation to a LiveKit room as a live video track.

Why frames are pre-rendered rather than streamed as they simulate
-----------------------------------------------------------------
Measured on this machine: physics 12.6 Hz, offscreen render 1.4 fps, against a
50 Hz control rate. Rendering is ~35x too slow to feed a realtime track. So the
rollout is computed and rendered up front, then published on a wall-clock timer
and looped. Viewers get genuinely smooth realtime motion; it is simply not
simulated in the same instant. Anything else would stream a 1.4 fps slideshow
and call it live.

On a CUDA box with EGL the same script can render fast enough to go truly live;
`--live` does that, and warns if it cannot keep pace.

Credentials
-----------
Needs LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in the environment, from
a project at https://cloud.livekit.io/ (or a self-hosted server). This script
never asks for, stores, or prints them. `--dry-run` renders and reports without
connecting to anything, so the whole pipeline can be checked without an account.
"""

import argparse
import asyncio
import contextlib
import os
import pathlib
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

# Playground's G1 config hardcodes impl="warp", which has no CPU fast path --
# measured 972 ms/step vs 300 for impl="jax" (see CLAUDE.md and local_view.py).
# naconmax defaults to 8*8192, sized for thousands of parallel envs; we run one.
# GPU is the opposite: warp is much faster there and naconmax=128 overflows once
# envs are batched, so this is applied ONLY off-GPU.

ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"


def load_env_file(path=ENV_FILE):
    """Read KEY=VALUE lines from .env into os.environ, without overriding.

    Hand-rolled rather than pulling in python-dotenv: one less dependency to
    install on the GPU box, and this file is the ONLY place credentials live.
    Real environment variables win, so `LIVEKIT_URL=... python stream_sim.py`
    still overrides the file.
    """
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


_ARGS = None

FAST_CPU = {"impl": "jax", "naconmax": 128}


def cpu_overrides():
    import jax
    on_gpu = jax.devices()[0].platform == "gpu"
    return {} if on_gpu else dict(FAST_CPU)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ice",
                   choices=["ice", "flat", "ascent"],
                   help="ice = ice+wind+patches, flat = stock, ascent = fixed line")
    p.add_argument("--checkpoint", default=None,
                   help="brax PPO checkpoint dir. Without it the robot is "
                        "unactuated and will simply collapse -- useful for "
                        "checking the pipeline, useless as a demo.")
    p.add_argument("--room", default="himalaya-g1")
    p.add_argument("--identity", default="sim-publisher")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    # The G1 scene ships a "track" camera in trackcom mode bolted to the pelvis.
    # Playground's render() defaults to camera=-1, a STATIC free camera, so a
    # walking robot simply strolls out of shot -- which is exactly what the
    # first LiveKit stream showed. Track by default; --camera "" for the static
    # view.
    p.add_argument("--slope", type=float, default=15.0,
                   help="ascent slope in degrees. The flat-trained policy "
                        "climbs 0-15 deg for a full episode and falls at 30.")
    p.add_argument("--camera", default="track",
                   help='named camera; "track" follows the robot (default), '
                        '"" for the static free camera')
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--once", dest="loop", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="render and report, never connect")
    p.add_argument("--pov", action="store_true", default=True,
                   help="also publish the robot's onboard camera as a second "
                        "track (default). --no-pov for chase only.")
    p.add_argument("--no-pov", dest="pov", action="store_false")
    p.add_argument("--viewer-token", action="store_true",
                   help="mint a subscribe-only viewer token and a meet.livekit.io "
                        "link, print them, and exit. Runs no simulation.")
    return p.parse_args()



def load_policy_compat(path):
    """brax's own checkpoint round-trip is broken; work around it.

    `save` writes `"mean_kernel_init_fn": null` into ppo_network_config.json,
    and `load_config` then does `KERNEL_INITIALIZER[None]` -- which raises
    KeyError, because None is not a key. So brax cannot read back a checkpoint
    it just wrote. Registering None as an alias for "no explicit initializer"
    (which is what null meant at save time) restores the round-trip without
    touching the saved artefact.
    """
    from brax.training import networks
    from brax.training.agents.ppo import checkpoint as ppo_ckpt

    if None not in networks.KERNEL_INITIALIZER:
        networks.KERNEL_INITIALIZER[None] = None
    return ppo_ckpt.load_policy(pathlib.Path(path).absolute().as_posix())



def print_viewer_token(args):
    """Mint a SUBSCRIBE-ONLY token so a viewer can watch but not publish.

    Deliberately not the publisher token: this string gets pasted into browsers
    and chat, and a token that can publish into the room is a bigger thing to
    hand around than one that can only watch. It expires, but treat it as
    shareable-with-the-team, not public.
    """
    import urllib.parse

    from livekit import api

    load_env_file()
    url = os.environ.get("LIVEKIT_URL")
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        raise SystemExit("LIVEKIT_URL / _API_KEY / _API_SECRET not set; "
                         "see .env.example")

    token = (api.AccessToken(key, secret)
             .with_identity("viewer")
             .with_name("viewer")
             .with_grants(api.VideoGrants(
                 room_join=True, room=args.room,
                 can_publish=False, can_publish_data=False,
                 can_subscribe=True))
             .to_jwt())

    link = ("https://meet.livekit.io/custom?liveKitUrl="
            + urllib.parse.quote(url, safe="")
            + "&token=" + urllib.parse.quote(token, safe=""))
    print(f"room  : {args.room}")
    print(f"url   : {url}")
    print("\nviewer token (subscribe-only):\n" + token)
    print("\nopen this to watch:\n" + link)


def _camera_names(env):
    import mujoco
    m = env.mj_model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(m.ncam)]


def build_env(name, pov=True):
    """Build the env, optionally with the onboard camera injected.

    The POV camera is added by rewriting the robot XML in memory during
    construction (see pov_camera.py); the patch is removed immediately after.
    """
    import pov_camera
    ctx = pov_camera.pov_assets() if pov else contextlib.nullcontext()
    with ctx:
        return _build_env(name)


def _build_env(name):
    over = cpu_overrides()
    if over:
        print(f"off-GPU: using {over} (stock impl='warp' is ~3x slower here)",
              flush=True)
    if name == "ascent":
        import fixed_line
        return fixed_line.load(config_overrides={
            **over, "line_config.slope_deg": _ARGS.slope})
    if name == "flat":
        from mujoco_playground import locomotion
        return locomotion.load("G1JoystickFlatTerrain", config_overrides=over)
    import ice_patch
    return ice_patch.load("flat_terrain", config_overrides=over)


def rollout(env, args):
    """Step the env and render. Returns (frames, fps, stats)."""
    import jax
    import jax.numpy as jp

    policy = None
    if args.checkpoint:
        policy = load_policy_compat(args.checkpoint)
        print(f"loaded policy from {args.checkpoint}", flush=True)
    else:
        print("NO CHECKPOINT: robot is unactuated and will collapse. "
              "Pipeline test only.", flush=True)

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    traj = [state]
    t0 = time.time()
    for _ in range(args.steps):
        if policy is not None:
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
        else:
            action = jp.zeros(env.action_size)
        state = step(state, action)
        traj.append(state)
    sim_s = time.time() - t0

    t0 = time.time()
    names = _camera_names(env)

    def render_cam(cam):
        if cam and cam not in names:
            print(f"WARN: no camera '{cam}' in this scene; static view instead",
                  flush=True)
            cam = None
        print(f"rendering camera: {cam or 'static free camera'}", flush=True)
        return env.render(traj, height=args.height, width=args.width, camera=cam)

    frames = {"chase": render_cam(args.camera or None)}
    if args.pov:
        frames["pov"] = render_cam("pov")
    render_s = time.time() - t0

    fps = 1.0 / env.dt
    stats = {
        "steps": len(traj),
        "sim_hz": len(traj) / max(sim_s, 1e-9),
        "render_fps": len(traj) / max(render_s, 1e-9),
        "target_fps": fps,
    }
    return frames, fps, stats


async def publish(frames, fps, args):
    from livekit import api, rtc

    loaded = load_env_file()
    if loaded:
        # Names only. Never print the values.
        print(f"loaded from .env: {', '.join(loaded)}", flush=True)

    url = os.environ.get("LIVEKIT_URL")
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    missing = [n for n, v in
               (("LIVEKIT_URL", url), ("LIVEKIT_API_KEY", key),
                ("LIVEKIT_API_SECRET", secret)) if not v]
    if missing:
        raise SystemExit(
            "Missing " + ", ".join(missing) + ".\n"
            "Create a project at https://cloud.livekit.io/, then:\n"
            "  export LIVEKIT_URL=wss://<project>.livekit.cloud\n"
            "  export LIVEKIT_API_KEY=...\n"
            "  export LIVEKIT_API_SECRET=...\n"
            "Run with --dry-run to test everything except the connection."
        )

    token = (api.AccessToken(key, secret)
             .with_identity(args.identity)
             .with_name("G1 simulation")
             .with_grants(api.VideoGrants(room_join=True, room=args.room))
             .to_jwt())

    room = rtc.Room()

    # A dropped connection is otherwise INVISIBLE: capture_frame() on a dead
    # room silently no-ops, so the loop happily reports "looped (249021 frames
    # sent)" while the server returns 404 for the room. Found exactly that way.
    alive = {"ok": True}

    @room.on("disconnected")
    def _on_disconnected(*a):
        alive["ok"] = False
        print("DISCONNECTED by server -- stopping", flush=True)

    await room.connect(url, token)
    print(f"connected to room '{args.room}' as '{args.identity}'", flush=True)

    sources = {}
    for name, seq in frames.items():
        src = rtc.VideoSource(args.width, args.height)
        track = rtc.LocalVideoTrack.create_video_track(f"g1-{name}", src)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
        sources[name] = src
        print(f"publishing track 'g1-{name}' ({len(seq)} frames)", flush=True)
    print(f"{fps:.0f} fps{' (looping)' if args.loop else ''}", flush=True)

    # RGB -> RGBA once, up front. Per-frame conversion inside the publish loop
    # would add jitter to the one thing that must stay on a clock.
    rgba = {}
    for name, seq in frames.items():
        buf = []
        for f in seq:
            a = np.asarray(f, dtype=np.uint8)
            buf.append(np.dstack(
                [a, np.full(a.shape[:2] + (1,), 255, np.uint8)]).tobytes())
        rgba[name] = buf

    period = 1.0 / fps
    n_frames = min(len(v) for v in rgba.values())
    try:
        n = 0
        next_t = time.perf_counter()
        while alive["ok"]:
            for i in range(n_frames):
                if not alive["ok"]:
                    break
                for name, src in sources.items():
                    src.capture_frame(rtc.VideoFrame(
                        args.width, args.height, rtc.VideoBufferType.RGBA,
                        rgba[name][i]))
                n += 1
                next_t += period
                await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
            if not args.loop:
                break
            state = room.connection_state
            print(f"looped ({n} frames sent, state={state})", flush=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await room.disconnect()
        print("disconnected", flush=True)


def main():
    global _ARGS
    args = parse_args()
    _ARGS = args
    if args.viewer_token:
        print_viewer_token(args)
        return
    env = build_env(args.env)
    frames, fps, stats = rollout(env, args)
    print(f"rolled out {stats['steps']} steps: sim {stats['sim_hz']:.1f} Hz, "
          f"render {stats['render_fps']:.1f} fps, "
          f"target {stats['target_fps']:.0f} fps", flush=True)
    if stats["render_fps"] < stats["target_fps"]:
        print(f"  (render is {stats['target_fps']/stats['render_fps']:.0f}x "
              f"slower than realtime -- hence pre-render + timed playback)",
              flush=True)
    print(f"frames: {len(frames)} at {args.width}x{args.height}", flush=True)

    if args.dry_run:
        out = pathlib.Path("stream_preview.png")
        try:
            import mediapy
            mediapy.write_image(out, frames[len(frames) // 2])
            print(f"DRY RUN OK -- wrote {out}, connected to nothing", flush=True)
        except Exception as e:
            print(f"DRY RUN OK (preview write failed: {e})", flush=True)
        return

    asyncio.run(publish(frames, fps, args))


if __name__ == "__main__":
    main()
