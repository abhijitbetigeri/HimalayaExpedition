# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda12]", "mujoco", "playground", "brax", "livekit", "numpy"]
# ///
"""Drive a trained G1 ice policy live over LiveKit, or benchmark the render.

This is the demo surface, deliberately OFF the training path: it loads a
checkpoint that `train_ice.py` already wrote and never imports brax's trainer.
Nothing here can affect an A/B number.

Two modes
---------
`--bench` measures step and render rate and exits. No network, no LiveKit
import, so it runs in the project .venv with nothing new installed:

    .venv/bin/python teleop_eval.py --bench --checkpoint <path>

Streaming mode joins a LiveKit room as participant `robot`:

    export LIVEKIT_URL=wss://<project>.livekit.cloud
    export LIVEKIT_TOKEN=$(lk token create --join --room himalaya \
        --identity robot --valid-for 24h --token-only --yes)
    python teleop_eval.py --checkpoint /mnt/himalaya-g1/runs/ice-v1/checkpoints

How LiveKit is used
-------------------
Four primitives, one room (`himalaya`), two participants (`robot`, `operator`):

  video track  "g1.view"       out   rendered MuJoCo frames
  data track   "robot.telemetry" out   foot_mu / torso_z / wind / survived
  data track   "robot.control"   in    [vx, vy, wz] joystick command
  rpc          estop|turn_back|reset in discrete acts, with a return value

The command channel is the whole trick: the Playground G1 policy is already
joystick-conditioned, so teleoperation is a matter of writing a different value
into `state.info["command"]` -- the same field `train_ice.py` pins to
[1, 0, 0] for its scripted eval. No retraining, no observation change, so the
216-vs-219 shape trap in CLAUDE.md does not come into play.

Why a data track and not RPC for the command: commands are a lossy, latest-wins
stream at ~50 Hz. `subscribe(buffer_size=1)` drops backlog rather than queuing
it, which is what you want on a bad link -- a stale command is worse than none.
RPC is request/response, so it carries the discrete acts that need an ack.

Link degradation (`--command-hz`, `--command-latency-ms`) throttles the ADOPTED
command to imitate a satellite uplink. That is the experiment: the policy runs
at full rate locally while the human's commands arrive slowly. Real latency is
measured separately from `user_timestamp` on the wire; the flags only shape the
uplink so the arms can be compared without renting a satellite.

Watchdog: if no command arrives within `--command-timeout-ms`, the command goes
to zero. For a joystick policy zero means stand in place, which the policy was
trained to do -- so a dropped link degrades to a G1 balancing on ice, not to
undefined behaviour. That safety property is free; do not replace it with a
freeze that holds the last command.

PREREQUISITE, same as every job here: `hf jobs uv run` ships only this file.
`ice_randomize.py`, `wind.py` and `ice_patch.py` must already be in the bucket.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

# The uv image ships no GL stack and `import mujoco` walks the EGL path eagerly,
# so this has to happen before any mujoco import. Same dance as train_ice.py --
# but gated on Linux, because unlike the training scripts this one is meant to
# also run on a laptop under --bench, and mujoco REJECTS MUJOCO_GL=egl on macOS
# rather than falling back.
if sys.platform.startswith("linux"):
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
        "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0",
        shell=True, check=False,
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

BUCKET = pathlib.Path(os.environ.get("HIMALAYA_OUT", "/mnt/himalaya-g1"))
if str(BUCKET) not in sys.path:
    sys.path.insert(0, str(BUCKET))

import jax          # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco       # noqa: E402
import numpy as np  # noqa: E402

VIDEO_TRACK = "g1.view"
TELEMETRY_TRACK = "robot.telemetry"
CONTROL_TRACK = "robot.control"
OPERATOR_IDENTITY = "operator"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="a brax PPO checkpoint dir, or the parent 'checkpoints' "
                        "dir (newest step is picked)")
    p.add_argument("--task", default="flat_terrain",
                   choices=["flat_terrain", "rough_terrain"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0,
                   help="video publish rate; the sim runs at its own pace")
    p.add_argument("--room", default="himalaya")
    p.add_argument("--bench", action="store_true",
                   help="measure step+render rate and exit. No LiveKit.")
    p.add_argument("--bench-steps", type=int, default=200)
    # Satellite-link shaping, applied to the ADOPTED command only.
    p.add_argument("--command-hz", type=float, default=0.0,
                   help="0 = adopt every command as it lands")
    p.add_argument("--command-latency-ms", type=float, default=0.0)
    p.add_argument("--command-timeout-ms", type=float, default=1500.0,
                   help="no command for this long -> zero command (stand)")
    p.add_argument("--out", default=None,
                   help="write a session summary here on exit")
    return p.parse_args()


def resolve_checkpoint(path: str) -> pathlib.Path:
    """brax writes checkpoints/<step>/; accept either that or its parent."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    steps = sorted((c for c in p.iterdir() if c.is_dir() and c.name.isdigit()),
                   key=lambda c: int(c.name))
    if steps:
        print(f"checkpoint: {len(steps)} steps present, using {steps[-1].name}",
              flush=True)
        return steps[-1]
    return p


class Sim:
    """Owns the policy, the env and a PERSISTENT renderer.

    mjx_env.render_array() builds a fresh mujoco.Renderer AND a fresh MjData on
    every call (mjx_env.py:333, :341). That is fine for the one batch render at
    the end of training and hopeless per-frame, so this holds both for the life
    of the process and only copies state into them.
    """

    def __init__(self, args):
        import ice_patch
        from brax.training.agents.ppo import checkpoint as ppo_checkpoint

        self.env = ice_patch.load(args.task)  # full stack, as in train_ice eval
        self.inference_fn = jax.jit(
            ppo_checkpoint.load_policy(resolve_checkpoint(args.checkpoint)))
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)

        self.mj_model = self.env.mj_model
        self.renderer = mujoco.Renderer(
            self.mj_model, height=args.height, width=args.width)
        self.mj_data = mujoco.MjData(self.mj_model)
        # Preallocated RGBA; MuJoCo renders RGB and LiveKit wants 4 channels.
        self.rgba = np.empty((args.height, args.width, 4), np.uint8)
        self.rgba[..., 3] = 255

        cfg = getattr(self.env, "_config", None)
        self.cmd_lo, self.cmd_hi = self._command_bounds(cfg)

        self.rng = jax.random.PRNGKey(args.seed)
        self.state = self._reset(self.rng)
        self.steps = 0
        self.survived = 0
        self.min_mu = 1.0

    @staticmethod
    def _command_bounds(cfg):
        """Pull the joystick ranges off the env so they cannot drift from it."""
        try:
            return (np.array([cfg.lin_vel_x[0], cfg.lin_vel_y[0],
                              cfg.ang_vel_yaw[0]], np.float32),
                    np.array([cfg.lin_vel_x[1], cfg.lin_vel_y[1],
                              cfg.ang_vel_yaw[1]], np.float32))
        except (AttributeError, TypeError):
            return (np.array([-1.0, -0.5, -1.0], np.float32),
                    np.array([1.0, 0.5, 1.0], np.float32))

    def clamp(self, cmd) -> np.ndarray:
        return np.clip(np.asarray(cmd, np.float32), self.cmd_lo, self.cmd_hi)

    def step(self, command: np.ndarray):
        self.state.info["command"] = jp.asarray(command)
        self.rng, act_rng = jax.random.split(self.rng)
        action, _ = self.inference_fn(self.state.obs, act_rng)
        self.state = self._step(self.state, action)
        self.state.info["command"] = jp.asarray(command)
        self.steps += 1
        if float(self.state.done) == 0.0:
            self.survived += 1

    def render(self) -> np.ndarray:
        """Render current state into the preallocated RGBA buffer."""
        d, s = self.mj_data, self.state
        d.qpos, d.qvel = np.asarray(s.data.qpos), np.asarray(s.data.qvel)
        d.mocap_pos, d.mocap_quat = (np.asarray(s.data.mocap_pos),
                                     np.asarray(s.data.mocap_quat))
        d.xfrc_applied = np.asarray(s.data.xfrc_applied)  # carries the wind
        mujoco.mj_forward(self.mj_model, d)
        self.renderer.update_scene(d, camera=-1)
        self.rgba[..., :3] = self.renderer.render()
        return self.rgba

    def telemetry(self, command) -> dict:
        info = self.state.info
        mu = float(jp.min(info.get("foot_mu", jp.ones(2))))
        self.min_mu = min(self.min_mu, mu)
        return {
            "step": self.steps,
            "foot_mu": round(mu, 4),
            "min_foot_mu": round(self.min_mu, 4),
            "torso_z": round(float(self.state.data.qpos[2]), 4),
            "wind_gust": round(float(info.get("wind_gust", 0.0)), 4),
            "wind_speed": round(float(info.get("wind_base_speed", 0.0)), 4),
            "done": float(self.state.done),
            "survived": self.survived,
            "command": [round(float(c), 3) for c in command],
        }

    def summary(self) -> dict:
        return {"steps": self.steps, "survived": self.survived,
                "min_foot_mu": round(self.min_mu, 4)}


class Shared:
    """Handoff between the sim thread and the asyncio side.

    The LiveKit SDK is asyncio; jitted JAX calls block. Stepping on the event
    loop stalls the video track regardless of link quality, so the sim gets its
    own thread and the two sides trade under a lock. The publisher always takes
    the newest frame -- a slow sim drops frames rather than backing up.
    """

    def __init__(self, args):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.telemetry = {}
        self.summary = {}
        self._incoming = np.zeros(3, np.float32)   # last received
        self._applied = np.zeros(3, np.float32)    # last adopted
        self._last_rx = 0.0
        self._last_adopt = 0.0
        self.estopped = False
        self.timeout_s = args.command_timeout_ms / 1000.0
        self.adopt_period = 1.0 / args.command_hz if args.command_hz > 0 else 0.0
        self.latency_s = args.command_latency_ms / 1000.0
        self.pending = []  # (deliver_at, command) when latency is simulated

    def submit(self, cmd: np.ndarray):
        now = time.monotonic()
        with self.lock:
            if self.latency_s > 0:
                self.pending.append((now + self.latency_s, cmd))
            else:
                self._incoming, self._last_rx = cmd, now

    def current(self) -> np.ndarray:
        """The command the sim should apply right now."""
        now = time.monotonic()
        with self.lock:
            if self.estopped:
                return np.zeros(3, np.float32)
            while self.pending and self.pending[0][0] <= now:
                _, cmd = self.pending.pop(0)
                self._incoming, self._last_rx = cmd, now
            # Uplink throttle: adopt at most once per period.
            if self.adopt_period and now - self._last_adopt < self.adopt_period:
                stale = self._applied
            else:
                self._applied = self._incoming
                self._last_adopt = now
                stale = self._applied
            # Watchdog: silence means stand, not "hold the last command".
            if self._last_rx and now - self._last_rx > self.timeout_s:
                return np.zeros(3, np.float32)
            return stale.copy()


def sim_thread(sim: Sim, shared: Shared, fps: float):
    period = 1.0 / fps if fps > 0 else 0.0
    next_render = 0.0
    while not shared.stop.is_set():
        cmd = shared.current()
        sim.step(cmd)
        now = time.monotonic()
        if now >= next_render:
            frame = sim.render().tobytes()
            with shared.lock:
                shared.frame = frame
                shared.telemetry = sim.telemetry(cmd)
            next_render = now + period
    with shared.lock:
        shared.summary = sim.summary()


def run_bench(args):
    """Answers the only question that decides where this can run."""
    sim = Sim(args)
    cmd = np.array([1.0, 0.0, 0.0], np.float32)

    sim.step(cmd); sim.render()          # warm the jit and the GL context
    t0 = time.time()
    for _ in range(args.bench_steps):
        sim.step(cmd)
    t_step = (time.time() - t0) / args.bench_steps

    t0 = time.time()
    for _ in range(args.bench_steps):
        sim.render()
    t_render = (time.time() - t0) / args.bench_steps

    combined = 1.0 / (t_step + t_render)
    print(f"\nstep   {t_step*1000:7.2f} ms  ({1/t_step:6.1f} Hz)")
    print(f"render {t_render*1000:7.2f} ms  ({1/t_render:6.1f} Hz)  "
          f"{args.width}x{args.height}")
    print(f"combined ceiling: {combined:.1f} fps")
    print("verdict:", "streams fine here" if combined >= 20 else
          "too slow for a live demo on this box -- render smaller, "
          "drop --fps, or move to the GPU box")
    return sim.summary()


async def run_livekit(args):
    import asyncio
    from livekit import rtc

    url, token = os.environ.get("LIVEKIT_URL"), os.environ.get("LIVEKIT_TOKEN")
    if not url or not token:
        raise SystemExit(
            "set LIVEKIT_URL and LIVEKIT_TOKEN. Mint one with:\n"
            f"  lk token create --join --room {args.room} --identity robot "
            "--valid-for 24h --token-only --yes")

    sim, shared = Sim(args), Shared(args)
    sim.step(np.zeros(3, np.float32)); sim.render()  # warm before anyone joins

    room = rtc.Room()

    # --- inbound: the joystick command stream -------------------------------
    def on_control_frame(track):
        async def reader():
            # buffer_size=1: keep the newest command, drop the backlog. On a
            # degraded link a stale command is worse than no command.
            async for frame in track.subscribe(buffer_size=1):
                try:
                    payload = json.loads(frame.payload.decode("utf-8"))
                    cmd = sim.clamp([payload["vx"], payload["vy"], payload["wz"]])
                except (json.JSONDecodeError, UnicodeDecodeError,
                        KeyError, TypeError, ValueError):
                    continue  # never trust the wire; drop and keep going
                shared.submit(cmd)
        return asyncio.create_task(reader())

    @room.on("data_track_published")
    def _(track: rtc.RemoteDataTrack):
        if (track.publisher_identity == OPERATOR_IDENTITY
                and track.info.name == CONTROL_TRACK):
            on_control_frame(track)

    # --- inbound: discrete acts, which need an ack --------------------------
    @room.local_participant.register_rpc_method("estop")
    async def _(data):
        with shared.lock:
            shared.estopped = True
        return json.dumps({"ok": True, "state": "estopped"})

    @room.local_participant.register_rpc_method("turn_back")
    async def _(data):
        # The turn-back decision engine CLAUDE.md scopes out of the policy:
        # the human makes the call, the policy handles the footing.
        shared.submit(sim.clamp([-1.0, 0.0, 0.0]))
        return json.dumps({"ok": True, "state": "returning"})

    @room.local_participant.register_rpc_method("resume")
    async def _(data):
        with shared.lock:
            shared.estopped = False
        return json.dumps({"ok": True, "state": "running"})

    await room.connect(url, token)
    print(f"connected to '{room.name}' as '{room.local_participant.identity}'",
          flush=True)

    # --- outbound: video + telemetry ----------------------------------------
    source = rtc.VideoSource(args.width, args.height)
    video = rtc.LocalVideoTrack.create_video_track(VIDEO_TRACK, source)
    await room.local_participant.publish_track(video, rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_encoding=rtc.VideoEncoding(
            max_framerate=int(args.fps), max_bitrate=3_000_000),
    ))
    telemetry = await room.local_participant.publish_data_track(
        name=TELEMETRY_TRACK)

    worker = threading.Thread(
        target=sim_thread, args=(sim, shared, args.fps), daemon=True)
    worker.start()

    period = 1.0 / args.fps
    try:
        while True:
            with shared.lock:
                frame, tele = shared.frame, dict(shared.telemetry)
            if frame is not None:
                source.capture_frame(rtc.VideoFrame(
                    args.width, args.height, rtc.VideoBufferType.RGBA, frame))
                telemetry.try_push(rtc.DataTrackFrame(
                    payload=json.dumps(tele).encode("utf-8"),
                    # Lets the operator compute real one-way latency instead of
                    # us asserting a number for the satellite comparison.
                    user_timestamp=int(time.time() * 1000),
                ))
            await asyncio.sleep(period)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        shared.stop.set()
        worker.join(timeout=5.0)
        telemetry.unpublish()
        await room.disconnect()
    return shared.summary or sim.summary()


def main():
    args = parse_args()
    print("jax:", jax.__version__, "devices:", jax.devices(), flush=True)

    if args.bench:
        summary = run_bench(args)
    else:
        import asyncio
        summary = asyncio.run(run_livekit(args))

    print("summary:", json.dumps(summary), flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
