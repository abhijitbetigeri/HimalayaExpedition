"""Fixed-line ascent env for the G1.

Task: climb a slope while clipped into a taut fixed rope with a mechanical
ascender, the way a real climber moves on a fixed line.

Reuses the Joystick machinery rather than inventing a reward from scratch. The
locomotion problem "walk forward at a commanded velocity" is already solved and
tuned in Playground; ascending is that, on an incline, with a tether. So the
command is pinned forward and the slope converts forward progress into height.
A hand-rolled climbing reward would be a second research project.

The tether
----------
One-sided spring-damper on the pelvis, applied through `xfrc_applied`. It is
slack until the robot drops `slack` metres below the ascender, then pulls up the
fall line. Rejected alternative: a rigid `connect` equality, which pins the
pelvis to a 1-DOF rail -- measured 327 N of tether load against a 330 N robot,
i.e. the rope carried the entire robot and the legs were decorative.

The ratchet
-----------
A jumar slides up freely and locks under load. So the ascender position is a
running maximum of how high the robot has been, and it never decreases within an
episode. This is what makes a fall recoverable rather than terminal: slip, and
you drop to the ascender, not to the bottom.

Kept at stock DOF on purpose: nq=36, nv=35, nu=29, neq=0, identical to the flat
G1. The ascender is a mocap marker with no DOF. So this composes with
ice_randomize.py unchanged, and a policy trained here has the same shape as
every other policy in this repo.
"""

import contextlib
import copy
import math
import pathlib
import tempfile
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import joystick

import fixed_line_scene as fls


def default_config() -> config_dict.ConfigDict:
    config = joystick.default_config()
    config.line_config = config_dict.create(
        enable=True,
        slope_deg=30.0,
        # Metres the robot may drop below the ascender before the rope comes
        # taut. Real slack between a harness and a jumar.
        slack=0.35,
        # One-sided spring-damper, N/m and N/(m/s). Stiff enough to arrest a
        # fall inside a step or two, soft enough not to fire the solver.
        stiffness=1200.0,
        damping=120.0,
        # Cap so a deep fall cannot inject an absurd impulse.
        max_force=1500.0,
        ratchet=True,
        # Recompute the tether force at every physics substep instead of once
        # per control step. See _with_substep_tether for why this is opt-in.
        substep=False,
        # Heading jitter about the fall line, radians. NOT Playground's +/-pi:
        # facing downhill makes the ascent reward pay for descending.
        start_yaw_jitter=0.35,
        start_xy_jitter=0.15,
    )
    # Reward per metre gained along the fall line. This is the actual task;
    # everything inherited from Joystick is there to keep it walking while it
    # does it.
    config.reward_config.scales.ascent = 10.0
    config.njmax = 160
    return config


class FixedLineAscent(joystick.Joystick):
    """G1 ascending a fixed line on a slope."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        config = default_config() if config is None else copy.deepcopy(config)
        # Apply overrides BEFORE reading slope_deg. G1Env applies them itself,
        # but only after __init__ has already built the scene XML -- so
        # config_overrides={"line_config.slope_deg": 0} silently produced a
        # 30 degree scene whose config claimed 0. The geometry and the config
        # disagreed, and nothing errored.
        if config_overrides:
            config.update_from_flattened_dict(dict(config_overrides))
            config_overrides = None
        slope = config.line_config.slope_deg
        mjx_env.ensure_menagerie_exists()

        # G1Env reads the XML off disk but resolves <include> from the in-memory
        # asset dict, so a generated scene in a temp file works fine.
        xml = fls.build_scene_xml(slope)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="himalaya-line-"))
        path = tmp / "scene_fixed_line.xml"
        path.write_text(xml)

        # Skip Joystick.__init__ (it hardcodes the flat/rough XML) but keep the
        # _post_init it performs.
        g1_base.G1Env.__init__(
            self, xml_path=path.as_posix(), config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

        a = math.radians(self._config.line_config.slope_deg)
        # Unit vector up the fall line, and the line's origin.
        self._line_dir = jp.array([math.cos(a), 0.0, math.sin(a)])
        self._line_origin = jp.array([0.0, 0.0, 0.785])
        self._pelvis_body_id = self._mj_model.body("pelvis").id

    def _project(self, data) -> jax.Array:
        """Distance of the pelvis along the fall line from the start anchor."""
        rel = data.xpos[self._pelvis_body_id] - self._line_origin
        return jp.dot(rel, self._line_dir)

    def _tether(self, data, asc_s: jax.Array):
        """One-sided spring-damper pulling the pelvis back up to the ascender."""
        cfg = self._config.line_config
        s = self._project(data)
        vel = jp.dot(data.cvel[self._pelvis_body_id][3:6], self._line_dir)

        # Violation is positive only when the robot has fallen past the slack.
        drop = jp.maximum(asc_s - cfg.slack - s, 0.0)
        taut = drop > 0.0
        mag = cfg.stiffness * drop - cfg.damping * vel * taut
        mag = jp.clip(mag, 0.0, cfg.max_force)  # a rope pulls, never pushes
        return mag * self._line_dir, s

    @contextlib.contextmanager
    def _with_substep_tether(self, asc_s: jax.Array):
        """Recompute the tether force inside the substep loop.

        The problem
        -----------
        `Joystick.step` calls `mjx_env.step(model, data, ctrl, n_substeps)`,
        which is a `lax.scan` holding both `ctrl` and `xfrc_applied` constant
        for all n_substeps. For the G1, ctrl_dt/sim_dt = 0.02/0.002 = **10
        substeps**, so the default path computes the spring-damper once from
        the state at the START of the control step and then holds that force
        for 20 ms of physics.

        MEASURED VERDICT: it barely matters. Keep this off.
        ---------------------------------------------------
        `check_tether_lag.py`, same seed and same action stream both ways:

            preloaded taut (0.30 m past slack, ~360 N):
                peak force  435.6 N -> 431.3 N
                ringing     35 -> 31 force reversals
                trajectory  3.04 cm max divergence, 0.59 cm final
            dynamic catch (slack rope, neutral action, rope catches at speed):
                peak force  314.4 N -> 304.5 N
                ringing     10 -> 10 force reversals
                trajectory  0.81 cm max divergence, 0.08 cm final

        The prior reasoning was that a stale damping term would show up as
        overshoot and ringing at the moment of arrest. It does not, and the
        dynamic catch -- the case predicted to be WORST -- is the one where
        the two modes agree most closely. At this stiffness the pelvis's
        inertia dominates over a 20 ms window, so the held force never gets
        far enough out of date to change the trajectory.

        So `ice_patch.py`'s original call ("not worth it; documented instead")
        was right, and is now measured rather than assumed. This code stays
        opt-in for one reason: if `stiffness` is ever raised substantially
        (or `ctrl_dt` lengthened), re-run check_tether_lag.py before trusting
        the per-step path -- the conclusion is a property of those numbers,
        not a general fact.

        This is the same lag ice_patch.py documents for friction and declines
        to fix ("stepping physics to find the positions before deciding the
        friction to step physics with"). That objection does NOT apply here,
        and it is worth being precise about why: friction there is a *Model*
        field and the patch reasoned it would need the contact solution. The
        tether force is a pure function of `xpos` and `cvel` -- forward
        kinematics, already valid at the top of each substep. Nothing has to
        be solved twice. Per-substep recomputation is just a scan.

        Why not port icefall-g1's split-step
        ------------------------------------
        npow/icefall-g1 solves the analogous problem with
        `mj_step1 -> apply force -> mj_step2`, because ITS force depends on
        `mj_contactForce`, which only exists after the collision solve.
        **MJX exposes no step1/step2** (verified on mujoco 3.12.0), so that
        pattern is unavailable here regardless. It is also unnecessary: our
        force needs no contact solution.

        The cost, and why it is not worth paying by default
        ---------------------------------------------------
        This patches the module-level `mjx_env.step` for the duration of the
        traced call, because that is the only interception point short of
        forking ~60 lines of `Joystick.step` (obs, rewards, done, info) into
        this file. It is the same swap trick `ice_patch._with_model` uses one
        level down, and it is restored in `finally`. It IS a global mutation,
        so it is only safe while a single trace is in flight -- fine under
        jit/vmap, not safe if two envs are ever traced from separate threads.
        Opt-in for that reason.
        """
        original = mjx_env.step

        def substep_tether_step(model, data, action, n_substeps=1):
            def one(d, _):
                force, _ = self._tether(d, asc_s)
                xfrc = jp.zeros_like(d.xfrc_applied)
                xfrc = xfrc.at[self._pelvis_body_id, 0:3].set(force)
                d = d.replace(ctrl=action, xfrc_applied=xfrc)
                return mjx.step(model, d), None

            return jax.lax.scan(one, data, (), n_substeps)[0]

        mjx_env.step = substep_tether_step
        try:
            yield
        finally:
            mjx_env.step = original

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Always climb. The slope turns forward velocity into height."""
        del rng
        return jp.array([0.8, 0.0, 0.0])


    def _face_uphill(self, state: mjx_env.State) -> mjx_env.State:
        """Point the robot up the fall line and put it back on the rope.

        Playground's reset randomizes yaw over the FULL +/-pi and offsets xy by
        +/-0.5 m (joystick.py:259-267). For flat walking that is harmless -- the
        joystick command is in the robot frame, so any heading is equivalent.

        For ascent it destroys the task. The command is pinned to "forward" and
        the slope is what converts forward into height, so a robot spawned
        facing downhill is rewarded for walking DOWN the mountain, and one
        spawned sideways traverses. Measured on the first four seeds: three of
        the four faced away from the slope. The lateral offset matters too --
        the tether anchors on the line, so starting 0.5 m off it begins the
        episode with the rope already loaded.

        So: reset the heading to up-slope with a modest jitter (keeps variety,
        keeps the task well-posed) and pull xy back near the line.
        """
        cfg = self._config.line_config
        rng, yaw_rng, xy_rng = jax.random.split(state.info["rng"], 3)

        yaw = jax.random.uniform(
            yaw_rng, minval=-cfg.start_yaw_jitter, maxval=cfg.start_yaw_jitter)
        # Fall line runs along +x, so up-slope heading is yaw = 0.
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        dxy = jax.random.uniform(
            xy_rng, (2,), minval=-cfg.start_xy_jitter, maxval=cfg.start_xy_jitter)

        qpos = state.data.qpos
        qpos = qpos.at[0:2].set(dxy)
        qpos = qpos.at[3:7].set(quat)
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))

        state.info["rng"] = rng
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        obs = self._get_obs(data, state.info, contact)
        return state.replace(data=data, obs=obs)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        state = self._face_uphill(state)
        s0 = self._project(state.data)
        state.info["asc_s"] = s0
        state.info["last_s"] = s0
        state.info["tether_force"] = jp.zeros(3)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config.line_config
        if not cfg.enable:
            return super().step(state, action)

        force, s = self._tether(state.data, state.info["asc_s"])

        xfrc = jp.zeros_like(state.data.xfrc_applied)
        xfrc = xfrc.at[self._pelvis_body_id, 0:3].set(force)
        # Drive the visual marker to the ascender point.
        mocap = (self._line_origin + state.info["asc_s"] * self._line_dir)[None]
        state = state.replace(
            data=state.data.replace(xfrc_applied=xfrc, mocap_pos=mocap)
        )
        # Logged at the START of the control step in BOTH modes on purpose, so
        # the substep A/B compares physics rather than two different telemetry
        # definitions.
        state.info["tether_force"] = force
        state.info["last_s"] = s

        if cfg.substep:
            with self._with_substep_tether(state.info["asc_s"]):
                state = super().step(state, action)
        else:
            state = super().step(state, action)

        # Ratchet AFTER stepping: the ascender records the high-water mark.
        new_s = self._project(state.data)
        if cfg.ratchet:
            state.info["asc_s"] = jp.maximum(state.info["asc_s"], new_s)
        else:
            state.info["asc_s"] = new_s
        return state

    def _get_reward(self, data, action, info, metrics, done, first_contact,
                    contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact)
        # Metres gained along the fall line this step. Clipped so a physics
        # glitch cannot pay out a huge one-step bonus.
        gained = self._project(data) - info["last_s"]
        rewards["ascent"] = jp.clip(gained / self.dt, -2.0, 2.0)
        return rewards


def load(**kwargs) -> FixedLineAscent:
    return FixedLineAscent(**kwargs)
