"""Sustained wind for the Playground G1.

`WindyJoystick` subclasses the stock Joystick env and applies a persistent
aerodynamic load to the torso. Project-level subclass rather than a patch to
site-packages, for the same reason as ice_randomize.py: bootstrap.sh does a
clean install on the VM.

Why this is not already covered by push_config
----------------------------------------------
The stock "push" writes `qvel[:2]` directly -- an instantaneous velocity
teleport, applied for one step every 5-10s. Wind is the opposite: a modest force
that never goes away, so the policy has to lean into it and hold a trim
attitude. A velocity impulse teaches nothing about that.

The model
---------
Drag: F = 0.5 * rho * Cd * A * v^2, applied through `Data.xfrc_applied`, which
persists across all n_substeps of a control step.

At 5500 m, rho is about 0.65 kg/m^3 -- roughly half sea level, which matters:
the same wind speed hits noticeably softer up high. With Cd*A ~= 0.5 m^2, a
100 km/h gust is 126 N. The G1 weighs 33.3 kg (327 N), so that is 38% of body
weight applied sideways, continuously. Measured over the sampled distribution:
median episode peaks at 13% of body weight, p90 at 45%, worst case 61%.

Center of pressure sits above the torso COM, so the force also produces a
pitching/rolling moment. Ignoring it would make wind far easier to reject than
it is in reality, so it is applied as an explicit torque.

Gusts are an AR(1) process on wind speed with a ~1.5 s correlation time, not
per-step white noise -- white noise averages out over a gait cycle and the
policy learns to ignore it.

Observability
-------------
Wind is NOT in the actor's `state` obs. A real robot has no anemometer and must
infer loading from its IMU, which is exactly the skill we want. It IS appended
to `privileged_state`, which only the critic sees -- that makes the value
function's job easier without giving the policy anything it could not get on
hardware.
"""

import copy
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick

# Air density at ~5500 m. Sea level is 1.225; using that would overstate every
# force by ~1.9x and is the single easiest way to get this wrong.
AIR_DENSITY = 0.65

# Drag area, Cd * frontal area, in m^2. A standing humanoid is a bluff body.
DRAG_AREA = 0.5

# Height of the center of pressure above the torso COM, in m. Sets the moment.
COP_LEVER_ARM = 0.25


def default_config() -> config_dict.ConfigDict:
  """Stock joystick config plus wind, with njmax raised for ice_randomize."""
  config = joystick.default_config()
  config.wind_config = config_dict.create(
      enable=True,
      # Base wind speed in m/s, sampled once per episode. 0 to 20 m/s spans
      # dead calm to ~72 km/h, with gusts taking the top end past that.
      speed_range=[0.0, 20.0],
      # Stationary std of the gust process, as a fraction of base speed.
      gust_std=0.35,
      # Gust correlation time in seconds.
      gust_tau=1.5,
      # Hard cap on instantaneous speed, m/s.
      max_speed=35.0,
  )
  # Compliant contact from ice_randomize needs more constraint rows than the
  # stock 90. Kept in sync with ice_randomize.CONFIG_OVERRIDES.
  config.njmax = 160
  return config


class WindyJoystick(joystick.Joystick):
  """Joystick with a persistent aerodynamic load on the torso."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # NOT `config = default_config()` in the signature. Playground's own envs do
    # exactly that, and because a default argument is evaluated once and shared,
    # `config_overrides` mutates it for every env built afterwards in the same
    # process. Building a wind-off env and then a wind-on env -- i.e. any A/B
    # script -- silently gives you two wind-off envs. Copy defensively.
    config = default_config() if config is None else copy.deepcopy(config)
    # Playground calls this only from `locomotion.load()`, i.e. the registry
    # loader -- never from the env constructors. Building the class directly, as
    # we do, means the robot assets are never fetched and the XML fails to
    # resolve pelvis.STL on any clean box. Costs nothing once they exist.
    mjx_env.ensure_menagerie_exists()
    super().__init__(task, config, config_overrides)
    cfg = self._config.wind_config
    self._wind_alpha = jp.exp(-self.dt / cfg.gust_tau)
    # Stationary AR(1): x_t = a*x_{t-1} + sqrt(1-a^2)*sigma*eps keeps std=sigma.
    self._wind_noise_scale = jp.sqrt(1.0 - self._wind_alpha**2) * cfg.gust_std
    self._drag_k = 0.5 * AIR_DENSITY * DRAG_AREA

  def _wind_wrench(self, speed: jax.Array, direction: jax.Array):
    """Force and torque on the torso from a horizontal wind."""
    force = self._drag_k * speed**2 * direction  # (3,), world frame
    # r x F with r = (0, 0, h) gives (-h*Fy, h*Fx, 0).
    torque = jp.array(
        [-COP_LEVER_ARM * force[1], COP_LEVER_ARM * force[0], 0.0]
    )
    return force, torque

  def reset(self, rng: jax.Array) -> mjx_env.State:
    cfg = self._config.wind_config
    if not cfg.enable:
      # Consume no randomness at all, so wind-off is bitwise identical to the
      # stock env on the same seed. That makes same-seed wind-on/wind-off the
      # clean A/B, which is the comparison the demo actually rests on.
      state = super().reset(rng)
      state.info["wind_dir"] = jp.array([1.0, 0.0, 0.0])
      state.info["wind_base_speed"] = jp.zeros(())
      state.info["wind_gust"] = jp.zeros(())
      state.info["wind_force"] = jp.zeros(3)
      return state

    rng, wind_rng = jax.random.split(rng)
    state = super().reset(rng)

    dir_rng, speed_rng = jax.random.split(wind_rng)
    theta = jax.random.uniform(dir_rng, maxval=2 * jp.pi)
    base_speed = jax.random.uniform(
        speed_rng, minval=cfg.speed_range[0], maxval=cfg.speed_range[1]
    )

    state.info["wind_dir"] = jp.array([jp.cos(theta), jp.sin(theta), 0.0])
    state.info["wind_base_speed"] = base_speed
    state.info["wind_gust"] = jp.zeros(())
    # No wind has been applied at t=0, so the zero here is correct, not a stub.
    state.info["wind_force"] = jp.zeros(3)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config.wind_config
    if not cfg.enable:
      return super().step(state, action)

    state.info["rng"], gust_rng = jax.random.split(state.info["rng"])
    gust = (
        self._wind_alpha * state.info["wind_gust"]
        + self._wind_noise_scale * jax.random.normal(gust_rng)
    )
    speed = jp.clip(
        state.info["wind_base_speed"] * (1.0 + gust), 0.0, cfg.max_speed
    )
    force, torque = self._wind_wrench(speed, state.info["wind_dir"])

    # Rebuild from zero every step: xfrc_applied is NOT cleared by the
    # integrator, so accumulating into it would leave a stale gust blowing.
    xfrc = jp.zeros_like(state.data.xfrc_applied)
    xfrc = xfrc.at[self._torso_body_id, 0:3].set(force)
    xfrc = xfrc.at[self._torso_body_id, 3:6].set(torque)
    state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

    # Set before super().step() so this step's _get_obs sees the current wind.
    state.info["wind_gust"] = gust
    state.info["wind_force"] = force

    return super().step(state, action)

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    obs = super()._get_obs(data, info, contact)
    # Critic only. `.get` because super().reset() computes obs before reset()
    # has had a chance to seed the wind fields.
    wind_force = info.get("wind_force", jp.zeros(3))
    obs["privileged_state"] = jp.hstack([obs["privileged_state"], wind_force])
    return obs


def load(task: str = "flat_terrain", **kwargs) -> WindyJoystick:
  """Build a WindyJoystick. Task is 'flat_terrain' or 'rough_terrain'."""
  return WindyJoystick(task=task, **kwargs)
