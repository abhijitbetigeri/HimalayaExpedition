"""SPIKE: mid-episode rock->ice transitions for the Playground G1.

The open question was whether per-step friction changes can work at all, given
that `pair_friction` lives on mjx.Model and `domain_randomize` only runs once,
per environment, before the episode starts.

Answer: yes, and it costs about ten lines.

How
---
`BraxDomainRandomizationVmapWrapper` does not pass the per-env model as an
argument. It temporarily assigns it to `env._mjx_model` inside a context manager
and then calls `env.step` (see wrapper.py:220-246). So by the time our `step`
runs, `self.mjx_model` IS this environment's randomized model, as a tracer under
vmap. We can derive a new model from it and swap it in the same way, one level
down. Nothing about the wrapper needs to change, and per-episode randomization
from ice_randomize still applies underneath -- the patch field modulates it
rather than replacing it.

The friction field
------------------
Friction is sampled per FOOT from the terrain at that foot's own position, not
once for the robot. Crossing a patch boundary therefore puts one foot on ice
while the other is still on rock, which is the case that actually breaks a gait
and the whole reason per-foot friction exists in ice_randomize.

Bands are sinusoidal in world x with a per-episode phase, interpolating in log
space between the episode's rock friction (whatever ice_randomize drew) and
ICE_MU. Sinusoidal rather than sharp-edged deliberately: a step discontinuity in
friction is not differentiable and tends to produce solver chatter at the
boundary. Real ice has a rime/verglas margin anyway.

Known limitation
----------------
Friction is computed from foot positions in `state.data`, i.e. the END of the
previous control step, then applied to this step. One control step of lag at
50 Hz = 20 ms. Removing it would mean stepping physics to find the positions
before deciding the friction to step physics with. Not worth it; documented
instead.
"""

import contextlib
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

import wind

# Friction of the icy bands. The rock end of the interpolation is whatever
# ice_randomize drew for that env, so the contrast varies across the batch.
ICE_MU = 0.05


def default_config() -> config_dict.ConfigDict:
  config = wind.default_config()
  config.patch_config = config_dict.create(
      enable=True,
      # Spatial period of the bands, metres. ~1.5 m is a little over one stride,
      # so a foot can land mid-transition rather than always clearing a band.
      band_length=1.5,
      # Fraction of the band that is fully icy, 0-1. Higher = more ice.
      ice_fraction=0.5,
  )
  return config


class PatchyIceJoystick(wind.WindyJoystick):
  """Windy joystick whose ground friction varies with position, per foot."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if config is None:
      config = default_config()
    super().__init__(task, config, config_overrides)

  @contextlib.contextmanager
  def _with_model(self, model):
    """Swap in a derived model for one step, mirroring wrapper.v_env_fn."""
    old = self._mjx_model
    try:
      self._mjx_model = model
      yield
    finally:
      self._mjx_model = old

  def _patch_friction(self, data, phase: jax.Array) -> jax.Array:
    """Per-foot friction from the band field at each foot's own x position."""
    foot_x = data.site_xpos[self._feet_site_id][:, 0]  # (2,)
    cfg = self._config.patch_config

    wave = jp.sin(2 * jp.pi * foot_x / cfg.band_length + phase)
    # Map sin to [0,1], then bias by ice_fraction so the knob means what it says.
    iciness = jp.clip(0.5 * (1.0 + wave) + (cfg.ice_fraction - 0.5), 0.0, 1.0)

    # Interpolate in log space -- friction spans a decade, so a linear lerp
    # would spend almost all its range on the rock end.
    rock_mu = self.mjx_model.pair_friction[0:2, 0]  # this env's per-foot draw
    log_mu = jp.log(rock_mu) + iciness * (jp.log(ICE_MU) - jp.log(rock_mu))
    return jp.exp(log_mu)  # (2,)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    cfg = self._config.patch_config
    if not cfg.enable:
      state = super().reset(rng)
      state.info["patch_phase"] = jp.zeros(())
      state.info["foot_mu"] = self.mjx_model.pair_friction[0:2, 0]
      return state

    rng, phase_rng = jax.random.split(rng)
    state = super().reset(rng)
    phase = jax.random.uniform(phase_rng, maxval=2 * jp.pi)
    state.info["patch_phase"] = phase
    state.info["foot_mu"] = self._patch_friction(state.data, phase)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if not self._config.patch_config.enable:
      return super().step(state, action)

    mu = self._patch_friction(state.data, state.info["patch_phase"])
    # (2,1) broadcasts across both tangential columns, rows 0:2 = the foot pairs.
    pair_friction = self.mjx_model.pair_friction.at[0:2, 0:2].set(mu[:, None])
    model = self.mjx_model.tree_replace({"pair_friction": pair_friction})

    with self._with_model(model):
      state = super().step(state, action)

    state.info["foot_mu"] = mu
    return state


def load(task: str = "flat_terrain", **kwargs) -> PatchyIceJoystick:
  return PatchyIceJoystick(task=task, **kwargs)
