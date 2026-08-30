"""Himalaya domain randomization for the Playground G1.

Drop-in replacement for
`mujoco_playground._src.locomotion.g1.randomize.domain_randomize` -- pass it as
`randomization_fn=` to brax PPO train. Kept as a project file rather than a
patch to site-packages so it survives the clean install bootstrap.sh does on
the VM.

Two deltas vs stock; everything else is copied verbatim so the only thing that
changes between runs is what we meant to change.

  friction  U(0.4, 1.0), one scalar shared by both feet
        ->  logU(0.05, 1.0), sampled independently per foot

      The stock floor of 0.4 is roughly an order of magnitude above ice, so the
      shipped policy has no training signal for the surface this whole project
      is about. Log-uniform because plain U(0.05, 1.0) puts ~95% of samples
      above 0.05 -- you would barely train on ice at all. Per-foot because the
      case that actually breaks a gait is one foot on rock and one on ice, not
      both on the same surface.

  solref/solimp   fixed hard contact
              ->  randomized softness, so the foot sinks instead of bouncing

      This is postholing in soft snow and breakable crust. Cheap: the contact
      solver parameters are Model fields, so they randomize in exactly the same
      place as friction, with no terrain work at all.

REQUIRES njmax >= ~96. Softer contact lets the feet penetrate further, which
raises the number of simultaneous constraint rows past the stock njmax of 90
(= 29*2 + 8*4). Verified: the stock randomizer never overflows, this one does,
and MuJoCo asks for 93. An overflow silently DROPS constraints, so this is a
wrong-physics bug, not a warning to ignore. Use `load()` below, or pass
`config_overrides=CONFIG_OVERRIDES` yourself.

Not modeled here, deliberately: mid-episode rock->ice transitions (pair_friction
lives on Model, not Data, so it needs the model rebuilt inside step() -- unproven
against brax's randomization wrapper), and wind (goes in the env's step() as
xfrc_applied, not here).
"""

import functools
import os

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16

# Rows 0 and 1 of the pair arrays are left_foot_floor and right_foot_floor.
# Rows 2-4 are the condim=1 self-collision pairs and must be left alone.
FEET = slice(0, 2)

# Tangential friction. 0.05 is bare ice; 1.0 is dry rock. Verified stock range
# is U(0.4, 1.0) -- see smoke_test.py check 3.
ICE_MU_MIN = float(os.environ.get("HIMALAYA_MU_MIN", "0.05"))
ROCK_MU_MAX = float(os.environ.get("HIMALAYA_MU_MAX", "1.0"))
# Overridable so an ablation can hold friction at the STOCK U(0.4, 1.0) range while
# varying one other factor. Defaults are unchanged.

# Sampling distribution over [ICE_MU_MIN, ROCK_MU_MAX]. Default stays "logu" so
# nothing that already depends on this module changes behaviour.
#
# Measured 2026-08-29, three seeds, existing checkpoints:
#   base-v2 (stock randomizer)  301/301 on seeds 0-1, but 196 on seed 2 -- the
#                               "every seed" version of this claim was wrong
#   ice-v2  (logu randomizer)   fails at 26-47/300 even on BENIGN ground
# ice-v2 did not learn a worse gait, it never learned to WALK. Meanwhile the Isaac
# arm, using plain uniform over the same range with nothing else enabled, hit 98%
# survival. Same range, different distribution, opposite outcome.
#
# So logu is right about exposure (37% of feet at ice-grade) and wrong about
# learnability: without enough easy ground early, walking never bootstraps and
# there is no gait for ice to degrade. "uniform" puts ~5% below mu=0.15, which is
# what the Isaac run had.
MU_DIST = os.environ.get("HIMALAYA_MU_DIST", "logu")  # "logu" | "uniform"

# Compliant ground is bundled with the friction change in this module, so the two
# can only be tested together -- which is exactly the mistake that has made every
# ice run so far uninterpretable. base-v2 (stock friction, hard ground, no wind,
# no patches) WALKS; ice-uniform (wide friction + compliant + wind + patches)
# fails even on stock flat ground. Four differences, never tested apart.
#
# Set HIMALAYA_COMPLIANT=0 to keep stock hard contact while still widening
# friction, so the friction change can be isolated. Default 1 = existing behaviour.
COMPLIANT = os.environ.get("HIMALAYA_COMPLIANT", "1") != "0"

# Contact solver softness. Stock solref is [0.02, 1.0] = (timeconst, dampratio).
# Raising timeconst makes contact slower and squishier. Floor stays at the stock
# 0.02; MuJoCo wants timeconst >= 2*sim_dt and sim_dt is 0.002, so this is safe.
SOLREF_TIMECONST = (0.02, 0.12)

# Stock solimp is [dmin, dmax, width, mid, power] = [0.9, 0.95, ...]. Lowering
# dmax lets the foot penetrate further before contact goes fully rigid, which is
# the part that actually reads as "sinking into snow" on video. dmin is carried
# down with it since MuJoCo requires dmin <= dmax.
SOLIMP_DMAX = (0.75, 0.95)
SOLIMP_DMIN_OFFSET = 0.05


def make_domain_randomize(mu_min: float = ICE_MU_MIN):
  """Randomizer with a configurable friction floor, for curriculum staging.

  Measured: with the floor at 0.05 from step zero, 37% of feet land on
  ice-grade friction and the policy never learns to walk AT ALL -- it scored
  43/501 on plain flat ground, against 455 for a policy trained on the stock
  range. It failed the prerequisite, not the hard case. Annealing the floor down
  across stages lets it learn walking first and ice second.
  """

  def domain_randomize(model: mjx.Model, rng: jax.Array):
    return _domain_randomize(model, rng, mu_min)

  return domain_randomize


def domain_randomize(model: mjx.Model, rng: jax.Array):
  return _domain_randomize(model, rng, ICE_MU_MIN)


def _domain_randomize(model: mjx.Model, rng: jax.Array, mu_min: float):
  @jax.vmap
  def rand_dynamics(rng):
    # CHANGED: per-foot friction spanning ice to rock. See MU_DIST above for why
    # the distribution -- not just the range -- decides whether this trains at all.
    rng, key = jax.random.split(rng)
    if MU_DIST == "uniform":
        friction = jax.random.uniform(
            key, shape=(2, 1), minval=ICE_MU_MIN, maxval=ROCK_MU_MAX
        )
    else:
        friction = jp.exp(
            jax.random.uniform(
                key,
                shape=(2, 1),
                minval=jp.log(mu_min),
                maxval=jp.log(ROCK_MU_MAX),
            )
        )
    # (2, 1) broadcasts across the two tangential friction columns.
    pair_friction = model.pair_friction.at[FEET, 0:2].set(friction)

    # NEW: compliant ground. Both feet share one softness -- snow depth is a
    # property of the terrain, unlike friction, which really can differ per foot.
    # Skipped entirely when COMPLIANT is off, so friction can be isolated.
    # NOTE: the rng splits still happen either way, so a given seed draws the same
    # friction whether or not compliance is on -- otherwise the two arms would
    # differ by their random stream as well as by the physics, and the ablation
    # would prove nothing.
    rng, key = jax.random.split(rng)
    timeconst = jax.random.uniform(
        key, minval=SOLREF_TIMECONST[0], maxval=SOLREF_TIMECONST[1]
    )
    rng, key2 = jax.random.split(rng)
    dmax = jax.random.uniform(
        key2, minval=SOLIMP_DMAX[0], maxval=SOLIMP_DMAX[1]
    )
    if COMPLIANT:
        pair_solref = model.pair_solref.at[FEET, 0].set(timeconst)
        pair_solimp = model.pair_solimp.at[FEET, 0].set(dmax - SOLIMP_DMIN_OFFSET)
        pair_solimp = pair_solimp.at[FEET, 1].set(dmax)
    else:
        pair_solref = model.pair_solref
        pair_solimp = model.pair_solimp

    # --- everything below is stock, unchanged ---

    # Scale static friction: *U(0.5, 2.0).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(29,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05)
    )

    return (
        pair_friction,
        pair_solref,
        pair_solimp,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )

  (
      pair_friction,
      pair_solref,
      pair_solimp,
      frictionloss,
      armature,
      body_mass,
      qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "pair_solref": 0,
      "pair_solimp": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "pair_friction": pair_friction,
      "pair_solref": pair_solref,
      "pair_solimp": pair_solimp,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  return model, in_axes


# Stock njmax is 90 and overflows under compliant contact; 160 leaves headroom
# for the rough-terrain hfield too.
CONFIG_OVERRIDES = {"njmax": 160}


def load(task: str = "G1JoystickFlatTerrain", **kwargs):
  """locomotion.load() with the njmax bump this randomizer needs."""
  from mujoco_playground import locomotion

  overrides = dict(CONFIG_OVERRIDES)
  overrides.update(kwargs.pop("config_overrides", {}))
  return locomotion.load(task, config_overrides=overrides, **kwargs)


def randomizer(rng: jax.Array):
  """Bind an rng batch, ready to pass as brax's `randomization_fn`."""
  return functools.partial(domain_randomize, rng=rng)
