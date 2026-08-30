# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda12]", "mujoco", "playground", "mediapy"]
# ///
"""Smoke test for the Himalaya traction-stack project.

Proves, in one cheap GPU job, the five things every later run depends on:
  1. JAX sees a GPU
  2. mujoco_playground imports and the G1 env loads
  3. the shipped friction randomization really is U(0.4, 1.0)
  4. headless rendering works (needed for every demo video)
  5. the org bucket is mounted read+write at /mnt

Run:
  hf jobs uv run --namespace iteratehack --flavor l4x1 --timeout 20m \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=smoke-test \
      smoke_test.py
"""

import ctypes
import ctypes.util
import inspect
import os
import pathlib
import subprocess
import sys
import traceback

# The uv bookworm image ships no GL stack at all, and `import mujoco` walks the
# EGL path eagerly -- so without this every mujoco import dies, not just renders.
# We run as root in the job, so just install them.
print("=== 0. installing GL libraries ===", flush=True)
subprocess.run(
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0",
    shell=True,
    check=False,
)


def pick_gl_backend():
    """EGL renders on the GPU; OSMesa is the CPU fallback. Probe, don't assume."""
    if ctypes.util.find_library("EGL"):
        try:
            ctypes.CDLL("libEGL.so.1")
            return "egl"
        except OSError:
            pass
    if ctypes.util.find_library("OSMesa"):
        return "osmesa"
    return "none"


GL = pick_gl_backend()
print(f"GL backend: {GL}", flush=True)
os.environ["MUJOCO_GL"] = GL
os.environ["PYOPENGL_PLATFORM"] = GL

OUT = pathlib.Path("/mnt/himalaya-g1/smoke")
results = {}


def check(name):
    """Run a step, record pass/fail, never abort the rest of the test."""
    def deco(fn):
        print(f"\n=== {name} ===", flush=True)
        try:
            fn()
            results[name] = "PASS"
        except Exception:
            results[name] = "FAIL"
            traceback.print_exc()
        return fn
    return deco


@check("1. JAX sees a GPU")
def _():
    import jax
    print("jax", jax.__version__, "devices:", jax.devices())
    assert jax.devices()[0].platform == "gpu", "no GPU visible to JAX"


@check("2. G1 env loads")
def _():
    from mujoco_playground import locomotion
    envs = [e for e in locomotion.ALL_ENVS if e.startswith("G1")]
    print("G1 envs:", envs)
    env = locomotion.load("G1JoystickFlatTerrain")
    print("obs:", env.observation_size)
    print("act:", env.action_size)


@check("3. ice randomization is live and sane")
def _():
    """The stock range really is U(0.4, 1.0); ours must actually replace it.

    This used to assert minval=0.4 to document the gap. Now that ice_randomize
    exists, asserting the stock range would fail on every run -- so we assert
    the stock source still says what we think it says (it is the baseline our
    whole claim rests on) and then check our replacement behaves.
    """
    import jax
    import numpy as np
    from mujoco_playground._src.locomotion.g1 import randomize

    src = inspect.getsource(randomize.domain_randomize)
    assert "minval=0.4" in src, "stock range moved -- the 'never seen ice' claim needs rechecking"
    print("baseline confirmed: stock G1 is still U(0.4, 1.0)")

    # `hf jobs uv run` ships THIS FILE ONLY, so a plain import fails on the VM.
    # Keep a copy of ice_randomize.py in the mounted bucket and we find it there.
    try:
        import ice_randomize as ir
    except ModuleNotFoundError:
        sys.path.insert(0, "/mnt/himalaya-g1")
        import ice_randomize as ir  # noqa: F811
    print("ice_randomize from:", ir.__file__)

    env = ir.load("G1JoystickFlatTerrain")
    assert env._config.njmax >= 96, "njmax too low -- compliant contact will drop constraints"

    n = 4096
    model, _ = ir.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(0), n)
    )
    mu = np.asarray(model.pair_friction[:, 0:2, 0])
    assert mu.shape == (n, 2), f"expected per-foot friction, got {mu.shape}"
    assert mu.min() >= ir.ICE_MU_MIN - 1e-6 and mu.max() <= ir.ROCK_MU_MAX + 1e-6

    on_ice = (mu < 0.15).mean()
    below_stock = (mu < 0.4).mean()
    split = (np.maximum(*mu.T) / np.minimum(*mu.T) > 5).mean()
    print(f"feet on ice (mu<0.15)   : {on_ice:.1%}")
    print(f"feet below stock floor  : {below_stock:.1%}")
    print(f"envs with >5x split mu  : {split:.1%}")
    assert on_ice > 0.25, "log-uniform sampling is not putting enough mass on ice"
    assert split > 0.10, "per-foot asymmetry is not happening -- check the (2,1) shape"

    solimp = np.asarray(model.pair_solimp[:, 0:2, :])
    assert (solimp[..., 0] <= solimp[..., 1]).all(), "solimp dmin > dmax is invalid"

    stock_pairs = np.asarray(env.mjx_model.pair_friction)[2:, :]
    assert np.allclose(np.asarray(model.pair_friction[:, 2:, :]), stock_pairs), \
        "self-collision pairs were modified -- only rows 0:2 are foot-floor"
    print("-> per-foot ice friction + compliant ground active, stock pairs untouched")


@check("4. headless render")
def _():
    import mujoco
    import numpy as np
    from mujoco_playground import locomotion

    assert GL != "none", "no GL backend available - cannot render at all"
    env = locomotion.load("G1JoystickFlatTerrain")
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as r:
        r.update_scene(data)
        frame = r.render()
    print("rendered frame:", frame.shape, frame.dtype)
    assert np.asarray(frame).any(), "frame is entirely black"

    # Visual proof, and it pre-flights the exact path the demo videos will use.
    import mediapy
    OUT.mkdir(parents=True, exist_ok=True)
    mediapy.write_image(OUT / "render_check.png", frame)
    print("wrote", OUT / "render_check.png")


@check("5. bucket is mounted read+write")
def _():
    mnt = pathlib.Path("/mnt")
    assert mnt.is_dir(), "/mnt not mounted - did you pass -v hf://buckets/...?"
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "smoke_ok.txt"
    marker.write_text("himalaya traction stack: smoke test reached the bucket\n")
    print("wrote", marker, "->", marker.read_text().strip())
    print("bucket top level:", sorted(p.name for p in mnt.iterdir())[:10])


print("\n" + "=" * 40)
for name, status in results.items():
    print(f"{status:4}  {name}")
print("=" * 40, flush=True)

sys.exit(1 if "FAIL" in results.values() else 0)
