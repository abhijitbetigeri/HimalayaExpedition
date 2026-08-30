"""Add an onboard (robot's-eye) camera to the G1.

The shipped G1 XML has exactly one camera, `track`: a chase cam in `trackcom`
mode that follows the pelvis from outside. Useful for watching the robot, but it
is not what the robot sees, and on a mountain the onboard view is the one that
carries the story -- it is also the only view a real deployment would actually
have.

The G1 has no head body, so the camera mounts on `torso_link`, which is roughly
where a real G1's perception stack lives.

How it is injected
------------------
The robot XML is not a file we may edit: it lives in site-packages, and
bootstrap.sh reinstalls over it (see CLAUDE.md). It is also pulled in by
`<include>`, so a parent scene cannot reopen its bodies to add a child element.

But `G1Env` builds the model with `from_xml_string(text, assets=get_assets())`,
and that asset dict carries the robot XML *as bytes*. So we rewrite those bytes
in memory and hand back a patched dict. Nothing on disk changes.

`get_assets` is called by `G1Env.__init__` itself rather than passed in, so the
patch is applied as a context manager around construction and removed straight
after -- never left installed globally.
"""

import contextlib
import re

from mujoco_playground._src.locomotion.g1 import base as g1_base

CAMERA_NAME = "pov"

# Mounted on the torso, looking along +x (the robot's forward), tilted slightly
# down so the ground the feet are about to land on is in shot -- on a slope the
# horizon is useless and the next foot placement is everything.
#   pos:    0.12 m forward, 0.35 m up from the torso origin (chest height).
#   xyaxes: camera right = -y, camera up = +z tilted forward by ~15 deg.
# Pitch and fov chosen empirically, not analytically: at 15 deg down the frame
# was 4% ground, at 27 deg still 5%, and only a wider fov with ~27 deg of pitch
# put the slope and the line both in shot (30% ground) once the slope itself was
# fixed. Verified on the ascent scene, seed 0.
POV_CAMERA = (
    '<camera name="{name}" pos="0.12 0 0.35" '
    'xyaxes="0 -1 0 0.45 0 0.89" fovy="75"/>'
)


def _inject(xml: str, name: str = CAMERA_NAME) -> str:
    """Insert the camera as the first child of the torso_link body."""
    if f'name="{name}"' in xml:
        return xml
    m = re.search(r'(<body\s+name="torso_link"[^>]*>)', xml)
    if not m:
        raise RuntimeError(
            "torso_link body not found in the G1 XML -- the model changed and "
            "the POV camera needs a new mount point."
        )
    cam = POV_CAMERA.format(name=name)
    return xml[: m.end()] + "\n      " + cam + xml[m.end():]


def assets_with_pov(name: str = CAMERA_NAME, source=None):
    # `source` must be the ORIGINAL get_assets. Calling g1_base.get_assets here
    # recurses forever once the patch below is installed, since that is the very
    # name it replaces.
    assets = (source or g1_base.get_assets)()
    key = "g1_mjx_feetonly.xml"
    assets[key] = _inject(assets[key].decode(), name).encode()
    return assets


@contextlib.contextmanager
def pov_assets(name: str = CAMERA_NAME):
    """Patch g1_base.get_assets for the duration of env construction only."""
    original = g1_base.get_assets
    g1_base.get_assets = lambda: assets_with_pov(name, source=original)
    try:
        yield
    finally:
        g1_base.get_assets = original
