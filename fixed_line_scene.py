"""Scene generation for fixed-line ascent: a slope, a taut rope, an ascender.

Why the rope is a static geom
-----------------------------
A fixed line is anchored top and bottom and kept taut -- that is the whole point
of fixing it. A taut line is, to the accuracy anything here needs, a straight
rigid segment. Simulating it as a rope (a chain of capsules, or MuJoCo's cable
composite) would add hundreds of DOF, wreck the MJX batching we depend on for
8192 parallel envs, and buy nothing: we are not studying rope dynamics, we are
studying whether the robot can climb one.

Why the ascender is a joint, not a grasp
----------------------------------------
The G1 in `feetonly` has 29 actuators and NO fingers. There is a `with_hands`
variant (43 actuators, 14 finger joints) but grasping a thin rope is a
contact-rich manipulation problem that would dominate the learning and is not
what fixed-line ascent is about anyway.

Real climbers do not grip the rope. They clip a mechanical ascender (a jumar)
to it: a cammed device that slides freely upward and locks under downward load.
So the honest model is a 1-DOF slider on the rope with a ratchet, tethered to
the robot -- not a hand closing on a cylinder. That is what this builds.

A rigid `connect` equality was tried first and rejected. It pins the pelvis to a
1-DOF slider on the line, so the robot hangs from a rail: measured tether load
was 327 N against a 330 N robot, i.e. the rope carried everything and the legs
did nothing. That is not ascent, it is a zip line.

A real tether has SLACK. It pulls only when it comes taut, and the climber's
weight rides on their feet the rest of the time. So the tether here is a
one-sided spring applied through `xfrc_applied` in the env, not a MuJoCo
constraint -- the same mechanism wind.py already uses. That keeps the model at
exactly the stock 36 DOF (no extra joints, no equalities), which means the
policy architecture stays compatible with everything else in this repo and no
MJX constraint-support questions arise at all.

The ascender is therefore a mocap body: a visual marker the env drives to the
ratchet position. It carries no DOF.
"""

import math

from mujoco_playground._src.locomotion.g1 import base as g1_base

# Rope runs up the fall line, offset from the slope surface by roughly the
# distance a clipped-in climber's harness sits from the rock.
ROPE_OFFSET = 0.9
ROPE_RADIUS = 0.012  # 12 mm, a realistic fixed line
ROPE_LENGTH = 12.0


def build_scene_xml(slope_deg: float = 30.0) -> str:
    """Flat-terrain G1 scene with the floor tilted and a fixed line added."""
    a = math.radians(slope_deg)
    # Slope rises along +x, tilted about the y axis; surface passes through the
    # origin, so surface height at x=0 is 0.
    ux, uz = math.cos(a), math.sin(a)  # unit vector up the fall line

    # The robot stands upright (gravity-aligned, not slope-aligned) at x=0, so
    # its pelvis sits at the usual standing height above the surface there.
    start_z = 0.785
    start_z_bent = 0.755

    # The line must pass THROUGH the start pelvis position. Anchoring it
    # anywhere else means the rigid tether has a large violation at t=0 and
    # yanks the robot off its feet before the episode begins -- which is
    # exactly what an earlier offset-from-the-rock version did (340 N of
    # constraint force at reset, i.e. the robot hanging rather than standing).
    x0, z0 = 0.0, start_z
    x1, z1 = x0 + ux * ROPE_LENGTH, z0 + uz * ROPE_LENGTH

    return f"""<mujoco model="g1 fixed line ascent">
  <include file="g1_mjx_feetonly.xml"/>

  <statistic center="2 0 2" extent="6" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="140" elevation="-20"/>
    <map force="0.01"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".7 .8 .9" rgb2="1 1 1"
      width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1=".9 .92 .95" rgb2=".8 .84 .88" markrgb=".3 .3 .3"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="8 8" reflectance="0.1"/>
    <material name="ropemat" rgba=".85 .3 .2 1"/>
    <material name="ascmat" rgba=".2 .5 .9 1"/>
  </asset>

  <worldbody>
    <!-- The slope. euler y rotates the plane normal off vertical. -->
    <geom name="floor" type="plane" size="0 0 0.01" material="groundplane"
      euler="0 {-slope_deg} 0"/>

    <!-- Fixed line: taut, anchored, therefore a static capsule. -->
    <geom name="rope" type="capsule" material="ropemat"
      fromto="{x0:.4f} 0 {z0:.4f} {x1:.4f} 0 {z1:.4f}" size="{ROPE_RADIUS}"
      contype="0" conaffinity="0"/>

    <!-- Ascender: a mocap marker the env drives to the ratchet point. No DOF,
         no constraint -- the tether force is applied in step(). -->
    <body name="ascender" mocap="true" pos="{x0:.4f} 0 {z0:.4f}">
      <geom type="box" size="0.04 0.03 0.06" material="ascmat"
        contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <include file="sensor.xml"/>

  <keyframe>
    <key name="home"
      qpos="
      0 0 {start_z:.4f}
      1 0 0 0
      -0.1 0 0 0.3 -0.2 0
      -0.1 0 0 0.3 -0.2 0
      0 0 0
      0.2 0.2 0 1.28 0 0 0
      0.2 -0.2 0 1.28 0 0 0
      "
      ctrl="
      -0.1 0 0 0.3 -0.2 0
      -0.1 0 0 0.3 -0.2 0
      0 0 0
      0.2 0.2 0 1.28 0 0 0
      0.2 -0.2 0 1.28 0 0 0
      "/>
    <!-- Playground's _post_init requires this keyframe by name; it is the
         reset pose every G1 env starts from. -->
    <key name="knees_bent"
      qpos="
      0 0 {start_z_bent:.4f}
      1 0 0 0
      -0.312 0 0 0.669 -0.363 0
      -0.312 0 0 0.669 -0.363 0
      0 0 0.073
      0.2 0.2 0 0.6 0 0 0
      0.2 -0.2 0 0.6 0 0 0
      "
      ctrl="
      -0.312 0 0 0.669 -0.363 0
      -0.312 0 0 0.669 -0.363 0
      0 0 0.073
      0.2 0.2 0 0.6 0 0 0
      0.2 -0.2 0 0.6 0 0 0
    "/>
  </keyframe>
</mujoco>
"""


def get_assets():
    return g1_base.get_assets()
