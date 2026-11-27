"""Collision primitives for the G1 links the policy's USD leaves without any.

`usd/g1_isaac.usd` carries collision geometry on the two ankle-roll links (a sole
box each) and on the torso (torso and head hulls): 4 of 53 shapes. Everything
else is visual only, so a robot that falls sinks into the ground until its torso
catches it, and two robots pass through each other everywhere but feet and
torso.

`add_body_collisions` adds capsules and boxes on the pelvis, thighs, shins, upper
arms, forearms, wrists, palms and fingers. Sizes come from the visual meshes:
`check_fit()` measures how far each primitive pokes out of the convex hull of its
link's visual mesh (a few millimetres at most). The head is not in the table
because the USD's own head hull on the torso link already covers it.

Three things keep an upright robot's walk bit-for-bit what it was:

- density 0, so no link's mass, centre of mass or inertia moves;
- every new shape is filtered against every other colliding shape of the same
  robot, which is what `add_usd(enable_self_collisions=False)` does for the
  shapes it imports. Arms swing through thighs exactly as before;
- nothing reaches below the sole while a foot is on the ground (the lower shin
  capsule stops 0.06 m above it).

So the new shapes touch only the ground, once the robot is down, and other
robots. The shapes are collision-only (no `ShapeFlags.VISIBLE`), like the USD's
own colliders under `hide_collision_shapes=True`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton import ShapeFlags

ROBOT_MU = 0.75

# Left side and centre line, in the link's own frame [m]. Right-side links get
# the mirror image in y. Capsules: (link, name, "capsule", end a, end b, radius).
# Boxes: (link, name, "box", centre, half extents).
_SHAPES = [
    ("pelvis", "pelvis", "box", (0.0, 0.0, -0.082), (0.040, 0.056, 0.048)),
    ("left_hip_yaw_link", "thigh", "capsule", (-0.006, -0.001, -0.038), (-0.076, -0.005, -0.165), 0.034),
    ("left_knee_link", "shin", "capsule", (0.012, 0.002, -0.025), (0.009, 0.0, -0.180), 0.026),
    ("left_knee_link", "shin_low", "capsule", (0.009, 0.0, -0.205), (0.009, 0.0, -0.272), 0.020),
    ("left_shoulder_roll_link", "upper_arm_a", "capsule", (0.0, 0.005, 0.000), (0.0, 0.006, -0.062), 0.030),
    ("left_shoulder_yaw_link", "upper_arm_b", "capsule", (0.002, 0.0, -0.005), (0.010, -0.009, -0.082), 0.030),
    ("left_elbow_link", "forearm", "capsule", (0.048, 0.002, -0.010), (0.072, 0.002, -0.010), 0.026),
    ("left_wrist_pitch_link", "wrist", "capsule", (-0.006, 0.0, 0.0), (0.052, 0.0, 0.0), 0.023),
    ("left_wrist_yaw_link", "palm", "box", (0.095, 0.0035, 0.0), (0.026, 0.015, 0.035)),
    ("left_hand_index_0_link", "finger", "capsule", (0.012, -0.001, 0.0), (0.043, -0.001, 0.0), 0.010),
    ("left_hand_index_1_link", "fingertip", "capsule", (0.004, 0.0, 0.0), (0.040, 0.0, 0.0), 0.007),
    ("left_hand_middle_0_link", "finger", "capsule", (0.012, -0.001, 0.0), (0.043, -0.001, 0.0), 0.010),
    ("left_hand_middle_1_link", "fingertip", "capsule", (0.004, 0.0, 0.0), (0.040, 0.0, 0.0), 0.007),
    ("left_hand_thumb_1_link", "thumb", "capsule", (0.001, -0.014, 0.0), (0.001, -0.043, 0.0), 0.010),
    ("left_hand_thumb_2_link", "thumbtip", "capsule", (0.0, -0.004, 0.0), (0.0, -0.040, 0.0), 0.007),
]


def _mirror(v):
    return (v[0], -v[1], v[2])


def shape_table() -> list[tuple]:
    """Both sides spelled out: (link, name, kind, a, b_or_half, radius_or_None)."""
    rows = []
    for link, name, kind, a, b, *rest in _SHAPES:
        radius = rest[0] if rest else None
        rows.append((link, name, kind, a, b, radius))
        if link.startswith("left_"):
            # A box's half extents are unsigned, so only its centre mirrors.
            rows.append(("right_" + link[5:], name, kind, _mirror(a), b if kind == "box" else _mirror(b), radius))
    return rows


def add_body_collisions(builder: newton.ModelBuilder, mu: float = ROBOT_MU) -> list[int]:
    """Add collision-only primitives to a single-robot builder. Returns the new shape indices.

    Call it after `add_usd(..., enable_self_collisions=False)` and before the
    robot is copied into a fleet with `add_builder`, which carries the filter
    pairs along. Mass properties are untouched; that is asserted.
    """
    body_of = {label.split("/")[-1]: i for i, label in enumerate(builder.body_label)}
    mass_before = (list(builder.body_mass), [np.array(c) for c in builder.body_com], [np.array(m) for m in builder.body_inertia])

    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0  # geometry only: the links keep the USD's mass and inertia
    cfg.mu = mu
    cfg.is_visible = False
    cfg.has_shape_collision = True

    existing = [s for s in range(builder.shape_count) if builder.shape_flags[s] & ShapeFlags.COLLIDE_SHAPES]
    new = []
    for link, name, kind, a, b, radius in shape_table():
        body = body_of[link]
        label = f"{builder.body_label[body]}/collisions/body_{name}/{kind}"
        if kind == "box":
            s = builder.add_shape_box(body, xform=wp.transform(wp.vec3(*a)), hx=b[0], hy=b[1], hz=b[2], cfg=cfg, label=label)
        else:
            a3, b3 = np.array(a), np.array(b)
            axis = b3 - a3
            length = float(np.linalg.norm(axis))
            rot = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(axis / length)))  # capsules lie along local z
            s = builder.add_shape_capsule(
                body, xform=wp.transform(wp.vec3(*(0.5 * (a3 + b3))), rot), radius=radius, half_height=0.5 * length,
                cfg=cfg, label=label,
            )  # fmt: skip
        new.append(s)

    # Self-collision stays off: filter each new shape against every other colliding
    # shape of the robot, old and new. Same-body pairs are filtered too, including
    # the USD's own torso and head hulls. They can never collide, but with every
    # pair inside a robot filtered SolverMuJoCo needs one contype bit per robot:
    # 8 of MuJoCo's 32 for four robots, snow and three ice patches, where the
    # model without these shapes uses 17.
    filtered = set(builder.shape_collision_filter_pairs)
    for i, s in enumerate(new):
        for t in existing + new[:i]:
            builder.add_shape_collision_filter_pair(s, t)
    for i, s in enumerate(existing):
        for t in existing[:i]:
            if builder.shape_body[s] == builder.shape_body[t] and (t, s) not in filtered:
                builder.add_shape_collision_filter_pair(s, t)

    for i in range(builder.body_count):
        if (
            builder.body_mass[i] != mass_before[0][i]
            or not np.array_equal(np.array(builder.body_com[i]), mass_before[1][i])
            or not np.array_equal(np.array(builder.body_inertia[i]), mass_before[2][i])
        ):
            raise RuntimeError(f"collision shapes changed the mass properties of {builder.body_label[i]}")
    return new


def _surface_points(kind, a, b, radius, n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "box":
        p = rng.uniform(-1.0, 1.0, (n, 3))
        face = rng.integers(0, 3, n)
        p[np.arange(n), face] = np.sign(p[np.arange(n), face])
        return np.array(a) + p * np.array(b)
    a3, b3 = np.array(a), np.array(b)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return a3 + rng.uniform(0.0, 1.0, (n, 1)) * (b3 - a3) + radius * d  # on or inside the surface; the hull test is one-sided


def check_fit(builder: newton.ModelBuilder) -> list[dict]:
    """How far each primitive pokes out of the convex hull of its link's visual mesh [m]."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    body_of = {label.split("/")[-1]: i for i, label in enumerate(builder.body_label)}
    out = []
    for link, name, kind, a, b, radius in shape_table():
        verts = []
        for s in builder.body_shapes[body_of[link]]:
            src = builder.shape_source[s]
            if not (builder.shape_flags[s] & ShapeFlags.VISIBLE) or src is None or not hasattr(src, "vertices"):
                continue
            v = np.asarray(src.vertices) * np.array(builder.shape_scale[s])
            t = builder.shape_transform[s]
            u, w = np.array(t[3:6]), float(t[6])
            verts.append(v + 2.0 * np.cross(u, np.cross(u, v) + w * v) + np.array(t[:3]))
        hull = ConvexHull(np.concatenate(verts))
        pts = _surface_points(kind, a, b, radius)
        outside = (pts @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
        out.append({"link": link, "shape": name, "kind": kind, "max_outside_mm": round(1000 * float(max(outside.max(), 0.0)), 1)})
    return out
