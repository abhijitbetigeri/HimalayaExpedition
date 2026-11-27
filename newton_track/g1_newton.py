"""Unitree G1 on Himalayan ground, in Newton.

A third, independent stack for the project's friction thesis. The MuJoCo
Playground track (`train_ice.py`) and the Isaac Lab track (`train_ice_isaac.py`)
each train their own policy. This one trains nothing: it takes the G1 29-DoF
walking policy that ships with Newton (trained by the Newton developers in
Isaac Lab on MuJoCo-Warp, flat grippy ground) and asks what ice, wind and slope
do to a policy we had no hand in.

Runs in its own environment, NOT the repo's `.venv` (see NEWTON_TRACK.md):

    ~/projects/newton-sandbox/.venv/bin/python newton_track/ice_sweep.py

Three things here are deliberate and easy to get wrong.

**Ground friction wins the contact.** MuJoCo combines the two geoms' friction
with `max`, so a 0.75 boot on 0.05 ice would be simulated as 0.75 -- no ice at
all. That is the same class of error as PhysX's `average` combine mode in
SIM_ACCURACY.md item 1. The ground gets `geom_priority=1`, which makes MuJoCo
take the ground's coefficient outright. `calibrate_friction.py` measures the
effective coefficient with a sliding block rather than trusting this comment.

**Slope is a tilted gravity vector, not a tilted plane.** For an infinite flat
plane the two are the same physics, and it avoids the radians trap that put an
81 degree wall in `fixed_line_scene.py`. The policy's projected-gravity
observation is computed from the true (tilted) gravity direction, as an IMU
would report it.

**Wind is a force on the torso the policy cannot observe**, as in `wind.py`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import warp as wp
import yaml

import newton
import newton.utils
from newton import JointTargetMode

wp.config.quiet = True

# Shapes of the two ankle-roll links are the only parts that should touch the
# ground while walking.
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_LINK = "torso_link"

POLICY_HZ = 50
SIM_HZ = 200
DECIMATION = SIM_HZ // POLICY_HZ

# Fall = pelvis below this height (stands at ~0.76 m) or tilted past 60 degrees.
# Friction cone used by BOTH the robot and calibrate_friction.py. MuJoCo's default
# pyramidal cone is anisotropic: measured with the sliding block, nominal 0.20
# delivers 0.187 along a contact axis and 0.143 on the diagonal. Elliptic
# delivers 0.200 and 0.202. On ice that 30 % matters, so elliptic it is;
# `ice_sweep.py --cone pyramidal` reruns under the cone the policy trained on.
CONE = "elliptic"

FALL_HEIGHT = 0.40
FALL_COS_TILT = 0.5


@dataclasses.dataclass
class Conditions:
    """One physical condition. Everything defaults to the policy's home turf."""

    mu_ground: float = 0.75
    slope_deg: float = 0.0  # uphill along +command direction
    wind_n: float = 0.0  # steady force on the torso, newtons
    wind_dir_deg: float = 180.0  # 180 = headwind, 90 = from the robot's right
    gust_n: float = 0.0  # std of an AR(1) gust added to wind_n
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)  # vx, vy, yaw rate

    def label(self) -> str:
        parts = [f"mu={self.mu_ground:g}"]
        if self.slope_deg:
            parts.append(f"slope={self.slope_deg:g}deg")
        if self.wind_n or self.gust_n:
            parts.append(f"wind={self.wind_n:g}N")
        return " ".join(parts)


@wp.kernel
def _obs_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_pos_initial: wp.array[float],
    gravity_dir: wp.vec3,
    command: wp.vec3,
    prev_act: wp.array2d[float],
    num_dofs: int,
    obs: wp.array2d[float],
):
    # Same layout as newton.examples.robot_policy: 12 + 3 * num_dofs.
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])
    vel_b = wp.quat_rotate_inv(q, wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2]))
    avel_b = wp.quat_rotate_inv(q, wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5]))
    grav_b = wp.quat_rotate_inv(q, gravity_dir)
    for a in range(3):
        obs[0, a] = vel_b[a]
        obs[0, 3 + a] = avel_b[a]
        obs[0, 6 + a] = grav_b[a]
        obs[0, 9 + a] = command[a]
    for k in range(num_dofs):
        obs[0, 12 + k] = joint_q[7 + k] - joint_pos_initial[k]
        obs[0, 12 + num_dofs + k] = joint_qd[6 + k]
        obs[0, 12 + 2 * num_dofs + k] = prev_act[0, k]


@wp.kernel
def _target_kernel(
    act: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    action_scale: float,
    out: wp.array[float],
):
    i = wp.tid()
    if i < 7:
        out[i] = 0.0
    else:
        out[i] = joint_pos_initial[i - 7] + action_scale * act[0, i - 7]


def gravity_vector(slope_deg: float, heading: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Gravity for a slope that rises along `heading` (unit xy), magnitude 9.81.

    Tilting gravity by theta toward -heading is the same physics as walking
    along +heading up a plane inclined by theta.
    """
    th = math.radians(slope_deg)
    hx, hy = heading
    return 9.81 * np.array([-hx * math.sin(th), -hy * math.sin(th), -math.cos(th)])


def paint_g1(builder: newton.ModelBuilder, asset_dir) -> int:
    """Give the G1 its real two-tone finish. Rendering only; returns shapes painted.

    The USD the policy runs on carries no materials, so every part imports at
    the same 0.7 grey and the viewer's default half-gloss turns the robot into a
    chrome mannequin. Unitree's own MJCF in the same asset folder states the
    finish per part: dark (0.2) pelvis, hip-pitch housings, feet, head and logo,
    light (0.7) shells everywhere else. Colours are read from that file rather
    than typed in, and the meshes are made matte.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.parse(asset_dir / "mjcf" / "g1_29dof_with_hand_rev_1_0.xml").getroot()
    rgb = {}
    for geom in root.iter("geom"):
        if geom.get("mesh") and geom.get("rgba") and geom.get("group") == "1":  # group 1 = visual
            rgb[geom.get("mesh")] = tuple(float(v) for v in geom.get("rgba").split()[:3])
    painted = 0
    for shape in range(builder.shape_count):
        link = builder.shape_label[shape].split("/")[-2].replace("_rev_1_0", "")
        if link not in rgb:
            continue
        c = rgb[link]
        # Unitree's two values are a flat 0.2 and 0.7; nudge them to the graphite
        # and warm off-white of the real shells so the parts separate on screen.
        builder.shape_color[shape] = (0.10, 0.11, 0.12) if c[0] < 0.5 else (0.84, 0.85, 0.86)
        src = builder.shape_source[shape]
        if src is not None and hasattr(src, "roughness"):
            src.roughness = 0.75 if c[0] < 0.5 else 0.55
            src.metallic = 0.0
        painted += 1
    return painted


def add_ground(builder: newton.ModelBuilder, mu: float) -> int:
    """Ground plane whose friction wins every contact (see module docstring)."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, ke=5.0e4, kd=5.0e2, kf=1.0e3)
    return builder.add_shape_plane(
        body=-1,
        xform=wp.transform_identity(),
        width=0.0,
        length=0.0,
        cfg=cfg,
        label="ground",
        # Rendering only: grey rock above the fall threshold, blue ice below it.
        color=(0.62, 0.78, 0.90) if mu < 0.2 else (0.55, 0.53, 0.50),
        custom_attributes={"mujoco:geom_priority": 1},
    )


class G1Himalaya:
    """The Newton-shipped G1 29-DoF policy under one `Conditions`.

    `step()` advances one policy tick (20 ms). `rollout()` runs a seeded episode
    and returns survival and tracking numbers.
    """

    def __init__(
        self,
        cond: Conditions,
        *,
        rigid: str = "mujoco",
        use_mujoco_cpu: bool | None = None,
        cone: str = CONE,
        builder_hook=None,
    ):
        if rigid not in ("mujoco", "steel"):
            raise ValueError(rigid)
        self.cond = cond
        self.rigid = rigid
        self.device = wp.get_device()
        if use_mujoco_cpu is None:
            # mujoco_warp has no CPU fast path (CLAUDE.md, the `impl` finding).
            use_mujoco_cpu = self.device.is_cpu
        self.use_mujoco_cpu = use_mujoco_cpu

        asset_dir = newton.utils.download_asset("unitree_g1")
        with open(asset_dir / "rl_policies" / "g1_29dof.yaml", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.num_dofs = int(self.config["num_dofs"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.rigid_gap = 0.0
        builder.add_usd(
            str(asset_dir / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")
        paint_g1(builder, asset_dir)
        if rigid == "steel":
            # Steel takes contacts from Newton's own pipeline, which AVERAGES the
            # two shapes' friction and has no priority or combine-mode setting
            # (measured: 0.75 block on 0.05 ground slides at 0.40). The only way
            # to get a pair coefficient of mu is to give both sides mu, so the
            # robot's shapes take the ground value. Self-collision is off, so
            # this changes nothing but robot-ground contact.
            for shape in range(builder.shape_count):
                builder.shape_material_mu[shape] = cond.mu_ground
        self.ground_shape = add_ground(builder, cond.mu_ground)

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]  # facing +y
        builder.joint_q[7:] = self.config["mjw_joint_pos"]
        for i in range(self.num_dofs):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        def body_index(name: str) -> int:
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith("/" + name))

        self.pelvis = 0
        self.torso = body_index(TORSO_LINK)
        self.feet = [body_index(n) for n in FOOT_LINKS]

        if builder_hook is not None:
            builder_hook(builder, self)

        self.model = builder.finalize()
        self.heading = (0.0, 1.0)  # the robot faces +y at reset
        self.gravity = gravity_vector(cond.slope_deg, self.heading)
        self.model.set_gravity(tuple(float(g) for g in self.gravity))
        self._gravity_dir = wp.vec3(*(self.gravity / np.linalg.norm(self.gravity)))

        self.cone = cone
        self.sim_dt = 1.0 / SIM_HZ
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_q = wp.zeros(self.num_dofs + 7, dtype=wp.float32, device=self.device)
        self._make_solver()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._q0 = self.state_0.joint_q.numpy().copy()
        self._qd0 = self.state_0.joint_qd.numpy().copy()
        self._joint_pos_initial = wp.array(self._q0[7:], dtype=wp.float32, device=self.device)
        self._q_start = self._slope_stance() if cond.slope_deg else self._q0

        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415

        policy_path = str(asset_dir / "rl_policies" / "mjw_g1_29DOF.onnx")
        self.policy = OnnxRuntime(policy_path, device=self.device)
        self._in = self.policy.inputs[0].name
        self._out = self.policy.outputs[0].name
        self._obs = wp.zeros((1, 12 + 3 * self.num_dofs), dtype=wp.float32, device=self.device)
        self._prev_act = wp.zeros((1, self.num_dofs), dtype=wp.float32, device=self.device)
        self.policy.prepare({self._in: self._obs})

        self._gust = 0.0
        self._rng = np.random.default_rng(0)

    def _make_solver(self):
        """Rigid solver and contact source. Overridden by the snow track."""
        if self.rigid == "steel":
            self.solver = newton.solvers.SolverSteel(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
            self.contacts = self.pipeline.contacts()
            return
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco_cpu,
            solver="newton",
            cone=self.cone,
            nconmax=30,
            # 100 in the stock example; raised for the same reason CLAUDE.md
            # raises Playground's: an overflow drops constraints silently.
            njmax=160,
        )
        if self.use_mujoco_cpu:
            self.contacts = None
        else:
            self.pipeline = newton.CollisionPipeline(
                self.model, rigid_contact_max=self.solver.get_max_contact_count(), soft_contact_max=0
            )
            self.contacts = self.pipeline.contacts()

    # -- episode control ---------------------------------------------------

    def _foot_frames(self, q: np.ndarray) -> tuple[float, float]:
        """(mean foot-link height, mean cos of the foot's tilt from the ground normal) at pose q."""
        self.state_1.joint_q.assign(q)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        bq = self.state_1.body_q.numpy()
        zs, cs = [], []
        for i in self.feet:
            zs.append(float(bq[i][2]))
            up = wp.quat_rotate(wp.quat(*[float(v) for v in bq[i][3:7]]), wp.vec3(0.0, 0.0, 1.0))
            cs.append(float(up[2]))
        return float(np.mean(zs)), float(np.mean(cs))

    def _slope_stance(self) -> np.ndarray:
        """Start pose for a slope: body along true vertical, feet flat on the ground.

        The scene's z axis is the ground NORMAL (gravity is what tilts), so the
        nominal pose stands perpendicular to the slope, i.e. leaning back from
        vertical by the slope angle. The first slope sweep started every rollout
        that way and all 32 seeds fell inside a second on 5 degrees at mu 0.2,
        where the same policy walks the flat with no falls: it was measuring
        recovery from a backward lean on low friction, not slope walking.
        Here the base is pitched to vertical about the point under the pelvis
        and the ankle-pitch joints take the slope angle back out.
        """
        th = math.radians(self.cond.slope_deg)
        hx, hy = self.heading
        tilt = wp.quat_from_axis_angle(wp.vec3(-hy, hx, 0.0), th)  # takes +z to the true vertical
        q = self._q0.copy()
        base = wp.quat(*[float(v) for v in q[3:7]])
        q[3:7] = list(tilt * base)
        q[:3] = list(wp.quat_rotate(tilt, wp.vec3(*[float(v) for v in q[:3]])))
        names = self.config["mjw_joint_names"]
        ankles = [7 + names.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
        z_flat, _ = self._foot_frames(self._q0)
        best = None
        for sign in (1.0, -1.0):  # the joint's sign convention is not documented: take the one that flattens the feet
            cand = q.copy()
            cand[ankles] += sign * th
            z, cos_tilt = self._foot_frames(cand)
            if best is None or cos_tilt > best[0]:
                best = (cos_tilt, z, cand)
        cos_tilt, z, q = best
        if cos_tilt < math.cos(math.radians(1.0)):
            raise RuntimeError(f"slope stance left the feet {math.degrees(math.acos(cos_tilt)):.1f} deg off the ground plane")
        q[2] += z_flat - z  # feet back to their flat-ground height
        return q

    def reset(self, seed: int, joint_noise: float = 0.05, vel_noise: float = 0.2):
        """Seeded reset. The SAME seed gives the SAME start under every condition,
        which is what makes a sweep across conditions paired."""
        rng = np.random.default_rng(seed)
        q = self._q_start.copy()
        qd = self._qd0.copy()
        q[7:] += rng.uniform(-joint_noise, joint_noise, self.num_dofs)
        qd[0:2] = rng.uniform(-vel_noise, vel_noise, 2)
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(q)
            s.joint_qd.assign(qd)
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
        self._prev_act.zero_()
        self._rng = np.random.default_rng(seed + 10_000)  # gust stream
        self._gust = 0.0
        self._start_xy = q[:2].copy()

    def _wind_force(self) -> np.ndarray:
        c = self.cond
        if not (c.wind_n or c.gust_n):
            return np.zeros(3)
        # AR(1) gust, correlation time ~1 s at the policy rate.
        rho = math.exp(-1.0 / POLICY_HZ)
        self._gust = rho * self._gust + math.sqrt(1 - rho * rho) * self._rng.normal() * c.gust_n
        mag = max(0.0, c.wind_n + self._gust)
        a = math.radians(c.wind_dir_deg)
        hx, hy = self.heading
        # direction the air MOVES, in world xy: 180 deg = against the heading.
        dx = hx * math.cos(a) - hy * math.sin(a)
        dy = hx * math.sin(a) + hy * math.cos(a)
        return mag * np.array([dx, dy, 0.0])

    def step(self):
        c = self.cond
        wp.launch(
            _obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial,
                self._gravity_dir,
                wp.vec3(*c.command),
                self._prev_act,
                self.num_dofs,
                self._obs,
            ],
            device=self.device,
        )
        act = self.policy({self._in: self._obs})[self._out]
        wp.launch(
            _target_kernel,
            dim=7 + self.num_dofs,
            inputs=[act, self._joint_pos_initial, float(self.config["action_scale"]), self.control.joint_target_q],
            device=self.device,
        )
        wp.copy(self._prev_act, act)

        self.advance(self._wind_force())

    def _apply_wind(self, wind: np.ndarray):
        self.state_0.clear_forces()
        if wind.any():
            f = self.state_0.body_f.numpy()
            f[self.torso, :3] = wind
            self.state_0.body_f.assign(f)

    def advance(self, wind: np.ndarray):
        """Physics for one policy tick (20 ms). Overridden by the snow track."""
        for _ in range(DECIMATION):
            self._apply_wind(wind)
            if self.contacts is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    # -- measurement -------------------------------------------------------

    def base(self) -> tuple[np.ndarray, float]:
        """Pelvis position and cos(tilt from the slope normal)."""
        q = self.state_0.joint_q.numpy()
        pos = q[:3]
        quat = wp.quat(*[float(v) for v in q[3:7]])
        up_body = wp.quat_rotate_inv(quat, wp.vec3(*(-self.gravity / np.linalg.norm(self.gravity))))
        return pos, float(up_body[2])

    def fallen(self) -> bool:
        pos, cos_tilt = self.base()
        return bool(pos[2] < FALL_HEIGHT or cos_tilt < FALL_COS_TILT or not np.isfinite(pos).all())

    def yaw_deg(self) -> float:
        """Heading change since reset, degrees. The policy is commanded zero yaw
        rate and has no heading reference, so this drifts."""
        q = self.state_0.joint_q.numpy()
        fwd = wp.quat_rotate(wp.quat(*[float(v) for v in q[3:7]]), wp.vec3(1.0, 0.0, 0.0))
        hx, hy = self.heading
        return math.degrees(math.atan2(hx * fwd[1] - hy * fwd[0], hx * fwd[0] + hy * fwd[1]))

    def rollout(self, seed: int, steps: int = 500) -> dict:
        self.reset(seed)
        survived = steps
        vx_body = []
        for t in range(steps):
            self.step()
            # obs[0] is the body-frame forward velocity the policy just saw.
            vx_body.append(float(self._obs.numpy()[0, 0]))
            if self.fallen():
                survived = t + 1
                break
        pos, _ = self.base()
        d = pos[:2] - self._start_xy
        return {
            "seed": seed,
            "survived": survived,
            "fell": survived < steps,
            # World-frame progress along the start heading. Over 20 s this mixes
            # slowing down with turning away: the first Steel sweep read 0.04 m/s
            # at mu 0.3 for a robot walking at 0.37 m/s in a slow arc.
            "forward_m": round(float(d @ np.array(self.heading)), 3),
            "lateral_m": round(float(d @ np.array([self.heading[1], -self.heading[0]])), 3),
            # Body-frame forward speed: what "tracks the 0.5 m/s command" means.
            "mean_speed": round(float(np.mean(vx_body)), 3),
            "yaw_deg": round(self.yaw_deg(), 1),
        }
