"""Several G1s in one Newton scene on a snowfield with hidden ice.

Physics half of the fleet track. `FleetSim` owns N robots in ONE MuJoCo model
(the C engine on CPU, see NEWTON_TRACK.md), each driven by its own copy of the
Newton-shipped walking policy through the usual command interface: forward
speed, lateral speed, yaw rate. Nothing here decides where anyone goes. That is
`fleet_search.py`.

The ground is firm snow as a friction surface, with rectangular ice patches set
flush into it. Snow here is NOT the MPM material from `snow_mpm.py`: a field
this size at 5 cm voxels is a few hundred thousand particles, hours per episode
on CPU, and that scene does not yet pass its own sanity check. What the fleet
needs from the ground is the thing the single-robot sweep measured: this policy
walks at mu 0.2 and above and falls below 0.15.

Ice patches use the same trick as the ground (`geom_priority`), one level up, so
a foot on a patch gets the patch's friction outright. `check_patch_friction()`
measures it with the sliding block.
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

from g1_collision import add_body_collisions
from g1_newton import CONE, DECIMATION, FALL_COS_TILT, FALL_HEIGHT, POLICY_HZ, SIM_HZ, paint_g1

wp.config.quiet = True

SNOW_MU = 0.4  # firm, settled snow under a boot
ICE_MU = 0.08  # wind-polished ice. Below the policy's 0.15-0.2 cliff
PATCH_TOP = 0.002  # patches stand 2 mm proud so a foot on one is not also on the snow
SNOW_COLOR = (0.93, 0.95, 0.98)
ICE_COLOR = (0.45, 0.70, 0.92)


@dataclasses.dataclass(frozen=True)
class IcePatch:
    x: float
    y: float
    hx: float  # half extents
    hy: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(x - self.x) <= self.hx + margin and abs(y - self.y) <= self.hy + margin


@dataclasses.dataclass
class Overlay:
    """What the mission layer wants drawn on one frame. Shared by fleet_search.py
    (which fills it) and film_fleet.py (which draws it). Purely for the viewer:
    nothing in here feeds back into physics or into any agent's decisions."""

    field: tuple[float, float, float, float] | None = None  # xmin, xmax, ymin, ymax of the search area
    beacon: tuple[float, float] | None = None  # true buried-beacon position
    beacon_found: bool = False
    hazards: list = dataclasses.field(default_factory=list)  # (x, y, radius) the fleet has mapped
    links: list = dataclasses.field(default_factory=list)  # (i, j) robot pairs that exchanged a message this tick
    targets: list = dataclasses.field(default_factory=list)  # per robot: (x, y) waypoint or None
    roles: list = dataclasses.field(default_factory=list)  # per robot: sweep | locate | converge | hold | backoff | down
    caption: str = ""


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the frame of quaternion q = (x, y, z, w)."""
    u, w = -q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def add_snowfield(builder: newton.ModelBuilder, patches: list[IcePatch], snow_mu: float, ice_mu: float):
    cfg = newton.ModelBuilder.ShapeConfig(mu=snow_mu, ke=5.0e4, kd=5.0e2, kf=1.0e3)
    builder.add_shape_plane(
        body=-1, xform=wp.transform_identity(), width=0.0, length=0.0, cfg=cfg, label="snow",
        color=SNOW_COLOR, custom_attributes={"mujoco:geom_priority": 1},
    )  # fmt: skip
    ice = newton.ModelBuilder.ShapeConfig(mu=ice_mu, ke=5.0e4, kd=5.0e2, kf=1.0e3)
    hz = 0.02
    for k, p in enumerate(patches):
        builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(p.x, p.y, PATCH_TOP - hz)), hx=p.hx, hy=p.hy, hz=hz, cfg=ice,
            label=f"ice_{k}", color=ICE_COLOR, custom_attributes={"mujoco:geom_priority": 2},
        )  # fmt: skip


class FleetSim:
    """N policy-driven G1s on a snowfield. One `step(commands)` is one 20 ms tick."""

    def __init__(
        self,
        starts: list[tuple[float, float, float]],  # x, y, yaw (rad, 0 = facing +x)
        patches: list[IcePatch] | tuple = (),
        *,
        snow_mu: float = SNOW_MU,
        ice_mu: float = ICE_MU,
        seed: int = 0,
        body_collisions: bool = True,
    ):
        self.n = len(starts)
        self.patches = list(patches)
        self.device = wp.get_device()
        asset_dir = newton.utils.download_asset("unitree_g1")
        with open(asset_dir / "rl_policies" / "g1_29dof.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.nj = nj = int(cfg["num_dofs"])
        self.action_scale = float(cfg["action_scale"])
        self.default_q = np.asarray(cfg["mjw_joint_pos"], dtype=np.float32)

        def new_builder():
            b = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(b)
            b.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
            b.default_shape_cfg.ke = 5.0e4
            b.default_shape_cfg.kd = 5.0e2
            b.default_shape_cfg.kf = 1.0e3
            b.default_shape_cfg.mu = 0.75
            b.rigid_gap = 0.0
            return b

        robot = new_builder()
        robot.add_usd(
            str(asset_dir / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        robot.approximate_meshes("convex_hull")
        paint_g1(robot, asset_dir)
        if body_collisions:
            # The USD collides at the soles and the torso only. See g1_collision.py.
            add_body_collisions(robot)
        for i in range(nj):
            robot.joint_target_ke[i + 6] = cfg["mjw_joint_stiffness"][i]
            robot.joint_target_kd[i + 6] = cfg["mjw_joint_damping"][i]
            robot.joint_armature[i + 6] = cfg["mjw_joint_armature"][i]
            robot.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        self.bodies_per_robot = robot.body_count
        self.foot_local = [
            next(i for i, lbl in enumerate(robot.body_label) if lbl.endswith("/" + name))
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]

        builder = new_builder()
        for _ in range(self.n):
            builder.add_builder(robot)
        add_snowfield(builder, self.patches, snow_mu, ice_mu)
        self.nq, self.nv = 7 + nj, 6 + nj
        if builder.joint_coord_count != self.n * self.nq or builder.joint_dof_count != self.n * self.nv:
            raise RuntimeError(
                f"unexpected layout: {builder.joint_coord_count} coords / {builder.joint_dof_count} dofs "
                f"for {self.n} robots of {self.nq} / {self.nv}"
            )

        rng = np.random.default_rng(seed)
        q0 = np.zeros(self.n * self.nq, dtype=np.float32)
        for r, (x, y, yaw) in enumerate(starts):
            o = r * self.nq
            q0[o : o + 3] = (x, y, 0.76)
            q0[o + 3 : o + 7] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            q0[o + 7 : o + self.nq] = self.default_q + rng.uniform(-0.03, 0.03, nj)
        builder.joint_q[:] = q0.tolist()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        # A robot lying on 33 shapes can make far more contacts than one standing on
        # two soles: up to 80 against flat ground, at 3 rows each plus joint limits.
        # Measured peak with four down is 47 contacts and 153 rows in total.
        ncon, nrow = (80, 320) if body_collisions else (30, 160)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            solver="newton",
            cone=CONE,
            nconmax=ncon * self.n,
            njmax=nrow * self.n,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_q = wp.zeros(self.n * self.nq, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415

        policy_path = str(asset_dir / "rl_policies" / "mjw_g1_29DOF.onnx")
        self.policy = OnnxRuntime(policy_path, device=self.device)
        self._in = self.policy.inputs[0].name
        self._out = self.policy.outputs[0].name
        self._obs = wp.zeros((1, 12 + 3 * nj), dtype=wp.float32, device=self.device)
        self.policy.prepare({self._in: self._obs})
        self.prev_act = np.zeros((self.n, nj), dtype=np.float32)
        self.down = np.zeros(self.n, dtype=bool)  # latched: no robot gets up in this track
        self.t = 0
        self._q = q0.copy()
        self._qd = np.zeros(self.n * self.nv, dtype=np.float32)
        self.slip = np.zeros(self.n)  # stance-foot sliding speed, m/s, low-passed

    # -- state queries (what a robot's own sensors would give it) ----------

    def pose(self, r: int) -> tuple[float, float, float]:
        """x, y, yaw of robot r. Yaw 0 faces +x."""
        o = r * self.nq
        q = self._q[o + 3 : o + 7]
        f = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        return float(self._q[o]), float(self._q[o + 1]), math.atan2(f[1], f[0])

    def upright(self, r: int) -> bool:
        o = r * self.nq
        up = quat_rotate(self._q[o + 3 : o + 7], np.array([0.0, 0.0, 1.0]))
        return bool(self._q[o + 2] > FALL_HEIGHT and up[2] > FALL_COS_TILT)

    def on_ice(self, r: int) -> bool:
        """Ground truth, for scoring and for the film. Agents never call this."""
        x, y, _ = self.pose(r)
        return any(p.contains(x, y) for p in self.patches)

    # -- stepping ----------------------------------------------------------

    def step(self, commands: np.ndarray):
        """commands: (n, 3) of forward m/s, lateral m/s, yaw rate rad/s, body frame."""
        nj, nq, nv = self.nj, self.nq, self.nv
        targets = np.zeros(self.n * nq, dtype=np.float32)
        obs = np.zeros(12 + 3 * nj, dtype=np.float32)
        for r in range(self.n):
            qo, vo = r * nq, r * nv
            jq = self._q[qo + 7 : qo + nq]
            if self.down[r]:
                # A fallen robot goes limp at its current pose: no thrashing into teammates.
                targets[qo + 7 : qo + nq] = jq
                continue
            quat = self._q[qo + 3 : qo + 7]
            obs[0:3] = quat_rotate_inv(quat, self._qd[vo : vo + 3])
            obs[3:6] = quat_rotate_inv(quat, self._qd[vo + 3 : vo + 6])
            obs[6:9] = quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
            obs[9:12] = commands[r]
            obs[12 : 12 + nj] = jq - self.default_q
            obs[12 + nj : 12 + 2 * nj] = self._qd[vo + 6 : vo + nv]
            obs[12 + 2 * nj :] = self.prev_act[r]
            self._obs.assign(obs[None, :])
            act = self.policy({self._in: self._obs})[self._out].numpy()[0]
            self.prev_act[r] = act
            targets[qo + 7 : qo + nq] = self.default_q + self.action_scale * act
        self.control.joint_target_q.assign(targets)

        dt = 1.0 / SIM_HZ
        for _ in range(DECIMATION):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._q = self.state_0.joint_q.numpy()
        self._qd = self.state_0.joint_qd.numpy()
        self.t += 1
        self._update_slip()
        for r in range(self.n):
            if not self.down[r] and not self.upright(r):
                self.down[r] = True

    def _update_slip(self):
        """Sliding speed of the lower foot of each robot: a proprioceptive ice cue.

        A planted foot should not move. This is computed from body velocities a
        real robot gets from leg kinematics plus its IMU. It reads nothing about
        the ground.
        """
        bq = self.state_0.body_q.numpy()
        bqd = self.state_0.body_qd.numpy()
        for r in range(self.n):
            if self.down[r]:
                continue
            feet = [r * self.bodies_per_robot + i for i in self.foot_local]
            low = min(feet, key=lambda b: bq[b][2])
            if bq[low][2] > 0.06:  # both feet in the air
                continue
            speed = float(np.hypot(bqd[low][0], bqd[low][1]))
            self.slip[r] = 0.8 * self.slip[r] + 0.2 * speed


def check_patch_friction(ice_mu: float = ICE_MU, snow_mu: float = SNOW_MU) -> dict:
    """Sliding-block measurement on the snow and on a patch. See calibrate_friction.py."""
    out = {}
    for name, pos in (("snow", (5.0, 0.0)), ("ice", (0.0, 0.0))):
        lo, hi = 0.0, 45.0
        for _ in range(11):
            mid = 0.5 * (lo + hi)
            b = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(b)
            add_snowfield(b, [IcePatch(0.0, 0.0, 1.0, 1.0)], snow_mu, ice_mu)
            body = b.add_body(xform=wp.transform(wp.vec3(pos[0], pos[1], 0.06)), mass=5.0)
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
            m = b.finalize()
            th = math.radians(mid)
            m.set_gravity((0.0, -9.81 * math.sin(th), -9.81 * math.cos(th)))
            s = newton.solvers.SolverMuJoCo(m, use_mujoco_cpu=True, solver="newton", cone=CONE)
            s0, s1, c = m.state(), m.state(), m.control()
            for _ in range(300):
                s0.clear_forces()
                s.step(s0, s1, c, None, 1.0 / 200)
                s0, s1 = s1, s0
            if float(s0.body_q.numpy()[body][1]) < pos[1] - 0.02:
                hi = mid
            else:
                lo = mid
        out[name] = round(math.tan(math.radians(0.5 * (lo + hi))), 3)
    return out


if __name__ == "__main__":
    print("effective friction, sliding block:", check_patch_friction(), f"(nominal snow {SNOW_MU}, ice {ICE_MU})")
