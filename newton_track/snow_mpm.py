# /// script
# requires-python = ">=3.11"
# dependencies = ["newton[examples]==1.8.0"]
# ///
"""G1 walking through simulated snow: two-way coupled MuJoCo + implicit MPM.

Everywhere else in this repo "snow" is a stand-in. The MuJoCo track uses
compliant ground (softened `solref`), the Isaac clips show a grid floor, and
the dataset card says so. Here the snow is material: a bed of particles with a
compressible elasto-plastic rheology (yield pressure + hardening, the same
model as Newton's `mpm_snow_ball`), solved by Newton's implicit MPM and coupled
both ways to the robot. A foot compacts the snow under it and the snow pushes
back on the foot, so sinkage, post-holing and drag come out of the simulation
instead of being put in as a spring.

    # Preflight, then a 25-step smoke, then a COARSE local run (5 cm voxels)
    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_block_test.py
    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_mpm.py --smoke
    ~/projects/newton-sandbox/.venv/bin/python newton_track/snow_mpm.py --voxel 0.05 \
        --bed-length 1.6 --bed-half-width 0.3 --seeds 1 --steps 120

    # GPU (ask before launching; see NEWTON_TRACK.md for the hf jobs line)
    python snow_mpm.py --voxel 0.025 --seeds 8 --steps 500

READ NEWTON_TRACK.md "Snow" BEFORE QUOTING ANYTHING FROM THIS SCRIPT. As of
2026-11-27 it still fails its own sanity check: on hard snow the robot sinks
under 2 cm and falls, where the bare-ground control walks. A day of one-seed
probes at 5 cm voxels did not fix it. What they showed, so nobody repeats them
blind:

- The fall is not specific to walking. Commanded to stand still, the robot on
  bare ground is at rest within a second. On hard snow it never settles: it
  marches in place, the loaded sole ends up rolled and pitched by 10 to 20
  degrees, and it falls or is about to by 2 s. With no policy at all (joint
  targets held) it topples within three ticks of when it does on bare ground,
  so the bed carries a passive robot. The trouble needs the policy in the loop.
- No single change cures it: walls round the bed, 8 coupled steps per tick,
  proxy relaxation 0.5, three Aitken passes, standing before walking, soles
  started on the surface, a taller or a wider foot collider, particles projected
  out of the feet, 10x and 100x the Young's modulus, the Q1 collider basis,
  3.5 cm voxels, a foot proxy given its measured articulated inertia. Nor did
  the sixteen combinations tried. All are arguments of `G1Snow`.
- The mean vertical force is right (0.98 to 1.00 of the weight while the robot
  is up; the 76 % in the table is averaged over the fall). The force is not
  one-sided as a contact should be: in a fifth of foot-ticks the snow pulls a
  lifting foot DOWN, by up to 70 N with the stock collider, and sideways by
  about 100 N.
- `snow_foot_test.py` asks the bed for what rigid ground gives a stance foot.
  It gives a fifth to a half: a sole-sized plate under the robot's weight has
  tilted 2 degrees by 20 to 55 % of the rigid tipping torque in roll, whatever
  the snow's stiffness or strength, the voxel size or the number of passes.
- `snow_friction_test.py`: the snow is not ice to the foot. The stock foot box
  does not slide under a pull of 1.5 x its load.
- The 1.9 cm "sinkage" on hard snow is the spawn pose, not the snow. The nominal
  pose puts the soles 2.4 cm below the surface. MuJoCo's ground pushes them
  out; the MPM contact is velocity-level and never does.
  `--spawn-clearance 0.024` starts them on the surface.
- The stock particle collider of a G1 foot is a box 2 cm thick. A grid node more
  than 1 cm inside it is nearer the top face, its contact normal points up, and
  the snow there holds the foot down. `--foot-collider-height 0.12` removes
  that. It does not make the robot stand.

The script is self-contained apart from `g1_newton.py`, which must sit next to
it (on HF Jobs: copy both into the bucket, CLAUDE.md "ships only the single
named script").

What is measured per rollout, beyond survival:
- `sink_m`: how far below the original snow surface the lower foot sole sits,
  median over the rollout after the first half second. With the default spawn
  the soles START 2.4 cm below the surface, so on firm snow this reads about
  2 cm for a robot that has not sunk at all;
- `mean_speed`: body-frame forward speed, to compare with the same seed on bare
  ground (the `bare` rows run that control through the same mujoco_warp solver);
- `snow_weight_fraction`: mean vertical force the snow delivers to the robot,
  over the robot's weight (~340 N). Near 0 means the robot is standing on the
  rigid floor under the snow and the coupling is decorative; near 1 means the
  snow is carrying it. This is the check the rigid `connect` tether failed in
  CLAUDE.md ("the rope carried the entire robot").

Limits, stated up front. The policy was trained on rigid ground and has never
felt sinkage. Snow parameters are order-of-magnitude (density 300 kg/m^3, three
named types in `SNOW`), not fitted to a snow pit. The voxel size
on GPU defaults to 2.5 cm against a ~7 cm wide foot.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, "/mnt/himalaya-g1")  # HF Jobs bucket mount
from g1_newton import DECIMATION, POLICY_HZ, Conditions, G1Himalaya  # noqa: E402

from newton.solvers import SolverImplicitMPM, SolverMuJoCo  # noqa: E402
from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: E402

ROBOT_WEIGHT_N = 340.0  # ~35 kg


class G1Snow(G1Himalaya):
    """G1 on a snow bed. `depth=0` is the bare-ground control on the same solver."""

    def __init__(
        self,
        cond: Conditions,
        *,
        depth: float = 0.15,
        yield_pressure: float = 2.0e4,
        cohesion: float = 0.0,
        mass_scale: float = 1.0,
        coupling_mode: str = "lagged",
        mpm_per_tick: int = 4,
        density: float = 300.0,
        voxel: float = 0.025,
        bed_half_width: float = 0.5,
        bed_length: float = 3.0,
        max_iterations: int = 50,
        walls: bool = False,
        proxy_relaxation: float = 1.0,
        relaxation_mode: str = "fixed",
        coupling_iterations: int = 1,
        snow_friction: float = 0.5,
        foot_mu: float | None = None,
        foot_collider_height: float = 0.0,
        young_modulus: float = 1.4e6,
        project_outside: bool = False,
        spawn_clearance: float = 0.0,
        foot_collider_half_width: float | None = None,
        collider_basis: str = "S2",
        mpm_tolerance: float = 1.0e-4,
        proxy_foot_inertia: tuple[float, float, float] | None = None,
    ):
        self.depth = depth
        self.yield_pressure = yield_pressure
        self.cohesion = cohesion
        self.mass_scale = mass_scale
        self.coupling_mode = coupling_mode
        self.mpm_per_tick = mpm_per_tick
        self.density = density
        self.voxel = voxel
        self.bed = (bed_half_width, bed_length)
        self.max_iterations = max_iterations
        self.walls = walls
        self.proxy_relaxation = proxy_relaxation
        self.relaxation_mode = relaxation_mode
        self.coupling_iterations = coupling_iterations
        self.snow_friction = snow_friction
        self.foot_mu = foot_mu
        self.foot_collider_height = foot_collider_height
        self.young_modulus = young_modulus
        self.project_outside = project_outside
        self.spawn_clearance = spawn_clearance
        self.foot_collider_half_width = foot_collider_half_width
        self.collider_basis = collider_basis
        self.mpm_tolerance = mpm_tolerance
        self.proxy_foot_inertia = proxy_foot_inertia
        self._snow_force_log: list[float] = []
        # The coupled path feeds Newton contacts to mujoco_warp, so the pure-C
        # CPU engine is not available here.
        super().__init__(cond, use_mujoco_cpu=False, builder_hook=self._add_snow)

    # -- scene -------------------------------------------------------------

    def _add_snow(self, builder: newton.ModelBuilder, _env):
        self._rigid_bodies = list(range(builder.body_count))
        # Only the feet meet the snow (the two ankle-roll links are the only
        # particle colliders that result); skipping the rest keeps the
        # collider small. A fallen robot is already terminated.
        keep = ("ankle_roll_link", "ankle_pitch_link", "knee_link")
        for body in range(builder.body_count):
            if not builder.body_label[body].endswith(keep):
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

        SolverImplicitMPM.register_custom_attributes(builder)
        if self.depth <= 0.0:
            return
        # Spawn ON the snow. The first smoke run left the robot at ground level,
        # i.e. buried to the shins from step 0: it reported the full depth as
        # "sinkage" while standing on the rigid floor.
        builder.joint_q[2] += self.depth + self.spawn_clearance
        if self.foot_mu is not None:
            for body in range(builder.body_count):
                if builder.body_label[body].endswith(keep):
                    for shape in builder.body_shapes[body]:
                        builder.shape_material_mu[shape] = self.foot_mu
        if self.foot_collider_height > 0.0:
            self._thicken_foot_colliders(builder, self.foot_collider_height, self.foot_collider_half_width)
        hw, length = self.bed
        lo = np.array([-hw, -0.4, 0.0])
        hi = np.array([hw, length - 0.4, self.depth])
        if self.walls:
            t, hz = 0.05, 0.5 * (self.depth + 0.05)
            cx, cy = 0.0, 0.5 * (lo[1] + hi[1])
            for x, y, hx, hy in (
                (lo[0] - t, cy, t, 0.5 * length + 2 * t),
                (hi[0] + t, cy, t, 0.5 * length + 2 * t),
                (cx, lo[1] - t, hw, t),
                (cx, hi[1] + t, hw, t),
            ):
                builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, hz)), hx=hx, hy=hy, hz=hz)
        ppc = 2  # particles per cell edge
        res = np.maximum(np.ceil(ppc * (hi - lo) / self.voxel).astype(int), 1)
        cell = (hi - lo) / res
        radius = float(cell.max() * 0.5)
        builder.add_particle_grid(
            pos=wp.vec3(*lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=float(np.prod(cell) * self.density),
            jitter=0.5 * radius,
            radius_mean=radius,
            custom_attributes={
                # Compressible snow, after newton.examples.mpm_snow_ball.
                "mpm:young_modulus": float(self.young_modulus),
                "mpm:poisson_ratio": 0.3,
                "mpm:friction": float(self.snow_friction),
                "mpm:damping": 0.01,
                "mpm:yield_pressure": float(self.yield_pressure),
                "mpm:tensile_yield_ratio": 0.2,
                "mpm:yield_stress": float(self.cohesion),
                "mpm:hardening": 5.0,
            },
        )

    @staticmethod
    def _thicken_foot_colliders(builder: newton.ModelBuilder, height: float, half_width: float | None = None):
        """Replace each foot's particle collider by a box of the same footprint, `height` tall."""
        for body in range(builder.body_count):
            if not builder.body_label[body].endswith("ankle_roll_link"):
                continue
            for shape in list(builder.body_shapes[body]):
                if not builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES:
                    continue
                if builder.shape_type[shape] != newton.GeoType.BOX:
                    raise RuntimeError(f"foot particle collider {builder.shape_label[shape]} is not a box")
                hx, hy, hz = (float(v) for v in builder.shape_scale[shape])
                if half_width is not None:
                    hy = half_width  # DIAGNOSTIC ONLY: a snowshoe, not the G1 foot
                xf = wp.transform(*builder.shape_transform[shape])
                builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
                lift = wp.quat_rotate(xf.q, wp.vec3(0.0, 0.0, 0.5 * height - hz))
                builder.add_shape_box(
                    body,
                    xform=wp.transform(xf.p + lift, xf.q),
                    hx=hx,
                    hy=hy,
                    hz=0.5 * height,
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=0.0,  # geometry only: the link's mass and inertia stay as shipped
                        mu=float(builder.shape_material_mu[shape]),
                        has_shape_collision=False,
                        has_particle_collision=True,
                        is_visible=False,
                    ),
                    label=builder.shape_label[shape] + "/snow_collider",
                )

    def _make_solver(self):
        model = self.model
        self.has_snow = model.particle_count > 0
        mj_kwargs = dict(use_mujoco_contacts=False, solver="newton", cone=self.cone, njmax=200, nconmax=60)

        if not self.has_snow:
            self.solver = SolverMuJoCo(model, **mj_kwargs)
            self.mpm = None
        else:
            cfg = SolverImplicitMPM.Config()
            cfg.voxel_size = self.voxel
            cfg.grid_type = "fixed" if self.device.is_cuda else "sparse"
            if cfg.grid_type == "fixed":
                cfg.grid_padding = 50
                cfg.max_active_cell_count = 1 << 17
            cfg.strain_basis = "P0"
            cfg.max_iterations = self.max_iterations
            cfg.tolerance = self.mpm_tolerance
            cfg.collider_basis = self.collider_basis
            cfg.critical_fraction = 0.0
            bodies = self._rigid_bodies
            self.solver = SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mjc",
                        solver=lambda v: SolverMuJoCo(model=v, **mj_kwargs),
                        bodies=bodies,
                        joints=list(range(model.joint_count)),
                        substeps=max(1, DECIMATION // self.mpm_per_tick),
                    ),
                    SolverCoupledProxy.Entry(
                        name="mpm",
                        solver=lambda v: SolverImplicitMPM(model=v, config=cfg),
                        particles=list(range(model.particle_count)),
                        in_place=True,
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc",
                            destination="mpm",
                            bodies=bodies,
                            mode=self.coupling_mode,
                            mass_scale=self.mass_scale,
                            proxy_relaxation=self.proxy_relaxation,
                            proxy_relaxation_mode=self.relaxation_mode,
                            collision_pipeline=lambda _m: None,  # MPM does its own collider contact
                        )
                    ],
                    iterations=self.coupling_iterations,
                ),
            )
            self.mpm = self.solver.solver("mpm")
            if self.proxy_foot_inertia is not None:
                self._override_proxy_foot_inertia(self.proxy_foot_inertia)
            # collider id -> model body index; -1 for static shapes (the floor)
            # and for the "no collider" id.
            cbi = np.asarray(self.mpm.collider_body_index.numpy() if hasattr(self.mpm.collider_body_index, "numpy")
                             else self.mpm.collider_body_index, dtype=int)
            self._collider_body = np.concatenate([cbi, [-1]])  # so id -1 indexes the sentinel
        self.pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
        self.contacts = self.pipeline.contacts()

    def _override_proxy_foot_inertia(self, diag: tuple[float, float, float]):
        """Give the MPM solver's foot proxies a chosen inertia (body-frame diagonal, kg m^2).

        By default a proxy gets the link's own inertia tensor scaled to MuJoCo's
        MEAN angular compliance. For the G1 foot that is 3x the articulated
        inertia in pitch and yaw and a third of it in roll (measured: push the
        foot of the floating robot, read the acceleration), and lagged feedback
        is only well behaved when the two match. Uses a private method of
        `SolverCoupled`; there is no public hook in Newton 1.8.0.
        """
        entry = self.solver._entries["mpm"]
        local = entry.body_global_to_local.numpy()
        ids = wp.array([int(local[i]) for i in self.feet], dtype=int, device=self.device)
        mass = self.mpm.model.body_mass.numpy()
        masses = wp.array([float(mass[int(local[i])]) for i in self.feet], dtype=float, device=self.device)
        inertias = wp.array([wp.mat33(np.diag(diag).astype(np.float32)) for _ in self.feet], dtype=wp.mat33, device=self.device)
        self.solver._apply_body_inertia_override(entry, ids, masses, inertias)

    # -- stepping ----------------------------------------------------------

    def advance(self, wind: np.ndarray):
        if not self.has_snow:
            return super().advance(wind)
        # `mpm_per_tick` coupled steps per 20 ms policy tick, each one implicit
        # MPM solve and `DECIMATION // mpm_per_tick` rigid substeps (at least
        # one). One coupled step per tick lets the feet fall straight through
        # (NEWTON_TRACK.md "Snow", point 3).
        fz = 0.0
        for _ in range(self.mpm_per_tick):
            self._apply_wind(wind)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_0, self.control, self.contacts, 1.0 / (POLICY_HZ * self.mpm_per_tick)
            )
            fz += self._snow_vertical_force() / self.mpm_per_tick
            if self.project_outside:
                self.mpm.project_outside(self.state_0, self.state_0, 0.0)
        self._snow_force_log.append(fz)

    def reset(self, seed: int, **kw):
        if self.has_snow and getattr(self, "_used", False):
            # Footprints are state. A second rollout on trampled snow is not the
            # same experiment, so one env = one rollout.
            raise RuntimeError("G1Snow holds compacted snow from the last rollout; build a new one per seed")
        self._used = True
        self._snow_force_log = []
        super().reset(seed, **kw)

    # -- measurement -------------------------------------------------------

    def _snow_vertical_force(self) -> float:
        """Vertical force [N] the snow applied to the ROBOT over the last tick.

        `collect_collider_impulses` covers every collider, including the static
        floor under the bed, which carries the snow's own weight (~330 N for the
        smoke bed, by coincidence the robot's weight). The first version summed
        them all and reported "the snow carries 98 % of the robot" for a robot
        standing on the floor in two post-holes. Only robot bodies count here.
        Sign: the impulse is the one applied TO the collider, so +z is support.
        """
        impulses, _pos, ids = self.mpm.collect_collider_impulses(self.state_0)
        imp = impulses.numpy()
        if imp.size == 0:
            return 0.0
        body = self._collider_body[ids.numpy()]
        return float(imp[body >= 0, 2].sum() * POLICY_HZ * self.mpm_per_tick)

    def lower_sole_height(self) -> float:
        bq = self.state_0.body_q.numpy()
        # ankle-roll link origin sits ~3.5 cm above the sole.
        return float(min(bq[i][2] for i in self.feet)) - 0.035

    def rollout(self, seed: int, steps: int = 500) -> dict:
        self.reset(seed)
        survived = steps
        soles, vxs = [], []
        for t in range(steps):
            self.step()
            vxs.append(float(self._obs.numpy()[0, 0]))
            soles.append(self.lower_sole_height())
            if self.fallen():
                survived = t + 1
                break
        pos, _ = self.base()
        forward = float((pos[:2] - self._start_xy) @ np.array(self.heading))
        vx_body = float(np.mean(vxs))
        settle = min(25, len(soles) - 1)  # skip the half second of landing
        out = {
            "seed": seed,
            "depth_m": self.depth,
            "yield_pressure": self.yield_pressure if self.has_snow else None,
            "cohesion": self.cohesion if self.has_snow else None,
            "particles": int(self.model.particle_count),
            "survived": survived,
            "fell": survived < steps,
            "forward_m": round(forward, 3),
            "mean_speed": round(vx_body, 3),  # body frame, as in g1_newton.rollout
            "sink_m": round(float(self.depth - np.median(soles[settle:])), 4),
        }
        if self.has_snow:
            f = np.array(self._snow_force_log[settle:])
            out["snow_force_n"] = round(float(f.mean()), 1) if f.size else None
            out["snow_weight_fraction"] = round(float(f.mean()) / ROBOT_WEIGHT_N, 3) if f.size else None
        return out


# Snow types as (yield pressure, cohesion), both Pa. Order of magnitude only.
# Cohesion is what separates them in practice: with none, a cohesionless
# frictional bed has almost no bearing capacity under a 7 cm wide foot whatever
# its yield pressure.
SNOW = {
    "powder": (2.0e4, 0.0),  # fresh, dry, unbonded
    "packed": (1.0e5, 2.0e4),  # settled or wind-packed
    "hard": (1.0e6, 1.0e5),  # sintered wind slab / refrozen crust
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snow", nargs="+", default=list(SNOW), choices=list(SNOW))
    ap.add_argument("--depth", type=float, default=0.15, help="snow depth, m")
    ap.add_argument("--density", type=float, default=300.0)
    ap.add_argument("--voxel", type=float, default=0.025)
    ap.add_argument("--bed-length", type=float, default=3.0)
    ap.add_argument("--bed-half-width", type=float, default=0.5)
    ap.add_argument("--mpm-per-tick", type=int, default=4, choices=(1, 2, 4, 8), help="coupled steps per 20 ms policy tick")
    # Diagnostic knobs. The defaults are the configuration behind results/snow_mpm.json.
    ap.add_argument("--walls", action="store_true", help="static walls around the bed")
    ap.add_argument("--mass-scale", type=float, default=1.0, help="proxy mass and inertia scale")
    ap.add_argument("--relaxation", type=float, default=1.0, help="proxy feedback relaxation factor")
    ap.add_argument("--coupling-iterations", type=int, default=1, help="proxy passes per coupled step; above 1 uses Aitken")
    ap.add_argument("--foot-collider-height", type=float, default=0.0, help="m; 0 keeps the stock 2 cm foot box")
    ap.add_argument("--spawn-clearance", type=float, default=0.0, help="m added to the spawn height above the snow")
    ap.add_argument("--young", type=float, default=1.4e6, help="snow Young's modulus, Pa")
    ap.add_argument("--project-outside", action="store_true", help="push particles out of the feet after every coupled step")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: coarse voxels, short bed, 1 seed, few steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = wp.get_device()
    kw = dict(
        density=args.density,
        voxel=args.voxel,
        bed_length=args.bed_length,
        bed_half_width=args.bed_half_width,
        mpm_per_tick=args.mpm_per_tick,
        walls=args.walls,
        mass_scale=args.mass_scale,
        proxy_relaxation=args.relaxation,
        coupling_iterations=args.coupling_iterations,
        relaxation_mode="aitken" if args.coupling_iterations > 1 else "fixed",
        foot_collider_height=args.foot_collider_height,
        spawn_clearance=args.spawn_clearance,
        young_modulus=args.young,
        project_outside=args.project_outside,
    )
    if args.smoke:
        args.seeds, args.steps = 1, min(args.steps, 25)
        args.snow = args.snow[:1]
        kw.update(voxel=0.05, bed_half_width=0.3, bed_length=1.0, max_iterations=20)
    elif not device.is_cuda and args.voxel < 0.05:
        print(f"No CUDA: voxel {args.voxel} m would take hours here. Pass --voxel 0.05 for a coarse local run.")
        return 2

    cond = Conditions(command=(args.speed, 0.0, 0.0))
    rows = []
    t0 = time.time()
    # depth 0 first: the paired bare-ground control on the same solver stack.
    cells = [("bare", 0.0, 0.0, 0.0)] + [(name, args.depth, *SNOW[name]) for name in args.snow]
    for name, depth, yp, coh in cells:
        for seed in range(args.seeds):
            t = time.time()
            env = G1Snow(cond, depth=depth, yield_pressure=yp, cohesion=coh, **kw)
            r = {"snow": name, **env.rollout(seed, args.steps)}
            r["wall_s"] = round(time.time() - t, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)

    control = [r for r in rows if r["snow"] == "bare"]
    if not args.smoke and any(r["fell"] for r in control):
        print("WARNING: the bare-ground control fell. Snow rows below have no valid baseline.")

    out_dir = pathlib.Path("/mnt/himalaya-g1/newton") if pathlib.Path("/mnt/himalaya-g1").is_dir() else (
        pathlib.Path(__file__).resolve().parent / "results"
    )
    out = pathlib.Path(args.out) if args.out else out_dir / ("snow_smoke.json" if args.smoke else "snow_mpm.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "args": vars(args), "rows": rows}, indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
