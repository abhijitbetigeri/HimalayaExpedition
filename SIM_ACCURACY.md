# Simulation fidelity — what is approximate, and what to fix first

Everything below is about whether the ice we are simulating behaves like ice. The
policies are only as trustworthy as the contact model underneath them, and several
of the defaults are wrong for low-friction contact specifically.

Ordered by **effect on this project per unit of effort**, not by general importance.

---

## 1. Friction combine mode — wrong default, one line to fix

PhysX combines the friction of the two materials in contact. Isaac Lab's default is
`average`, so a foot at μ=0.8 stepping on ice at μ=0.06 gets an effective
**μ = 0.43** — not ice at all.

Physically the *slipperier* surface dominates: you do not get purchase on ice
because your boot is grippy. `min` is the right mode for this project, and its
absence means **every ice number so far is measured on ground roughly 7× grippier
than intended**.

```python
self.scene.terrain.physics_material.friction_combine_mode = "min"
self.scene.terrain.physics_material.restitution_combine_mode = "min"
```

This is the single highest-value fidelity fix here. It probably also explains part
of why "μ=0.06" still allowed any walking at all.

## 2. Solver iterations — low friction is where solvers fail

Contact resolution is iterative, and grazing low-friction contacts are the
worst-conditioned case: the solver has to decide stick-vs-slip with very little
normal force to work from. Isaac Lab's locomotion default is tuned for grippy
ground.

```python
PhysxCfg(solver_position_iteration_count=8,    # default 4
         solver_velocity_iteration_count=2)    # default 0
```

Cost: roughly 20–35% throughput. Worth it for the final policy; leave the defaults
for exploratory ablations.

## 3. Timestep — contact events are brief on ice

Default `sim.dt = 1/200 s` with `decimation = 4` (50 Hz control). A slipping foot
changes state faster than a planted one, so the contact is under-resolved exactly
when it matters.

```python
self.sim.dt = 1 / 400
self.decimation = 8          # keep control at 50 Hz
```

Halves throughput. Use it to *validate* a policy trained at the default dt rather
than to train — if behaviour changes materially between dt values, the result was
a solver artefact, not a gait.

## 4. Velocity-dependent friction — the real physics we are not modelling

Ice friction is **not constant**. Sliding generates frictional heating, meltwater
lubricates the interface, and μ drops as sliding speed rises — which is why a slip
on ice accelerates rather than self-arrests. PhysX has no native model for this.

Approximations, cheapest first:
- sample μ per episode from a distribution skewed low (a partial proxy; what we do
  now)
- an event term that lowers μ when foot tangential velocity exceeds a threshold —
  crude but captures the runaway
- the `newton_mjwarp` backend, which is MuJoCo underneath and exposes `solref` /
  `solimp`, so compliant and velocity-dependent contact become expressible

Worth stating as a known limitation in any write-up. The current model makes ice
*more* forgiving than reality, so results are conservative in the useful direction.

## 5. Actuator model — currently idealised

The G1 uses `ImplicitActuator`: a perfect PD source with no motor dynamics, no
backlash, no torque-speed curve. On ice, recovery from a slip is a fast
high-torque transient, exactly where an idealised actuator flatters the policy.

`DCMotorCfg` with `saturation_effort` and `velocity_limit` from the real G1
datasheet is the fix, and matters most for anything intended to reach hardware.

## 6. Terrain scale versus reality

`HfPyramidSlopedTerrainCfg(slope_range=(0.25, 0.35))` is ~14–19°. Himalayan
approach terrain runs 25–40°, and couloirs steeper still. The current slope is a
reasonable first target — do not quote it as expedition-representative.

---

## Recommended order

1. **friction_combine_mode = "min"** — corrects a real error, ~free
2. **solver iterations 8/2** — for the final policy only
3. re-run the cross-eval with 1 and 2 to see how much the numbers move
4. dt sweep as a *validation* check, not a training change
5. velocity-dependent friction, if there is time
6. DC motor model, only if hardware is in scope

Items 1 and 2 are in `train_ice_hifi.py`, ready to launch. Expect the numbers to get
**worse**, not better — that is the point. A policy that only works because the
contact model was generous is not a result.
