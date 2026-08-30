# Himalaya Expedition — teaching a Unitree G1 to handle ice

Track 1 (Movement) entry for the Himalaya Robotics Hack, Aug 2026.

## The finding

Both major humanoid locomotion benchmarks train the Unitree G1 on ground roughly an
order of magnitude grippier than ice — and one of them doesn't vary the friction at all.

**Isaac Lab** — `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`:

```python
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    params={"static_friction_range":  (0.8, 0.8),   # min == max
            "dynamic_friction_range": (0.6, 0.6)})
```

Minimum equals maximum, so the humanoid sees exactly one surface for the whole of
training. Neither G1 config overrides it. In the same source tree the *quadruped*
Spot gets `(0.3, 1.0)` / `(0.3, 0.8)` — the four-legged robot gets friction
randomisation, the two-legged one doesn't.

**MuJoCo Playground** — `_src/locomotion/g1/randomize.py` samples `U(0.4, 1.0)`: a
real range, but with a floor still ~8× above ice.

Real ice is μ = 0.05–0.15.

## Results

**Isaac Lab — widening the range works, at no cost on normal ground.**

| | baseline (μ=0.8 fixed) | ice-trained (μ ∈ 0.05–1.0) |
|---|---|---|
| success rate | 1.000 | 1.000 |
| survives full episode | 98.0% | 97.8% |
| velocity tracking | 0.913 | 0.912 |

Each column is measured on its own training distribution, so this shows the ice
policy pays no tax for a 20× friction range — not that it beats the baseline on ice.
Cross-evaluation on identical ground is unfinished.

**MuJoCo Playground — the same experiment fails, and we can't yet say why.**
Six explanations were tested and all six are dead:

| # | hypothesis | verdict |
|---|---|---|
| 1 | the eval harness is broken | refuted — baseline reaches 301/300 through the same code path |
| 2 | log-uniform sampling starves it of easy ground | refuted — uniform also fails (35.1 ± 6.4) |
| 3 | the slip penalty punishes the surface, not the gait | refuted — `feet_slip` −0.25→−0.05 changed nothing |
| 4 | too many novel conditions at once | refuted — three factors break walking *individually* |
| 5 | undertrained | refuted — 25M→150M steps moved 7.1% → 10.7%; Isaac hits 98% at the same budget |
| 6 | a gait-phase reward forces a fixed cadence | refuted — zeroing `feet_phase` changed nothing |

**Which factor breaks walking** (one variable per run, 16 rollouts, scored on stock
flat ground):

| run | changed | survival of 501 | |
|---|---|---|---|
| A2 | compliant ground (soft snow) | 390.5 ± 137.5 | 78% — harmless |
| A3 | wind | 65.8 ± 15.9 | 13% |
| A1 | friction range alone | 35.6 ± 7.2 | 7% |
| A4 | mid-episode rock→ice patches | 11.6 ± 2.6 | 2% — worst |

Deformable ground is fine. **Sudden transitions between surfaces are the most
destructive thing you can do to a humanoid gait** — worse than ice itself.

## Watch it

Interactive viewer, on a laptop, no GPU:

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install mujoco playground brax mediapy imageio-ffmpeg
.venv/bin/mjpython local_view.py --backend warp            # interactive window
.venv/bin/python  local_view.py --backend warp --mu 0.06 --headless -o ice.mp4
```

`--mu 0.8` is dry rock, `--mu 0.06` bare ice. Checkpoints download from the HF
bucket on first run.

⚠️ **Use `--backend warp` for anything you quote or film.** `jax` is ~3× faster on
CPU but is not the backend these policies were trained under: the same policy at
μ=0.8 survives 301/301 under warp and *falls at step 112* under jax.

## Layout

| file | what |
|---|---|
| `ice_randomize.py` | drop-in `randomization_fn` — per-foot friction, compliant ground |
| `wind.py` | `WindyJoystick` — drag, centre-of-pressure torque, AR(1) gusts |
| `ice_patch.py` | `PatchyIceJoystick` — position-dependent, mid-episode rock→ice |
| `fixed_line.py` / `fixed_line_scene.py` | fixed-line ascent: slope, tether, ratcheting ascender |
| `train_ice.py` | MuJoCo PPO training (`--baseline` for the A/B arm) |
| `train_baseline.py` / `train_ice_isaac.py` / `train_ice_slope.py` | Isaac Lab training |
| `local_view.py` | watch a policy on your own machine |
| `isaac_film.py` / `render_long.py` | rendering |
| `diagnose_eval*.py` / `ablate_eval.py` | the controls and ablations behind the tables above |
| `CLAUDE.md` | MuJoCo track: conventions, traps, findings |
| `ISAAC_TRACK.md` | Isaac track: the full HF Jobs recipe and measured facts |

Read both markdown files before changing anything — they record several traps that
each cost a GPU run to find.

## Honest status

- The friction finding is verified in both source trees and does not depend on any
  policy training well.
- One simulator walks across a 20× friction range at no cost to normal-ground
  performance. The other refuses to, for reasons not yet identified.
- `base-v2` is **not** a reliable walker: only 3 of the first 8 seeds reach 300 steps
  even on rock. Single rollouts are illustrations, not measurements.
- No hardware was used. Fall-step counts are not comparable across physics backends.
- The μ̂ estimator — infer friction from proprioception, adapt gait, report it — is
  designed but not built. That's the gap between this and a system.
