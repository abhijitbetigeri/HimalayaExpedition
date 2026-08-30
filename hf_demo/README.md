---
license: mit
task_categories:
- reinforcement-learning
tags:
- robotics
- humanoid
- locomotion
- isaac-sim
- unitree-g1
- sim2real
pretty_name: "Himalaya G1 — humanoid locomotion on ice"
---

# Himalaya G1 — teaching a Unitree G1 to handle ice

Track 1 (Movement) entry for the Himalaya Robotics Hack, August 2026.
Code: [github.com/abhijitbetigeri/HimalayaExpedition](https://github.com/abhijitbetigeri/HimalayaExpedition)

---

## The finding

**Humanoid locomotion benchmarks train the Unitree G1 on ground an order of
magnitude grippier than ice — and one of them does not vary the friction at all.**

Isaac Lab, `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`:

```python
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    params={"static_friction_range":  (0.8, 0.8),   # min == max
            "dynamic_friction_range": (0.6, 0.6)})
```

Minimum equals maximum: the humanoid sees exactly **one** surface for the whole of
training, and neither G1 config overrides it. In the same source tree the
*quadruped* Spot gets `(0.3, 1.0)` / `(0.3, 0.8)`.

**The four-legged robot gets friction randomisation. The two-legged one does not.**

MuJoCo Playground is better but not by much — `U(0.4, 1.0)`, a floor still ~8×
above ice. Real ice is μ = 0.05–0.15.

This is verifiable in both source trees and depends on no policy training well.

---

## Result: widening the range works, at no cost on normal ground

Cross-evaluation, both policies on identical surfaces, **256 environments × 1000
steps, all six cells measured inside a single job**:

| falls per env | rock (μ=0.80) | ice (μ=0.08) | icy slope |
|---|---|---|---|
| **baseline** (μ=0.8 fixed) | **0.00** | 20.96 | 19.41 |
| **ice-trained** (μ ∈ 0.05–1.0) | **0.00** | **9.58** | **13.69** |

- **2.2× fewer falls on ice**, 1.4× on the icy slope
- **Zero regression on rock** — the cell people forget to check
- Training metrics: 97.8% vs 98.0% episode survival, 0.912 vs 0.913 velocity tracking

At walkable winter frictions the gap is wider still: on packed snow (μ=0.20) the
ice policy records **0.00 falls** against the baseline's 2.5; on glacial ice
(μ=0.12), **2.17 against 20.0**.

### Why single-job measurement matters

GPU rollouts here are **not reproducible run to run** — the same policy, seeds and
environment gave mean survivals of 289.4 / 220.6 / 255.9 across three identical
runs (±35 in the mean at 16 seeds). Chaotic amplification of nondeterministic float
ordering in contact detection. A paired A/B *inside one job* is valid because both
arms see the same nondeterminism; numbers from *different* jobs are not comparable.

---

## Terrain from the real Everest DEM

Built from NASA SRTM 1-arc-second tile `N27E086`, verified to contain Everest —
**8748 m** at the summit coordinates, the accepted SRTM value.

```
patch     27.9133°N, 86.5400°E  (Khumbu)
slope     15.8° mean (sd 15.4)
real      1440 m across, 607 m relief
scaled    ×0.025 → 36 m across, 15.2 m relief, slope preserved at 15.8°
```

**Honest framing:** SRTM samples every ~30 m and a G1 foot is ~0.25 m, so at 1:1
the robot walks a plane with occasional 30 m cliffs. The patch is scaled
**uniformly** — both axes by the same factor — which preserves the true slope
*angle* exactly while compressing the footprint to something crossable. The
gradient and ground shape are real; the absolute size is not. This is *"terrain
profile from the real Everest DEM"*, **not** *"the robot is walking on Everest"*.

---

## What is NOT claimed

- **Fall recovery does not work.** Many attempts; the failures were infrastructure
  (see the repo's `STATE.md`) and the learning question is still open.
- **Bare wet ice (μ≈0.06) is near the physical limit** for any legged system —
  humans need crampons. A robot falling there is physics, not a policy failure.
  Results are quoted at μ = 0.12–0.20, which is what an approach march crosses.
- **No hardware.** Simulation only.
- Fall-step counts are **not comparable across physics backends** (MuJoCo `warp`
  vs `jax` diverge substantially on long rollouts).
- The **μ̂ estimator** classifies ice vs rock from proprioception at 93.7%, but
  **cannot regress μ** (held-out R² is negative). It is a detector, not an estimator.

---

## Videos

The A/B pairs. Same surface, same conditions, different policy — watch 1 against 2,
and 3 against 4.

| file | surface | falls per env |
|---|---|---|
| `videos/1_ICE-POLICY_on_snow_ZERO-falls.mp4` | packed snow, μ=0.20 | **0.00** |
| `videos/2_BASELINE_on_snow_2.5-falls.mp4` | packed snow, μ=0.20 | 2.50 |
| `videos/3_ICE-POLICY_on_glacial-ice_2.2-falls.mp4` | glacial ice, μ=0.12 | **2.17** |
| `videos/4_BASELINE_on_glacial-ice_20-falls.mp4` | glacial ice, μ=0.12 | 20.0 |

### Read these before judging the footage

**The numbers are sound; the cinematography is not.** These are early renders and
they have known defects:

- **Six robots per frame**, at different depths and in different episode states.
  Some are walking, some mid-fall, some just reset. Limbs that appear to fly off
  belong to *other* robots. The fall counts are per-environment averages across all
  six, not something you can count by eye.
- **The large green shape** is Isaac Lab's velocity-command debug marker, not part
  of the scene.
- **The ground renders as a dark grid**, not snow. Friction is set correctly in
  physics (μ=0.20 / 0.12); appearance and physics are configured separately in
  Isaac Lab and only the physics was set for these clips.
- The robots are distant, so the gait is hard to read.

Better renders are in progress — one robot, snow that looks like snow, alpine HDRI
lighting. They are not published yet because a rendering bug (limb meshes not
following the articulation) is still being fixed, and clips are only added here
after being checked frame by frame.

Every clip is verified visually before publishing: several earlier batches passed
automated quality scoring and were unusable on inspection.

## Reproducing

Everything runs on Hugging Face Jobs; the full Isaac Sim recipe (and the traps that
cost real GPU time) is in the repo's `ISAAC_TRACK.md`.

```bash
git clone https://github.com/abhijitbetigeri/HimalayaExpedition
```
