# Isaac Sim / Isaac Lab track — working notes

Companion to `CLAUDE.md`, which covers the MuJoCo Playground track. This file
covers the Isaac side plus the cross-cutting experimental findings. Read both.

Everything below was established empirically on 2026-08-29 and each claim names
the evidence. Where something is unverified it says so.

---

## 1. The finding the project rests on

The claim is stronger in Isaac Lab than in MuJoCo, and both are verified in-tree.

**Isaac Lab** — `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
around line 200:

```python
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    params={"static_friction_range":  (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6), ...})
```

**min == max.** The humanoid locomotion task does not randomise ground friction at
all — it trains at exactly one value. Neither `config/g1/flat_env_cfg.py` nor
`rough_env_cfg.py` overrides it (verified by grep: "NO friction/physics_material
override" in both).

For contrast, `config/spot/flat_env_cfg.py` in the same tree uses
`(0.3, 1.0)` / `(0.3, 0.8)`. **The quadruped gets friction randomisation; the
humanoid does not.**

**MuJoCo Playground** — `_src/locomotion/g1/randomize.py` line 26: friction is
`U(0.4, 1.0)`, a real range but with a floor ~8x above ice.

Real ice is mu = 0.05-0.15. Both stacks are trained well above it.

---

## 2. Isaac Sim on Hugging Face Jobs — the complete recipe

This took seven failed attempts. Every line below is load-bearing.

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
```

| Setting | Why it is required |
|---|---|
| `>=3.12,<3.13` | isaacsim **6.0.1.0 publishes ONLY cp312 wheels**; the 5.x line is cp310/cp311. Isaac Lab main declares `requires-python >=3.12`. 3.12 is the only version satisfying both. |
| `==6.0.1.0` pin | Unpinned under `unsafe-best-match`, uv walks down to isaacsim 5.0.0.0 on PyPI, whose packages are `wheel_stub` placeholders that fail to fetch real wheels. |
| `index-strategy` | On 3.12 the dependency closure spans BOTH PyPI and NVIDIA's index (e.g. `mujoco-usd-converter`). uv refuses to mix indexes by default as a dependency-confusion guard. This is a real security relaxation — keep it scoped to the script. |
| `prerelease = "allow"` | `isaacsim-core==6.0.1.0` hard-depends on `tinyobjloader==2.0.0rc13`, a pre-release. pip accepts pinned pre-releases silently; uv refuses unless told. |
| `"pip"` in deps | uv's ephemeral venv ships **no pip at all**, and Isaac Lab's installer shells out to `python -m pip`. |

Runtime requirements:

```bash
--env OMNI_KIT_ACCEPT_EULA=YES     # else Kit blocks on an interactive prompt and
                                   # dies on EOF: "Unable to bootstrap inner kit kernel"
apt-get install -y libgl1 libglu1-mesa libegl1 libvulkan1 \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libsm6 libice6 libxt6 libgomp1
```

```python
SimulationApp({"headless": True, "multi_gpu": False,
               "active_gpu": 0, "physics_gpu": 0})
```

**`multi_gpu: False` is mandatory.** With the default, Kit CRASHES in the RTX Hydra
renderer at startup (`librtx.hydra.so`, `libusdrt.hydra.fabric_scene_delegate`),
because `vulkaninfo` enumerates GPU0 **and** GPU1 for a single physical L4. Startup
is ~13.6 s once correct.

**PhysX runs on CPU unless you ask for a device.** `"CUDA libs are present, but no
suitable CUDA GPU was found!"` means the World was built without one. Pass
`device="cuda:0"` to `World(...)` or `AppLauncher(...)` and you get
`use_gpu_pipeline: True`, `use_gpu_sim: True`.

### Isaac Lab install

Not on PyPI — clone from git. The physics backends are **not optional**
(`isaaclab_tasks` imports `isaaclab_ovphysx` at registration time), and `play.py`
needs `isaaclab_visualizers`:

```
isaaclab, isaaclab_ov, isaaclab_physx, isaaclab_ovphysx, isaaclab_newton,
isaaclab_assets, isaaclab_rl, isaaclab_tasks, isaaclab_visualizers
```
then `pip install rsl-rl-lib`. Whole install ~4 min.

### Registering a custom env

Each job re-clones Isaac Lab, so registration does **not** persist between jobs —
re-inject it every time. Write a cfg module into
`isaaclab_tasks/.../locomotion/velocity/config/g1/` and append a `gym.register`
block to that package's `__init__.py`. See `train_ice_isaac.py`.

Always **verify the override applied before spending GPU time** — assert the
config value in a tiny subprocess and exit non-zero if it did not.

---

## 3. Measured facts about the Isaac G1

| | value |
|---|---|
| Task ids | `Isaac-Velocity-Flat-G1-v0`, `Isaac-Velocity-Rough-G1-v0` (+ `-Play-` variants) |
| Observation | `(123,)` |
| Action | **`(37,)`** — 29 DOF + 8 finger joints |
| Bodies | 44 |
| Foot contact bodies | `left_ankle_roll_link`, `right_ankle_roll_link` |
| Throughput | 23.7k env-steps/s @1024 envs on `l4x1`; ~59k on `a100-large` |

**The Isaac G1 is NOT the same robot as COLA's or Playground's** (both 29 DOF,
obs 103). Do not assume dimensions transfer between the tracks or to hardware.

`config/g1/flat_env_cfg.py` also carries a dual physics backend: `PhysxCfg`
(default) and `NewtonCfg(MJWarpSolverCfg(njmax=95, ...))`. The Newton backend is
MuJoCo underneath, so MuJoCo-style compliant contact (`solref`/`solimp`) IS
reachable there — contrary to a PhysX-only reading. Note `njmax=95`, just under
the >=96 that `CLAUDE.md` documents as necessary under compliant contact.

Reward terms that matter for ice: `feet_slide -0.1`, `termination_penalty -200`.
Playground's equivalent is `feet_slip -0.25` — 2.5x harsher.

---

## 4. Results

### Isaac (works)

| | baseline (mu=0.8 fixed) | ice (mu in 0.05-1.0 uniform) |
|---|---|---|
| success_rate | 1.0000 | 1.0000 |
| survive full episode | 98.0% | 97.8% |
| velocity tracking | 0.9129 | 0.9119 |
| error_vel_xy | 0.1332 | 0.1342 |

The ice policy holds baseline performance across a 20x friction range.
**Caveat: these are each policy's metrics on its OWN distribution.** A real A/B
needs cross-evaluation on identical ground — that is `cross_eval_isaac.py`.

The ice run changed TWO things (friction range AND `feet_slide` -0.1 -> -0.01), so
its success cannot be attributed to the friction range alone.

### MuJoCo (does not walk yet)

Control eval, 3 seeds, existing checkpoints, `diagnose_eval2.py`:

| policy | condition | seeds | mean | % |
|---|---|---|---|---|
| ice-v2 | benign | 33, 47, 26 | 35.3 | 11.8% |
| ice-v2 | full stack | 35, 22, 34 | 30.3 | 10.1% |
| base-v2 | benign | 301, 301, 196 | 266.0 | **88.7%** |
| base-v2 | full stack | 47, 39, 148 | 78.0 | 26.0% |

This rules out three hypotheses at once:
- **harness is fine** — the baseline walks through the same code path
- **25M steps is enough** — same budget and seed as the failing arm
- **ice-v2 never learned to walk at all**, on any surface

`base-v2` full-stack 26.0% vs benign 88.7% is the gap the project exists to close,
now with error bars.

**A hypothesis that was tested and REFUTED:** that log-uniform friction sampling
was the cause. Switching to uniform (`ice-uniform`) still failed —
30.5 +/- 11.8 on the Himalayan eval and **35.1 +/- 6.4 on stock flat terrain**.
Distribution alone is not the explanation.

---

### MuJoCo ablation (2026-08-29) — which factor breaks walking

25M steps each, 16 rollouts, control column is stock flat terrain, out of 501:

| run | isolates | control | himalayan |
|---|---|---|---|
| A1-friction | wide friction alone | 35.6 +/- 7.2 (7%) | 28.2 +/- 10.4 |
| **A2-compliant** | compliant ground | **390.5 +/- 137.5 (78%)** | 172.4 +/- 164.1 |
| A3-wind | wind alone | 65.8 +/- 15.9 (13%) | 54.6 +/- 16.6 |
| A4-patches | patches alone | 11.6 +/- 2.6 (2%) | 10.7 +/- 2.7 |
| A5-slip | friction + feet_slip -0.05 | 35.9 +/- 7.0 (7%) | 28.0 +/- 10.5 |

Conclusions, including two REFUTED hypotheses:

- **`feet_slip` is NOT the culprit.** A5 == A1 within noise. Relaxing the slip
  penalty changed nothing. (Hypothesis refuted.)
- **"Too many factors at once" is NOT the explanation.** Three factors break
  walking INDIVIDUALLY; it is not an interaction effect. (Hypothesis refuted.)
- **Compliant ground is harmless** (78%, the only run that walks). The
  solref/solimp work and the njmax>=96 fix are sound — keep that layer.
- **Patches are the single most damaging factor** (2%), worse than ice itself, and
  that is with STOCK friction. `ice_patch.py`'s mid-episode rock->ice transitions
  need scrutiny before they are used again.

**The step-count hypothesis is back, for the HARD task only.** A1 (friction
U(0.05,1.0), nothing else) is exactly what the Isaac ice run did, and Isaac reached
98%. The difference is budget: Isaac trained ~147M steps (1500 x 24 x 4096), MuJoCo
A1 trained 25M — 6x fewer. base-v2 walks at 25M because the STOCK task is easy;
widening friction makes it much harder. Earlier this hypothesis was declared dead —
it was dead for the easy task, not the hard one. `A1-long-150M` tests it.

## 5. Methodology lessons — these cost real time

**Never change more than one thing per run.** Every MuJoCo ice run varied four
factors simultaneously (friction range, compliant ground, wind, patches), so none
of them is interpretable. The Isaac run varied one and worked first try. The
ablation (`A1`-`A5`) exists to undo this mistake.

**n=1 is not a result.** A single rollout gave ice 38 vs baseline 47 and looked
like "ice is worse". Three seeds gave 35.3 vs 266.0 and reversed the meaning.
Survival on ice is high variance; always report mean and spread.

**Always run the control.** The benign-ground baseline is what proved the harness
was sound. Without it the obvious (wrong) conclusion was "the eval is broken".

**Get data out via files in the bucket, not stdout.** Kit's startup is thousands of
lines, so anything printed early is lost to `tail`. Three separate diagnostics were
thrown away this way. Checkpoints survived only because a 3-minute rsync loop wrote
them to the bucket.

**Physics backends diverge; never compare fall steps across them.** The same
policy, same seed, same friction fell at step 178 under `impl="warp"` (the
Playground default, used by jobs) and step 82 under `impl="jax"` (used by
`local_view.py` for CPU speed). Verified NOT a contact-overflow bug: `naconmax=128`
and stock `naconmax` give identical results (both 82). It is ordinary numerical
divergence amplified over a long humanoid rollout. Local footage is for looking at;
quoted numbers must come from one backend consistently.

**Results before artefacts.** A render failure must never destroy a training run —
write the numbers first, then the video.

**Verify before spending.** Assert the config actually changed in a cheap
subprocess before launching an hour of GPU.

---

## 6. Traps that will bite again

**brax writes checkpoints it cannot read back.** Unset optional initializers are
serialised as `null`; `load_config` then does `KERNEL_INITIALIZER[None]` ->
`KeyError: None`. **Every checkpoint this project has produced is affected** —
rendering, resuming and hardware deployment all hit it. Fix: copy the checkpoint,
rewrite any `*_kernel_init_fn: null` to `"lecun_uniform"`, load the copy. Safe
because initializers only produce initial values. See `diagnose_eval.py`.

**`ffmpeg` is a system binary.** `mediapy` shells out to it, so listing `mediapy`
in dependencies is not enough. A 10-minute GPU run trained fully, rendered every
frame, then died on `write_video`.

**The eval loop does not auto-reset.** Once `done=1` the robot sinks through the
floor, so ~90% of a failed run's mp4 is empty ground. Stop at termination before
using any clip as a demo asset.

**HF Jobs cost model.** Billed per minute while Starting or Running; default
timeout is only 30 min, so long runs need `--timeout`. Credits live on the **org**,
not the user — `--namespace iteratehack` is mandatory, and the token needs
`job.write` scoped to the org (user-scope alone gives 403; personal namespace gives
402). Cost per step is roughly flat across flavors, so pick by wall-clock, not price.

---

## 7. Where things live

```
hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/
  baseline/rsl_rl/g1_flat/2026-08-29_20-54-12/model_1499.pt   Isaac baseline
  ice-isaac/rsl_rl/g1_flat/2026-08-29_21-48-51/model_1499.pt  Isaac ice
  runs/<name>/checkpoints/...                                 MuJoCo (orbax)
  runs/<name>/eval.json, eval.mp4                             MuJoCo evals
  diagnostics/                                                control evals
  videos/                                                     Isaac clips
  ice_randomize.py, wind.py, ice_patch.py                     env modules (hf cp these)
```

Jobs are labelled `name=himalaya-traction`; filter on that at
`huggingface.co/organizations/iteratehack/settings/jobs`.

Scripts in this repo: `train_baseline.py`, `train_ice_isaac.py`,
`cross_eval_isaac.py`, `play_isaac.py`, `diagnose_eval.py`, `diagnose_eval2.py`,
`setup_isaaclab.py`, `probe_isaac.py`, `introspect_mdp.py`.

---

## 8. Open items

- **Layer 1 of the original pitch does not exist.** The plan was estimate friction
  from proprioception -> adapt gait -> report it in a 340-byte Iridium packet. What
  exists is domain randomisation. COLA (arXiv 2510.14293) supplies the architecture:
  teacher on privileged contact/foot-velocity, BC-distilled to a proprioception-only
  student, history length 25, joint-tracking-error as the interaction-force proxy.
  This is what turns a config change into a system.
- **"Stand back up after falling on ice"** is an entire unclaimed Track 1 bullet and
  visually the most compelling clip available.
- Isaac cross-evaluation (the 2x2) not yet completed.
- MuJoCo ablation `A1`-`A5` in flight.
- Neither track has a usable demo video yet.
