# himalaya-hack

Make the MuJoCo Playground **Unitree G1** locomotion policy survive Himalayan
conditions: ice, soft snow, wind, and mixed terrain.

## The thesis everything rests on

The G1 policy that ships with Playground randomizes foot-floor friction over
**U(0.4, 1.0)**. Ice is ~0.05 — roughly an order of magnitude below that floor.
The shipped policy has never had a training signal for the surface this entire
project is about.

`smoke_test.py` check 3 asserts the stock source still says `minval=0.4`. If that
assertion ever fails, the baseline moved and the claim needs rechecking before
anyone repeats it publicly.

## Hard conventions

**Never patch `.venv/` site-packages.** `bootstrap.sh` does a clean install on
the remote GPU box and would silently wipe any such edit. Extend Playground from
project-level files instead — a `randomization_fn` passed to brax, or an env
subclass. Every module here follows that pattern.

**Deepcopy any config you accept.** Playground's envs declare
`config: ConfigDict = default_config()` as a *default argument*, so one config
object is shared by every env built in the process and `config_overrides`
mutates it permanently. Build a wind-off env and then a wind-on env and you get
**two wind-off envs, silently**. `wind.py` guards against this; anything new must
too. This cost real debugging time.

**`hf jobs uv run` ships only the single named script.** Helper modules are not
uploaded. Copy them into the mounted bucket (`/mnt/himalaya-g1`) or the import
fails on the remote box. `smoke_test.py` has a `sys.path` fallback for exactly
this.

## Module map

| File | What it is |
|---|---|
| `ice_randomize.py` | Drop-in `randomization_fn`. Per-foot log-uniform friction 0.05–1.0 + compliant ground. Use `ice_randomize.load()`, not `locomotion.load()`. |
| `wind.py` | `WindyJoystick` env subclass. Sustained drag force + center-of-pressure torque via `xfrc_applied`, AR(1) gusts. |
| `ice_patch.py` | `PatchyIceJoystick`, subclasses `WindyJoystick`. Position-dependent per-foot friction — mid-episode rock→ice transitions. |
| `train_ice.py` | PPO training as an HF job. `--baseline` trains the stock arm for the A/B; `--dry-run` validates the whole path on cpu-basic without spending GPU time. |
| `fixed_line_scene.py` | Generates the ascent scene XML: tilted plane, taut rope as a static capsule, ascender as a zero-DOF mocap marker. |
| `fixed_line.py` | `FixedLineAscent` env. Slope + one-sided tether + ratcheting ascender. Stock DOF, so it composes with `ice_randomize`. |
| `smoke_test.py` | Five-check preflight, runs as an HF job. Validates GPU, env load, our randomizer, headless render, bucket mount. |
| `local_view.py` | Watch a checkpoint walk on your own Mac, one seed, live window or mp4. CPU-tuned (see below); needs `mjpython` for the interactive window. |
| `gpu_view.py` | The same rollout as an HF job: N seeds vmapped + scanned, survival spread and an mp4 to the bucket. Refuses to run off GPU. |
| `stream_sim.py` | Publishes a rollout to a LiveKit room as a live video track for the demo. `--dry-run` exercises everything but the connection. |
| `bootstrap.sh` | Provisioning for the Nebius VM (the other compute path). |
| `recon.html` | Original project recon and track breakdown. |

They compose: `PatchyIceJoystick` → `WindyJoystick` → stock `Joystick`, with
`ice_randomize` supplying the per-episode model randomization underneath.
Verified working together under the brax wrapper.

## Non-obvious findings

**`njmax` must be ≥ 96.** Compliant contact lets the feet penetrate further,
pushing constraint rows past the stock `njmax=90`. An `nefc` overflow **silently
drops constraints** — wrong physics, not a cosmetic warning. `ice_randomize.load()`
and `wind.default_config()` both set 160. Direct `locomotion.load()` calls do not.

**The G1 joystick config hardcodes `impl="warp"`, which is the wrong backend
off-GPU.** `mujoco_warp` has no CPU fast path -- it emits GPU-shaped kernels and
runs them serially -- so on a Mac the shipped default is the slowest of the three
backends. Measured, stock `G1JoystickFlatTerrain`, 50 steps, identical final
`qpos` to 3dp:

| impl | ms/step | 600-step rollout |
|---|---|---|
| `warp` (playground default) | 972 | ~10 min |
| `jax` | 394 | ~4 min |
| `jax`, `naconmax=128` | 300 | ~3 min |

`naconmax` defaults to `8*8192`, sized for thousands of parallel envs. `local_view.py`
carries these as `FAST_CPU` and `gpu_view.py` applies them only when the backend is
not GPU. **Do not put them on a GPU path**: warp is much faster there and
`naconmax=128` overflows once the envs are batched. `impl="cpp"` is not an escape
hatch -- playground's `reset()` calls `mjx.forward`, which raises `forward requires
JAX backend implementation`.

**The brax randomization wrapper does not pass the model as an argument.**
`BraxDomainRandomizationVmapWrapper` assigns the per-env model to
`env._mjx_model` inside a context manager, then calls `env.step`
(`wrapper.py:220-246`). So inside `step`, `self.mjx_model` is already this env's
randomized model as a tracer under vmap. That is what makes `ice_patch.py`
possible in ~10 lines — the same swap trick, one level down.

**`ensure_menagerie_exists()` is only called by `locomotion.load()`.** The env
constructors never call it. Our modules build the classes directly, so on any
clean box the robot assets are never fetched and the XML dies on a missing
`pelvis.STL`. `WindyJoystick.__init__` now calls it, which covers every subclass.
This is invisible locally once the assets are cached — it only bites on a fresh
remote box.

**brax 0.14.2 is broken against current jax.** It calls
`jax.device_put_replicated` (`ppo/train.py:756`), which jax removed — present in
0.9.2, gone by 0.10.2 — and brax declares `jax>=0.4.6` with no upper bound, so
uv resolves the newest jax and PPO dies the moment training starts. 0.14.2 is
the newest brax, so there is nothing to upgrade to. `train_ice.py` shims the one
function rather than pinning jax back three minor versions, which would drag
`mujoco` and `mujoco_warp` with it. The shim is single-device by design and
RAISES on multi-GPU: replicating across devices means building a properly
sharded global array, and a subtly wrong one corrupts training silently. If you
move to `l40sx4` or similar, fix the shim or pin `jax==0.9.2` — do not just
delete the guard.

**A dry run must reach `ppo.train`.** The brax break lives past env construction,
so an env-only preflight sails straight through it and the job dies on GPU
instead. `--dry-run` now executes a real, tiny PPO pass for exactly this reason.
Two GPU launches were lost learning this.

**The job image has no `ffmpeg`.** `mediapy.write_video` shells out to it, so
the mp4 write throws at the very end of a run — after training has completed and
the frames are rendered. A 10-minute GPU run died this way with nothing saved.
`train_ice.py` now installs ffmpeg, writes `eval.json` BEFORE the video, and
falls back to PNGs. General rule for this repo: results before artefacts, and a
render failure must never destroy a training run.

**GPU rollouts are NOT reproducible run to run.** Three `gpu_view.py` runs with
the same policy, the same 16 seeds, the same env, the same `l4x1` and the same
script logic gave mean survival **289.4 / 220.6 / 255.9**. Individual seeds moved
far more: seed 0 was 436, 262, 536. Seeds that fall early are stable — seed 7 was
48 in all three, seed 6 was 68/69/68 — so this is chaotic amplification of
nondeterministic float ordering (GPU atomics in contact detection), not a harness
bug. Three consequences, and the third is the one that bites:

1. A survival number from a single GPU run is a **draw, not a measurement**.
2. A paired A/B inside **one job** is still valid — both arms see the same
   nondeterminism — but the delta must clear the run-to-run spread, which is about
   **±35 in the mean** at 16 seeds × 600 steps.
3. **Never compare a number from one job against a number from another job.** Most
   of the survival figures in this file were produced that way.

**Stock ground and `ice_patch` benign are the same environment, to the policy.**
Paired inside one job, 16 seeds × 600 steps: mean 255.9 stock vs 245.4
patchy-benign, delta −10.6 — well inside the ±35 spread above. So
`diagnose_eval2.py`'s "benign" cell is not measurably different from
`locomotion.load`, and the config difference between the two is not an explanation
for anything.

**"301/300 benign steps on EVERY seed" was never true — the run it cites says
otherwise.** `diagnostics/eval_control.json`, the artifact that claim comes from,
records `base-v2-baseline|benign` as **seeds [301, 301, 196], mean 266**. One of
its three seeds fell at 196. The claim is a misreading of its own data, and it is
repeated in `ice_randomize.py`, `local_view.py` and this file.

A rerun on `cpu-basic` with the shipped `impl=warp` config and the same
`ice_patch` benign env — i.e. `diagnose_eval2.py`'s exact setup — reproduces that
file to within 4 steps: **[301, 301, 200]** on seeds 0–2. Extending to 8 seeds,
only 0, 1 and 5 reach 300; seeds 2, 3, 4, 6 and 7 fall at 200, 148, 125, 68 and
48. At 600 steps on GPU, **16/16 fall**.

The early-falling seeds agree closely across CPU and GPU — seed 7 is 48 on both,
seed 6 is 68 vs 68/69/68, seed 4 is 125 vs 126/130/128 — so the backend is NOT an
explanation and the nondeterminism above only affects long-horizon runs. The
honest summary: the baseline walks reliably on a **minority of seeds**, and on
none of them for 600 steps. Anything that treats it as a known-good control needs
requalifying.

**`hf jobs uv run` swallows `--env`, even after the script name.** It has its own
`-e/--env` for environment variables, and typer parses options interspersed with
positional args, so a script flag called `--env` never reaches the script — which
then runs with its DEFAULT and reports a perfectly plausible wrong answer. Flags it
does not recognise (`--steps`, `--seeds`) pass through fine, so the failure is
silent and partial. `gpu_view.py` calls the flag `--arena` for this reason. Check
`hf jobs inspect <id>` and read the `command` array before trusting any job that
took flags.

**`hf cp`, not `hf upload`, for the bucket.** `upload` targets repos. Re-run the
cp after ANY edit to `ice_randomize.py` / `wind.py` / `ice_patch.py`, or the job
silently trains against a stale copy.

**Angles in any scene XML here are RADIANS.** `g1_mjx_feetonly.xml` declares
`<compiler angle="radian"/>`, and that governs the whole compiled model
including the parent scene that includes it. `fixed_line_scene.py` emitted
`euler="0 -30 0"` intending 30 degrees and got **30 radians = 81 degrees**: the
floor normal compiled to (0.988, 0, 0.154) instead of (-0.5, 0, 0.866), so the
"30 degree slope" was a near-vertical wall.

This invalidates everything run against the ascent env before it was fixed --
including the `ascent-v1` NaN, which was training a humanoid to walk up an 81
degree face. It also explains the "robot falls through the floor" behaviour in
the ascent scene specifically, which was NOT the stock feetonly artefact
described below. It surfaced only because the onboard camera showed sky where
the ground should have been; nothing in the physics complained.

**Zero-action rollouts fall through the floor (torso z ≈ −0.67).** This is stock
behavior, not a bug in our code. `done=1` fires but nothing resets without the
brax auto-reset wrapper. Do not go debugging it.

**Wind is critic-only.** It is in `privileged_state`, never the actor's `state`.
A real robot has no anemometer and must infer loading from its IMU — that is the
skill being trained. `enable=False` is bitwise-identical to stock on the same
seed, so same-seed wind-on/off is a clean A/B.

## Fixed-line ascent: two rejected designs

The G1 `feetonly` model has 29 actuators and NO fingers. A `with_hands` variant
exists (43 actuators, 14 finger joints) but grasping a 12 mm rope is a
contact-rich manipulation problem that would dominate the learning.

It does not matter, because **real climbers do not grip the rope** — they clip a
mechanical ascender (jumar) that slides up freely and locks under load. Modeling
grip as a mechanism is the faithful choice, not a shortcut.

**Rejected: rigid `connect` equality** pinning the pelvis to a slider on the
rope. Measured 327 N of tether load against a 330 N robot — the rope carried the
entire robot and the legs were decorative. A zip line, not an ascent.

**Rejected: simulating the rope.** A fixed line is anchored and taut, therefore
straight. A cable composite would add hundreds of DOF and wreck MJX batching to
model something we are not studying.

**Shipped:** a one-sided spring-damper tether through `xfrc_applied` (same
mechanism as wind), slack until the robot drops 0.35 m, plus a ratchet that
tracks the high-water mark so a slip drops you to the ascender, not the bottom.
The ascender is a mocap marker. This keeps nq=36, nv=35, nu=29, neq=0 — exactly
stock — so obs stays 103/216 and everything composes.

Also deliberate: the command is pinned forward and the slope converts forward
velocity into height, plus a reward per metre gained. Playground's walking
reward is already tuned; a hand-rolled climbing reward would be a second
research project.

## LiveKit demo streaming

**There is no bare `python` on this Mac.** `python3` is a homebrew 3.14 with none
of the project's dependencies; everything here runs on `.venv/bin/python`
(3.12.13). Any command in this file written as `python foo.py` means
`.venv/bin/python foo.py`.

**Verified working end to end** (2026-08-29): connects, mints its own publisher
token, publishes an RGBA video track, disconnects cleanly. Watch it with
`.venv/bin/python stream_sim.py --viewer-token --room himalaya-g1`, which prints a
SUBSCRIBE-ONLY token and a meet.livekit.io link — never hand out the publisher
token.

Credentials live in `.env` (gitignored, and `.env.example` is the template).
**Never put them in a `.py` file and never `hf cp` the `.env`**: every module here
gets copied into the shared team bucket, and `LIVEKIT_API_SECRET` is a signing key
that mints room tokens.

`stream_sim.py` publishes the sim to a LiveKit room as a video track. Needs
`LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` from a project at
cloud.livekit.io; the script never stores or prints them, and `--dry-run` tests
the whole pipeline while connecting to nothing.

**Frames are pre-rendered, then published on a wall-clock timer and looped.**
Rendering, not physics, is the bottleneck: ~14 fps at 320x240 on this Mac against
a 50 Hz control rate, and the FAST_CPU backend switch does not help because it
speeds up physics, not the renderer. Streaming as it simulates would put out a
slideshow and call it live. On a CUDA box with EGL, `--live` can go truly live.

**brax cannot read back its own checkpoints.** `save` writes
`"mean_kernel_init_fn": null`; `load_config` then does `KERNEL_INITIALIZER[None]`
and raises KeyError. `stream_sim.load_policy_compat` registers the alias rather
than editing saved artifacts.

## Deliberately out of scope

- **Crevasse as randomized terrain** — ~1 GB to vmap 256×256 hfields over 4096
  envs. Doing one hand-authored hfield as a fixed eval terrain instead.
- **Avalanche, serac collapse, GLOF** — no controller recovers from these. They
  are routing problems, not control problems, and belong in the turn-back
  decision engine. Keeping that split explicit is a stronger story than implying
  the policy handles them.

## Running on GPU (Hugging Face Jobs)

Local CPU runs at roughly 0.3 s/step single-env on the tuned backend (0.97 on the
shipped one — see the `impl` finding above) — fine for correctness checks, useless
for training. Real runs go to HF Jobs. On `l4x1`, MJX/warp steady-state is
**1.16 ms per env-step (860 env-steps/s)** at 16 envs — but the FIRST env built in
a process pays ~80 s of warp CUDA kernel compilation, so a single short rollout
measures compilation, not simulation. A second env in the same job costs 11 s.

```bash
hf jobs uv run \
  --namespace iteratehack --flavor l4x1 --timeout 40m \
  -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
  --label name=himalaya-traction --label task=<task> \
  <script>.py
```

- `hf` CLI is installed (v1.19.0). `hf jobs hardware` lists flavors.
- Flavors seen in use: `l4x1`. Bigger options include `l40sx1`, `a100-large`,
  `h200`. Pick by cost, not reflex.
- The bucket mounts read+write at `/mnt`; artifacts go to `/mnt/himalaya-g1/`.
- Add `-d` to detach for long runs.
- Renders are headless: set `MUJOCO_GL=egl`, write mp4, copy it down.
- **Ask before launching a paid GPU job.** These cost money and are not free to
  retry.

## The A/B, and a shape trap

The demo rests on comparing a policy trained on ice against the stock one.

**Do not quote the training reward curves as the A/B.** brax applies the same
`randomization_fn` to `eval_env` as to the training env (`ppo/train.py:759-769`)
and offers no separate eval randomizer, so each arm is evaluated on its own
terrain distribution -- the ice arm on friction 0.05-1.0 with compliant ground,
the baseline on 0.4-1.0 with hard ground. The curves measure different things.

The fair comparison is the final eval in `train_ice.py`, which drives the raw
`eval_env` object rather than brax's wrapped copy: identical terrain, identical
seed, both arms. `eval.json` (survived steps, min friction) and `eval.mp4` are
the numbers to quote.

The baseline arm does NOT use `locomotion.load()`. brax builds the value network
from the *train* env, so a 216-dim stock env against our 219-dim eval env
crashes at the first eval. The baseline instead uses our env with wind and
patches disabled, which is bitwise identical to stock on the same seed and keeps
the shapes comparable. Both arms must share `--seed` and `--num-timesteps`.

## Results so far — read before quoting anything

**First 50M pair (`ice-v3`) was a negative result.** Both arms fell over in about
one second on the Himalayan eval:

| | training reward | shared eval survival |
|---|---|---|
| ice | −2.56 | 37 / 501 steps |
| baseline | −15.45 | 64 / 501 steps |

Two things to understand before repeating those numbers:

1. **The 37 vs 64 was a single rollout per arm.** Survival on ice is high
   variance; n=1 cannot separate two policies that both fail immediately. That
   comparison was noise. `ice-v4` runs 16 rollouts and reports mean, std, median
   and every raw value.
2. **The training-reward gap is narrower than it looks.** brax evaluates each arm
   with its own randomizer, so the baseline is scored on ice and wind it never
   trained on. "Ice policy beats a policy that never saw ice" is nearly
   tautological, and it did NOT transfer to the shared test.

Working hypotheses for why both fail, in order:

- **50M is a quarter of Playground's 200M default**, on harder terrain than the
  default. Most likely just undertrained.
- **The friction distribution may be miscalibrated.** Log-uniform 0.05–1.0 puts
  37% of feet on ice-grade friction; the policy may never get enough easy
  experience to learn walking at all. If so the fix is a curriculum (start near
  the stock range, anneal down), not more steps.

`ice-v4` added a **flat-terrain control eval** to separate these. Result, 16
rollouts each:

| | Himalayan | flat control |
|---|---|---|
| ice-trained | 39.9 ± 15.2 | 43.2 ± 9.8 (median 42) |
| baseline | 122.1 ± 144.6 | 454.9 ± 108.6 (median 501) |

**These two arms ran as SEPARATE JOBS**, which the nondeterminism finding above
says never to compare directly. The comparison survives only because of its size:
the flat-control delta is ~410 against a ±35 run-to-run spread, an order of
magnitude clear of the noise. So "the ice-trained policy cannot walk on flat
ground" holds. The Himalayan delta (82) clears the floor far less comfortably and
should be re-measured paired inside one job before anyone quotes it.

The surviving conclusion: log-uniform 0.05–1.0 puts 37% of feet on ice-grade
friction and the policy never learns to walk at all. The fix is a curriculum
(start near the stock range, anneal the floor down), not more steps.

Note also that `ice-v4-baseline`'s flat control has std 108.6 around a median of
501 — so several seeds fell well short. Consistent with the finding above that
the baseline walks on a **minority of seeds**, not all of them. Do not describe
any baseline here as a known-good control.

## Status

Built and verified: per-foot ice friction, compliant ground, wind, mid-episode
rock→ice patches, fixed-line ascent, and the training pipeline.

No policy yet walks on ice. Do not put ice survival numbers in a deck until
`ice-v4` reports with error bars.

Open: crevasse hfield for the eval video; longer runs; the curriculum question
above.
