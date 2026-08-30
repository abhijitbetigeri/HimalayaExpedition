---
name: himalaya
description: Work on the Himalaya G1 ice-locomotion project — train, evaluate, film or diagnose Isaac Lab / MuJoCo Playground policies on Hugging Face Jobs. Use when the task touches this repo's training runs, HF Jobs, ice/slope/wind/tether policies, or the demo artifacts.
---

# Himalaya G1 — working rules

Read `STATE.md` (live state), `MORNING.md` (runbook), `ISAAC_TRACK.md` (Isaac + HF
Jobs recipe) and `CLAUDE.md` (MuJoCo track) before acting. They record traps that
each cost a GPU run.

## Non-negotiables

**One variable per run.** Every MuJoCo ice run changed four things at once and none
of them is interpretable as a result. The Isaac runs changed one and worked first
try. If a run would change two things, split it.

**n=1 is not a result.** A single rollout gave "ice 38 vs baseline 47" and looked
like ice was *better*. Three seeds gave 35.3 vs 266.0 and reversed the meaning.
Report mean and spread, always. Survival on ice is high-variance.

**Run the control.** The benign-ground baseline is what proved the harness was
sound; without it the obvious and wrong conclusion was "the eval is broken".

**Verify before spending.** Assert the config actually changed — and, for anything
structural, that the *state* changed too (e.g. the getup env asserts robots really
start at pelvis height < 0.45 m) — in a cheap subprocess that exits non-zero.
Never discover at minute 40 that an override silently did not apply.

**Results before artefacts.** Write the numbers, then the video. A render failure
must never destroy a training run.

**Get data out via the bucket, not stdout.** Kit prints thousands of startup lines;
anything early is lost to `tail`. Three diagnostics were thrown away this way.

## Aim at the right surface

μ = 0.05–0.08 is bare wet ice, near the physical limit for a legged system — humans
need crampons. A robot falling there is physics, not a policy failure. Train across
the range, but **film and quote at μ = 0.12–0.20** (glacial ice to packed snow),
which is walkable and is what an expedition actually crosses.

## HF Jobs

Credits are on the **org**: always `--namespace iteratehack` (personal namespace
402s). The org is shared with other hackathon participants — check
`hf jobs ps --namespace iteratehack` before blaming a job for being slow;
`SCHEDULING` usually means the flavor is contended, and `l4x1` is the most popular.
Prefer `l40sx1` / `a100-large` when `l4x1` is busy.

Long runs need `--timeout` (default is 30 min). Mirror checkpoints to the bucket on
a timer — jobs are killed at timeout with an ephemeral filesystem.

The full uv/Isaac header (python 3.12, `isaacsim==6.0.1.0`, mixed index strategy,
prereleases allowed, `pip` as a dep) is in `ISAAC_TRACK.md` §2. Every line is
load-bearing.

## Traps that will recur

- `play.py --video` records **nothing**. Use `render_mode="rgb_array"` +
  `AppLauncher(enable_cameras=True)` and an explicit capture loop.
- Exported TorchScript lands **next to the checkpoint** (`<ckpt>/exported/policy.pt`,
  i.e. in the bucket), not in `IsaacLab/logs`.
- Isaac **auto-resets**, so "steps without a termination event" is ~99% even for a
  robot falling constantly. Count cumulative terminations.
- Default Isaac camera is a wide overhead shot. Set `viewer.origin_type="asset_root"`,
  `asset_name="robot"`, and one env when on generated terrain.
- brax writes checkpoints it cannot read back (`*_kernel_init_fn: null`). Patch a
  copy before loading.
- MuJoCo `impl=jax` is 3× faster on CPU and **degrades the policy** (301/301 under
  warp, falls at 112 under jax). Film and quote under **warp**.
- `ffmpeg` is a system binary; having `mediapy` is not enough.
- zsh does not word-split unquoted vars — flags in `$VAR` arrive as one argument.

## Do not repeat these wrong claims

- `base-v2` is not a reliable walker (3 of 8 seeds reach 300 steps). "301/300 on
  every seed" was a misreading of `[301, 301, 196]`.
- Fall-step counts are not comparable across physics backends.
- Isaac training metrics (97.8% survival) were measured on the **mixed** friction
  distribution, not on bare ice. At pinned μ=0.08 the ice policy still falls often.

## Safety

`.env` holds real LiveKit credentials and the GitHub repo is **public**. It is
gitignored — never overwrite `.gitignore` wholesale. Ask before pushing to GitHub;
the user asked to hold pushes.
