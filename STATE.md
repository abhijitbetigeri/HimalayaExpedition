# Checkpoint — 2026-08-29 ~19:45 PDT

Resume point for whoever picks this up: this session, a fresh session, or another
agent. Read `CLAUDE.md` (MuJoCo track) and `ISAAC_TRACK.md` (Isaac track + the HF
Jobs recipe) alongside it. This file is the *live* state; those two are the durable
knowledge.

**Hackathon ends 2026-08-30.** Spend so far ≈ $17 of $30; the user has said not to
optimise for credits.

---

## 1. Jobs in flight

All under org namespace `iteratehack`, label `name=himalaya-traction`.

| task | job id | delivers | if it fails |
|---|---|---|---|
| `train-ice-slope` | `6a937f59984507d9db4ec424` | G1 on ice **+ slopes** (rough terrain, height scanner) | fall back to the flat ice policy; incline is a stretch |
| `train-getup` | `6a938e3345686a1580c16aed` | **fall recovery on ice** — Track 1 bullet 2 | reward shape is the first suspect; height/orientation weights |
| `isaac-film3` | `6a938eb8984507d9db4ec510` | footage w/ tracking camera + real fall counts | 5th attempt; see §4 for the four prior failure modes |
| `cross-eval2` | `6a938fa0984507d9db4ec51b` | **2 policies × 3 surfaces** — the headline table | |
| `estimate-mu` | `6a938ff1984507d9db4ec51d` | **layer 1**: friction from proprioception | |

Check any of them:

```bash
hf jobs inspect iteratehack/<id> | tail -1
hf jobs logs iteratehack/<id> 2>&1 | grep -avE "neuraylib|material_library|Warning" | tail -40
```

Results land in the bucket, not just stdout — stdout tails lose early output behind
Kit's startup noise:

```
hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/
  ice-slope/rsl_rl/...            slope policy checkpoints
  getup/rsl_rl/...                recovery policy checkpoints
  videos/isaac_*.mp4              footage + isaac_film_report.json
  videos/isaac_cross_eval.json    the 2x3 matrix
  mu_estimator/                   mu_estimator.pt + results json
  <run>/.../exported/policy.pt    TorchScript, loadable without rsl-rl
```

---

## 2. What is established

**The finding (solid, verified in both source trees, independent of any training):**
Isaac Lab's humanoid velocity task sets friction as a POINT value —
`static (0.8, 0.8)`, `dynamic (0.6, 0.6)`, min == max — and neither G1 config
overrides it. Spot in the same tree gets `(0.3, 1.0)`. MuJoCo Playground samples
`U(0.4, 1.0)`. Real ice is 0.05–0.15.

**Isaac works.** Ice-trained policy holds baseline performance across a 20× friction
range: 97.8% vs 98.0% survival, 0.912 vs 0.913 tracking. Caveat: each measured on
its own distribution — that is what `cross-eval2` fixes.

**MuJoCo does not, and six explanations are dead:** harness, sampling distribution,
slip penalty, factor stacking, step count (25M→150M gave 7.1%→10.7%), gait-phase
reward. Ablation: compliant ground harmless (78%), wind 13%, friction alone 7%,
**mid-episode rock→ice patches worst at 2%**.

**Quantified ice effect (MuJoCo, 12 paired seeds):** rock 266.5 → ice 129.6 mean
steps, worse on 9/12. Ice roughly halves survival.

---

## 3. Queue — what to do next

**P3 — three seeds of the Isaac ice policy.** Every Isaac number is n=1. Today
proved repeatedly that n=1 misleads. `train_ice_isaac.py` with `--seed 1/2/3`.

**P4 — wind in Isaac.** Wind is built for MuJoCo, where nothing trains. Porting it
to the stack that works claims Track 1 bullet 4 properly. Pattern: copy
`train_ice_slope.py`, add an external-force event.

**P5 — the Iridium packet.** No GPU. Pack μ̂ + slip rate + gait state into 340
bytes. Closes the pitch; only meaningful once `estimate-mu` reports.

**Then:** update the artifact page (`build_page.py`, republish `friction_gap.html`
via the Artifact tool at the SAME url), and consider pushing to GitHub — the user
asked to hold pushes, so **ask first**.

---

## 4. Traps already paid for — do not re-learn these

- **`play.py --video` records nothing.** Use an explicit capture loop with
  `render_mode="rgb_array"` + `AppLauncher(enable_cameras=True)`.
- **Exported policies go NEXT TO THE CHECKPOINT**, i.e. into the bucket at
  `<ckpt_dir>/exported/policy.pt` — not `IsaacLab/logs`. Cost one whole job.
- **`OnPolicyRunner` hand-rolled raises `unexpected keyword 'stochastic'`.** Load
  the TorchScript export instead; no rsl-rl in the loop.
- **Isaac auto-resets**, so "fraction of steps without a termination event" is ~99%
  even for a robot falling constantly. Count cumulative terminations.
- **The default Isaac camera is a wide overhead shot** — robots are specks. Set
  `viewer.origin_type="asset_root"`, `asset_name="robot"`, close `eye`.
- **brax writes checkpoints it cannot read back** (`*_kernel_init_fn: null` →
  `KERNEL_INITIALIZER[None]` KeyError). Patch a copy before loading.
- **`ffmpeg` is a system binary**, not covered by having `mediapy` installed.
- **MuJoCo `impl=jax` is ~3× faster on CPU but degrades the policy**: same policy at
  μ=0.8 survives 301/301 under warp, falls at 112 under jax. Film and quote under
  **warp** only.
- **zsh does not word-split unquoted vars** — flags in a `$VAR` arrive as one
  argument. Cost five job launches.
- **`.env` holds real LiveKit credentials** and the repo is PUBLIC. It is in
  `.gitignore`; never `cat >` that file again (doing so once nearly leaked them).

---

## 5. Known-wrong things to not repeat

- `base-v2` is **not** a reliable walker. "301/300 on every seed" was a misreading
  of `[301, 301, 196]`; only 3 of the first 8 seeds reach 300 steps. Single rollouts
  are illustrations, never measurements.
- Isaac's `mean_alive_frac` in the first `isaac_film_report.json` is meaningless
  (see §4). Ignore those numbers; the reissued run counts falls.
- Fall-step counts are **not comparable across physics backends**.

---

## 6. Deliverables as they stand

- **Artifact page** (private, share from its menu):
  `https://claude.ai/code/artifact/4852c73c-6e55-4ab6-913c-0da459e7702e`
  Rebuild with `python build_page.py`, republish the same file path to keep the URL.
- **GitHub** `abhijitbetigeri/HimalayaExpedition` — pushed once, PUBLIC.
  User has asked to hold further pushes.
- **Footage**: `demo/seed2_rock.mp4` (513 steps) and `demo/seed2_ice.mp4` (170) —
  the clearest of twelve seeds, rendered locally under warp.
- **Local viewer**: `local_view.py --backend warp --mu 0.06 --seed 2` — runs on a
  Mac, no GPU.

## 7. The gap

Layer 1 (μ̂ estimation) is in flight; layer 3 (telemetry) is unbuilt. Until those
land, this is careful domain randomisation plus a well-controlled negative result —
defensible and honest, but not yet the "traction stack" the pitch describes.
