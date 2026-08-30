# Morning runbook — 2026-08-30

Everything below runs on HF Jobs, which is **independent of any Claude session**.
Close the laptop; the jobs keep going and write to the bucket. Nothing here needs a
human overnight. Work through this top to bottom when you wake.

Read `STATE.md` for the durable state and `ISAAC_TRACK.md` for the HF Jobs recipe.

---

## Step 1 — what survived the night (2 minutes)

```bash
cd ~/projects/himalaya-hack
hf jobs ps --namespace iteratehack           # the whole org, incl. other people
```

Then check ours specifically:

```bash
for id in \
  6a93a107984507d9db4ec601 \
  6a93a10945686a1580c16dab \
  6a938e3345686a1580c16aed \
  6a939fd8984507d9db4ec5ed \
  6a93943d45686a1580c16be4 \
  6a9393e745686a1580c16bd4 \
  6a938fa0984507d9db4ec51b ; do
  printf "%s  %s\n" "$id" \
    "$(hf jobs inspect iteratehack/$id 2>/dev/null | tail -1 | grep -oE "'stage': '[A-Z]*'")"
done
```

| id | task | objective |
|---|---|---|
| `6a93a107984507d9db4ec601` | **train-wind** | C — wind resistance |
| `6a93a10945686a1580c16dab` | **train-tether** | D — fixed-line ascent |
| `6a938e3345686a1580c16aed` | **train-getup** | B — fall recovery |
| `6a939fd8984507d9db4ec5ed` | film-slope | A — footage |
| `6a93943d45686a1580c16be4` | film snow/hard-ice | A — footage |
| `6a9393e745686a1580c16bd4` | estimate-mu | layer 1 |
| `6a938fa0984507d9db4ec51b` | cross-eval | headline table |

For any that say `ERROR`, the failure is almost always in the last 30 lines:

```bash
hf jobs logs iteratehack/<id> 2>&1 \
  | grep -avE "neuraylib|material_library|Warning" | tail -30
```

---

## Step 2 — collect the results (5 minutes)

```bash
hf buckets ls -R hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/ | grep -E "\.mp4|\.json"
```

The numbers that matter, each printable directly:

```bash
B=hf://buckets/iteratehack/jobs-artifacts/himalaya-g1
hf buckets cp $B/videos/isaac_cross_eval.json -          # headline 2x3 table
hf buckets cp $B/mu_estimator/mu_estimator_results.json - # can it sense friction?
hf buckets cp $B/videos/slope_film_report.json -          # incline falls/steps
hf buckets cp $B/videos/isaac_film_report.json -          # snow + hard ice
```

Training metrics are in the job logs:

```bash
hf jobs logs iteratehack/<id> 2>&1 \
  | grep -aE "success_rate|time_out:|base_contact:|track_lin_vel|Training time"
```

---

## Step 3 — film whatever trained (the one thing that could NOT be chained)

Overnight jobs cannot film policies that did not exist when they were launched. Any
training that completed needs an export + capture pass. `film_slope.py` is the
template — it exports TorchScript via `play.py`, then drives its own capture loop.

For a new policy, copy `film_slope.py` and change three things:

1. `CKPT` → the new `model_*.pt` path in the bucket
2. the env registration block → the config that policy was trained on
3. the shot list at the bottom

```bash
hf jobs uv run --detach --namespace iteratehack --flavor l40sx1 --timeout 50m \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
  --label name=himalaya-traction --label task=film-<name> \
  film_<name>.py
```

**Film at μ = 0.12–0.20, not 0.06.** Bare wet ice is near the physical limit for
any legged system — humans need crampons — so a robot falling there is physics,
not a policy failure. Packed snow and glacial ice are both walkable and what an
expedition actually crosses.

---

## Step 4 — update the deliverables

```bash
python build_page.py        # regenerates friction_gap.html from demo/*.mp4
```
then republish via the Artifact tool with the **same file path** to keep the URL:
`https://claude.ai/code/artifact/4852c73c-6e55-4ab6-913c-0da459e7702e`

GitHub has **one unpushed commit**. The user asked to hold pushes — **ask first**.

---

## Step 5 — if there is time left

In value order:

1. **Three seeds of the best policy.** Every Isaac number is n=1, and today proved
   repeatedly that n=1 misleads.
2. **The Iridium packet** (no GPU): pack μ̂ + slip rate + gait state into 340 bytes.
   Closes the pitch; only meaningful once `estimate-mu` reports.
3. **A combined recovery→walk demo**: reset fallen, recover with the getup policy,
   hand off to the ice policy. A stitched clip is honest if labelled as two policies.

---

## Objectives — status at 20:30

| # | objective | state |
|---|---|---|
| A | **walk on ice + slopes clearly** | policy trained, 93.5% on rough terrain across ice→rock. Footage filming at walkable friction. |
| B | **fall → recovery → walk** | recovery training in flight. Verified robots start fallen (pelvis 0.29 m) with contact-termination removed. |
| C | **wind resistance** | training launched: sustained 40–90 N lateral, interval mode, on icy slopes. Wind deliberately NOT observable — must be inferred from IMU. |
| D | **rope / fixed-line ascent** | training launched: tether as an 80–140 N upward assist (~⅓ body weight), the ascender modelled as a mechanism, not a grip. |

## What is already solid, whatever the night does

- The finding: Isaac Lab's humanoid trains at a **point** friction value while the
  quadruped in the same tree gets a range. Verified in source, needs no policy.
- Isaac ice policy: **2.1× fewer falls than baseline on ice, zero falls on rock**.
- MuJoCo: ice halves survival across 12 paired seeds; six hypotheses tested and
  refuted; ablation shows patches worse than ice itself.
