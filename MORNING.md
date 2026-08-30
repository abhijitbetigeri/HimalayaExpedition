# Morning runbook — 2026-08-30

Everything below runs on HF Jobs, which is **independent of any Claude session**.
Close the laptop; the jobs keep going and write to the bucket. Nothing here needs a
human overnight. Work through this top to bottom when you wake.

Read `STATE.md` for the durable state and `ISAAC_TRACK.md` for the HF Jobs recipe.

---

## Step 0 — read the watchdog first (30 seconds)

A supervisor job runs on cpu-basic all night, polls every 10 min, and RELAUNCHES
anything that ERRORs (max 2 retries each, manifest-restricted, never cancels).
Its decisions are the fastest summary of the night:

```bash
hf buckets cp hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/watchdog_log.txt - | tail -40
hf buckets cp hf://buckets/iteratehack/jobs-artifacts/himalaya-g1/watchdog_state.json -
```

`watchdog_state.json` holds the CURRENT job id per task (which may be a retry, not
the id in the table below) plus how many retries each needed. A task showing
`retries: 2` failed three times and needs a human — do not just relaunch it.

Watchdog job: `6a93a50a984507d9db4ec661`

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
| `6a93a31045686a1580c16dff` | **train-getup2** | B — fall recovery, RETRY |
| `6a939fd8984507d9db4ec5ed` | film-slope | A — incline footage |
| `6a93a576984507d9db4ec66b` | **train-ice-hifi** | corrected contact model — see below |
| `6a93a2cc984507d9db4ec641` | cross-eval3 | headline table, RETRY |
| `6a93a2ce984507d9db4ec643` | estimate-mu3 | layer 1, RETRY |

Wind and tether are on `l40sx1` with a 3 h timeout. The comparable rough-terrain
run took 102 min on an A100, so they may need 2.5-3.5 h and could be KILLED at the
cap. The 3-minute checkpoint mirror means a partial policy survives -- check for
`model_*.pt` in the bucket even if the job says ERROR.

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

## Step 4.5 — the contact-model correction (read before quoting any ice number)

`train_ice_hifi.py` is retraining with `friction_combine_mode = "min"`.

PhysX combines the two contacting materials, and Isaac Lab defaults to `average`.
So a foot at mu=0.8 on ice at mu=0.06 was actually experiencing **mu=0.43** — not
ice. **Every ice result produced before this ran on ground roughly 7x grippier than
intended**, including the 0-falls-on-snow headline.

Expect the hifi numbers to be WORSE. That is the correct outcome: a policy that
only worked because the contact model was generous is not a result. Compare
`ice-slope-hifi/` against `ice-slope/` and quote the hifi numbers if they differ
materially. Full reasoning and the remaining fidelity gaps are in `SIM_ACCURACY.md`.

## Step 5 — if there is time left

In value order:

1. **Three seeds of the best policy.** Every Isaac number is n=1, and today proved
   repeatedly that n=1 misleads.
2. **The Iridium packet** (no GPU): pack μ̂ + slip rate + gait state into 340 bytes.
   Closes the pitch; only meaningful once `estimate-mu` reports.
3. **A combined recovery→walk demo**: reset fallen, recover with the getup policy,
   hand off to the ice policy. A stitched clip is honest if labelled as two policies.

---

## RESULT OF THE NIGHT — objective A is done

Filmed at realistic winter surfaces (the earlier mu=0.06 runs were aiming at bare
wet ice, which is near the physical limit for any legged system):

| shot | falls per env, 1500 steps |
|---|---|
| **ice policy on packed snow, mu=0.20** | **0.00 — never falls** |
| baseline on snow | 2.5 |
| **ice policy on hard glacial ice, mu=0.12** | **2.17** |
| baseline on hard ice | **20.0 — 9.2x worse** |
| **ice policy on snowy slope** | **3.33 — it climbs** |

Clips: `videos/isaac_ice_on_snow.mp4`, `isaac_baseline_on_hardice.mp4`,
`isaac_ice_on_slopesnow.mp4`. These are the demo.

## Two failures to be aware of

**Fall recovery v1 failed**: 3000 iterations, `success_rate 0.000`,
`base_height -1.83` — it never got off the ground. Prone-to-standing is
long-horizon and a flat reward from a fully-sprawled start gives almost no
gradient. v2 (`6a93a310...`) samples start poses across the whole range from
near-upright to fully prone, so easy episodes bootstrap a behaviour the hard ones
extend. If v2 also reports `success_rate 0`, the next lever is a staged reward
(torso off the ground -> hips up -> stand), not more iterations.

**mu estimator v1 failed**: held-out R2 **-2.10**, worse than predicting the mean,
and `ice_vs_rock_accuracy NaN` because the held-out band [0.295, 0.565] contained
no ice at all. v2 splits by EPISODE (new episode, same friction range = the
deployment question) rather than by value, with 1024 envs and twice the windows.
If it still fails, that is a real finding: 25 frames of proprioception may simply
not carry friction, and the honest move is to report it rather than tune until it
looks good.

## Objectives — status at 21:00

| # | objective | state |
|---|---|---|
| A | **walk on ice + slopes clearly** | ✅ **DONE** — 0 falls on snow, 9.2× better than baseline on glacial ice, climbs a snowy slope. Filmed. |
| B | **fall → recovery → walk** | ❌ v1 failed (success 0.000); v2 retraining with graded start poses |
| C | **wind resistance** | training launched: sustained 40–90 N lateral, interval mode, on icy slopes. Wind deliberately NOT observable — must be inferred from IMU. |
| D | **rope / fixed-line ascent** | training launched: tether as an 80–140 N upward assist (~⅓ body weight), the ascender modelled as a mechanism, not a grip. |

## What is already solid, whatever the night does

- The finding: Isaac Lab's humanoid trains at a **point** friction value while the
  quadruped in the same tree gets a range. Verified in source, needs no policy.
- Isaac ice policy: **2.1× fewer falls than baseline on ice, zero falls on rock**.
- MuJoCo: ice halves survival across 12 paired seeds; six hypotheses tested and
  refuted; ablation shows patches worse than ice itself.
