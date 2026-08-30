# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.35"]
# ///
"""Six-hour improvement orchestrator: keep escalating until the deliverables are good.

Runs on cpu-basic (~$0.01/h) and outlives any session. Unlike the watchdog -- which
only reacts to jobs that ERROR -- this reacts to jobs that SUCCEED BADLY: a policy
that trains to completion but never stands up, a clip that renders fine but has the
robot out of shot. Those are the failures that actually cost the night.

Two ladders, each escalating only when the previous rung is measurably not working.

RECOVERY (the harder problem; three attempts have already failed)
  rung 0  graded starts, standing_bonus            <- getup-v3c, running
  rung 1  STUMBLE ONLY: start nearly upright. Recovering from a stumble is a much
          shorter horizon than prone-to-standing; learn that first.
  rung 2  stumble + 20 s episodes + 4000 iters, for more time per attempt
  rung 3  half-down starts, inheriting whatever rung 1/2 established
  Judged on `Metrics/success_rate` and `Episode_Reward/standing` in the job log --
  not on the job exiting 0, which two earlier runs did while having learned nothing.

FILM
  Relaunch autofilm whenever its own report marks a clip not-ok. autofilm already
  retries the CAMERA internally; this retries the whole shoot, which also picks up
  any newer policy that has appeared since.

Stops early if everything passes. Never runs past its deadline.

  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 6h \
      --secrets HF_TOKEN \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=improve-loop \
      improve_loop.py
"""

import glob
import json
import os
import pathlib
import re
import subprocess
import time

BUCKET = "hf://buckets/iteratehack/jobs-artifacts/himalaya-g1"
MNT = pathlib.Path("/mnt/himalaya-g1")
LOG = MNT / "improve_log.txt"
STATE = MNT / "improve_state.json"

HOURS = float(os.environ.get("LOOP_HOURS", "5.5"))
DEADLINE = time.time() + HOURS * 3600
POLL = 300

# Each rung is only tried if the one before it demonstrably did not work.
RECOVERY_LADDER = [
    dict(tag="v4-stumble", env={"GETUP_Z_MIN": "-0.20", "GETUP_Z_MAX": "-0.02",
                                "GETUP_ROLL": "0.7", "GETUP_EPISODE_S": "8.0",
                                "GETUP_TAG": "v4", "MAX_ITER": "3000"},
         why="prone-to-standing is long-horizon; a stumble is a much shorter one"),
    dict(tag="v5-stumble-long", env={"GETUP_Z_MIN": "-0.25", "GETUP_Z_MAX": "-0.02",
                                     "GETUP_ROLL": "0.9", "GETUP_EPISODE_S": "20.0",
                                     "GETUP_TAG": "v5", "MAX_ITER": "4000"},
         why="more time per episode and more iterations at the easier start"),
    dict(tag="v6-halfdown", env={"GETUP_Z_MIN": "-0.35", "GETUP_Z_MAX": "-0.05",
                                 "GETUP_ROLL": "1.4", "GETUP_EPISODE_S": "20.0",
                                 "GETUP_TAG": "v6", "MAX_ITER": "4000"},
         why="push toward genuinely fallen once the easier case works"),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(c, t=900):
    return subprocess.run(c, shell=True, timeout=t, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


_api = None


def stage(jid):
    global _api
    try:
        if _api is None:
            from huggingface_hub import HfApi
            _api = HfApi(token=os.environ.get("HF_TOKEN"))
        st = getattr(getattr(_api.inspect_job(job_id=jid, namespace="iteratehack"),
                             "status", None), "stage", None)
        if st:
            return str(st).upper()
    except Exception:
        pass
    out = run(f"hf jobs inspect iteratehack/{jid} 2>&1", 300).stdout.upper()
    for s in ("COMPLETED", "SCHEDULING", "RUNNING", "CANCELED", "ERROR"):
        if s in out:
            return s
    return "UNKNOWN"


def launch(script, flavor, timeout, task, env=None):
    if not pathlib.Path(f"/tmp/{script}").exists():
        run(f"hf buckets cp {BUCKET}/scripts/{script} /tmp/{script}")
    if not pathlib.Path(f"/tmp/{script}").exists():
        log(f"  !! {script} not in bucket/scripts -- cannot launch")
        return None
    envs = " ".join(f"--env {k}={v}" for k, v in (env or {}).items())
    p = run(f"cd /tmp && hf jobs uv run --detach --namespace iteratehack "
            f"--flavor {flavor} --timeout {timeout} "
            f"--env OMNI_KIT_ACCEPT_EULA=YES {envs} "
            f"-v hf://buckets/iteratehack/jobs-artifacts:/mnt "
            f"--label name=himalaya-traction --label task={task} {script}")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)
    m = (re.search(r"\bid=([0-9a-f]{16,})", clean)
         or re.search(r"iteratehack/([0-9a-f]{16,})", clean))
    if m:
        log(f"  launched {task} -> {m.group(1)}")
        return m.group(1)
    log(f"  !! launch of {task} returned no id: {clean.strip()[-200:]}")
    return None


def recovery_worked(job_id):
    """Did it actually learn to stand, or merely exit cleanly?

    Two earlier runs completed with success_rate 0.000 -- exiting 0 proves nothing.
    The standing bonus is the honest signal: it is strictly positive and near zero
    while the robot stays down.
    """
    txt = run(f"hf jobs logs iteratehack/{job_id} 2>&1 | tail -400", 300).stdout
    std = re.findall(r"Episode_Reward/standing:\s*([-\d.]+)", txt)
    suc = re.findall(r"Metrics/success_rate:\s*([\d.]+)", txt)
    s_val = float(std[-1]) if std else 0.0
    ok = s_val > 2.0                     # meaningful fraction of the 8.0 weight
    return ok, dict(standing=s_val, success=float(suc[-1]) if suc else None)


def film_scores():
    p = run(f"hf buckets cp {BUCKET}/videos/autofilm_report.json - 2>/dev/null", 300)
    try:
        return json.loads(p.stdout[p.stdout.index("{"):])
    except Exception:
        return None


state = {"recovery_rung": -1, "recovery_job": None, "film_job": None,
         "recovery_done": False, "film_done": False, "history": []}
if STATE.exists():
    try:
        state.update(json.loads(STATE.read_text()))
        log("resumed improve-loop state")
    except Exception:
        pass

# The currently running attempts, passed in so the loop does not duplicate them.
state.setdefault("recovery_job", os.environ.get("SEED_RECOVERY_JOB") or None)
state.setdefault("film_job", os.environ.get("SEED_FILM_JOB") or None)

log(f"improve-loop up for {HOURS}h. recovery_job={state['recovery_job']} "
    f"film_job={state['film_job']}")

while time.time() < DEADLINE:
    # ---------------- recovery ladder ----------------
    if not state["recovery_done"]:
        jid = state["recovery_job"]
        st = stage(jid) if jid else "NONE"
        if st in ("COMPLETED", "ERROR", "CANCELED", "NONE"):
            # A CRASH is not evidence that the approach failed. The first ladder
            # run escalated through all three rungs in fifteen minutes because a
            # NameError looked identical to "did not learn to stand" -- so retry
            # the SAME rung on a crash instead of moving on.
            if jid and st in ("ERROR", "CANCELED"):
                tail = run(f"hf jobs logs iteratehack/{jid} 2>&1 | tail -60", 300).stdout
                crashed = any(k in tail for k in
                              ("Traceback", "NameError", "ImportError", "SyntaxError",
                               "ModuleNotFoundError", "AttributeError"))
                if crashed and state.get("crash_retries", 0) < 2:
                    state["crash_retries"] = state.get("crash_retries", 0) + 1
                    rung = (RECOVERY_LADDER[state["recovery_rung"]]
                            if state["recovery_rung"] >= 0 else RECOVERY_LADDER[0])
                    log(f"recovery {jid} CRASHED (not a learning failure) -- "
                        f"retrying rung '{rung['tag']}' unchanged "
                        f"({state['crash_retries']}/2)")
                    new = launch("train_getup.py", "a100-large", "3h",
                                 f"getup-{rung['tag']}-r", rung["env"])
                    if new:
                        state["recovery_job"] = new
                    try:
                        STATE.write_text(json.dumps(state, indent=2))
                    except Exception:
                        pass
                    time.sleep(POLL)
                    continue
            worked, metrics = (recovery_worked(jid) if jid and st == "COMPLETED"
                               else (False, {}))
            if worked:
                log(f"RECOVERY WORKS: {metrics} -- ladder stops here")
                state["recovery_done"] = True
            else:
                if jid:
                    log(f"recovery attempt {jid} did not stand ({st}, {metrics})")
                nxt = state["recovery_rung"] + 1
                if nxt < len(RECOVERY_LADDER):
                    rung = RECOVERY_LADDER[nxt]
                    log(f"escalating to rung {nxt} '{rung['tag']}': {rung['why']}")
                    new = launch("train_getup.py", "a100-large", "3h",
                                 f"getup-{rung['tag']}", rung["env"])
                    if new:
                        state["recovery_rung"] = nxt
                        state["recovery_job"] = new
                        state["history"].append(
                            {"rung": rung["tag"], "job": new, "t": time.strftime("%H:%M")})
                else:
                    log("recovery ladder exhausted. The remaining lever is motion "
                        "imitation (AMP with AMASS references, as in TeamHOI) -- a "
                        "bigger change than this loop should make unsupervised.")
                    state["recovery_done"] = True   # stop retrying

    # ---------------- film ----------------
    if not state["film_done"]:
        jid = state["film_job"]
        st = stage(jid) if jid else "NONE"
        if st in ("COMPLETED", "ERROR", "CANCELED", "NONE"):
            rep = film_scores()
            bad = []
            if rep:
                for k, v in rep.items():
                    if "error" in v:
                        bad.append(f"{k}:{v['error']}")
                    elif not v.get("score", {}).get("ok", False):
                        bad.append(f"{k}:{v.get('score', {}).get('why')}")
            if rep and not bad:
                log(f"FILM PASSES: {list(rep)}")
                state["film_done"] = True
            else:
                log(f"film needs another pass: {bad or 'no report yet'}")
                new = launch("autofilm.py", "h200", "2h", "autofilm-retry")
                if new:
                    state["film_job"] = new

    try:
        STATE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass
    if state["recovery_done"] and state["film_done"]:
        log("both deliverables settled -- exiting early")
        break
    time.sleep(POLL)

log("improve-loop finished")
log(json.dumps(state, indent=2))
