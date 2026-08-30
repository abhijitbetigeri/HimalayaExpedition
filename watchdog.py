# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.35"]
# ///
"""Self-healing overnight supervisor. Runs AS an HF Job, so it outlives any session.

The problem it solves: HF Jobs run unattended fine, but nothing notices when one
ERRORs at 3am, and a failed run is dead until a human looks. This polls the tracked
jobs and relaunches failures on its own.

It runs on cpu-basic (~$0.01/h, so ~$0.08 for a night) and needs HF_TOKEN passed as
a SECRET so it can launch jobs in the org:

  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 8h \
      --secrets HF_TOKEN \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=watchdog \
      watchdog.py

Guard rails, because an unsupervised thing that launches GPU jobs deserves them:
  * MAX_RETRIES per task (default 2). A job that fails three times has a real bug
    that relaunching will not fix, and burning GPU on it overnight is waste.
  * Only tasks in the manifest are ever launched -- it cannot invent work.
  * Every decision is appended to a log in the bucket, so the morning shows exactly
    what it did and why.
  * It never cancels or deletes anything.

Scripts are fetched from the bucket, so `hf cp` them there before starting; a
relaunch cannot upload a file the watchdog does not have.
"""

import json
import os
import pathlib
import subprocess
import time

BUCKET_URI = "hf://buckets/iteratehack/jobs-artifacts/himalaya-g1"
MNT = pathlib.Path("/mnt/himalaya-g1")
STATE = MNT / "watchdog_state.json"
LOG = MNT / "watchdog_log.txt"

POLL_S = int(os.environ.get("WATCHDOG_POLL_S", "600"))       # 10 min
MAX_RETRIES = int(os.environ.get("WATCHDOG_MAX_RETRIES", "2"))
DEADLINE_S = int(os.environ.get("WATCHDOG_HOURS", "8")) * 3600

# task -> how to relaunch it. Only these can ever be started.
MANIFEST = {
    "train-wind":    dict(script="train_wind.py",    flavor="l40sx1",     timeout="3h"),
    "train-tether":  dict(script="train_tether.py",  flavor="l40sx1",     timeout="3h"),
    "train-getup2":  dict(script="train_getup.py",   flavor="a100-large", timeout="3h"),
    "cross-eval3":   dict(script="cross_eval2.py",   flavor="l4x1",       timeout="40m"),
    "estimate-mu3":  dict(script="estimate_mu.py",   flavor="l4x1",       timeout="50m",
                          env={"MU_SPLIT": "episode"}),
    "film-slope":    dict(script="film_slope.py",    flavor="l40sx1",     timeout="50m"),
}

# Job ids as launched before the watchdog started.
WATCHING = {
    "train-wind":   "6a93a107984507d9db4ec601",
    "train-tether": "6a93a10945686a1580c16dab",
    "train-getup2": "6a93a31045686a1580c16dff",
    "cross-eval3":  "6a93a2cc984507d9db4ec641",
    "estimate-mu3": "6a93a2ce984507d9db4ec643",
    "film-slope":   "6a939fd8984507d9db4ec5ed",
}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(cmd, timeout=300):
    return subprocess.run(cmd, shell=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


_api = None
_raw_logged = False


def stage(job_id):
    """RUNNING / COMPLETED / ERROR / SCHEDULING, or UNKNOWN if we cannot tell.

    Uses the Python API first: shelling out to `hf jobs inspect` and scraping the
    text returned UNKNOWN for every job on the first attempt, and a watchdog that
    cannot read a status is a watchdog that never fires.
    """
    global _api, _raw_logged
    try:
        if _api is None:
            from huggingface_hub import HfApi
            _api = HfApi(token=os.environ.get("HF_TOKEN"))
        info = _api.inspect_job(job_id=job_id, namespace="iteratehack")
        st = getattr(getattr(info, "status", None), "stage", None)
        if st:
            return str(st).upper()
    except Exception as e:
        if not _raw_logged:
            log(f"  (python api unavailable: {e!r}; falling back to CLI)")
            _raw_logged = True

    p = run(f"hf jobs inspect iteratehack/{job_id} 2>&1")
    out = p.stdout.upper()
    for s in ("COMPLETED", "SCHEDULING", "RUNNING", "CANCELED", "ERROR"):
        if s in out:
            return s
    if not _raw_logged:
        log(f"  (unparsed inspect output: {p.stdout.strip()[:300]})")
        _raw_logged = True
    return "UNKNOWN"


def relaunch(task):
    spec = MANIFEST[task]
    script = spec["script"]
    got = run(f"hf buckets cp {BUCKET_URI}/scripts/{script} /tmp/{script}")
    if not pathlib.Path(f"/tmp/{script}").exists():
        log(f"  !! cannot relaunch {task}: {script} missing from bucket/scripts/")
        log(f"     {got.stdout.strip()[:200]}")
        return None
    envs = " ".join(f"--env {k}={v}" for k, v in spec.get("env", {}).items())
    cmd = (f"cd /tmp && hf jobs uv run --detach --namespace iteratehack "
           f"--flavor {spec['flavor']} --timeout {spec['timeout']} "
           f"--env OMNI_KIT_ACCEPT_EULA=YES {envs} "
           f"-v hf://buckets/iteratehack/jobs-artifacts:/mnt "
           f"--label name=himalaya-traction --label task={task}-retry "
           f"{script}")
    p = run(cmd, timeout=600)
    for tok in p.stdout.split():
        if tok.startswith("id="):
            return tok[3:]
    log(f"  !! relaunch of {task} produced no job id: {p.stdout.strip()[-300:]}")
    return None


state = {t: {"job": j, "retries": 0, "history": [j]} for t, j in WATCHING.items()}
if STATE.exists():
    try:
        state.update(json.loads(STATE.read_text()))
        log("resumed prior watchdog state")
    except Exception:
        pass

log(f"watchdog up: {len(state)} tasks, poll {POLL_S}s, max {MAX_RETRIES} retries, "
    f"deadline {DEADLINE_S//3600}h")

t0 = time.time()
while time.time() - t0 < DEADLINE_S:
    summary = []
    for task, rec in state.items():
        st = stage(rec["job"])
        rec["stage"] = st
        summary.append(f"{task}={st}")

        if st in ("ERROR", "CANCELED"):
            if rec["retries"] >= MAX_RETRIES:
                log(f"{task}: {st}, retries exhausted ({rec['retries']}) — leaving it. "
                    f"This needs a human; relaunching again would just burn GPU.")
                continue
            if task not in MANIFEST:
                log(f"{task}: {st} but not in manifest — not launching")
                continue
            log(f"{task}: {st} after {rec['retries']} retries — relaunching")
            new = relaunch(task)
            if new:
                rec["job"] = new
                rec["retries"] += 1
                rec["history"].append(new)
                log(f"  -> relaunched {task} as {new} (retry {rec['retries']})")

    log("status: " + "  ".join(summary))
    try:
        STATE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log(f"  (could not persist state: {e})")

    if all(r.get("stage") == "COMPLETED" for r in state.values()):
        log("all tracked tasks COMPLETED — watchdog exiting early")
        break
    time.sleep(POLL_S)

log("watchdog finished")
log(json.dumps({t: {"stage": r.get("stage"), "job": r["job"], "retries": r["retries"]}
                for t, r in state.items()}, indent=2))
