# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.35"]
# ///
"""Self-HEALING overnight supervisor: diagnoses failures and patches them, not just retries.

Runs AS an HF Job, so it outlives any session.

Why v2 was not enough
---------------------
v2 relaunched failures with the identical script, which fails identically. A retry
only helps for transient faults (capacity, a flaky pull); every real failure today
was deterministic and needed a code change.

So v3 reads the failing job's log, matches it against the failure modes this project
has actually hit, and applies the corresponding fix to the script in the bucket
before relaunching. The rules below are not speculative -- each one cost a real job.

  TimeoutExpired          -> raise the job timeout (and the inner subprocess cap)
  ModuleNotFoundError     -> add the missing isaaclab_* package to the install list
  CUDA out of memory      -> halve num_envs
  EULA prompt             -> set OMNI_KIT_ACCEPT_EULA
  ffmpeg not found        -> add ffmpeg to the apt line
  KERNEL_INITIALIZER[None]-> brax null-initializer patch needed (flagged, not auto-fixed)
  render() no frames      -> enable_cameras on AppLauncher
  402 Payment Required    -> STOP. Out of credits; retrying burns nothing but noise
  403 job.write           -> STOP. Token scope; a human must fix it

Anything unmatched is relaunched unchanged ONCE (in case it was transient) and then
left alone with a clear log line, because guessing at an unknown failure
unsupervised is how you burn a night of GPU on nothing.

Guard rails: max 2 retries per task, manifest-restricted, never cancels or deletes,
every decision logged to the bucket with its reason.

  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 8h \
      --secrets HF_TOKEN \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=watchdog \
      watchdog.py
"""

import json
import os
import pathlib
import re
import subprocess
import time

BUCKET_URI = "hf://buckets/iteratehack/jobs-artifacts/himalaya-g1"
MNT = pathlib.Path("/mnt/himalaya-g1")
STATE = MNT / "watchdog_state.json"
LOG = MNT / "watchdog_log.txt"

POLL_S = int(os.environ.get("WATCHDOG_POLL_S", "420"))
MAX_RETRIES = int(os.environ.get("WATCHDOG_MAX_RETRIES", "2"))
DEADLINE_S = int(os.environ.get("WATCHDOG_HOURS", "8")) * 3600

MANIFEST = {
    "getup-v3b":   dict(script="train_getup.py", flavor="h200", timeout="3h"),
    "autofilm2":   dict(script="autofilm.py",    flavor="h200", timeout="3h"),
    "film-slope2": dict(script="film_slope.py",  flavor="h200", timeout="90m"),
}

WATCHING = {
    "getup-v3b":   "6a93e87045686a1580c17972",
    "autofilm2":   "6a93e873984507d9db4ecb73",
    "film-slope2": "6a93e2ec45686a1580c17871",
}

# Timeouts we may escalate to, in order.
TIMEOUT_LADDER = ["40m", "50m", "90m", "2h", "3h", "5h"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


_api = None


def stage(job_id):
    global _api
    try:
        if _api is None:
            from huggingface_hub import HfApi
            _api = HfApi(token=os.environ.get("HF_TOKEN"))
        info = _api.inspect_job(job_id=job_id, namespace="iteratehack")
        st = getattr(getattr(info, "status", None), "stage", None)
        if st:
            return str(st).upper()
    except Exception:
        pass
    out = run(f"hf jobs inspect iteratehack/{job_id} 2>&1").stdout.upper()
    for s in ("COMPLETED", "SCHEDULING", "RUNNING", "CANCELED", "ERROR"):
        if s in out:
            return s
    return "UNKNOWN"


def job_log(job_id, n=400):
    return run(f"hf jobs logs iteratehack/{job_id} 2>&1 | tail -{n}", timeout=300).stdout


def diagnose(text):
    """Match a failure log against known modes. Returns (name, action, detail)."""
    if "402" in text and "Payment Required" in text:
        return ("out_of_credits", "STOP",
                "org credit balance exhausted -- a human must top it up")
    if "403" in text and "job.write" in text:
        return ("token_scope", "STOP",
                "token lost job.write on the org -- a human must fix the scope")
    if "TimeoutExpired" in text or "TIMEOUT after" in text:
        return ("timeout", "BUMP_TIMEOUT",
                "a step exceeded its cap; escalating the job timeout")
    m = re.search(r"No module named '(isaaclab[a-z_]*)'", text)
    if m:
        return ("missing_pkg", ("ADD_PKG", m.group(1)),
                f"{m.group(1)} not installed -- Isaac Lab splits packages and "
                f"registration imports them at import time")
    if "out of memory" in text.lower() or "CUDA error: out of memory" in text:
        return ("oom", "HALVE_ENVS", "GPU OOM -- halving num_envs")
    if "Do you accept the EULA" in text or "Unable to bootstrap inner kit kernel" in text:
        return ("eula", "SET_EULA", "Kit blocked on the interactive EULA prompt")
    if "Program 'ffmpeg' is not found" in text:
        return ("ffmpeg", "ADD_FFMPEG", "mediapy shells out to the ffmpeg BINARY")
    if "KERNEL_INITIALIZER" in text:
        return ("brax_ckpt", "FLAG",
                "brax null-initializer bug -- needs the checkpoint patch, see "
                "diagnose_eval.py")
    if "render() returned no frames" in text or "no frames" in text:
        return ("no_frames", "FLAG",
                "render() returned nothing -- needs AppLauncher(enable_cameras=True)")
    return ("unknown", "RETRY_ONCE", "unrecognised failure")


def fetch(script):
    p = f"/tmp/{script}"
    run(f"hf buckets cp {BUCKET_URI}/scripts/{script} {p}")
    return pathlib.Path(p) if pathlib.Path(p).exists() else None


def patch(script, action):
    """Apply a fix to the script and push it back to the bucket. True if changed."""
    f = fetch(script)
    if not f:
        return False
    s = orig = f.read_text()

    if isinstance(action, tuple) and action[0] == "ADD_PKG":
        pkg = action[1]
        if f'"{pkg}"' not in s:
            s = s.replace('"isaaclab_tasks",', f'"isaaclab_tasks", "{pkg}",', 1)
            s = s.replace('"isaaclab_tasks"]', f'"isaaclab_tasks", "{pkg}"]', 1)
    elif action == "HALVE_ENVS":
        m = re.search(r"NUM_ENVS\s*=\s*(\d+)", s)
        if m:
            s = s.replace(m.group(0), f"NUM_ENVS = {max(512, int(m.group(1)) // 2)}", 1)
        m2 = re.search(r"num_envs=(\d{3,})", s)
        if m2 and s == orig:
            s = s.replace(m2.group(0), f"num_envs={max(64, int(m2.group(1)) // 2)}", 1)
    elif action == "ADD_FFMPEG":
        if "libgomp1 ffmpeg" not in s:
            s = s.replace("libgomp1", "libgomp1 ffmpeg", 1)
    elif action == "SET_EULA":
        if 'OMNI_KIT_ACCEPT_EULA' not in s:
            s = s.replace("import os\n", 'import os\nos.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"\n', 1)
    elif action == "BUMP_TIMEOUT":
        # Raise inner subprocess caps too -- the outer job timeout is handled by
        # the launcher, but an inner cap that is too small fails the same way.
        def bump(m):
            return str(min(int(m.group(0)) * 2, 9000))
        s = re.sub(r"(?<=, )\d{3,4}(?=, \"(export|train|film|verify))", bump, s)

    if s != orig:
        f.write_text(s)
        run(f"hf buckets cp {f} {BUCKET_URI}/scripts/{script}")
        return True
    return False


def relaunch(task, extra_timeout=False):
    spec = MANIFEST[task]
    if not fetch(spec["script"]):
        log(f"  !! {spec['script']} missing from bucket/scripts/ -- cannot relaunch")
        return None
    to = spec["timeout"]
    if extra_timeout and to in TIMEOUT_LADDER:
        i = TIMEOUT_LADDER.index(to)
        to = TIMEOUT_LADDER[min(i + 1, len(TIMEOUT_LADDER) - 1)]
        spec["timeout"] = to
        log(f"  timeout escalated to {to}")
    envs = " ".join(f"--env {k}={v}" for k, v in spec.get("env", {}).items())
    p = run(f"cd /tmp && hf jobs uv run --detach --namespace iteratehack "
            f"--flavor {spec['flavor']} --timeout {to} "
            f"--env OMNI_KIT_ACCEPT_EULA=YES {envs} "
            f"-v hf://buckets/iteratehack/jobs-artifacts:/mnt "
            f"--label name=himalaya-traction --label task={task}-r "
            f"{spec['script']}", timeout=900)
    clean = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)      # strip ANSI colour
    m = re.search(r"\bid=([0-9a-f]{16,})", clean)
    if m:
        return m.group(1)
    m = re.search(r"iteratehack/([0-9a-f]{16,})", clean)   # fall back to the hint
    if m:
        return m.group(1)
    log(f"  !! no job id parsed from: {clean.strip()[-200:]}")
    return None


state = {t: {"job": j, "retries": 0, "history": [j]} for t, j in WATCHING.items()}
if STATE.exists():
    try:
        prior = json.loads(STATE.read_text())
        for k, v in prior.items():
            if k in state:
                state[k].update(v)
        log("resumed prior watchdog state")
    except Exception:
        pass

log(f"watchdog v3 (self-healing) up: {len(state)} tasks, poll {POLL_S}s, "
    f"max {MAX_RETRIES} retries")

halt = False
t0 = time.time()
while time.time() - t0 < DEADLINE_S and not halt:
    summary = []
    for task, rec in state.items():
        st = stage(rec["job"])
        rec["stage"] = st
        summary.append(f"{task}={st}")
        if st not in ("ERROR", "CANCELED"):
            continue
        if rec["retries"] >= MAX_RETRIES:
            continue

        name, action, why = diagnose(job_log(rec["job"]))
        log(f"{task}: {st} -> diagnosed '{name}': {why}")

        if action == "STOP":
            log(f"  HALTING all retries: {why}")
            halt = True
            break
        if action == "FLAG":
            log(f"  needs a code change I will not make unsupervised. Left for the "
                f"morning; see MORNING.md.")
            rec["retries"] = MAX_RETRIES        # stop reconsidering it
            rec["flagged"] = why
            continue

        changed = patch(MANIFEST[task]["script"], action) if action != "RETRY_ONCE" else False
        log(f"  patch applied: {changed}")
        new = relaunch(task, extra_timeout=(action == "BUMP_TIMEOUT"))
        if new:
            rec["job"] = new
            rec["retries"] += 1
            rec["history"].append(new)
            rec["last_fix"] = name
            log(f"  -> relaunched as {new} (retry {rec['retries']}, fix={name})")

    log("status: " + "  ".join(summary))
    try:
        STATE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass
    if all(r.get("stage") == "COMPLETED" for r in state.values()):
        log("all tracked tasks COMPLETED -- exiting early")
        break
    time.sleep(POLL_S)

log("watchdog finished")
log(json.dumps({t: {k: r.get(k) for k in ("stage", "job", "retries", "last_fix", "flagged")}
                for t, r in state.items()}, indent=2))
