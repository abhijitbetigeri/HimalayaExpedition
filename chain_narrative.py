# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.35"]
# ///
"""Wait for the recovery policy, then launch the narrative film job.

Exists because of a timing gap: autofilm2 waits 70 min for the recovery policy but
getup-v3c needs ~85, so autofilm would abandon the walk/slip/recover clip about ten
minutes before the policy it needs appears. Rather than leave that for a human at
6am, this idles on cpu-basic (~$0.01/h) and fires the GPU job the moment the export
lands.

Cheap by construction: the waiting happens on a CPU box, and the expensive H200 job
only starts when there is actually something to film.

  hf jobs uv run --detach --namespace iteratehack --flavor cpu-basic --timeout 4h \
      --secrets HF_TOKEN \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=chain-narrative \
      chain_narrative.py
"""
import glob
import pathlib
import re
import subprocess
import time

BUCKET = "hf://buckets/iteratehack/jobs-artifacts/himalaya-g1"
MNT = pathlib.Path("/mnt/himalaya-g1")
LOG = MNT / "chain_log.txt"
DEADLINE = time.time() + 3.5 * 3600


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(c, t=600):
    return subprocess.run(c, shell=True, timeout=t, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


log("chain: waiting for the recovery policy export")
found = None
while time.time() < DEADLINE:
    hits = sorted(glob.glob(str(MNT / "getup-v*/rsl_rl/*/*/exported/policy.pt")))
    if hits:
        found = hits[-1]
        log(f"recovery policy exported: {found}")
        break
    ck = sorted(glob.glob(str(MNT / "getup-v*/rsl_rl/*/*/model_*.pt")))
    log(f"  not yet ({len(ck)} checkpoints present)")
    time.sleep(240)

if not found:
    # A checkpoint without an export still gives the narrative something to use, so
    # say so explicitly rather than exiting silent.
    ck = sorted(glob.glob(str(MNT / "getup-v*/rsl_rl/*/*/model_*.pt")))
    log(f"deadline reached with no export; {len(ck)} raw checkpoints exist. "
        f"A human can export one with play.py and rerun autofilm.py.")
    raise SystemExit(0)

run(f"hf buckets cp {BUCKET}/scripts/autofilm.py /tmp/autofilm.py")
if not pathlib.Path("/tmp/autofilm.py").exists():
    log("!! autofilm.py missing from bucket/scripts -- cannot launch")
    raise SystemExit(1)

p = run("cd /tmp && hf jobs uv run --detach --namespace iteratehack --flavor h200 "
        "--timeout 2h --env OMNI_KIT_ACCEPT_EULA=YES "
        "-v hf://buckets/iteratehack/jobs-artifacts:/mnt "
        "--label name=himalaya-traction --label task=narrative-final autofilm.py",
        t=900)
clean = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)
m = re.search(r"\bid=([0-9a-f]{16,})", clean) or re.search(r"iteratehack/([0-9a-f]{16,})", clean)
log(f"launched narrative job: {m.group(1) if m else 'ID NOT PARSED -- ' + clean[-200:]}")
