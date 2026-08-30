# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["isaacsim[all,extscache]==6.0.1.0", "pip"]
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# index-strategy = "unsafe-best-match"
# prerelease = "allow"
# ///
"""The narrative shot: walk on snow -> fall -> recover -> walk on.

One continuous take with a policy HANDOFF, which is the honest way to show this:
the walking policy and the recovery policy are two networks, and stitching them is
the system working, not a trick -- provided the caption says so.

    phase 1  ice policy walking on visible snow
    phase 2  a shove hard enough to put it down (external force, not a cut)
    phase 3  handoff to the recovery policy the moment the torso is down
    phase 4  handoff back to the ice policy once it is upright, and walk on

Waits for the recovery checkpoint rather than failing if it is not ready -- the
training may still be running when this starts.

  hf jobs uv run --detach --namespace iteratehack --flavor h200 --timeout 90m \
      --env OMNI_KIT_ACCEPT_EULA=YES \
      -v hf://buckets/iteratehack/jobs-artifacts:/mnt \
      --label name=himalaya-traction --label task=film-narrative \
      film_narrative.py
"""
import os, pathlib, subprocess, sys, textwrap, time, json, glob

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("HOME", "/root")
PY, LAB = sys.executable, "/tmp/IsaacLab"
BUCKET = pathlib.Path("/mnt/himalaya-g1")
WALK = BUCKET / "ice-isaac/rsl_rl/g1_flat/2026-08-29_21-48-51/exported/policy.pt"


def sh(cmd, t, label, tail=10):
    print(f"\n$ {cmd}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, timeout=t, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        print("\n".join(p.stdout.splitlines()[-tail:]), flush=True)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT -- continuing", flush=True)
        return False


# Wait for the recovery policy: training may still be in flight.
DEADLINE = time.time() + 55 * 60
recov = None
while time.time() < DEADLINE:
    hits = sorted(glob.glob(str(BUCKET / "getup-v3/rsl_rl/*/*/exported/policy.pt")))
    if hits:
        recov = hits[-1]
        print(f"recovery policy found: {recov}", flush=True)
        break
    hits = sorted(glob.glob(str(BUCKET / "getup-v3/rsl_rl/*/*/model_*.pt")))
    if hits:
        print(f"checkpoint exists but not exported yet ({len(hits)} found); "
              f"waiting for the export", flush=True)
    else:
        print("waiting for the recovery policy to appear...", flush=True)
    time.sleep(120)

if not recov:
    sys.exit("recovery policy never appeared -- cannot film the narrative. "
             "The walk-only snowscape clips are unaffected.")
print("READY to compose the narrative", flush=True)
