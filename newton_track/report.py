"""Turn newton_track/results/*.json into the markdown tables in NEWTON_TRACK.md.

    ~/projects/newton-sandbox/.venv/bin/python newton_track/report.py

Prints to stdout. Tables are generated, never typed, so a number in the write-up
can always be traced to a results file.
"""

from __future__ import annotations

import json
import pathlib
import statistics as st

R = pathlib.Path(__file__).resolve().parent / "results"


def load(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def cell_row(c: dict, steps: int) -> str:
    early = sum(1 for s in c["survived"] if s < 150)
    late = c["falls"] - early
    return (
        f"{c['falls']}/{c['n']} | {early} | {late} | {c['survived_median']:.0f} | {c['speed_mean']:+.2f} | "
        f"{c['abs_yaw_deg_median']:.0f}"
    )


def friction():
    mj = load("ice_sweep_mujoco_elliptic.json")
    steel = load("ice_sweep_steel_elliptic_friction.json")
    pyr = load("ice_sweep_mujoco_pyramidal_friction.json")
    if not mj:
        return
    steps = mj["steps"]
    cols = [("MuJoCo", mj)] + ([("Steel", steel)] if steel else []) + ([("MuJoCo, pyramidal cone", pyr)] if pyr else [])
    print(f"### Friction ({mj['seeds']} paired seeds x {steps} steps = {steps // 50} s, flat, no wind)\n")
    names = " / ".join(n for n, _ in cols)
    print("| ground mu | " + " | ".join(f"falls, {n}" for n, _ in cols) + f" | median steps ({names}) | body speed m/s, MuJoCo |")
    print("|---|" + "---|" * (len(cols) + 2))
    for i, c in enumerate(mj["results"]["friction"]):
        falls, med = [], []
        for _, d in cols:
            cc = d["results"]["friction"][i]
            assert cc["mu_ground"] == c["mu_ground"]
            falls.append(f"{cc['falls']}/{cc['n']}")
            med.append(f"{cc['survived_median']:.0f}")
        print(f"| {c['mu_ground']:g} | " + " | ".join(falls) + " | " + " / ".join(med) + f" | {c['speed_mean']:.2f} |")
    print()
    for name, d in cols:
        for c in d["results"]["friction"]:
            if 0 < c["falls"] < c["n"]:
                fell = sorted(s for s in c["survived"] if s < steps)
                print(
                    f"- {name}, mu={c['mu_ground']:g}: the {len(fell)} falls happen at steps "
                    f"{fell[0]}-{fell[-1]} (median {st.median(fell):.0f}); the other "
                    f"{c['n'] - c['falls']} seeds are still walking at step {steps}."
                )
    print()


def sweep(kind: str, key: str, unit: str, source: str = "ice_sweep_mujoco_elliptic.json"):
    mj = load(source)
    if not mj or not mj["results"][kind]:
        return
    steps = mj["steps"]
    title = "Commanded speed on ice" if kind == "speed" else kind.capitalize()
    print(f"### {title} (MuJoCo, same seeds)\n")
    print("| condition | falls | of which before step 150 | after | median steps | body speed m/s | heading drift deg |")
    print("|---|---|---|---|---|---|---|")
    for c in mj["results"][kind]:
        print(f"| {c['label']} | {cell_row(c, steps)} |")
    print()


def slope():
    fr = load("ice_sweep_mujoco_elliptic.json")
    # A full sweep writes its slope cells into the main file; `--only slope` writes its own.
    d = fr if fr and fr["results"]["slope"] else load("ice_sweep_mujoco_elliptic_slope.json")
    if not d:
        return
    rows = d["results"]["slope"]
    print("### Slope at mu 0.75 (MuJoCo, same seeds, uphill)\n")
    print("| condition | falls | of which before step 150 | after | median steps | body speed m/s | heading drift deg |")
    print("|---|---|---|---|---|---|---|")
    for c in rows:
        if "margin" not in c:
            print(f"| {c['label']} | {cell_row(c, d['steps'])} |")
    print("\n### Friction threshold on a slope (MuJoCo, same seeds)\n")
    print("Margin = mu - tan(slope): the friction left over after holding position on the slope.\n")
    print("| slope | mu | margin | falls | median steps |")
    print("|---|---|---|---|---|")
    grid = [(0.0, c["mu_ground"], c["mu_ground"], c) for c in (fr["results"]["friction"] if fr else [])
            if c["mu_ground"] <= 0.4]
    grid += [(c["slope_deg"], c["mu_ground"], c["margin"], c) for c in rows if "margin" in c]
    for sl, mu, margin, c in sorted(grid, key=lambda g: (g[0], -g[1])):
        print(f"| {sl:g} deg | {mu:g} | {margin:.3f} | {c['falls']}/{c['n']} | {c['survived_median']:.0f} |")
    print()


def snow():
    d = load("snow_mpm.json")
    if not d:
        return
    rows = d["rows"]
    a = d["args"]
    print(
        f"### Snow (depth {a['depth']} m, voxel {a['voxel']} m, {a['mpm_per_tick']} coupled steps per tick, "
        f"{a['seeds']} seeds x {a['steps']} steps, device {d['device']})\n"
    )
    print("| ground | yield pressure / cohesion | falls | fell at step | sinkage cm | body speed m/s | snow share of weight |")
    print("|---|---|---|---|---|---|---|")
    for name in dict.fromkeys(r["snow"] for r in rows):
        g = [r for r in rows if r["snow"] == name]
        bare = name == "bare"
        mat = "-" if bare else f"{g[0]['yield_pressure'] / 1000:g} kPa / {g[0]['cohesion'] / 1000:g} kPa"
        share = "-" if bare else f"{st.mean(r['snow_weight_fraction'] for r in g):.0%}"
        fell = ", ".join(str(r["survived"]) for r in g if r["fell"]) or "-"
        print(
            f"| {name} | {mat} | {sum(r['fell'] for r in g)}/{len(g)} | {fell} | "
            f"{100 * st.mean(r['sink_m'] for r in g):.1f} | {st.mean(r['mean_speed'] for r in g):.2f} | {share} |"
        )
    print()


def fleet():
    """Paired fleet missions: coordinated vs independent, without and with the slip reflex."""
    files = [("no reflex", load("fleet_search.json")), ("slip reflex", load("fleet_search_reflex.json"))]
    files = [(n, d) for n, d in files if d]
    if not files:
        return
    d0 = files[0][1]
    print(f"### Fleet search ({d0['robots']} robots, {d0['summary'][next(iter(d0['summary']))]['missions']} paired seeds, "
          f"snow mu {d0['snow_mu']}, ice mu {d0['ice_mu']})\n")
    print("| robots | radio | beacon located | dig team assembled | robots lost per mission | median time to locate s | median time to assemble s |")
    print("|---|---|---|---|---|---|---|")
    for name, d in files:
        for mode, sm in d["summary"].items():
            print(
                f"| {name} | {'on' if mode == 'coordinated' else 'off'} | {sm['located']}/{sm['missions']} | "
                f"{sm['assembled']}/{sm['missions']} | {sm['robots_lost_mean']:.2f} | "
                f"{sm['t_located_median_s'] if sm['t_located_median_s'] is not None else '-'} | "
                f"{sm['t_assembled_median_s'] if sm['t_assembled_median_s'] is not None else '-'} |"
            )
    print()
    for name, d in files:
        rows = d["rows"]
        seeds = sorted({r["seed"] for r in rows})
        by = {(r["seed"], r["mode"]): r for r in rows}
        if not all((s, "coordinated") in by and (s, "independent") in by for s in seeds):
            continue
        win = sum(by[s, "coordinated"]["assembled"] and not by[s, "independent"]["assembled"] for s in seeds)
        lose = sum(by[s, "independent"]["assembled"] and not by[s, "coordinated"]["assembled"] for s in seeds)
        more = sum(by[s, "coordinated"]["robots_lost"] > by[s, "independent"]["robots_lost"] for s in seeds)
        fewer = sum(by[s, "coordinated"]["robots_lost"] < by[s, "independent"]["robots_lost"] for s in seeds)
        print(
            f"- {name}, paired by seed: the team assembles with the radio on and not off on {win} seeds, the "
            f"reverse on {lose}. Radio on loses more robots on {more} seeds and fewer on {fewer}."
        )
    print()


if __name__ == "__main__":
    cal = load("friction_calibration.json")
    if cal:
        print("### Friction calibration (sliding block, effective mu)\n")
        print("| nominal ground mu | MuJoCo + ground priority | MuJoCo default (max) | Steel, both sides mu | Steel, 0.75 block (average) |")
        print("|---|---|---|---|---|")
        for r in cal["rows"]:
            print(f"| {r['nominal']:g} | {r['priority']:.3f} | {r['max']:.3f} | {r['steel']:.3f} | {r['steel_avg']:.3f} |")
        print()
    friction()
    sweep("wind", "wind_n", "N")
    slope()
    sweep("speed", "command_vx", "m/s", "ice_sweep_mujoco_elliptic_speed.json")
    snow()
    fleet()
