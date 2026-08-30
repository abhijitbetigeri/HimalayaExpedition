"""Does the one-control-step tether lag actually matter? Measure it.

Runs the SAME seed and the SAME action sequence through `substep=False` (force
computed once per 20 ms control step, current behaviour) and `substep=True`
(recomputed every 2 ms physics substep), with the rope forced taut at reset so
the spring-damper is loaded from step 0.

If the trajectories match, the lag is cosmetic and substep should stay off --
it costs compile time and a global patch for nothing. If they diverge, the
current tether is modelling a softer, laggier rope than its own config says.

    python check_tether_lag.py --steps 120 --preload 0.30
"""

import argparse
import copy

import jax
import jax.numpy as jp
import numpy as np

import fixed_line


def rollout(substep, steps, preload, seed, fall=False):
    cfg = copy.deepcopy(fixed_line.default_config())
    cfg.line_config.substep = substep
    env = fixed_line.FixedLineAscent(config=cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    st = reset(jax.random.PRNGKey(seed))
    # Force the rope taut immediately: put the ascender preload metres above
    # where the robot actually is, past the slack. Otherwise the tether is
    # slack for the whole rollout and both modes trivially agree.
    st.info["asc_s"] = st.info["asc_s"] + cfg.line_config.slack + preload

    s_hist, f_hist = [], []
    rng = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        rng, ar = jax.random.split(rng)
        # Same action stream in both modes -- deterministic given the seed, so
        # any divergence is physics, not exploration noise.
        # --fall: neutral action, rope slack at reset, so the robot drops
        # and the rope catches it DYNAMICALLY. That is the case where a stale
        # damping term should hurt most -- high velocity at the instant the
        # spring engages.
        act = (jp.zeros(env.action_size) if fall
               else 0.3 * jax.random.normal(ar, (env.action_size,)))
        st = step(st, act)
        s_hist.append(float(env._project(st.data)))
        f_hist.append(float(jp.linalg.norm(st.info["tether_force"])))
    env_free = env
    del env_free
    return np.array(s_hist), np.array(f_hist)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--preload", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fall", action="store_true",
                   help="Neutral action + slack rope: dynamic catch at speed.")
    a = p.parse_args()

    print(f"rope preloaded {a.preload} m past slack "
          f"(~{1200 * a.preload:.0f} N at k=1200 N/m)\n", flush=True)

    out = {}
    for mode in (False, True):
        s, f = rollout(mode, a.steps, a.preload, a.seed, a.fall)
        out[mode] = (s, f)
        label = "substep" if mode else "per-step"
        # Sign changes in the force derivative = ringing.
        d = np.diff(f)
        reversals = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
        print(f"[{label:8s}] peak force {f.max():7.1f} N | mean {f.mean():7.1f} N "
              f"| force reversals {reversals:3d} | net travel {s[-1] - s[0]:+.4f} m",
              flush=True)

    s0, f0 = out[False]
    s1, f1 = out[True]
    print(f"\nmax |position difference| : {np.abs(s1 - s0).max() * 100:.2f} cm")
    print(f"max |force difference|    : {np.abs(f1 - f0).max():.1f} N")
    print(f"final position difference : {abs(s1[-1] - s0[-1]) * 100:.2f} cm")


if __name__ == "__main__":
    main()
