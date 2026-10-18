"""Validate the simulator against the analytic best-case model (Chapter 4.2).

With both people always available, nothing overlooked and no clarifications,
a topic needing K turns should take about K*(D + T) minutes, where T is the
mean reply time.
"""
import copy, math
import numpy as np
import sim

p = copy.deepcopy(sim.PARAMS)
p.update(windows=[(0, sim.EARTH_DAY)], p_skip=0.0, attend_base=1.0, clarify_base=0.0,
         topic_rate=0.05, k_mean=10.0, event_rate=0.0)
T = p["reply_median"] * math.exp(0.6**2 / 2)        # mean of the lognormal reply time

print(f"{'D (min)':>8} {'sim min/turn':>13} {'analytic D+T':>13} {'error':>7}")
rows = []
for D in [0.0, 3.2, 13.9, 22.3]:
    cond = dict(delay=lambda t, D=D: D, drift=False)
    per_turn = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        r = sim.run(cond, "baseline", seed, days=200, warmup=0, p=p, detail=True)
        # recover per-turn time: duration / turns is unknown per topic, so use K-free check:
        per_turn += [d for _, d in r["res_log"]]
    # mean duration of a topic = E[K] * (D + T); E[K] = 1 + k_mean
    sim_turn = np.mean(per_turn) * 60 / (1 + p["k_mean"])
    rows.append((D, sim_turn, D + T))
    print(f"{D:8.1f} {sim_turn:13.2f} {D+T:13.2f} {100*(sim_turn/(D+T)-1):6.1f}%")
