"""Run all simulation experiments and save results to results/."""
import json, os, sys, copy
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy as np
import sim

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- orbital delay (Appendix A model)
def _kepler(M, e):
    E = M.copy()
    for _ in range(30):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    return E
def _pos(d, a, e, varpi, L0, n):
    w = np.radians(varpi); M = (np.radians(L0 + n*d) - w) % (2*np.pi); E = _kepler(M, e)
    x = a*(np.cos(E)-e); y = a*np.sqrt(1-e*e)*np.sin(E)
    return x*np.cos(w)-y*np.sin(w), x*np.sin(w)+y*np.cos(w)
START = datetime(2026, 1, 1)
DAYS_ORBIT = 1400
_d = np.array([(START + timedelta(days=i) - datetime(2000, 1, 1, 12)).total_seconds()/86400 for i in range(DAYS_ORBIT)])
_ex, _ey = _pos(_d, 1.00000261, 0.01671123, 102.93768, 100.46457, 0.98560028)
_mx, _my = _pos(_d, 1.52371034, 0.09339410, -23.94363, -4.55343, 0.52402068)
ORBIT_MIN = np.hypot(_mx-_ex, _my-_ey) * 499.004784 / 60.0   # one-way delay, minutes, per day

CYCLE_OFFSET = 400        # start the cycle ~Feb 2027 so the 2028 conjunction falls mid-run
class OrbitDelay:
    def __call__(self, t):
        x = t / sim.EARTH_DAY + CYCLE_OFFSET
        i = min(int(x), DAYS_ORBIT - 2); f = x - i
        return ORBIT_MIN[i]*(1-f) + ORBIT_MIN[i+1]*f
class Const:
    def __init__(self, m): self.m = m
    def __call__(self, t): return self.m

CYCLE_DAYS = 794          # one synodic cycle (~780 d) + 14 d warm-up
peak = int(np.argmax(ORBIT_MIN[CYCLE_OFFSET:CYCLE_OFFSET+CYCLE_DAYS]))   # days after cycle start
BLACKOUT = [((peak-7)*sim.EARTH_DAY, (peak+7)*sim.EARTH_DAY)]

def condition(name):
    return {
        "Earth (same clock)":        dict(delay=Const(0.0), drift=False),
        "Moon (1.3 s)":              dict(delay=Const(1.3/60), drift=False),
        "Drift only (0 delay)":      dict(delay=Const(0.0), drift=True),
        "Delay only (13.9 min)":     dict(delay=Const(13.9), drift=False),
        "Mars near (3.2 min)":       dict(delay=Const(3.2), drift=True),
        "Mars average (13.9 min)":   dict(delay=Const(13.9), drift=True),
        "Mars far (22.3 min)":       dict(delay=Const(22.3), drift=True),
        "Mars full cycle":           dict(delay=OrbitDelay(), drift=True, blackouts=BLACKOUT),
    }[name]

FIXED = ["Earth (same clock)", "Moon (1.3 s)", "Drift only (0 delay)", "Delay only (13.9 min)",
         "Mars near (3.2 min)", "Mars average (13.9 min)", "Mars far (22.3 min)"]
DESIGNS = ["baseline", "latency_native"]

def job(args):
    name, design, seed, days, params, detail = args
    p = copy.deepcopy(sim.PARAMS); p.update(params or {})
    r = sim.run(condition(name), design, seed, days=days, p=p, detail=detail)
    return name, design, seed, r

def summarize(rows):
    keys = [k for k in rows[0] if not k.endswith("_log")]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float); v = v[~np.isnan(v)]
        m = v.mean(); se = v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0
        out[k] = dict(mean=m, ci_lo=m-1.96*se, ci_hi=m+1.96*se, median=float(np.median(v)), n=len(v))
    return out

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    only = sys.argv[2] if len(sys.argv) > 2 else "all"
    with Pool() as pool:
      if only in ("all", "main"):
        # 1. main experiment: fixed conditions, paired seeds across designs
        jobs = [(c, d, s, 120, None, False) for c in FIXED for d in DESIGNS for s in range(N)]
        res = pool.map(job, jobs, chunksize=8)
        main = {}
        for c in FIXED:
            for d in DESIGNS:
                main[f"{c}|{d}"] = summarize([r for n, dd, s, r in res if n == c and dd == d])
        # paired differences for the key outcomes at Mars conditions
        paired = {}
        for c in FIXED:
            b = {s: r for n, dd, s, r in res if n == c and dd == "baseline"}
            l = {s: r for n, dd, s, r in res if n == c and dd == "latency_native"}
            for k in ["median_resolution_h", "p90_resolution_h", "events_shared_2way", "events_shared_1way"]:
                diff = np.array([l[s][k]-b[s][k] for s in b if not (np.isnan(l[s][k]) or np.isnan(b[s][k]))])
                paired[f"{c}|{k}"] = dict(mean=diff.mean(), ci_lo=diff.mean()-1.96*diff.std(ddof=1)/np.sqrt(len(diff)),
                                          ci_hi=diff.mean()+1.96*diff.std(ddof=1)/np.sqrt(len(diff)),
                                          share_ln_better=float(np.mean(diff < 0) if "resolution" in k else np.mean(diff > 0)))
        json.dump(dict(N=N, main=main, paired=paired), open(f"{OUT}/main.json", "w"), indent=1)
        print("main done")

      if only in ("all", "cycle"):
        # 2. full synodic cycle with blackout: weekly time series
        NC = max(20, N // 10)
        jobs = [("Mars full cycle", d, 10000+s, CYCLE_DAYS, None, True) for d in DESIGNS for s in range(NC)]
        res = pool.map(job, jobs, chunksize=2)
        weeks = np.arange(2, CYCLE_DAYS // 7)
        series = {}
        for d in DESIGNS:
            med = []; sh2 = []
            for w in weeks:
                lo, hi = w*7*sim.EARTH_DAY, (w+1)*7*sim.EARTH_DAY
                durs = [x[1] for n, dd, s, r in res if dd == d for x in r["res_log"] if lo <= x[0] < hi]
                evs = [x[1] for n, dd, s, r in res if dd == d for x in r["ev_log"] if lo <= x[0] < hi]
                med.append(float(np.median(durs)) if durs else None)
                sh2.append(float(np.mean([e == "shared2" for e in evs])) if evs else None)
            series[d] = dict(median_resolution_h=med, shared2=sh2)
        cyc_summary = {d: summarize([r for n, dd, s, r in res if dd == d]) for d in DESIGNS}
        json.dump(dict(weeks=weeks.tolist(), start=(START+timedelta(days=CYCLE_OFFSET)).isoformat(), delay_min=[float(ORBIT_MIN[CYCLE_OFFSET+w*7]) for w in weeks],
                       blackout_days=[peak-7, peak+7], series=series, summary=cyc_summary, n=NC),
                  open(f"{OUT}/cycle.json", "w"), indent=1)
        print("cycle done")

      if only in ("all", "sens"):
        # 3. sensitivity analysis at Mars average
        NS = max(50, N // 2)
        base = sim.PARAMS
        sens_params = ["attend_base", "attend_ln", "clarify_base", "clarify_ln", "branch_ln", "leak_base",
                       "leak_ln", "replay_p", "topic_rate", "k_mean", "p_skip", "reply_median"]
        settings = [("default", {})]
        for k in sens_params:
            for f in (0.5, 1.5):
                v = base[k]*f
                if k.startswith(("attend", "clarify", "branch", "leak", "replay", "p_skip")):
                    v = min(v, 0.99)
                settings.append((f"{k}x{f}", {k: v}))
        jobs = [("Mars average (13.9 min)", d, 20000+s, 120, prm, False) for _, prm in settings for d in DESIGNS for s in range(NS)]
        res = pool.map(job, jobs, chunksize=8)
        sens = {}
        for i, (lab, prm) in enumerate(settings):
            block = res[i*2*NS:(i+1)*2*NS]
            b = [r for n, d, s, r in block if d == "baseline"]; l = [r for n, d, s, r in block if d == "latency_native"]
            mb = np.nanmean([r["median_resolution_h"] for r in b]); ml = np.nanmean([r["median_resolution_h"] for r in l])
            sb = np.mean([r["events_shared_2way"] for r in b]); sl = np.mean([r["events_shared_2way"] for r in l])
            sens[lab] = dict(params=prm, res_ratio=mb/ml, res_base=mb, res_ln=ml, sh2_base=sb, sh2_ln=sl, sh2_gain=sl-sb)
        json.dump(sens, open(f"{OUT}/sensitivity.json", "w"), indent=1)
        print("sensitivity done")
