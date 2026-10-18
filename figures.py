"""Make paper figures S1-S5 from results/*.json."""
import json, os
from datetime import datetime, timedelta
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "results")
F = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(F, exist_ok=True)
BASE, LN, INK, MUTED, GRID = "#2f6bc4", "#c2410c", "#1f2937", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9ca3af", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
main = json.load(open(f"{R}/main.json")); M = main["main"]
CONDS = ["Earth (same clock)", "Moon (1.3 s)", "Drift only (0 delay)", "Delay only (13.9 min)",
         "Mars near (3.2 min)", "Mars average (13.9 min)", "Mars far (22.3 min)"]
LAB = ["Earth\n(same clock)", "Moon\n1.3 s", "Drift only\n0 delay", "Delay only\n13.9 min",
       "Mars near\n3.2 min", "Mars avg\n13.9 min", "Mars far\n22.3 min"]
note = lambda fig, s: fig.text(0.01, 0.01, s, fontsize=7.2, color=MUTED)

def grouped(ax, key, scale=1.0, fmt="{:.0f}", label_idx=None):
    x = np.arange(len(CONDS)); w = 0.38
    for j, (d, c, nm) in enumerate([("baseline", BASE, "Baseline (real-time habits)"),
                                    ("latency_native", LN, "Latency-native design")]):
        m = np.array([M[f"{k}|{d}"][key]["mean"] for k in CONDS]) * scale
        lo = np.array([M[f"{k}|{d}"][key]["ci_lo"] for k in CONDS]) * scale
        hi = np.array([M[f"{k}|{d}"][key]["ci_hi"] for k in CONDS]) * scale
        xs = x + (j - 0.5) * (w + 0.02)
        ax.bar(xs, m, w, color=c, label=nm, edgecolor="white", linewidth=1.0)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        for i in (label_idx if label_idx is not None else range(len(CONDS))):
            ax.text(xs[i], hi[i] + ax.get_ylim()[1] * 0.0, fmt.format(m[i]), ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=8)
    ax.axvspan(3.5, 6.5, color="#fff7ed", zorder=0)
    ax.grid(axis="x", visible=False)

# ---------------- S1: conversation
fig, ax = plt.subplots(figsize=(8.4, 3.9))
grouped(ax, "median_resolution_h")
ax.set_ylabel("Median time to resolve a topic (hours)")
ax.set_title("Latency-native messaging roughly halves how long conversations take on Mars",
             loc="left", fontweight="bold", fontsize=10.5)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
note(fig, f"Simulation, {main['N']} pairs per condition, 106 measured days each. Bars: mean of per-pair medians; whiskers: 95% CI. Shaded: Mars conditions.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{F}/S1_conversation.png", dpi=220); plt.close(fig)

# ---------------- S2: shared events
fig, ax = plt.subplots(figsize=(8.4, 3.9))
grouped(ax, "events_shared_2way", 100, "{:.0f}%")
ax.set_ylim(0, 105)
ax.set_ylabel("Events shared in both directions (%)")
ax.set_title("Co-watching restores two-way sharing of events that live viewing loses on Mars",
             loc="left", fontweight="bold", fontsize=10.5)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
note(fig, "Both directions = each side experiences the other's reactions at the matching moment, with no spoiler. Whiskers: 95% CI.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{F}/S2_shared_events.png", dpi=220); plt.close(fig)

# ---------------- S3: play (small multiples)
fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.3), sharey=False)
x = np.arange(len(CONDS))
short = ["Earth", "Moon", "Drift", "Delay", "M near", "M avg", "M far"]
for ax, (key, d, c, title) in zip(axs, [
        ("rt_sessions_per_week", "baseline", BASE, "Real-time game sessions"),
        ("tb_games_per_week", "baseline", BASE, "Turn-based games finished"),
        ("ghost_matches_per_week", "latency_native", LN, "Ghost matches (latency-native)")]):
    m = np.array([M[f"{k}|{d}"][key]["mean"] for k in CONDS])
    ax.bar(x, m, 0.62, color=c, edgecolor="white")
    for i in (0, 5):
        ax.text(x[i], m[i], f"{m[i]:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="x", visible=False)
    ax.axvspan(3.5, 6.5, color="#fff7ed", zorder=0)
axs[0].set_ylabel("Per week")
fig.suptitle("Real-time play disappears beyond Earth; ghost play keeps a steady rhythm", x=0.01, ha="left",
             fontweight="bold", fontsize=10.5)
note(fig, "Real-time games assume one-way delay must stay under 0.15 s. A turn-based game = 20 moves. Separate y-scales per panel.")
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(f"{F}/S3_play.png", dpi=220); plt.close(fig)

# ---------------- S4: full synodic cycle
cy = json.load(open(f"{R}/cycle.json"))
st = datetime.fromisoformat(cy["start"])
dates = [st + timedelta(days=w*7 + 3.5) for w in cy["weeks"]]
b0, b1 = [st + timedelta(days=d) for d in cy["blackout_days"]]
fig, axs = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
axs[0].plot(dates, cy["delay_min"], color=INK, lw=1.6)
axs[0].set_ylabel("Delay (min)"); axs[0].set_ylim(0, 25)
for d, c, nm in [("baseline", BASE, "Baseline"), ("latency_native", LN, "Latency-native")]:
    y = np.array([v if v is not None else np.nan for v in cy["series"][d]["median_resolution_h"]], float)
    axs[1].plot(dates, y, color=c, lw=1.8, label=nm)
    axs[1].text(dates[-1], y[~np.isnan(y)][-1], f"  {nm}", color=INK, va="center", fontsize=8)
for ax in axs:
    ax.axvspan(b0, b1, color="#e5e7eb", zorder=0)
axs[1].text(b0, axs[1].get_ylim()[1]*0.92, "conjunction\nblackout  ", fontsize=7.5, color=MUTED, va="top", ha="right")
axs[1].set_ylabel("Median topic resolution (h)")
axs[1].legend(frameon=False, loc="upper left", fontsize=8.5)
axs[0].set_title("Over a full Earth–Mars cycle the design gap holds, but the blackout stalls both designs",
                 loc="left", fontweight="bold", fontsize=10)
note(fig, f"Time-varying delay from the Appendix A orbital model, Feb 2027 to Apr 2029; 14-day blackout at peak delay. {cy['n']} pairs per design, weekly medians.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{F}/S4_full_cycle.png", dpi=220); plt.close(fig)

# ---------------- S5: sensitivity tornado
se = json.load(open(f"{R}/sensitivity.json"))
d0 = se["default"]
names = sorted({k.split("x")[0] for k in se if k != "default"})
nice = dict(attend_base="Topics overlooked (baseline)", attend_ln="Topics overlooked (latency-native)",
            clarify_base="Clarification turns (baseline)", clarify_ln="Clarification turns (latency-native)",
            branch_ln="Branching answers", leak_base="Spoiler leak rate (baseline)", leak_ln="Spoiler leak (shield)",
            replay_p="Earth replays reactions", topic_rate="New topics per day", k_mean="Turns per topic",
            p_skip="Busy (skipped) windows", reply_median="Reply time")
fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
spread = lambda n, key, ref: max(abs(se[f"{n}x0.5"][key]-ref), abs(se[f"{n}x1.5"][key]-ref))
order = sorted(names, key=lambda n: (spread(n, "res_ratio", d0["res_ratio"]) / max(d0["res_ratio"], 1e-9)
                                     + spread(n, "sh2_gain", d0["sh2_gain"]) / max(d0["sh2_gain"], 1e-9)))
rows = [(n,) for n in order]
for ax, key, ref, title in [
        (axs[0], "res_ratio", d0["res_ratio"], "Conversation speed-up\n(baseline time ÷ latency-native time)"),
        (axs[1], "sh2_gain", d0["sh2_gain"], "Gain in two-way shared events\n(percentage points)")]:
    sc = 100 if key == "sh2_gain" else 1
    for i, n in enumerate(order):
        lo = se[f"{n}x0.5"][key]*sc; hi = se[f"{n}x1.5"][key]*sc
        ax.plot([lo, hi], [i, i], color="#cbd5e1", lw=6, solid_capstyle="round", zorder=1)
        ax.plot(lo, i, "o", color=BASE, ms=6, zorder=2); ax.plot(hi, i, "o", color=LN, ms=6, zorder=2)
    ax.axvline(ref*sc, color=INK, lw=1, ls="--")
    ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="y", visible=False)
    if key == "res_ratio":
        ax.axvline(1.0, color=MUTED, lw=0.8); ax.set_xlim(0.6, 5)
        ax.text(1.0, -0.75, " break-even", fontsize=7, color=MUTED)
    else:
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlim(-5, 95)
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([nice[r[0]] for r in rows], fontsize=8)
axs[1].plot([], [], "o", color=BASE, label="parameter × 0.5"); axs[1].plot([], [], "o", color=LN, label="parameter × 1.5")
axs[1].plot([], [], "--", color=INK, label="default value"); axs[1].legend(frameon=True, framealpha=0.95, edgecolor="none", fontsize=7.5, loc="upper left")
fig.suptitle("Sensitivity at average Mars delay: the result depends most on question tracking actually working", x=0.01, ha="left",
             fontweight="bold", fontsize=10.5)
note(fig, "Each assumption scaled to 0.5× and 1.5× (probabilities capped at 0.99) with all others at default. Speed-up > 1 and gain > 0 mean latency-native is better.")
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(f"{F}/S5_sensitivity.png", dpi=220); plt.close(fig)
print("figures done")
