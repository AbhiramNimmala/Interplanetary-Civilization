# Latency-Native Media Simulator (LNMS)

Supplementary code and data for *Latency-Native Media: Designing Shared Experience for an
Interplanetary Civilization* (Abhiram Nimmala, 2026).

An agent-based, discrete-event simulation of two people, one on Earth and one off-Earth,
communicating through a delayed link with daily schedules, Mars-sol drift and solar-conjunction
blackouts. It compares real-time habits with latency-native designs for conversation, shared
events and play.

## Files

| File | What it does |
| --- | --- |
| `sim.py` | The simulator. All behavioural assumptions are in `PARAMS`. |
| `experiments.py` | Runs every experiment: 7 fixed conditions, the full orbital cycle, and the sensitivity analysis. |
| `validate.py` | Checks the simulator against the analytic model `turn time = D + T`. |
| `figures.py` | Draws Figures 4–8 of the paper from `results/`. |
| `results/main.json` | Means, 95% CIs and paired differences for every condition and design. |
| `results/cycle.json` | Weekly time series over Feb 2027 – Apr 2029. |
| `results/sensitivity.json` | Each assumption at 0.5x and 1.5x. |
| `results/validation.txt` | Validation table (paper Table 11). |
| `figures/` | Paper figures as PNG. |

## Reproduce

Requires Python 3.9+ with NumPy and Matplotlib.

```
pip install numpy matplotlib
python3 validate.py              # about 1 minute
python3 experiments.py 200       # all experiments, a few minutes on 2 cores
python3 figures.py
```

`python3 experiments.py 200 main|cycle|sens` runs one part only.

## What it measures

Communication **opportunity**: time to resolve conversation topics, share of events
experienced together, and play sessions per week. It does **not** measure how connected
people feel; that needs the human study described in the paper's Appendix E.
