"""
Latency-Native Media Simulator (LNMS)
=====================================
Discrete-event agent simulation of two people, one on Earth and one off-Earth,
communicating through a delayed link.

It models:
  * one-way signal delay D (fixed, or time-varying from an orbital model)
  * solar-conjunction blackouts (link down; messages queue)
  * daily free windows on each person's own clock (Earth day 1440 min,
    Mars sol 1479.58 min -> schedules drift ~40 min/day)
  * randomly skipped windows (busy days)

and three activities, each under a BASELINE (real-time habits) and a
LATENCY-NATIVE design:
  1. Conversation  - topics that need K back-and-forth turns to resolve
                     baseline: per-topic replies, some topics overlooked,
                               some turns spent on clarification
                     latency-native (batched): open-question tracking,
                               fewer clarifications, branching answers
  2. Shared events - Earth-side live events (games, shows)
                     baseline: live stream + live chat
                     latency-native: co-watching with timeline-anchored
                               reactions and a spoiler shield
  3. Play          - real-time game sessions vs turn-based vs ghost runs

All behavioural parameters are explicit in PARAMS and tested in the
sensitivity analysis. Outputs are measures of communication *opportunity*
(what is possible), not of felt connection.
"""
import heapq, math
import numpy as np

EARTH_DAY = 1440.0
MARS_SOL = 1479.58  # 24 h 39 m 35 s

PARAMS = dict(
    # schedule: free windows in local minutes [start, end)
    windows=[(7*60, 7*60+30), (12*60, 12*60+30), (18*60+30, 22*60+30)],
    p_skip=0.2,               # chance a given window is lost to other commitments
    # conversation
    topic_rate=2.0,           # new topics per person per day
    k_mean=5.0,               # mean turns needed to resolve a topic (1 + geometric)
    reply_median=1.5,         # minutes to read + reply (lognormal median)
    attend_base=0.75,         # P(address a waiting topic in a session), baseline
    attend_ln=0.97,           # same, with open-question tracking
    clarify_base=0.25,        # P(a turn is spent on clarification), baseline
    clarify_ln=0.10,          # same, batched rich messages
    branch_ln=0.25,           # P(branching answer resolves two turns at once)
    # shared events
    event_rate=2.0/7,         # Earth-side events per day
    event_len=120.0,          # minutes
    sync_tolerance_s=5.0,     # max one-way delay (s) for reactions to feel simultaneous
    leak_base=0.30,           # P(an incoming Earth message spoils an unwatched event), baseline
    leak_ln=0.03,             # same, with spoiler shield
    watch_deadline_h=72.0,    # an event not watched within this many hours is missed
    replay_p=0.6,             # P(Earth side replays the off-Earth partner's reactions)
    replay_deadline_h=48.0,
    # play
    rt_game_max_delay_s=0.15, # max one-way delay for a playable real-time game
    rt_min_overlap=30.0,      # minutes of joint free time needed for a session
    game_moves=20,            # moves in a turn-based game
    ghost_p=0.7,              # P(play a ghost run in a free window when a new ghost is waiting)
)


class Person:
    def __init__(self, name, day_len, phase, horizon, rng, p):
        self.name, self.day_len, self.phase = name, day_len, phase
        self.windows = p["windows"]
        ndays = int(horizon / day_len) + 3
        self.skip = rng.random((ndays, len(self.windows))) < p["p_skip"]

    def _local(self, t):
        x = t + self.phase
        return int(x // self.day_len), x % self.day_len

    def window_at(self, t):
        """(start, end) of the free window containing t, or None."""
        day, loc = self._local(t)
        if day < 0 or day >= len(self.skip):
            return None
        for i, (a, b) in enumerate(self.windows):
            if a <= loc < b and not self.skip[day, i]:
                base = t - loc
                return base + a, base + b
        return None

    def windows_between(self, t0, t1):
        """All non-skipped free windows overlapping [t0, t1)."""
        out = []
        day0, _ = self._local(t0)
        day1, _ = self._local(t1)
        for day in range(max(day0, 0), min(day1 + 1, len(self.skip))):
            base = day * self.day_len - self.phase
            for i, (a, b) in enumerate(self.windows):
                if self.skip[day, i]:
                    continue
                s, e = base + a, base + b
                if e > t0 and s < t1:
                    out.append((s, e))
        return out

    def next_window(self, t, min_left=0.0):
        """Earliest window (start, end) with at least min_left minutes available after t."""
        for s, e in self.windows_between(t, t + 4 * self.day_len):
            start = max(s, t)
            if e - start >= min_left:
                return start, e
        return None


class Link:
    """One-way delay D(t) in minutes plus blackout intervals."""
    def __init__(self, delay_fn, blackouts=()):
        self.delay_fn, self.blackouts = delay_fn, list(blackouts)

    def arrival(self, t):
        for a, b in self.blackouts:
            if a <= t < b:
                t = b
        return t + self.delay_fn(t)


def run(condition, design, seed, days=120, warmup=14, p=PARAMS, detail=False):
    """Simulate one pair for `days`; return metrics over days (warmup, days]."""
    rng = np.random.default_rng(seed)
    H = days * EARTH_DAY
    T0 = warmup * EARTH_DAY
    ln = (design == "latency_native")

    earth = Person("E", EARTH_DAY, 0.0, H, rng, p)
    other_day = MARS_SOL if condition["drift"] else EARTH_DAY
    other_phase = rng.uniform(0, other_day) if condition["drift"] else condition.get("phase", 0.0)
    other = Person("M", other_day, other_phase, H, rng, p)
    people = {"E": earth, "M": other}
    link = Link(condition["delay"], condition.get("blackouts", ()))
    peer = {"E": "M", "M": "E"}

    attend = p["attend_ln"] if ln else p["attend_base"]
    clarify = p["clarify_ln"] if ln else p["clarify_base"]
    branch = p["branch_ln"] if ln else 0.0

    # ------------------------------------------------------------------ conversation
    topics = {}           # id -> dict(rem, turn, created, game)
    pending = {"E": set(), "M": set()}
    resolved, durations = 0, []
    res_log, ev_log = [], []
    games_done = 0
    arrivals_at_M = []    # times Earth messages arrive at M (for spoilers)
    ev = []
    seq = [0]
    def push(t, kind, who, data=None):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, who, data))

    tid = [0]
    def new_topic(t, who, rem, game=False):
        tid[0] += 1
        topics[tid[0]] = dict(rem=rem, turn=who, created=t, game=game)
        pending[who].add(tid[0])
        return tid[0]

    # initial turn-based game
    new_topic(0.0, "E", p["game_moves"], game=True)
    # topic arrivals (Poisson) for each person
    for who in ("E", "M"):
        n = rng.poisson(p["topic_rate"] * days)
        for t in np.sort(rng.uniform(0, H, n)):
            push(t, "topic", who)
    # session starts at every free window
    for who, per in people.items():
        for s, e in per.windows_between(0, H):
            push(s, "session", who)

    compose_at = {"E": None, "M": None}
    def sched_compose(t, who):
        if compose_at[who] is None or compose_at[who] > t:
            compose_at[who] = t
            push(t, "compose", who)

    def reply_time():
        return float(rng.lognormal(math.log(p["reply_median"]), 0.6))

    while ev:
        t, _, kind, who, data = heapq.heappop(ev)
        if t > H:
            break
        per = people[who]
        if kind == "topic":
            k = 1 + rng.geometric(1.0 / p["k_mean"])
            new_topic(t, who, int(k))
            if per.window_at(t):
                sched_compose(t + reply_time(), who)
        elif kind == "session":
            if pending[who]:
                sched_compose(t + reply_time(), who)
        elif kind == "arrive":
            for (i, adv) in data:
                tp = topics[i]
                if tp["rem"] <= 0:
                    continue
                pending[who].add(i)
            if who == "M":
                arrivals_at_M.append(t)
            if per.window_at(t):
                sched_compose(t + reply_time(), who)
        elif kind == "compose":
            if compose_at[who] is None or abs(compose_at[who] - t) > 1e-9:
                continue
            compose_at[who] = None
            if not per.window_at(t):
                continue  # window closed; next session will pick up
            out = []
            for i in list(pending[who]):
                tp = topics[i]
                if tp["turn"] != who or tp["rem"] <= 0:
                    pending[who].discard(i)
                    continue
                if not tp["game"] and rng.random() > attend:
                    continue  # overlooked this session
                pending[who].discard(i)
                if tp["game"]:
                    tp["rem"] -= 1
                elif rng.random() < clarify:
                    pass          # turn spent clarifying; no progress
                else:
                    tp["rem"] -= 2 if rng.random() < branch else 1
                tp["turn"] = peer[who]
                out.append((i, 1))
            if out:
                arr = link.arrival(t)
                for i, _ in out:
                    tp = topics[i]
                    if tp["rem"] <= 0:
                        if arr > T0:
                            if tp["game"]:
                                games_done += 1
                            else:
                                resolved += 1
                                durations.append((arr - tp["created"]) / 60.0)
                                res_log.append((arr, (arr - tp["created"]) / 60.0))
                        if tp["game"]:
                            new_topic(arr, peer[who], p["game_moves"], game=True)
                live = [(i, a) for i, a in out if topics[i]["rem"] > 0]
                push(arr, "arrive", peer[who], live)

    weeks = (days - warmup) / 7.0
    arrivals_at_M = np.array(sorted(arrivals_at_M))

    # ------------------------------------------------------------------ shared events
    n_ev = rng.poisson(p["event_rate"] * days)
    shared1 = shared2 = alone = missed = spoiled = 0
    counted = 0
    for _ in range(n_ev):
        day = rng.integers(0, days - 4)
        s = day * EARTH_DAY + rng.uniform(19 * 60, 20.5 * 60)
        if s < T0:
            continue
        counted += 1
        a = link.arrival(s)          # stream reaches off-Earth partner
        L = p["event_len"]
        Dsec = (a - s) * 60
        win = other.window_at(a)
        live_ok = win is not None and win[1] - a >= L
        leak = p["leak_ln"] if ln else p["leak_base"]
        def replayed(w_end):
            back = link.arrival(w_end)
            ew = earth.next_window(back, 20)
            return (ew is not None and ew[0] - back <= p["replay_deadline_h"] * 60
                    and rng.random() < p["replay_p"])
        if live_ok:                  # both designs allow watching live
            shared1 += 1             # Earth reactions arrive alongside the stream
            two = Dsec <= p["sync_tolerance_s"] or (ln and replayed(a + L))
            shared2 += two
            ev_log.append((s, "shared2" if two else "shared1"))
            continue
        # watch later
        nw = other.next_window(a, L)
        if nw is None or nw[0] - a > p["watch_deadline_h"] * 60:
            missed += 1
            ev_log.append((s, "missed"))
            continue
        w0 = nw[0]
        n_msgs = np.count_nonzero((arrivals_at_M > a + L) & (arrivals_at_M < w0))
        if rng.random() < 1 - (1 - leak) ** n_msgs:
            spoiled += 1
            ev_log.append((s, "spoiled"))
            continue
        if not ln:
            alone += 1        # watched later, no reactions in sync
            ev_log.append((s, "alone"))
            continue
        shared1 += 1          # anchored reactions from Earth side
        two = replayed(w0 + L)
        shared2 += two
        ev_log.append((s, "shared2" if two else "shared1"))
    counted = max(counted, 1)

    # ------------------------------------------------------------------ play
    rt_sessions = 0
    D_mid = condition["delay"](H / 2) * 60
    if D_mid <= p["rt_game_max_delay_s"]:
        ew = earth.windows_between(T0, H)
        for s, e in other.windows_between(T0, H):
            for s2, e2 in ew:
                if min(e, e2) - max(s, s2) >= p["rt_min_overlap"]:
                    rt_sessions += 1
    # ghost runs
    matches = 0
    if ln:
        inbox = {"E": [], "M": []}   # arrival times of ghosts
        last_run = {"E": -1.0, "M": -1.0}
        wins = sorted([(s, w) for w in ("E", "M") for s, _ in people[w].windows_between(0, H)])
        heap = []
        for s, w in wins:
            # deliver arrived ghosts up to time s
            while heap and heap[0][0] <= s:
                ta, to = heapq.heappop(heap)
                inbox[to].append(ta)
            new = [x for x in inbox[w] if x > last_run[w]]
            first = last_run[w] < 0 and not inbox[w]
            if (new or first) and rng.random() < p["ghost_p"]:
                if new and s > T0:
                    matches += 1
                last_run[w] = s
                heapq.heappush(heap, (link.arrival(s + 10), peer[w]))
    out = dict(
        topics_per_week=resolved / weeks,
        median_resolution_h=float(np.median(durations)) if durations else float("nan"),
        p90_resolution_h=float(np.percentile(durations, 90)) if durations else float("nan"),
        events_shared_1way=shared1 / counted,
        events_shared_2way=shared2 / counted,
        events_alone=alone / counted,
        events_spoiled=spoiled / counted,
        events_missed=missed / counted,
        rt_sessions_per_week=rt_sessions / weeks,
        tb_games_per_week=games_done / weeks,
        ghost_matches_per_week=matches / weeks,
    )
    if detail:
        out["res_log"], out["ev_log"] = res_log, ev_log
    return out
