"""Forecast GTA VI PS5 trophy earn rates on Exophase.

For each case the program reads /input/<id>.in (one JSON object: the game's
trophies and every valid read of Exophase's trophy page before the case's
checkpoint) and writes /output/<id>.out: five quantiles (levels 0.05, 0.25,
0.5, 0.75, 0.95) of each trophy's earn rate at the case's target, in percent.

Model
-----
An earn rate is a ratio: the players who hold a trophy over the players
Exophase tracks. Both sides move, and they move on different clocks. New
players dilute every rate (they hold almost nothing yet); existing players
push rates up as they progress. So the program models the two sides
separately, in counts:

* Distinct reads. Exophase refreshes a game's counts on its own schedule,
  hours apart, so consecutive identical reads carry no new information.
  The program collapses them to distinct states first.
* Players tracked. It fits log N against log(hours since release + c),
  grid-searching the time scale c (1 to 48 hours) for the best fit, and
  extrapolates to the target anchored at the last state, the slope damped
  a little because launch growth saturates. The time scale matters: a
  launch curve is a power law in its own clock, not in hours since
  midnight, and the wrong clock bends the fit and the forecast with it.
* Trophy counts. Players start at different times and earn a trophy
  after a random time of their own, so a trophy's holders are a
  convolution: count_i(t) = p_i * M(t; tau_i), where p_i is the share of
  players who ever earn it, tau_i is the mean time from starting the
  game to earning it, and M(t; tau) is the players who started long
  enough ago, weighted by an exponential time-to-earn with mean tau:
  M(t; tau) = integral of N'(s) * (1 - exp(-(t - s) / tau)) over s <= t.
  For each trophy the program grid-searches tau (0.5 to 256 hours) and
  fits p_i by least squares on log counts, with a small penalty pulling
  p_i toward a tier prior (platinum lowest, bronze highest) so a flat
  count series cannot be read as "everyone will earn it, just later".
  The forecast rate is p_i * M(target; tau_i) / N(target). This shape
  matters: while the player count grows fast, a late trophy's rate falls
  (dilution outruns progress); as growth slows, the rate recovers toward
  p_i. A straight trend line on the rate would extrapolate the fall
  forever and miss the recovery, so the program does not use one when
  the player count is visibly growing.
* Stale display. When the whole history is one flat state (launch night:
  no refresh yet), neither side can be fitted, so the program expects the
  pending refresh instead: it assumes the tracked-player count jumps
  about six-fold by the target (the house saw 638 -> 3,868 at the first
  refresh) and the trophy counts barely move, except that fresh players
  earn the earliest story trophies first (a decaying new-earns rate by
  story position, platinum last). The median rate is then
  (old count + expected new earns) / expected players.
* Spread. Quantiles are the median plus/minus normal z-scores times a
  sigma that combines a random-walk term (0.5 * sqrt(horizon in days)),
  the fit's residual, a term when the last distinct state is stale, one
  when the time-to-earn is poorly identified, and one for the
  player-count forecast itself on long horizons. A trophy every tracked
  player already holds (rate 100%) is pinned near 100%: it survived a
  six-fold dilution at 100% on launch night, so every new player earns
  it before syncing.

Everything is a deterministic function of the case file: no clock, no
randomness, no network. Standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Expected jump in players tracked when a stale display finally refreshes
# on launch night: the house saw 638 -> 3,868 at the first refresh, a
# 6.06-fold jump, so the prior centres there (a sweep over the practice
# reconstruction put the best jump at 6 with a flat optimum from 5 to 8).
REFRESH_JUMP = 6.0
# Share of the new players expected to hold a non-intro trophy in the
# flat-history case, decaying with story position (see story_position).
NEW_EARN_BASE = 0.009
NEW_EARN_DECAY = 4.0
# Spread in the flat-history case (logit units). A sweep over the practice
# reconstruction put the best spread near 0.7-0.9; 0.85 hedges the
# reconstruction's own uncertainty.
FLAT_SIGMA = 0.85
# Random-walk spread per square-root day (the house baseline's 0.5).
WALK_SIGMA = 0.5
# Candidate mean times-to-earn (hours) for the exponential kernel.
TAU_GRID = [0.5, 2, 4, 8, 16, 32, 64, 128, 256]
# Time scales (hours) tried when fitting the players-tracked curve.
TIME_SCALES = [1, 2, 4, 6, 12, 24, 48]
# Damping applied to the player-growth slope when extrapolating.
GROWTH_DAMP = 0.85
# Weight of the tier prior on a trophy's eventual share p (pseudo-counts).
PRIOR_WEIGHT = 2.0
# Tier prior for the eventual share of players who earn the trophy.
TIER_PRIOR = {"platinum": 0.05, "gold": 0.3, "silver": 0.4, "bronze": 0.6}
# Cap on the total spread (logit units) for the structural path.
SIGMA_CAP = 1.3


def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, like the house baseline's rounding."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def distinct_states(history):
    """Collapse consecutive reads with identical counts to one state each."""
    states = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if not states or states[-1][0] != key:
            states.append((key, read))
    return [read for _, read in states]


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, residual_std, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n


def story_position(trophy, count):
    """Position in the game's progression: 0 for the first trophy after the
    intro one, rising through the story; the platinum sits last."""
    if trophy["tier"] == "platinum":
        return float(count - 1)
    return float(trophy["id"] - 2)


def prior_tau(trophy, count):
    """Prior mean time-to-earn in hours, from the trophy's position."""
    return 6.0 + story_position(trophy, count) * 2.5


def fit_players_curve(hours, counts):
    """Fit log N = a + b*log(h + c) over the distinct states, trying each
    time scale c. Returns (n_at, n_prime, b, c): n_at(h) is the fitted
    players tracked (damped extrapolation past the last state), n_prime(h)
    its derivative per hour."""
    best = None
    for c in TIME_SCALES:
        xs = [math.log(h + c) for h in hours]
        ys = [math.log(max(1, n)) for n in counts]
        a, b, resid, _ = fit_line(xs, ys)
        b = max(0.0, min(3.0, b))
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, a, b, c)
    _, a, b, c = best
    h_last = max(hours)

    def n_fit(h):
        return math.exp(a + b * math.log(max(h + c, 0.05)))

    def n_at(h):
        if h <= h_last:
            return n_fit(h)
        anchor = n_fit(h_last)
        return anchor * ((h + c) / (h_last + c)) ** (b * GROWTH_DAMP)

    def n_prime(h):
        if h <= h_last:
            return n_fit(h) * b / max(h + c, 0.05)
        return n_at(h) * (b * GROWTH_DAMP) / max(h + c, 0.05)

    return n_at, n_prime, b, c


def kernel_masses(n_prime, hours, h_target):
    """For each tau, M(h; tau) = integral of N'(s)*(1-exp(-(h-s)/tau)) ds
    over s from -1 to h, on an hourly grid. Returns {tau: ([M(h_j)], M(h_target))}.
    M does not depend on the trophy, so it is computed once per case."""
    grid = [float(k) for k in range(-1, int(math.ceil(h_target)) + 1)]
    primes = [max(0.0, n_prime(s)) for s in grid]
    out = {}
    for tau in TAU_GRID:
        # Cumulative kernel mass at each grid hour.
        masses = []
        for h in grid:
            total = 0.0
            for s, pr in zip(grid, primes):
                if s > h:
                    break
                total += pr * (1.0 - math.exp(-(h - s) / tau))
            masses.append(total)
        def interp(h):
            if h <= grid[0]:
                return masses[0]
            if h >= grid[-1]:
                return masses[-1]
            lo = int(math.floor(h)) + 1  # grid[k] == k - 1
            frac = h - grid[lo]
            return masses[lo] * (1.0 - frac) + masses[lo + 1] * frac
        out[tau] = ([interp(h) for h in hours], interp(h_target))
    return out


def trend_forecast(series, x_target, horizon_days, stale_hours):
    """Fallback median/sigma: a shrunk logit trend line. Used only when the
    player count is flat, so there is no dilution dynamics to model."""
    xs = [x for x, _ in series]
    ys = [y for _, y in series]
    _, slope, resid, _ = fit_line(xs, ys)
    shrink = len(xs) / (len(xs) + 3.0)
    slope = max(-2.5, min(2.5, slope * shrink))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    median_logit = min(my + slope * (x_target - mx), logit(99.0))
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    fit_sigma = min(0.7, max(0.25, resid)) if len(xs) >= 3 else 0.8
    stale = 0.5 if stale_hours > 2.0 else 0.0
    sigma = min(SIGMA_CAP, max(0.5, math.sqrt(walk * walk + fit_sigma ** 2 + stale ** 2)))
    return median_logit, sigma


def structural_forecast(trophy, counts, masses, n_target, n_trophies,
                        horizon_days, stale_hours):
    """Fit count_i(t) = p * M(t; tau) and forecast the rate at the target.

    counts: [(state index, hour, holders)] over the distinct states
    (holders > 0 only).
    masses: {tau: ([M(h_j)], M(h_target))} from kernel_masses.
    Returns (median_logit, sigma), or None when the fit is not usable.
    """
    if len(counts) < 3:
        return None
    prior = prior_tau(trophy, n_trophies)
    log_p_prior = math.log(TIER_PRIOR.get(trophy["tier"], 0.6))

    def fit_for(tau):
        ms, _ = masses[tau]
        diffs = []
        for idx, h, c in counts:
            mval = ms[idx]
            if mval <= 0:
                continue
            diffs.append(math.log(c) - math.log(mval))
        if len(diffs) < 3:
            return None
        n = len(diffs)
        log_p = (sum(diffs) + PRIOR_WEIGHT * log_p_prior) / (n + PRIOR_WEIGHT)
        log_p = min(0.0, log_p)
        sse = sum((d - log_p) ** 2 for d in diffs)
        sse += PRIOR_WEIGHT * (log_p - log_p_prior) ** 2
        return sse, tau, log_p, n

    scored = [f for f in (fit_for(tau) for tau in TAU_GRID) if f is not None]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0])
    best_sse, best_tau, best_log_p, n = scored[0]
    # Near-tie: prefer the tau nearer the prior.
    for sse, tau, log_p, _ in scored[1:]:
        if sse <= best_sse * 1.05 + 0.05 and abs(tau - prior) < abs(best_tau - prior):
            best_sse, best_tau, best_log_p = sse, tau, log_p
    resid = math.sqrt(best_sse / max(1, n - 2))
    identified = len(scored) < 2 or scored[1][0] >= best_sse * 1.10 + 0.05

    p = math.exp(best_log_p)
    m_target = masses[best_tau][1]
    median_rate = p * m_target / n_target * 100.0
    median_logit = min(logit(median_rate), logit(99.0))
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    fit_sigma = min(0.7, max(0.25, resid))
    stale = 0.5 if stale_hours > 2.0 else 0.0
    tau_pen = 0.0 if identified else 0.25
    sigma = math.sqrt(walk * walk + fit_sigma ** 2 + stale ** 2 + tau_pen ** 2)
    sigma = min(SIGMA_CAP, max(0.5, sigma))
    return median_logit, sigma


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    if not case["history"]:
        raise ValueError("case has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    h_target = hours_between(release, target)
    x_target = math.log1p(max(0.0, h_target))
    horizon_days = case["horizon_minutes"] / 1440.0

    states = distinct_states(case["history"])
    n_states = len(states)
    last = states[-1]
    n_last = last["players_tracked"]
    h_last = hours_between(release, parse_time(last["at"]))
    # How long the last distinct state had already been showing at the
    # checkpoint: a long-stale display may refresh before the target.
    stale_hours = hours_between(parse_time(states[-1]["at"]), checkpoint)

    hours_n = [hours_between(release, parse_time(s["at"])) for s in states]
    n_at, n_prime, b_n, _ = fit_players_curve(
        hours_n, [s["players_tracked"] for s in states])
    n_target = max(n_at(h_target), n_last)

    masses = None
    if b_n > 0.05 and n_states > 1:
        masses = kernel_masses(n_prime, hours_n, h_target)

    trophies = case["trophies"]
    tier_prior = {"platinum": 0.3, "gold": 0.4, "silver": 0.5, "bronze": 0.8}
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        last_count = last_rate * n_last / 100.0
        if last_rate >= 99.99:
            # A trophy every tracked player holds stays held.
            rows.append({"id": trophy["id"],
                         "q": [99.5, 99.9, 99.99, 99.99, 99.99]})
            continue

        series = []   # (x, logit rate) for the trend fallback
        counts = []   # (state index, hour, holders) for the structural fit
        for si, read in enumerate(states):
            rate = read["rates"][i]
            if rate is None:
                continue
            h = hours_between(release, parse_time(read["at"]))
            series.append((math.log1p(max(0.0, h)), logit(rate)))
            c = rate * read["players_tracked"] / 100.0
            if c >= 0.5:
                counts.append((si, h, c))

        if n_states <= 1 or len(series) < 2:
            # Flat history: the display is stale and a refresh is pending.
            pos = story_position(trophy, len(trophies))
            new_players = n_last * (REFRESH_JUMP - 1.0)
            extra = NEW_EARN_BASE * math.exp(-pos / NEW_EARN_DECAY) * new_players
            median_rate = (last_count + extra) / (n_last * REFRESH_JUMP) * 100.0
            median_logit, sigma = logit(median_rate), FLAT_SIGMA
        elif masses is not None:
            got = structural_forecast(trophy, counts, masses, n_target,
                                      len(trophies), horizon_days, stale_hours)
            if got is None:
                median_logit, sigma = trend_forecast(series, x_target,
                                                     horizon_days, stale_hours)
            else:
                median_logit, sigma = got
                if last_count < 5:
                    sigma = min(SIGMA_CAP, sigma + 0.15)
        elif len(series) >= 2:
            median_logit, sigma = trend_forecast(series, x_target,
                                                 horizon_days, stale_hours)
        else:
            median_logit = logit(tier_prior.get(trophy["tier"], 0.8))
            sigma = 1.5
        qs = [round4(clamp_rate(inv_logit(median_logit + z * sigma))) for z in Z]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
