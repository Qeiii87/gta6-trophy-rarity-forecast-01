# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Qeiii87/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Qeiii87/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of
Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as
five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. `main.py` (Python 3.13,
standard library only) reads a case at `/input/<id>.in` and writes
`/output/<id>.out`. It is deterministic: two runs on the same case write the
same forecast.

## How it works

An earn rate is a ratio — holders over players tracked — and the two sides
move on different clocks. New players dilute every rate (they hold almost
nothing yet); existing players push rates up as they progress. So the
program models the two sides separately, in counts:

- **Players tracked.** It fits log N against log(hours since release + c),
  grid-searching the time scale c, and extrapolates to the target anchored
  at the last state, the slope damped because launch growth saturates.
- **Trophy counts.** Players start at different times and earn a trophy
  after a random time of their own, so a trophy's holders are a
  convolution: count(t) = p · M(t; tau), where p is the share of players
  who ever earn it, tau is the mean time from starting the game to earning
  it, and M weights past player starts by an exponential time-to-earn.
  Each trophy's tau is fitted by grid search and p by least squares on log
  counts, with a small tier prior (platinum lowest, bronze highest).
- **Stale display.** When the whole history is one flat state (launch
  night: no refresh yet), the program expects the pending refresh instead:
  it assumes the tracked-player count jumps about six-fold by the target
  (the house saw 638 → 3,868 at the first refresh) and the trophy counts
  barely move, except that fresh players earn the earliest story trophies
  first. A trophy every tracked player already holds is pinned near 100%.
- **Spread.** Quantiles are the median plus/minus normal z-scores times a
  sigma combining a random-walk term, the fit's residual, staleness, lag
  identification, and player-count forecast uncertainty.

Local validation (pinball loss on the logit scale, lower is better): on the
kit's practice cases the program scores about 0.19–0.23 against a
reconstructed launch-night refresh, versus about 1.51–1.69 for the house
persistence baseline. On synthetic final-like cases (two days of reads,
4.5-day horizon, hard-lag, distributed-lag and saturating-growth ground
truths) it scores 0.30–0.71 versus 0.46–1.74 for the baseline.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6yb92059&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
