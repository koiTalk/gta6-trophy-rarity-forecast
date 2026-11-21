# GTA VI trophy rarity forecast

Entry for the make.fun bounty
**gta6-trophy-rarity-forecast-01** — forecast, for each of the 55 GTA VI
PlayStation 5 trophies, the Exophase earn rate (percent of tracked players
holding the trophy) at 2026-11-26T00:00:00Z, as five quantiles per trophy.

## Run

```
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`/output/<id>.out` (else `$MAKEFUN_OUTPUT_DIR`). Python 3.13, standard
library only, no network, deterministic.

## Method

A trophy's rate is **earners / players tracked**, and Exophase refreshes
its page irregularly, so most reads repeat the previous page exactly.
The forecaster therefore works on the decomposition, in log time
`x = ln(1 + hours since release)`:

1. **Fresh reads only.** Consecutive reads with identical
   players-tracked and rates are stale and carry no information; they
   are dropped before any fitting.
2. **Two least-squares fits** on the fresh reads: `ln(players) ~ x` and,
   per trophy, `ln(earners) ~ x` (earners = rate × players / 100; a
   read showing 0% is censored, so it is left out of the earners fit).
3. **Shrinkage to priors.** Each fitted slope is shrunk toward a prior:
   players grow about `(1+t)^1.35` on Exophase; a trophy's earners grow
   faster the further players have got with it, so the prior slope rises
   with the trophy's current rate, with explicit priors for the platinum
   and 100%-completion trophies (nobody finishes those in week one) and
   for the chapter trophies (Chapter 1 is reached in hours, Chapter 4
   in days). The priors were calibrated on the public Exophase page
   between launch night (638 players) and 2026-11-21 (~110k players).
4. **Extrapolation from the last fresh read**, with the slope damped
   the further ahead we look, because earners growth saturates for
   early trophies and players growth slows.
5. **Quantile spreads in logit space**, growing with the horizon, with
   the fit's residual, and with how few earners a trophy has so far —
   so the 4.5-day final gets a much wider band than a 90-minute
   practice case, as the horizon demands.

Edge cases: a trophy at 100% (everyone holds the opening fishing
trophy) is forecast to stay at 100%; a trophy with no earners yet is
forecast near the 0.01% floor; a case with no usable history falls back
to its last rates with a horizon-scaled spread; every output row is
exactly `{"id": n, "q": [q05, q25, q50, q75, q95]}`, non-decreasing,
within [0.01, 99.99].

## Validation

See `BUILD_LOG.md` in the build directory: practice runs on p00/p01
(byte-identical across runs), a 48-hour backtest from the flat
launch-night history to the live Exophase page of 2026-11-21 scored
with the contest's logit pinball loss against the persistence and
house-trend baselines, and edge-case checks (no history, all-null
rates, zero earners, missing trophy).
