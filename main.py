"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the 55 GTA VI PS5 trophies, the Exophase earn rate
(percent of tracked players holding the trophy) at the case's target time,
as five quantiles (5/25/50/75/95%).

Method -- earners / players decomposition in log time:
  * A trophy's rate is earners / players_tracked.  Both grow, and Exophase
    refreshes its page irregularly, so most reads repeat the previous page
    exactly (stale reads).  Only reads whose players_tracked or rates
    changed (fresh reads) carry new information; stale reads are dropped.
  * Time is x = ln(1 + hours since the game's release).  On fresh reads we
    fit, by least squares, ln(players) ~ x and, per trophy,
    ln(earners) ~ x (earners = rate * players / 100; a read showing 0% is
    censored, so it is left out of the earners fit, and a 100% trophy's
    earners are its players by definition).
  * Each fitted slope is shrunk toward a prior: players grow about
    (1+t)^1.35 on Exophase; a trophy's earners grow faster the further
    players have got with it, so the prior slope rises with the trophy's
    current rate, with explicit priors for the platinum / 100%-completion
    trophies (nobody finishes those in week one) and for the chapter
    trophies (Chapter 1 is reached in hours, Chapter 4 in days).
  * Extrapolation starts at the last fresh read (so a stale tail before
    the checkpoint is bridged the same way as the horizon) and the slope
    is damped the further ahead we look, because earners growth
    saturates for early trophies and players growth slows.
  * Quantile spread is in logit space and grows with the horizon, with
    the fit's residual, and with how few earners a trophy has so far.

Everything is deterministic, stdlib only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)  # normal quantiles, levels .05..95
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Prior for the players-tracked growth slope, ln P ~ x.
PRIOR_P_SLOPE = 1.35
# Prior for a trophy's earners slope from its current rate r (percent):
#   b0 = clip(1.15 + 0.55 * log10(max(r, 0.01)), 0, 2).  Calibrated on the
#   public Exophase page: launch night (638 players) to 2026-11-21 ~03:30Z
#   (~110k players), where earners slopes ran from ~0 for the rarest
#   trophies to ~2.0 for Chapter 1.
PRIOR_E_A = 1.15
PRIOR_E_B = 0.55
# Explicit priors (description / tier), used instead of the rate prior:
PRIOR_E_PLATINUM = 0.05   # platinum and 100%-completion: ~nobody in week 1
CHAPTER_E_SLOPE = {1: 1.95, 2: 1.60, 3: 0.65, 4: 0.40, 5: 0.40}  # 5 = final
# Slope damping: effective increment = slope * (1 - exp(-LAMBDA*dx)) / LAMBDA
# LAMBDA = 0: no damping.  The launch->2026-11-21 calibration showed the
# power-law growth holding undamped over ~52 h (backtest A: 1.49 at
# LAMBDA=0 vs 1.79 at 0.25), and the final's dx is only ~1.0, so damping
# moves the final medians by ~5% either way; the undamped fit wins.
LAMBDA = 0.0
# Fitted slopes are clipped: no trophy's earners or the player count can
# credibly grow faster than (1+t)^2 over the forecast window.
SLOPE_E_LO, SLOPE_E_HI = -0.5, 2.0
SLOPE_P_LO, SLOPE_P_HI = 0.0, 2.0
# Shrinkage strengths (pseudo-counts of fresh reads).
SHRINK_P = 2.0
SHRINK_E = 4.0
# Spread (logit sd) components.
SPREAD_BASE = 0.12
SPREAD_HORIZON = 0.55   # times sqrt(horizon in days); wider won on both
                        # backtests (0.42 -> A 1.49, 0.55 -> 1.43, 0.70 ->
                        # 1.35) and 0.55 is the central choice, not the argmin
SPREAD_NOFIT = 0.30
SPREAD_STALE = 0.15     # added when the last fresh read is > 3 h old
SPREAD_LOWE = 0.30      # times max(0, log10(30 / earners_last))

RATE_LO = 0.01   # scorer clamps rates to [0.01, 99.99] before the logit
RATE_HI = 99.99
EARNER_FLOOR = 0.5  # half an earner, for a positive rate that rounds small


# ---------------------------------------------------------------- helpers

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    p = clamp(p, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        e = math.exp(-l)
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return clamp(p * 100.0, RATE_LO, RATE_HI)


def parse_time(s):
    """ISO 8601 UTC -> aware datetime; date alone -> midnight UTC."""
    if not s:
        return None
    s = s.strip()
    try:
        if len(s) == 10:  # a bare date
            return datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]),
                            tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, IndexError):
        return None


def ols(xs, ys, ws=None):
    """Least squares y ~ a + b*x.  Returns (a, b, resid_sd, n, x_span)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0, 0.0)
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    if n > 2 and sxx > 1e-12:
        rss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        sd = math.sqrt(rss / max(1, n - 2))
    else:
        sd = 0.0
    return (a, b, sd, n, max(xs) - min(xs))


def damp_factor(dx, lam):
    """(1 - exp(-lam*dx)) / (lam*dx): 1 at dx=0, decaying for large dx."""
    if dx <= 0:
        return 1.0
    if lam <= 0:
        return 1.0
    t = lam * dx
    return (1.0 - math.exp(-t)) / t

# ---------------------------------------------------------------- case work

def _chapter_of(trophy):
    """Progression depth from the description: Chapter N -> N, the final
    mission -> 5, anything else -> None."""
    desc = (trophy.get("description") or "").lower()
    if "final mission" in desc:
        return 5
    for n in (1, 2, 3, 4):
        if "chapter %d" % n in desc:
            return n
    return None


def _prior_e_slope(trophy, last_rate):
    """Prior earners slope for one trophy from its tier / description /
    current rate."""
    tier = (trophy.get("tier") or "").lower()
    desc = (trophy.get("description") or "").lower()
    if tier == "platinum" or "100% completion" in desc or "100 percent" in desc:
        return PRIOR_E_PLATINUM
    ch = _chapter_of(trophy)
    if ch is not None:
        return CHAPTER_E_SLOPE[ch]
    r = max(last_rate, 0.01) if last_rate is not None else 0.01
    return clamp(PRIOR_E_A + PRIOR_E_B * math.log10(r), 0.0, 2.0)


def _hours_since_release(t, release):
    dt = parse_time(t)
    if dt is None or release is None:
        return None
    return max(0.0, (dt - release).total_seconds() / 3600.0)


def forecast_case(case, params=None):
    """Forecast one case.  Returns the output dict {"trophies": [...]}."""
    p = dict(
        prior_p=PRIOR_P_SLOPE,
        prior_e_a=PRIOR_E_A,
        prior_e_b=PRIOR_E_B,
        lam=LAMBDA,
        shrink_p=SHRINK_P,
        shrink_e=SHRINK_E,
        spread_base=SPREAD_BASE,
        spread_horizon=SPREAD_HORIZON,
        spread_nofit=SPREAD_NOFIT,
        spread_stale=SPREAD_STALE,
        spread_lowe=SPREAD_LOWE,
        chapter=True,
    )
    if params:
        p.update(params)

    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = max(0.0, horizon_min / 1440.0)

    game = case.get("game") or {}
    release = parse_time(game.get("release_date"))
    if release is None and history:
        release = parse_time(history[0].get("at"))
    target = parse_time(case.get("target"))
    checkpoint = parse_time(case.get("at"))

    # ---- fresh reads: drop consecutive reads that repeat the page ----
    fresh = []
    prev_key = None
    for h in history:
        try:
            players = float(h.get("players_tracked") or 0)
        except (TypeError, ValueError):
            players = 0.0
        rates = h.get("rates") or []
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        if players <= 0:
            continue
        t = _hours_since_release(h.get("at"), release)
        if t is None:
            continue
        fresh.append({"x": math.log1p(t), "t": t, "at": h.get("at"),
                      "players": players, "rates": rates})
    if not fresh:
        # No usable history at all: fall back to the last read's rates, or 1%.
        last_rates = []
        if history:
            last_rates = history[-1].get("rates") or []
        rows = []
        for i, tr in enumerate(trophies):
            r = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 1.0
            rows.append(_row(tr["id"], r, p, horizon_days, None, None))
        return {"trophies": rows}

    last = fresh[-1]
    x_last = last["x"]
    # Target x: hours since release at the target instant.
    if target is not None and release is not None:
        x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    else:
        x_target = x_last + math.log1p(horizon_min / 60.0)
    dx = max(0.0, x_target - x_last)
    damp = damp_factor(dx, p["lam"])

    # ---- players fit: ln P ~ x on fresh reads ----
    xs = [f["x"] for f in fresh]
    ys_p = [math.log(f["players"]) for f in fresh]
    _a, d_fit, _sd, n_p, span_p = ols(xs, ys_p)
    if n_p >= 3 and span_p >= 0.3:
        w = n_p / (n_p + p["shrink_p"])
        d_hat = w * d_fit + (1.0 - w) * p["prior_p"]
    else:
        d_hat = p["prior_p"]
    d_hat = clamp(d_hat, SLOPE_P_LO, SLOPE_P_HI)
    ln_p_target = math.log(last["players"]) + d_hat * damp * dx
    p_target = math.exp(ln_p_target)

    # Stale tail: how long the page had sat unchanged before the checkpoint.
    stale_extra = 0.0
    if checkpoint is not None:
        last_fresh_dt = parse_time(last["at"])
        if last_fresh_dt is not None:
            gap_h = (checkpoint - last_fresh_dt).total_seconds() / 3600.0
            if gap_h > 3.0:
                stale_extra = p["spread_stale"]

    rows = []
    for i, tr in enumerate(trophies):
        r_last = None
        if i < len(last["rates"]):
            r_last = last["rates"][i]
        if r_last is None:
            # No rate in the last read: use the latest fresh read that has it.
            for f in reversed(fresh):
                if i < len(f["rates"]) and f["rates"][i] is not None:
                    r_last = f["rates"][i]
                    break
        if r_last is None:
            r_last = 1.0
        r_last = clamp(float(r_last), 0.0, 100.0)

        if r_last >= 99.99:
            # Everyone holds it (the opening fishing trophy): it stays 100%.
            rows.append({"id": tr["id"], "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue

        # ---- per-trophy earners fit: ln E ~ x, rate-0 reads censored ----
        exs, eys = [], []
        for f in fresh:
            if i >= len(f["rates"]):
                continue
            r = f["rates"][i]
            if r is None or r <= 0:
                continue
            e = max(float(r) * f["players"] / 100.0, EARNER_FLOOR)
            exs.append(f["x"])
            eys.append(math.log(e))
        _a, b_fit, resid, n_e, span_e = ols(exs, eys)
        prior = _prior_e_slope(tr, r_last) if p["chapter"] else \
            clamp(p["prior_e_a"] + p["prior_e_b"] * math.log10(max(r_last, 0.01)),
                  0.0, 2.0)
        if n_e >= 3 and span_e >= 0.3:
            w = n_e / (n_e + p["shrink_e"])
            b_hat = w * b_fit + (1.0 - w) * prior
            fit_var = resid * math.sqrt(1.0 + (dx / span_e) ** 2)
        else:
            b_hat = prior
            fit_var = p["spread_nofit"]
        b_hat = clamp(b_hat, SLOPE_E_LO, SLOPE_E_HI)
        fit_var = clamp(fit_var, 0.0, 0.7)

        e_last = max(r_last * last["players"] / 100.0, EARNER_FLOOR)
        ln_e_target = math.log(e_last) + b_hat * damp * dx
        e_target = math.exp(ln_e_target)
        rate = 100.0 * e_target / p_target if p_target > 0 else r_last
        rate = clamp(rate, RATE_LO, RATE_HI)

        low_e = p["spread_lowe"] * max(0.0, math.log10(30.0 / max(e_last, 1.0)))
        s2 = (p["spread_base"] ** 2
              + (p["spread_horizon"] * math.sqrt(horizon_days)) ** 2
              + fit_var ** 2 + low_e ** 2 + stale_extra ** 2)
        rows.append(_row(tr["id"], rate, p, horizon_days, math.sqrt(s2), None))
    return {"trophies": rows}


def _row(tid, centre, p, horizon_days, spread, _unused):
    """Five quantiles around a centre rate (percent) with logit spread."""
    if spread is None:
        spread = math.sqrt(p["spread_base"] ** 2
                           + (p["spread_horizon"] * math.sqrt(horizon_days)) ** 2)
    lc = logit_pct(centre)
    qs = []
    for z in Z:
        q = inv_logit_pct(lc + z * spread)
        qs.append(round(q, 4))
    # Enforce non-decreasing after rounding.
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": tid, "q": qs}

# ---------------------------------------------------------------- entry point

def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except FileNotFoundError:
        names = []
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"), sort_keys=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
