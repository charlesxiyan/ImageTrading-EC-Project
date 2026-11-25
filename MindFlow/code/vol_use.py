"""Step 6c: what the volatility forecast is worth when used.

Three uses of the same scores (the Step 6b predictions, identical rows for every model):

1. **Forecast quality (IC, ICIR).** At every decision time, the Spearman correlation across stocks between a model's
   score and what happened next. Mean IC, its standard deviation, ICIR = mean / std, t = ICIR * sqrt(times).
2. **Volatility as a return factor (long-short).** Each decision time, stocks are sorted on predicted volatility;
   long the lowest fifth, short the highest fifth, equal weights, held for the label horizon. This tests whether
   predicted volatility says anything about the *sign* of returns (the low-volatility anomaly); Steps 2a/2b found no
   direction signal, so it is expected to be near zero. Costs are charged on full turnover every period.
3. **Volatility for risk control (sizing).** A long-only book over the same stocks, sized four ways: equal weight,
   inverse of each stock's training-median |return| (no forecast), inverse predicted volatility with gross exposure
   held at 1 (cross-sectional risk parity), and inverse predicted volatility with a fixed risk budget (gross exposure
   moves with predicted volatility: volatility targeting). A better forecast should give steadier realized risk and
   smaller drawdowns for the same budget.

**From a probability to a volatility level.** A model's score is p = P(|r| > m), where m is the stock's training
median |r| over the horizon. If |r| is exponential with mean s (a Laplace return), P(|r| > m) = exp(-m / s), so
s = m / (-ln p). This is monotone in p, uses no future data, and puts every stock on one scale.

Returns are the unadjusted horizon returns of the label (`forward_return`), so they match the labels exactly. Sharpe
ratios are annualised with the number of decision times per year actually traded; the book holds only during the
label horizon after each decision time, so it is not a full-day strategy, and overnight returns are not included.
"""
import numpy as np
import pandas as pd

__version__ = "2026-11-25.1"

P_CLIP = (0.02, 0.98)
DAYS_PER_YEAR = 252


# ----------------------------------------------------------------------------- inputs

def model_columns(pred):
    """Score columns of a predictions frame, without the constant prior."""
    return [c for c in pred.columns if c.startswith("score_") and c != "score_training_prior"]


def implied_vol(p, median):
    """Expected |return| implied by p = P(|r| > median) under an exponential |r|."""
    p = np.clip(np.asarray(p, np.float64), *P_CLIP)
    return np.asarray(median, np.float64) / -np.log(p)


def attach(pred, median):
    """Add the stock's training median |r| (`m`), the realised |r| scaled by it, and the time key."""
    out = pred.copy()
    out["m"] = out.security_index.map(median).astype(np.float64)
    out = out[out.m > 0].reset_index(drop=True)
    out["abs_r"] = out.forward_return.abs()
    out["abs_r_scaled"] = out.abs_r / out.m
    return out


# ----------------------------------------------------------------------------- 1. IC

def _spearman_by_time(frame, x, y, min_names=20):
    g = frame[["end_index", x, y]].copy()
    g["rx"] = g.groupby("end_index")[x].rank()
    g["ry"] = g.groupby("end_index")[y].rank()
    n = g.groupby("end_index").size()
    ic = g.groupby("end_index")[["rx", "ry"]].corr().xs("rx", level=1)["ry"]
    return ic[n.reindex(ic.index) >= min_names].dropna()


def ic_summary(series):
    s = pd.Series(series).dropna()
    mean, std = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    icir = mean / std if std > 0 else float("nan")
    return dict(mean_ic=mean, std_ic=std, icir=icir, t_stat=icir * np.sqrt(len(s)), share_positive=float((s > 0).mean()),
                times=len(s))


def ic_table(frame, models):
    """Per model: IC of the score vs |r| / own median (what the model was trained on), IC of the implied volatility
    vs raw |r| (the level, which also carries each stock's usual volatility), and IC vs the signed return."""
    rows, series = [], {}
    for col in models:
        f = frame.assign(_vol=implied_vol(frame[col], frame.m))
        for target, x, y in (("|r| / own median", col, "abs_r_scaled"), ("|r| (level)", "_vol", "abs_r"),
                             ("signed return", "_vol", "forward_return")):
            s = _spearman_by_time(f, x, y)
            series[(col, target)] = s
            rows.append(dict(model=col, target=target, **ic_summary(s)))
    base = _spearman_by_time(frame, "m", "abs_r")             # no forecast: the stock's usual volatility only
    series[("score_static_median", "|r| (level)")] = base
    rows.append(dict(model="score_static_median", target="|r| (level)", **ic_summary(base)))
    return pd.DataFrame(rows), series


# ----------------------------------------------------------------------------- performance helpers

def periods_per_year(frame):
    times = frame.groupby("decision_date").end_index.nunique()
    return float(times.mean() * DAYS_PER_YEAR)


def performance(r, ppy):
    """Annualised mean, volatility, Sharpe, max drawdown (on cumulative simple sum), hit rate, from period returns."""
    r = pd.Series(r).dropna()
    mean, std = float(r.mean()), float(r.std(ddof=1))
    cum = r.cumsum()
    dd = cum - cum.cummax()
    return dict(ann_return=mean * ppy, ann_vol=std * np.sqrt(ppy), sharpe=mean / std * np.sqrt(ppy) if std > 0 else np.nan,
                max_drawdown=float(dd.min()), hit_rate=float((r > 0).mean()), periods=len(r))


def daily_vol_stability(r, frame):
    """Std / mean across days of each day's realised volatility of the period returns (lower = steadier risk)."""
    day = frame.drop_duplicates("end_index").set_index("end_index").decision_date
    d = pd.Series(r).groupby(day.reindex(pd.Series(r).index).to_numpy()).std(ddof=0)
    d = d[d > 0]
    return float(d.std() / d.mean()) if len(d) > 2 else np.nan


# ----------------------------------------------------------------------------- 2. long-short on predicted volatility

def long_short(frame, col, q=5, min_names=20):
    """Period returns of long the lowest-predicted-volatility fifth, short the highest fifth (equal weights)."""
    f = frame[["end_index", "forward_return"]].copy()
    f["v"] = implied_vol(frame[col], frame.m) if col != "score_static_median" else frame.m
    n = f.groupby("end_index").v.transform("size")
    f = f[n >= min_names]
    f["bucket"] = f.groupby("end_index").v.transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    lo = f[f.bucket == 0].groupby("end_index").forward_return.mean()
    hi = f[f.bucket == q - 1].groupby("end_index").forward_return.mean()
    return (lo - hi).dropna()


def long_short_table(frame, models, costs_bp=(0.0, 1.0, 2.0)):
    """Sharpe and drawdown of the low-minus-high predicted-volatility portfolio, before and after costs.
    Cost per period: each of the two legs is opened and closed (4 one-way trades of the leg's notional)."""
    ppy, rows, curves = periods_per_year(frame), [], {}
    for col in list(models) + ["score_static_median"]:
        r = long_short(frame, col)
        curves[col] = r
        for c in costs_bp:
            rows.append(dict(model=col, cost_bp=c, **performance(r - 4 * c * 1e-4, ppy)))
    return pd.DataFrame(rows), curves


# ----------------------------------------------------------------------------- 3. sizing a long-only book

def sized_returns(frame, col, scheme):
    """Period return and gross exposure of a long-only book over the stocks of each decision time."""
    f = frame[["end_index", "forward_return", "m"]].copy()
    if scheme == "equal weight":
        w = pd.Series(1.0, index=f.index)
    elif scheme == "inverse median vol (no forecast)":
        w = 1.0 / f.m
    else:
        w = 1.0 / implied_vol(frame[col], frame.m)
        w = pd.Series(w, index=f.index)
    f["w"] = w
    n = f.groupby("end_index").w.transform("size")
    if scheme == "volatility target":
        f["w"] = f.w / n                                       # fixed risk budget per name: exposure floats
    else:
        f["w"] = f.w / f.groupby("end_index").w.transform("sum")   # gross exposure 1
    ret = (f.w * f.forward_return).groupby(f.end_index).sum()
    gross = f.groupby("end_index").w.sum()
    return ret, gross


def sizing_table(frame, models):
    ppy, rows, curves = periods_per_year(frame), [], {}
    plans = [("equal weight", None), ("inverse median vol (no forecast)", None)]
    plans += [(s, col) for col in models for s in ("risk parity", "volatility target")]
    for scheme, col in plans:
        ret, gross = sized_returns(frame, col, scheme)
        if scheme == "volatility target":                     # scale so the average gross exposure is 1
            k = 1.0 / gross.mean()
            ret, gross = ret * k, gross * k
        name = scheme if col is None else f"{scheme} | {col}"
        curves[name] = ret
        rows.append(dict(book=name, model=col or "-", scheme=scheme, **performance(ret, ppy),
                         vol_of_daily_vol=daily_vol_stability(ret, frame), mean_gross=float(gross.mean()),
                         min_gross=float(gross.min()), max_gross=float(gross.max())))
    return pd.DataFrame(rows), curves


def risk_forecast_ic(frame, models):
    """Time-series rank correlation between the book's predicted risk (mean implied volatility over its stocks)
    and the realised |return| of the equal-weight book: does the model know when the whole market will move?"""
    ew = frame.groupby("end_index").forward_return.mean().abs()
    rows = []
    for col in list(models) + ["score_static_median"]:
        v = frame.m if col == "score_static_median" else pd.Series(implied_vol(frame[col], frame.m), index=frame.index)
        pred = v.groupby(frame.end_index).mean()
        rho = pred.rank().corr(ew.reindex(pred.index).rank())
        rows.append(dict(model=col, spearman_pred_risk_vs_abs_book_return=float(rho), times=len(pred)))
    return pd.DataFrame(rows)
