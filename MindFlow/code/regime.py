"""Step 6a: does market-regime information raise the volatility AUC of HGB?

The vol label compares |r(t -> t+H)| with each stock's FIXED training-period median |r|. In a calm
year most labels are 0, in a turbulent year most are 1, so the level of volatility (the regime) is
part of what the label measures. The 38 hand + SPY features only look back one session (78 bars).
This module adds slower regime features, all known before the decision bar:

  VIX (Cboe, daily close, 16:15 ET)  previous session's close only (never the decision day's own)
      vix_log        log VIX close of the previous session
      vix_chg1       log change of that close vs the session before
      vix_chg5       log change over 5 sessions
      vix_rel20      log close minus log of its 20-session mean
  multi-day realized volatility, from 5-minute bars of COMPLETED sessions before the decision session
      rv5d, rv20d               stock: sqrt(mean over the last 5 / 20 sessions of the session's
                                sum of squared intraday 5-minute log returns)
      rv_ratio                  log(rv5d / rv20d)
      spy_rv5d, spy_rv20d       the same for SPY

Only intraday returns (bar to bar inside one session) enter the realized volatility, so no overnight
gap and no ex-date (ex-dates take effect at the open) is ever inside a return, and only ratios of two
bars of the same session are used (price-adjustment policy). A session counts when at least
`min_bars` of its 77 intraday returns are observed.

VIX comes from Cboe's public daily history (no WRDS needed); the file is cached on Drive on first use.
"""
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import feature_select as fs

__version__ = "2026-11-25.1"   # + VIXY intraday features

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
BARS = 78
VIX_COLS = ["vix_log", "vix_chg1", "vix_chg5", "vix_rel20"]
RV_COLS = ["rv5d", "rv20d", "rv_ratio"]
SPY_RV_COLS = ["spy_rv5d", "spy_rv20d"]
VIXY_COLS = [f"vixy_r{k}" for k in (1, 2, 3, 6, 12)] + [f"vixy_cum{w}" for w in (6, 12, 39, 77)] + \
    ["vixy_realized_vol", "vixy_mean_abs_r12"]
GROUPS = {"VIX": VIX_COLS, "multi-day RV": RV_COLS + SPY_RV_COLS}


def load_vixy(prepared):
    """log close of the VIXY add-on panel (tools/build_panel_bundle.py), or None if it is not there."""
    path = Path(prepared) / "vixy_features.npy"
    if not path.exists():
        return None
    return np.log(np.load(path)[:, 3].astype(np.float64))


def vixy_features(vixy_log_close, frame):
    """VIXY over the same 78-bar input window as the SPY features (bars up to the decision bar only).
    Stored prices are on one basis, so these bar-to-bar ratios equal the point-in-time ratios."""
    t = frame.end_index.to_numpy(np.int64)
    lc = vixy_log_close
    r = np.stack([lc[t - k + 1] - lc[t - k] for k in range(1, 78)])
    cols = {f"vixy_r{k}": r[k - 1] for k in (1, 2, 3, 6, 12)}
    cols.update({f"vixy_cum{w}": lc[t] - lc[t - w] for w in (6, 12, 39, 77)})
    cols["vixy_realized_vol"] = r.std(axis=0)
    cols["vixy_mean_abs_r12"] = np.abs(r[:12]).mean(axis=0)
    return pd.DataFrame(cols)


def load_vix(cache, url=VIX_URL):
    """Daily VIX closes (index: date string YYYY-MM-DD). Read from `cache` if present, else download it."""
    cache = Path(cache)
    if not cache.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research notebook)"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except OSError as err:
            raise RuntimeError(f"could not download {url} ({err}). Put a daily VIX file at {cache} instead: Cboe's "
                               "VIX_History.csv, or a WRDS cboe.cboe export with columns date, vix") from err
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    v = pd.read_csv(cache)
    v.columns = [c.strip().upper() for c in v.columns]
    v = v.rename(columns={"VIX": "CLOSE"}) if "CLOSE" not in v and "VIX" in v else v      # WRDS cboe.cboe export
    v["DATE"] = pd.to_datetime(v["DATE"]).dt.strftime("%Y-%m-%d")
    v = v.dropna(subset=["CLOSE"]).drop_duplicates("DATE").set_index("DATE").sort_index()
    return v["CLOSE"].astype(float)


def vix_table(vix, sessions):
    """VIX features per trading session, each from closes of sessions strictly before it."""
    lv = np.log(vix)
    f = pd.DataFrame({"vix_log": lv, "vix_chg1": lv.diff(1), "vix_chg5": lv.diff(5),
                      "vix_rel20": lv - np.log(vix.rolling(20).mean())})
    # the feature used on day d is the row of the last VIX date < d
    sessions = pd.Index(sorted(set(sessions)))
    pos = np.searchsorted(f.index.to_numpy(), sessions.to_numpy(), side="left") - 1
    out = pd.DataFrame(np.nan, index=sessions, columns=VIX_COLS)
    ok = pos >= 0
    out.loc[ok] = f.iloc[pos[ok]].to_numpy()
    return out


def _session_rv2(log_close, bad, min_bars):
    """(n_sessions,) sum of squared intraday 5-minute log returns per session (NaN if too few bars)."""
    lc = log_close.reshape(-1, BARS)
    b = bad.reshape(-1, BARS)
    r = np.diff(lc, axis=1)                                         # 77 intraday returns per session
    ok = ~(b[:, 1:] | b[:, :-1]) & np.isfinite(r)
    n = ok.sum(1)
    s = np.where(ok, r, 0.0) ** 2
    rv2 = s.sum(1) * (BARS - 1) / np.maximum(n, 1)                  # scale up for missing bars
    return np.where(n >= min_bars, rv2, np.nan)


def _trailing(rv2, w):
    """sqrt of the mean over the previous w sessions (excluding the current one), NaN-aware."""
    x = pd.DataFrame(rv2.T) if rv2.ndim == 2 else pd.DataFrame(rv2[:, None])
    m = x.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
    out = np.sqrt(m.to_numpy())
    return out.T if rv2.ndim == 2 else out[:, 0]


def rv_tables(store, min_bars=60):
    """Per (stock, session) and per session (SPY) trailing realized volatility arrays."""
    T = store.T
    assert T % BARS == 0, "slots must be whole sessions of 78 bars"
    valid_stock = np.diff(store.bad_cumsum, axis=1) == 0           # (S, T) observed slots
    rv2 = np.stack([_session_rv2(store.log_close[s], ~valid_stock[s], min_bars) for s in range(store.log_close.shape[0])])
    out = {"rv5d": _trailing(rv2, 5), "rv20d": _trailing(rv2, 20)}
    if store.has_market:
        mvalid = np.diff(store.market_bad_cumsum) == 0
        m2 = _session_rv2(store.market_log_close, ~mvalid, min_bars)
        out["spy_rv5d"], out["spy_rv20d"] = _trailing(m2, 5), _trailing(m2, 20)
    return out


def regime_features(store, frame, rv, vixf):
    s = frame.security_index.to_numpy(np.int64)
    d = frame.end_index.to_numpy(np.int64) // BARS                  # session index of the decision bar
    cols = {"rv5d": rv["rv5d"][s, d], "rv20d": rv["rv20d"][s, d]}
    cols["rv_ratio"] = np.log(cols["rv5d"] / cols["rv20d"])
    if "spy_rv5d" in rv:
        cols["spy_rv5d"], cols["spy_rv20d"] = rv["spy_rv5d"][d], rv["spy_rv20d"][d]
    out = pd.DataFrame(cols)
    v = vixf.reindex(frame.decision_date.to_numpy())
    for c in VIX_COLS:
        out[c] = v[c].to_numpy()
    return out


def build(store, cell, settings, rv, vixf, vixy=None):
    """Train sample (2016-2022) and validation (2023) frames with 38 + regime features."""
    from chartcnn import build_split
    train, median = build_split(store, "train", cell["target"], cell["horizon"], cell["stride"])
    valid, _ = build_split(store, "validation", cell["target"], cell["horizon"], cell["stride"], median)
    train = train.sample(min(settings["max_train"], len(train)), random_state=settings["seed"]).reset_index(drop=True)
    X = {}
    for k, f in (("train", train), ("valid", valid)):
        parts = [fs.features(store, f), regime_features(store, f, rv, vixf)]
        if vixy is not None:
            parts.append(vixy_features(vixy, f))
        X[k] = pd.concat(parts, axis=1)
    return dict(train=train, valid=valid), X


def compare(d, X, subsets, seed, n_boot=300, reference="38 hand + SPY"):
    """Fit HGB on each subset (same rows), score 2023, paired day bootstrap against `reference`."""
    ytr, yva = d["train"].label.to_numpy(), d["valid"].label.to_numpy()
    scores, rows, models = {}, [], {}
    for name, cols in subsets.items():
        t0 = time.perf_counter()
        m = fs.hgb(seed).fit(X["train"][cols], ytr)
        scores[name] = m.predict_proba(X["valid"][cols])[:, 1]
        models[name] = m
        rows.append(dict(model=name, n_features=len(cols), tree_nodes=fs.tree_size(m), auc=fs._auc(yva, scores[name]),
                         minutes=(time.perf_counter() - t0) / 60))
    boot = fs.day_bootstrap(d["valid"], scores, n_boot, seed)
    out = pd.DataFrame(rows).set_index("model")
    out["auc_low"], out["auc_high"] = boot.quantile(0.025), boot.quantile(0.975)
    diff = boot.sub(boot[reference], axis=0)
    out["diff_vs_ref"] = diff.mean()
    out["diff_low"], out["diff_high"] = diff.quantile(0.025), diff.quantile(0.975)
    out["verdict"] = np.where(out.diff_low > 0, "better", np.where(out.diff_high < 0, "worse", "tied"))
    out.loc[reference, "verdict"] = "reference"
    return out.reset_index(), scores, models


def by_regime(valid, scores, vix_level, names, n_bins=3):
    """AUC of each model within VIX terciles of the validation days (by the previous close)."""
    q = pd.Series(pd.qcut(vix_level, n_bins, labels=False, duplicates="drop"))
    rows = []
    y = valid.label.to_numpy()
    for g in sorted(q.dropna().unique()):
        m = (q == g).to_numpy()
        r = dict(regime=f"VIX tercile {int(g) + 1}", rows=int(m.sum()), positive_share=float(y[m].mean()) if m.any() else np.nan,
                 vix_mean=float(np.exp(vix_level[m]).mean()) if m.any() else np.nan)
        for n in names:
            r[n] = fs._auc(y[m], scores[n][m])
        rows.append(r)
    return pd.DataFrame(rows)
