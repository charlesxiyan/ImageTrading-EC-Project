"""Frame correlation analysis: how similar is image (stock s, time t) to image (s, t + k)?

Three views, from input to output:
  1. Image similarity  - pixel correlation, raw and with the shared bars aligned
  2. Label correlation - autocorrelation of the target at lag k (drives effective sample size)
  3. Model similarity  - correlation of CNN embeddings / predicted probabilities at lag k

Plus the cross-sectional view: how much label variance is common to all stocks at one time.
"""
import numpy as np
import pandas as pd
import torch

from .data import LOOKBACK, ChartDataset, build_split

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 39, 78, 156)     # in 5-minute bars; 78 = one session


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def sample_anchors(store, split, n, lags, seed=0):
    """Random (stock, time) anchors whose shifted windows t + k are complete for every lag."""
    f = store.windows(split)
    rng = np.random.default_rng(seed)
    s = f.security_index.to_numpy(np.int64); t = f.end_index.to_numpy(np.int64)
    k_max = max(lags)
    ok = t + k_max < store.T
    for k in lags:                                   # input window of the shifted frame complete
        tk = np.minimum(t + k, store.T - 1)
        ok &= store.complete(s, tk - LOOKBACK + 1, tk + 1)
    idx = rng.choice(np.flatnonzero(ok), size=min(n, ok.sum()), replace=False)
    return pd.DataFrame({"security_index": s[idx], "end_index": t[idx]})


def image_similarity(prepared, anchors, lags=DEFAULT_LAGS, models=None, device="cpu", batch=64):
    """Per-lag similarity between frame t and frame t+k.

    pixel_raw      : Pearson correlation of the two images pixel by pixel
    pixel_aligned  : same, after shifting so the bars both windows share line up
                     (measures content overlap; NaN once windows no longer overlap)
    <model>_embed  : correlation of the CNN's flattened feature maps
    <model>_prob   : correlation (across anchors) of predicted probabilities at t and t+k
    """
    models = models or {}
    frames = {0: anchors}
    for k in lags:
        frames[k] = anchors.assign(end_index=anchors.end_index + k)
    images = {k: np.stack([ChartDataset(prepared, f).image(i) for i in range(len(f))]).astype(np.float32)
              for k, f in frames.items()}

    def run(model, imgs):
        emb, prob = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(imgs[i:i + batch][:, None] / 255.0).to(device)
                e = model.embed(x)
                emb.append(e.cpu().numpy()); prob.append(model.head[-1](e).softmax(1)[:, 1].cpu().numpy())
        return np.concatenate(emb), np.concatenate(prob)

    outputs = {name: {k: run(m.eval().to(device), imgs) for k, imgs in images.items()} for name, m in models.items()}

    rows = []
    width = images[0].shape[2]
    for k in lags:
        a, b = images[0], images[k]
        shift = 3 * k
        row = dict(lag=k,
                   bars_shared=max(0, LOOKBACK - k),
                   pixel_raw=np.nanmean([_corr(a[i], b[i]) for i in range(len(a))]),
                   pixel_aligned=np.nanmean([_corr(a[i][:, shift:], b[i][:, :width - shift]) for i in range(len(a))])
                   if shift < width else np.nan)
        for name, out in outputs.items():
            (e0, p0), (ek, pk) = out[0], out[k]
            row[f"{name}_embed"] = np.nanmean([_corr(e0[i], ek[i]) for i in range(len(e0))])
            row[f"{name}_prob"] = _corr(p0, pk)
        rows.append(row)
    return pd.DataFrame(rows)


def label_autocorrelation(store, target, horizon, split="train", lags=DEFAULT_LAGS, median=None):
    """Correlation between label(s, t) and label(s, t + k), pooled over stocks (stride 1)."""
    f, median = build_split(store, split, target, horizon, stride=1, train_abs_median=median)
    y = f.set_index(["security_index", "end_index"]).label.astype(float)
    rows = []
    for k in lags:
        shifted = y.copy()
        shifted.index = pd.MultiIndex.from_arrays([y.index.get_level_values(0), y.index.get_level_values(1) - k])
        both = pd.concat([y, shifted], axis=1, join="inner").to_numpy()
        rows.append(dict(lag=k, label_autocorr=_corr(both[:, 0], both[:, 1]), pairs=len(both)))
    return pd.DataFrame(rows), f, median


def cross_sectional_share(frame):
    """ICC: share of label variance common to all stocks at the same timestamp.

    ICC = (Var(timestamp mean) - sigma^2 / m) / sigma^2, with m stocks per timestamp.
    """
    y = frame.label.astype(float)
    g = frame.groupby("end_index").label
    m = g.size().mean()
    icc = (g.mean().var(ddof=1) - y.var(ddof=1) / m) / y.var(ddof=1)
    return float(max(icc, 0.0)), float(m)


def effective_sample_size(n_stride1, stride, acf, icc, stocks_per_time):
    """Approximate effective N of a stride-thinned sample (Kish design effects).

    acf : Series of label autocorrelation indexed by lag 1..78 (stride-1 sample)
    time-series factor   1 + 2 * sum_j rho(j * stride)   (same stock, overlapping windows)
    cross-section factor 1 + (m - 1) * ICC               (all stocks move together at one time)
    """
    lags = [j * stride for j in range(1, LOOKBACK // stride + 1) if j * stride in acf.index]
    ts = 1 + 2 * float(np.clip(acf.loc[lags], 0, None).sum())
    cs = 1 + (stocks_per_time - 1) * icc
    n = n_stride1 // stride
    return dict(stride=stride, samples=n, time_series_factor=round(ts, 2),
                cross_section_factor=round(cs, 2), effective_samples=int(n / (ts * cs)))
