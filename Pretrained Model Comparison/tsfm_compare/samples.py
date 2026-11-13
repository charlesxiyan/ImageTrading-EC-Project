"""Evaluation samples and model inputs, built on the shared `chartcnn` data code.

Every model in the comparison (CNN, HGB, foundation models) is scored on exactly the
same (stock, decision bar) pairs, with the same labels as `chartcnn.build_split`.

To keep inference affordable we evaluate on a random subset of whole *timestamps*
(all stocks at a chosen decision bar), so cross-sectional targets stay intact.
"""
import numpy as np
import pandas as pd

from chartcnn import build_split

KEY = ["security_index", "end_index"]


def pick_timestamps(frames, n, seed):
    """Choose n decision bars present in every frame (shared across experiments)."""
    common = set.intersection(*(set(f.end_index.unique()) for f in frames))
    common = np.array(sorted(common))
    if n and n < len(common):
        common = np.sort(np.random.default_rng(seed).choice(common, n, replace=False))
    return common


def build_samples(store, experiments, stride, n_valid_times, n_calib_times, seed=69):
    """Labelled frames per experiment.

    Returns {exp_name: dict(train=..., calib=..., valid=..., median=...)}
      train : full training split (for the HGB baseline)
      calib : subset of training timestamps; the foundation models are run on it to fit
              a 1-feature logistic calibration (zero-shot score -> probability)
      valid : subset of validation timestamps, identical across experiments
    """
    out = {}
    for exp in experiments:
        train, median = build_split(store, "train", exp["target"], exp["horizon"], stride)
        valid, _ = build_split(store, "validation", exp["target"], exp["horizon"], stride, median)
        out[exp["name"]] = dict(train=train, valid=valid, median=median)

    valid_times = pick_timestamps([d["valid"] for d in out.values()], n_valid_times, seed)
    calib_times = pick_timestamps([d["train"] for d in out.values()], n_calib_times, seed + 1)
    for d in out.values():
        d["valid"] = d["valid"][d["valid"].end_index.isin(valid_times)].reset_index(drop=True)
        d["calib"] = d["train"][d["train"].end_index.isin(calib_times)].reset_index(drop=True)
    return out


def union_keys(samples, part):
    """Unique (stock, bar) pairs over all experiments -> forecast each pair only once."""
    keys = pd.concat([d[part][KEY] for d in samples.values()]).drop_duplicates()
    return keys.sort_values(KEY).reset_index(drop=True)


def align(keys, frame):
    """Row positions in `keys` for each row of `frame`."""
    pos = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys[KEY]))
    return pos.loc[pd.MultiIndex.from_frame(frame[KEY])].to_numpy()


# ----------------------------------------------------------------------------- inputs
def _window_index(t, length):
    """Slot indices t-length+1 .. t for every decision bar t -> (n, length)."""
    return t[:, None] + np.arange(-length + 1, 1)[None, :]


def _fill(a):
    """Forward-fill NaN along time, then back-fill the start (only matters for context > 78)."""
    a = pd.DataFrame(a.T).ffill().bfill().to_numpy().T
    return a


def close_contexts(store, keys, length):
    """Close prices of the last `length` bars, divided by the decision bar's close.

    The last value is exactly 1, so a forecast q of the price h bars ahead means a
    return of q - 1. Scaling does not matter to the models (they normalise internally)
    but keeps every series on the same footing.
    """
    s = keys.security_index.to_numpy(np.int64)
    t = keys.end_index.to_numpy(np.int64)
    if (t - length + 1).min() < 0:
        raise ValueError("context longer than the available history")
    idx = _window_index(t, length)
    c = store.close[s[:, None], idx].astype(np.float64)
    if length > 78:
        valid = np.isfinite(c) & (c > 0)
        c = _fill(np.where(valid, c, np.nan))
    return (c / c[:, -1:]).astype(np.float32)


def ohlcv_contexts(prepared, store, keys, length):
    """Raw OHLCV bars (n, length, 5) in prediction-time units (for Kronos, which models candles)."""
    feats = np.load(f"{prepared}/features.npy", mmap_mode="r")
    mult = np.load(f"{prepared}/multipliers.npy", mmap_mode="r")
    s = keys.security_index.to_numpy(np.int64)
    t = keys.end_index.to_numpy(np.int64)
    idx = _window_index(t, length)
    x = np.empty((len(s), length, 5), np.float64)
    for stock in np.unique(s):                       # one contiguous read per stock
        rows = np.flatnonzero(s == stock)
        block = np.asarray(feats[stock])
        x[rows] = block[idx[rows]]
        m = np.asarray(mult[stock])[t[rows]]           # (rows, 2): price, volume multiplier
        x[rows, :, :4] /= m[:, None, 0:1]
        x[rows, :, 4] /= m[:, None, 1]
    if length > 78:
        for j in range(5):
            x[:, :, j] = _fill(np.where(np.isfinite(x[:, :, j]), x[:, :, j], np.nan))
    return x.astype(np.float32)


def slot_times(store, keys, length, horizon):
    """Bar timestamps (New York time) for the context and the forecast horizon."""
    clock = pd.to_datetime(store.manifest["decision_clock_ns"], unit="ns", utc=True)
    clock = clock.tz_convert("America/New_York").tz_localize(None)
    t = keys.end_index.to_numpy(np.int64)
    past = _window_index(t, length)
    future = t[:, None] + np.arange(1, horizon + 1)[None, :]
    return clock.values[past], clock.values[future]


def training_series(store, max_stocks=None, seed=0):
    """Whole training-period close series per stock (NaN where a bar is missing).

    Used only for the optional Chronos-2 fine-tune. Series stop at the training end date,
    so nothing from validation or test leaks into fine-tuning.
    """
    last = np.flatnonzero(store.dates <= store.config["train_end"]).max() + 1
    c = store.close[:, :last].astype(np.float64)
    observed = (store.bad_cumsum[:, 1:last + 1] - store.bad_cumsum[:, :last]) == 0
    c = np.where(observed & (c > 0), c, np.nan)
    keep = np.flatnonzero(np.isfinite(c).mean(1) > 0.5)
    if max_stocks and max_stocks < len(keep):
        keep = np.random.default_rng(seed).choice(keep, max_stocks, replace=False)
    return [(c[i] / np.nanmedian(c[i])).astype(np.float32) for i in keep]
