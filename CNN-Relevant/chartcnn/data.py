"""Samples and labels built from the prepared CNN78u bundle (`prepared/`).

The bundle already contains audited 78-bar input windows (`train/validation/test.parquet`).
Here we keep those windows, optionally thin them with a stride, and attach a new label:

    vol     1 if |r(t -> t+H)| > that stock's TRAINING-period median |r|     (volatility)
    xs_dir  1 if r(t -> t+H) > cross-sectional median at the same timestamp  (relative direction)
    dir     1 if r(t -> t+H) > 0, zero returns dropped                       (raw direction)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .images import prediction_basis, render_window

LOOKBACK = 78          # bars per image (one trading day of 5-minute bars)
TARGETS = ("vol", "xs_dir", "dir")


class Store:
    """In-memory arrays needed to build labels (close, volume, validity, calendar)."""

    def __init__(self, prepared):
        self.root = Path(prepared)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.config = self.manifest["config"]
        assert self.config["lookback"] == LOOKBACK and self.config["bar_minutes"] == 5

        features = np.load(self.root / "features.npy", mmap_mode="r")   # (stocks, slots, OHLCV)
        self.close = np.array(features[:, :, 3])
        self.volume = np.array(features[:, :, 4])
        self.log_close = np.log(self.close)
        self.T = self.close.shape[1]

        # bad_cumsum[s, j] = number of missing slots before slot j  ->  O(1) "is range complete?"
        valid = np.load(self.root / "valid.npy")
        self.bad_cumsum = np.concatenate(
            [np.zeros((valid.shape[0], 1), np.int32), np.cumsum(~valid, axis=1, dtype=np.int32)], axis=1)

        clock = pd.to_datetime(self.manifest["decision_clock_ns"], unit="ns", utc=True)
        self.dates = np.array(clock.tz_convert("America/New_York").strftime("%Y-%m-%d"))

    def complete(self, s, start, stop):
        """True where slots [start, stop) of stock s are all observed."""
        return (self.bad_cumsum[s, stop] - self.bad_cumsum[s, start]) == 0

    def windows(self, split):
        cols = ["security_index", "permno", "end_index", "decision_time"]
        return pd.read_parquet(self.root / f"{split}.parquet", columns=cols)


def build_split(store, split, target, horizon, stride, train_abs_median=None):
    """Labelled samples for one split.

    Returns (frame, train_abs_median). For target='vol' pass the training split's
    median into validation/test so thresholds never use future data.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    f = store.windows(split)
    f = f[f.end_index % stride == stride - 1]                  # thin overlapping windows
    s, t, h = f.security_index.to_numpy(np.int64), f.end_index.to_numpy(np.int64), int(horizon)

    # The future horizon must be fully observed and must not leak into the next split.
    keep = t + h < store.T
    s, t, f = s[keep], t[keep], f[keep]
    end_dates = store.dates[t + h]
    boundary = {"train": store.config["train_end"], "validation": store.config["validation_end"]}
    keep = store.complete(s, t + 1, t + h + 1)
    if split in boundary:
        keep &= end_dates <= boundary[split]
    f, s, t = f[keep].reset_index(drop=True), s[keep], t[keep]

    r = store.close[s, t + h] / store.close[s, t] - 1.0
    f["forward_return"] = r
    f["xs_return"] = r - f.groupby("end_index").forward_return.transform("median").to_numpy()
    f["decision_date"] = store.dates[t]

    if target == "dir":
        f = f[f.forward_return != 0]
        f["label"] = (f.forward_return > 0).astype("int64")
    elif target == "xs_dir":
        n_at_time = f.groupby("end_index").end_index.transform("size")
        f = f[(n_at_time >= 20) & (f.xs_return != 0)]
        f["label"] = (f.xs_return > 0).astype("int64")
    else:  # vol
        abs_r = f.forward_return.abs()
        if train_abs_median is None:
            train_abs_median = abs_r.groupby(f.security_index).median()
        threshold = f.security_index.map(train_abs_median)
        f = f[threshold.notna() & (abs_r != threshold)]
        f["label"] = (f.forward_return.abs() > f.security_index.map(train_abs_median)).astype("int64")
    return f.reset_index(drop=True), train_abs_median


class ChartDataset(Dataset):
    """Renders chart images on the fly from the memory-mapped bundle."""

    def __init__(self, prepared, frame, include_volume=True):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self.y = frame.label.to_numpy(np.int64) if "label" in frame else np.zeros(len(frame), np.int64)
        self.include_volume = include_volume
        self._x = self._m = None            # opened lazily in each DataLoader worker

    def __len__(self):
        return len(self.s)

    def __getstate__(self):                 # do not pickle memmaps into workers
        state = self.__dict__.copy()
        state["_x"] = state["_m"] = None
        return state

    def image(self, i):
        if self._x is None:
            self._x = np.load(self.root / "features.npy", mmap_mode="r")
            self._m = np.load(self.root / "multipliers.npy", mmap_mode="r")
        s, t = self.s[i], self.t[i]
        window = prediction_basis(self._x[s, t - LOOKBACK + 1:t + 1], self._m[s, t])
        return render_window(window, self.include_volume)

    def __getitem__(self, i):
        x = torch.from_numpy(self.image(i).astype(np.float32)[None] / 255.0)   # [1, 234, 234] in [0, 1]
        return x, int(self.y[i]), int(i)
