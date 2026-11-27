"""Step 6d: a pure-image model. Can the picture alone carry what the final model gets as numbers?

The Step 6b final model reads two inputs: the chart + EWSD band, and 43 numbers through a small MLP. Here the numbers
input is removed. What the numbers know is drawn into the picture instead, on fixed scales fitted on training windows
only (a fixed scale is what makes a level visible: the chart itself is stretched to each window's own low and high).

Two pictures, both under the Step 6b chart + EWSD band (rows 0-311 are identical to Step 6b's `ewsd` picture):

  pure_key   + a session-open marker in the chart (time of day: a grey vertical line at the first bar of the
               decision session), and three strips:
               SPY EWSD band   the market's 5-minute volatility per bar (own fixed log scale)
               volume band     log(volume of each bar / mean volume of the 78 bars before the window), fixed scale
               multi-day RV    four horizontal bars: stock 5- and 20-session RV, SPY 5- and 20-session RV
                               (one shared fixed log scale, so stock and SPY lengths compare)
  pure_all   pure_key + a "number strip": all 43 standardised numbers of Step 6b (the same values its MLP reads),
             one 3-pixel bar each, up for positive and down for negative. This is the upper bound: every number
             is in the picture, so a gap to Step 6b measures how well the CNN reads numbers from pixels.

The CNN is Step 6b's (same channels, head, training rule, windows, rows); only the canvas is taller.
Everything drawn uses bars up to the decision bar t (the multi-day RV uses completed sessions only, as in Step 6a).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import Dataset

import indicator_cnn as ic
from chartcnn.images import point_in_time_window
from chartcnn.models import ChartCNN

__version__ = "2026-11-27.1"

GAP = 3                 # black rows between strips
SPY_BAND = 36
VOL_BAND = 36
RV_BAR, RV_GAP = 6, 2   # 4 bars of 6 rows with 2-row gaps
RV_ROWS = 4 * RV_BAR + 3 * RV_GAP
NUM_STRIP = 40
NUM_WIDTH = 5           # pixels per number (3 lit + 2 dark)
MARKER = 90             # grey level of the session-open line
RV_NAMES = ["rv5d", "rv20d", "spy_rv5d", "spy_rv20d"]
BASE = ic.SIZE + ic.BAND                                             # 312: Step 6b's picture
H_KEY = BASE + GAP + SPY_BAND + GAP + VOL_BAND + GAP + RV_ROWS       # 423
H_ALL = H_KEY + GAP + NUM_STRIP                                      # 466

VARIANTS = {"pure_key": False, "pure_all": True}     # name: draw the number strip
LABELS = {"pure_key": "pure image: chart + EWSD + time, SPY vol, volume, multi-day RV (no numbers)",
          "pure_all": "pure image: pure_key + all 43 numbers drawn as bars (no numbers input)"}
CHECKS = [
    ("pure_pure_key", "s4a_ewsd", "key strips added to the chart + EWSD picture (both image only)"),
    ("pure_pure_all", "pure_pure_key", "all 43 numbers drawn vs the key strips"),
    ("pure_pure_key", "final", "pure image (key) vs the final model (image + numbers input)"),
    ("pure_pure_all", "final", "pure image (all) vs the final model (image + numbers input)"),
    ("pure_pure_key", "hgb_tab43", "pure image (key) vs HGB 43"),
    ("pure_pure_all", "hgb_tab43", "pure image (all) vs HGB 43"),
    ("pure_pure_all", "tab_only_rv", "pure image (all) vs the 43-number MLP"),
]


def height(variant):
    return H_ALL if VARIANTS[variant] else H_KEY


def pretty(name):
    if name.startswith("pure_"):
        return LABELS.get(name[5:], name)
    return {"final": "final model (6b): image + 43 numbers", "tab_only_rv": "43-number MLP (6b)",
            "hgb_tab43": "HGB 43 (6b)", "hgb_tab38": "HGB 38 (6b)", "s4a_ewsd": "chart + EWSD, image only (4a)",
            "s4a_plain": "original chart, image only (4a)"}.get(name, name)


# ----------------------------------------------------------------------------- drawing

def ffill(x):
    """Forward-fill NaN (then back-fill a leading gap); all-NaN stays NaN."""
    x = np.asarray(x, np.float64).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    x[:np.argmax(ok)] = x[np.argmax(ok)]
    return x


def band(values, scale, rows):
    """Bars of a per-bar series on a fixed scale (value lo -> 1 pixel, hi -> `rows`); NaN draws nothing."""
    out = np.zeros((rows, ic.SIZE), np.uint8)
    lo, hi = scale
    v = np.asarray(values, np.float64)
    z = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    h = np.where(np.isfinite(v), 1 + np.rint(np.nan_to_num(z) * (rows - 1)), 0).astype(int)
    for i, k in enumerate(h):
        if k:
            out[rows - k:, 3 * i + 1] = 255
    return out


def rv_bars(log_rv, scale):
    """Four horizontal bars (length on one fixed log scale); NaN draws nothing."""
    out = np.zeros((RV_ROWS, ic.SIZE), np.uint8)
    lo, hi = scale
    for j, v in enumerate(log_rv):
        if np.isfinite(v):
            n = 1 + int(np.rint(np.clip((v - lo) / (hi - lo), 0, 1) * (ic.SIZE - 1)))
            r = j * (RV_BAR + RV_GAP)
            out[r:r + RV_BAR, :n] = 255
    return out


def number_strip(z):
    """Standardised numbers (clipped to +-5) as bars from the middle row: up if positive, down if negative."""
    out = np.zeros((NUM_STRIP, ic.SIZE), np.uint8)
    mid = NUM_STRIP // 2
    for j, v in enumerate(np.asarray(z, np.float64)):
        k = int(np.rint(min(abs(v), 5.0) / 5.0 * (mid - 1)))
        c = NUM_WIDTH * j
        out[mid, c:c + 3] = 120                                       # the zero line of each number
        if v > 0 and k:
            out[mid - k:mid, c:c + 3] = 255
        elif v < 0 and k:
            out[mid + 1:mid + 1 + k, c:c + 3] = 255
    return out


def strips(long_window, spy_close, bar_of_day, log_rv, scales, halflife, numbers=None):
    """Chart + EWSD (Step 6b) with the session marker, then the strips; uint8 (H_KEY or H_ALL, 234)."""
    window = long_window[-ic.LOOKBACK:]
    _, sigma = ic.indicators(long_window, halflife)
    chart = ic.draw_chart(window).copy()
    col = 3 * (ic.LOOKBACK - 1 - int(bar_of_day))                     # first bar of the decision session
    line = chart[:, col]
    line[line == 0] = MARKER
    spy = ffill(spy_close)
    spy_sigma = ic.ew_sigma(spy, halflife)[-ic.LOOKBACK:] if np.isfinite(spy).all() else np.full(ic.LOOKBACK, np.nan)
    v = np.asarray(long_window[:, 4], np.float64)
    base = v[:-ic.LOOKBACK].mean()
    rel = np.log(np.maximum(v[-ic.LOOKBACK:], 1e-3 * base) / base) if base > 0 else np.full(ic.LOOKBACK, np.nan)
    gap = np.zeros((GAP, ic.SIZE), np.uint8)
    parts = [chart, ic.draw_band(sigma, scales["ewsd"]), gap, band(np.log(spy_sigma), scales["spy_ewsd"], SPY_BAND), gap,
             band(rel, scales["volume"], VOL_BAND), gap, rv_bars(log_rv, scales["rv"])]
    if numbers is not None:
        parts += [gap, number_strip(numbers)]
    return np.vstack(parts)


# ----------------------------------------------------------------------------- data

class PureCharts(Dataset):
    """Renders a pure-image picture on the fly; returns (image [1, H, 234], empty tab, label, i) like ic.Charts."""

    def __init__(self, prepared, frame, variant, scales, halflife, market_close, log_rv, numbers):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self.y = frame.label.to_numpy(np.int64) if "label" in frame else np.zeros(len(frame), np.int64)
        self.variant, self.scales, self.halflife = variant, scales, halflife
        self.market_close = np.asarray(market_close, np.float64)
        self.log_rv = np.asarray(log_rv, np.float32)
        self.numbers = np.asarray(numbers, np.float32) if VARIANTS[variant] else None
        assert len(self.log_rv) == len(self.s)
        self.tab = np.zeros((len(self.s), 0), np.float32)
        self._x = self._m = None

    def __len__(self):
        return len(self.s)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x"] = state["_m"] = None
        return state

    def long_window(self, i):
        if self._x is None:
            self._x = np.load(self.root / "features.npy", mmap_mode="r")
            self._m = np.load(self.root / "multipliers.npy", mmap_mode="r")
        return point_in_time_window(self._x[self.s[i]], self._m[self.s[i]], self.t[i], ic.LOOKBACK + ic.HISTORY)

    def image(self, i):
        t = self.t[i]
        n = ic.LOOKBACK + ic.HISTORY
        return strips(self.long_window(i), self.market_close[t - n + 1:t + 1], t % ic.LOOKBACK, self.log_rv[i],
                      self.scales, self.halflife, None if self.numbers is None else self.numbers[i])

    def __getitem__(self, i):
        x = torch.from_numpy(self.image(i).astype(np.float32)[None] / 255.0)
        return x, torch.from_numpy(self.tab[i]), int(self.y[i]), int(i)


def log_rv(raw43):
    """log of the 4 multi-day RV numbers (NaN where missing), from Step 6b's raw 43-number frame."""
    x = raw43[RV_NAMES].to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(x), np.nan)


def dataset(prepared, frame, variant, scales, settings, market_close, lrv, numbers):
    rows = frame.row.to_numpy()
    return PureCharts(prepared, frame, variant, scales, settings["halflife"], market_close, lrv[rows],
                      numbers[rows] if numbers is not None else None)


def fit_scales(prepared, train_windows, lrv_train, market_close, halflife, ewsd_scale, n=20_000, q=(0.5, 99.5), seed=69):
    """Fixed scales of the new strips, from training windows only (the stock EWSD scale is Step 6b's)."""
    sample = train_windows.sample(min(n, len(train_windows)), random_state=seed)
    rows = sample.index.to_numpy()
    sample = sample.reset_index(drop=True)
    ds = ic.Charts(prepared, sample, "plain", (0.0, 1.0), halflife)
    m = ic.LOOKBACK + ic.HISTORY
    spy, vol = [], []
    for i in range(len(sample)):
        t = int(sample.end_index[i])
        c = ffill(market_close[t - m + 1:t + 1])
        if np.isfinite(c).all():
            spy.append(np.log(ic.ew_sigma(c, halflife)[-ic.LOOKBACK:]))
        v = ds.long_window(i)[:, 4].astype(np.float64)
        b = v[:-ic.LOOKBACK].mean()
        if b > 0:
            vol.append(np.log(np.maximum(v[-ic.LOOKBACK:], 1e-3 * b) / b))
    rv = lrv_train[rows].ravel()
    rv = rv[np.isfinite(rv)]
    pct = lambda a: tuple(float(x) for x in np.percentile(np.concatenate(a) if isinstance(a, list) else a, q))
    return dict(ewsd=tuple(ewsd_scale), spy_ewsd=pct(spy), volume=pct(vol), rv=pct(rv))


# ----------------------------------------------------------------------------- model and training

class PureCNN(nn.Module):
    """Step 6b's CNN on the taller canvas, no numbers input (same interface as ic.IndicatorCNN: cnn / n_tab)."""

    def __init__(self, settings, variant):
        super().__init__()
        self.cnn = ChartCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                            head_height=settings["head_height"], arch="plain", input_hw=(height(variant), ic.SIZE))
        self.input_shape = self.cnn.input_shape
        self.n_tab = 0

    def forward(self, x, t=None):
        return self.cnn(x)


def run_variant(variant, prepared, data, settings, device, local_dir, drive_dir=None):
    """Train one pure-image variant (skipped if done); returns (validation scores, info)."""
    local_dir = Path(local_dir) / variant
    drive_dir = Path(drive_dir) / variant if drive_dir is not None else None
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings["seed"]); np.random.seed(settings["seed"])
    model = PureCNN(settings, variant).to(device)
    n = ic.count(model)
    if (home / "valid_prob.npy").exists() and (home / "variant.json").exists():
        print(f"  {variant}: already trained, loaded")
        return np.load(home / "valid_prob.npy"), json.loads((home / "variant.json").read_text())
    print(f"  --- {LABELS[variant]} ({variant}): {n['total']:,} parameters, canvas {height(variant)}x{ic.SIZE}")
    w, bs, ebs = settings["num_workers"], settings["batch_size"], settings["eval_batch_size"]
    def mk(part, split, shuffle, b):          # split: which window table `row` points into ("train" or "valid")
        ds = dataset(prepared, data[part], variant, data["scales"], settings, data["market_close"], data["lrv"][split],
                     data["numbers"][split])
        return ic.make_loader(ds, b, w, shuffle, settings["seed"])

    t0 = time.perf_counter()
    history = ic.fit(model, mk("train", "train", True, bs), mk("es", "valid", False, ebs), data["es"].label.to_numpy(),
                     settings, device, local_dir, drive_dir)
    model.load_state_dict(torch.load(local_dir / "best.pt", map_location=device, weights_only=False)["model"])
    p = ic.predict(model, mk("valid", "valid", False, ebs), device, settings["amp"])
    info = dict(parameters=n, canvas=[height(variant), ic.SIZE], minutes=(time.perf_counter() - t0) / 60,
                best_epoch=int(history.loc[history.es_log_loss.idxmin(), "epoch"]), epochs=int(history.epoch.max()))
    np.save(local_dir / "valid_prob.npy", p)
    (local_dir / "variant.json").write_text(json.dumps(info))
    for f in ("valid_prob.npy", "variant.json"):
        ic._mirror(local_dir / f, drive_dir)
    print(f"  {variant}: validation AUC {roc_auc_score(data['valid'].label, p):.4f}, best epoch {info['best_epoch']}, "
          f"{info['minutes']:.1f} min")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, info


@torch.no_grad()
def score(variant, prepared, frame, settings, scales, market_close, lrv, numbers, ckpt, device):
    model = PureCNN(settings, variant).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = dataset(prepared, frame, variant, scales, settings, market_close, lrv, numbers)
    return ic.predict(model, ic.make_loader(ds, settings["eval_batch_size"], settings["num_workers"], False), device,
                      settings["amp"])


# ----------------------------------------------------------------------------- comparison

REF_COLS = {"score_ind_{final}": "score_final", "score_ind_tab_only_rv": "score_tab_only_rv",
            "score_hgb_tab43": "score_hgb_tab43", "score_hgb_tab38": "score_hgb_tab38",
            "score_s4a_ewsd": "score_s4a_ewsd", "score_s4a_plain": "score_s4a_plain"}


def join_6b(pred, path, final):
    """Step 6b's scores for the same rows (they must be identical rows)."""
    ref = pd.read_parquet(path)
    cols = {k.format(final=final): v for k, v in REF_COLS.items()}
    ref = ref[["security_index", "end_index", "label"] + [c for c in cols if c in ref]].rename(columns=cols | {"label": "ref_label"})
    out = pred.merge(ref, on=["security_index", "end_index"], how="left")
    assert out.ref_label.notna().all() and (out.ref_label == out.label).all(), "Step 6b rows differ"
    return out.drop(columns="ref_label")


def checks_table(boots):
    rows = []
    for a, b, label in CHECKS:
        p = ic.paired(boots, a, b)
        for _, r in p.iterrows():
            rows.append(dict(check=label, a=pretty(a), b=pretty(b), cell=r.cell, diff=r["diff"], low=r.low, high=r.high,
                             verdict=r.verdict))
    return pd.DataFrame(rows)


def auc_table(pred, names, boot):
    y = pred.label.to_numpy()
    rows = []
    for n in names:
        s = pred[f"score_{n}"].to_numpy(float)
        lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
        rows.append(dict(model=pretty(n), raw=n, roc_auc=roc_auc_score(y, s), auc_ci_low=lo, auc_ci_high=hi,
                         log_loss=log_loss(y, np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1])))
    return pd.DataFrame(rows)


def save_png(image, path, zoom=2):
    """A picture as a PNG (nearest-neighbour zoom), for the report."""
    from PIL import Image
    im = Image.fromarray(np.asarray(image, np.uint8))
    im.resize((im.width * zoom, im.height * zoom), Image.NEAREST).save(path, optimize=True)
