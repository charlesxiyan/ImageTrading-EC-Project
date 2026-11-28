"""Step 6f: compact pictures. Step 6d's pure_all picture is 466 x 234 pixels and about 90% of it is black.
Can the same information be drawn more densely, and is a denser picture as accurate and faster?

Every variant shows exactly what Step 6d's `pure_all` shows (chart with session-open line, stock EWSD band, SPY EWSD
band, volume band, four multi-day RV bars, the 43 numbers), on Step 6d's strip scales. Only the layout changes:

  quarter    pure_all, max-pooled 4 x 1 in height before the CNN (116 x 234): a quarter of the rows. Max pooling keeps
             one-pixel lines. (The 7b+ half-height student lost nothing at 2 x 1.)
  channels   the strips are stacked as channels instead of below the chart: 6 x 234 x 234 (chart, stock EWSD, SPY
             EWSD, volume, multi-day RV, numbers). Each strip is drawn on the full 234 rows, so one pixel column holds
             price, volatility and volume of the same bar.
  filled     pure_all with the empty space filled: the area under each bar's close in grey, and every band bar
             3 pixels wide instead of 1. More lit pixels, the same information.
  compact    channels, max-pooled 2 x 1 in height (6 x 117 x 234): the smallest picture.

Width is not changed: the CNN's first layer already reads each bar's 3 columns (open tick, high-low line, close tick)
with a 3-wide, stride-3 filter, so the 234 columns become 78 bar columns at once and no time is spent on empty width.

The CNN is Step 6d's (64-128-256, plain, same training rule, windows and rows); only the first layer's input channels
and the input size change. Step 6d's `pure_all` scores on the same rows are the reference.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import indicator_cnn as ic
import pure_image as pi
from chartcnn.models import ChartCNN

__version__ = "2026-11-28.1"

ROWS = ic.SIZE                    # 234: every channel of the channel pictures
N_CHANNELS = 6
FILL = 60                         # grey level under the close in `filled`
PRICE_ROWS = ic.SIZE - max(1, ic.SIZE // 5) - 1    # 187: price area of the chart (the volume bars are below)

# name: (picture, pool rows, channels, input height)
VARIANTS = {
    "quarter": ("pure_all", 4, 1, pi.H_ALL // 4),
    "channels": ("channels", 1, N_CHANNELS, ROWS),
    "filled": ("filled", 1, 1, pi.H_ALL),
    "compact": ("channels", 2, N_CHANNELS, ROWS // 2),
}
LABELS = {
    "quarter": "pure_all, a quarter of the height (4x1 max pool)",
    "channels": "strips as 6 channels of 234 x 234",
    "filled": "pure_all with the empty space filled",
    "compact": "6 channels, half height (6 x 117 x 234)",
}
CHECKS = [
    ("cmp_quarter", "pure_all", "quarter height vs pure_all (6d)"),
    ("cmp_channels", "pure_all", "strips as channels vs pure_all (6d)"),
    ("cmp_filled", "pure_all", "filled picture vs pure_all (6d)"),
    ("cmp_compact", "pure_all", "6 channels at half height vs pure_all (6d)"),
    ("cmp_compact", "cmp_channels", "half height vs full height (both channels)"),
    ("cmp_quarter", "hgb_tab43", "quarter height vs HGB 43"),
    ("cmp_compact", "hgb_tab43", "6 channels at half height vs HGB 43"),
]


def pretty(name):
    if name.startswith("cmp_"):
        return LABELS.get(name[4:], name)
    return {"pure_all": "pure_all, 466 x 234 (6d)", "pure_key": "pure_key, 423 x 234 (6d)",
            "final": "final model, image + numbers (6b)", "hgb_tab43": "HGB 43 (6b)", "hgb_tab38": "HGB 38 (6b)",
            "tab_only_rv": "43-number MLP (6b)"}.get(name, name)


def input_shape(variant):
    _, _, c, h = VARIANTS[variant]
    return (c, h, ic.SIZE)


# ----------------------------------------------------------------------------- drawing

def stretch(strip, rows=ROWS):
    """A strip resized to `rows` rows (nearest row): band heights keep their proportion."""
    idx = (np.arange(rows) * strip.shape[0]) // rows
    return strip[idx]


def channel_picture(pure_all):
    """Step 6d's pure_all picture (466 x 234) re-laid out as 6 channels of 234 x 234."""
    g = pi.GAP
    cuts = [0, ic.SIZE, pi.BASE]                                          # chart, stock EWSD band
    cuts += [cuts[-1] + g, cuts[-1] + g + pi.SPY_BAND]                    # SPY band
    cuts += [cuts[-1] + g, cuts[-1] + g + pi.VOL_BAND]                    # volume band
    cuts += [cuts[-1] + g, cuts[-1] + g + pi.RV_ROWS]                     # RV bars
    cuts += [cuts[-1] + g, cuts[-1] + g + pi.NUM_STRIP]                   # numbers
    assert cuts[-1] == pi.H_ALL, cuts
    parts = [(0, ic.SIZE), (ic.SIZE, pi.BASE), (cuts[3], cuts[4]), (cuts[5], cuts[6]), (cuts[7], cuts[8]),
             (cuts[9], cuts[10])]
    return np.stack([pure_all[a:b] if b - a == ROWS else stretch(pure_all[a:b]) for a, b in parts])


def band_rows():
    """Row ranges of the three per-bar bands in the pure_all picture (stock EWSD, SPY EWSD, volume)."""
    g = pi.GAP
    spy0 = pi.BASE + g
    vol0 = spy0 + pi.SPY_BAND + g
    return [(ic.SIZE, pi.BASE), (spy0, spy0 + pi.SPY_BAND), (vol0, vol0 + pi.VOL_BAND)]


def filled_picture(pure_all):
    """pure_all with grey under each close and 3-pixel-wide band bars (same information, more lit pixels)."""
    img = pure_all.copy()
    chart = img[:PRICE_ROWS]
    for i in range(ic.LOOKBACK):
        close_col = chart[:, 3 * i + 2]
        lit = np.flatnonzero(close_col == 255)
        if len(lit):
            c = int(lit[0])
            block = chart[c + 1:, 3 * i:3 * i + 3]
            block[block == 0] = FILL
    for a, b in band_rows():
        mid = img[a:b, 1::3]                                         # the bar column of each of the 78 bars
        img[a:b, 0::3] = np.maximum(img[a:b, 0::3], mid)
        img[a:b, 2::3] = np.maximum(img[a:b, 2::3], mid)
    return img


class CompactCharts(pi.PureCharts):
    """Step 6d's pure_all rendering, re-laid out per variant; returns (image [C, H, 234], empty tab, label, i)."""

    def __init__(self, prepared, frame, variant, scales, halflife, market_close, log_rv, numbers):
        super().__init__(prepared, frame, "pure_all", scales, halflife, market_close, log_rv, numbers)
        self.layout = VARIANTS[variant][0]

    def picture(self, i):
        img = self.image(i)
        if self.layout == "channels":
            return channel_picture(img)
        if self.layout == "filled":
            return filled_picture(img)[None]
        return img[None]

    def __getitem__(self, i):
        x = torch.from_numpy(self.picture(i).astype(np.float32) / 255.0)
        return x, torch.from_numpy(self.tab[i]), int(self.y[i]), int(i)


def dataset(prepared, frame, variant, scales, settings, market_close, lrv, numbers):
    rows = frame.row.to_numpy()
    return CompactCharts(prepared, frame, variant, scales, settings["halflife"], market_close, lrv[rows], numbers[rows])


def pooled(picture, variant):
    """What the CNN sees after the variant's height pooling (numpy, for the report)."""
    k = VARIANTS[variant][1]
    if k == 1:
        return picture
    h = picture.shape[-2] // k * k
    return picture[..., :h, :].reshape(*picture.shape[:-2], h // k, k, picture.shape[-1]).max(-2)


def lit_share(picture):
    return float((np.asarray(picture) > 0).mean())


# ----------------------------------------------------------------------------- model and training

class CompactCNN(nn.Module):
    """Step 6d's CNN on a variant's input: optional k x 1 max pool in height, then ChartCNN with C input channels."""

    def __init__(self, settings, variant):
        super().__init__()
        _, self.pool, c, h = VARIANTS[variant]
        self.cnn = ChartCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                            head_height=settings["head_height"], arch="plain", input_hw=(h, ic.SIZE), in_channels=c)
        self.n_tab = 0

    def forward(self, x, t=None):
        if self.pool > 1:
            x = F.max_pool2d(x, (self.pool, 1))
        return self.cnn(x)


def run_variant(variant, prepared, data, settings, device, local_dir, drive_dir=None):
    """Train one variant (skipped if done); returns (validation scores, info)."""
    local_dir = Path(local_dir) / variant
    drive_dir = Path(drive_dir) / variant if drive_dir is not None else None
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings["seed"]); np.random.seed(settings["seed"])
    model = CompactCNN(settings, variant).to(device)
    n = ic.count(model)
    if (home / "valid_prob.npy").exists() and (home / "variant.json").exists():
        print(f"  {variant}: already trained, loaded")
        return np.load(home / "valid_prob.npy"), json.loads((home / "variant.json").read_text())
    print(f"  --- {LABELS[variant]} ({variant}): {n['total']:,} parameters, CNN input {input_shape(variant)}")
    w, bs, ebs = settings["num_workers"], settings["batch_size"], settings["eval_batch_size"]

    def mk(part, split, shuffle, b):
        ds = dataset(prepared, data[part], variant, data["scales"], settings, data["market_close"], data["lrv"][split],
                     data["numbers"][split])
        return ic.make_loader(ds, b, w, shuffle, settings["seed"])

    t0 = time.perf_counter()
    history = ic.fit(model, mk("train", "train", True, bs), mk("es", "valid", False, ebs), data["es"].label.to_numpy(),
                     settings, device, local_dir, drive_dir)
    model.load_state_dict(torch.load(local_dir / "best.pt", map_location=device, weights_only=False)["model"])
    p = ic.predict(model, mk("valid", "valid", False, ebs), device, settings["amp"])
    info = dict(parameters=n, input=list(input_shape(variant)), minutes=(time.perf_counter() - t0) / 60,
                best_epoch=int(history.loc[history.es_log_loss.idxmin(), "epoch"]), epochs=int(history.epoch.max()),
                seconds_per_epoch=float(history.train_seconds.mean()))
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
    model = CompactCNN(settings, variant).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = dataset(prepared, frame, variant, scales, settings, market_close, lrv, numbers)
    return ic.predict(model, ic.make_loader(ds, settings["eval_batch_size"], settings["num_workers"], False), device,
                      settings["amp"])


# ----------------------------------------------------------------------------- comparison

def ll_day_bootstrap(pred, names, n_boot=300, seed=0):
    """Log loss draws on the same resampled days as `ic.day_bootstrap` (same seed, so the same day draws)."""
    from sklearn.metrics import log_loss
    import pandas as pd
    rng = np.random.default_rng(seed)
    y = pred.label.to_numpy()
    codes, _ = pd.factorize(pred.decision_date)
    days = [np.flatnonzero(codes == d) for d in range(codes.max() + 1)]
    cols = {n: np.clip(pred[f"score_{n}"].to_numpy(float), 1e-7, 1 - 1e-7) for n in names}
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
        draws.append({n: log_loss(y[idx], cols[n][idx], labels=[0, 1]) for n in names})
    return pd.DataFrame(draws)


def checks_table(boots, ll_boots=None):
    """Paired checks: AUC difference (a - b, higher is better) and, given `ll_boots`, log loss difference (a - b,
    lower is better), each with a 95% paired day-bootstrap interval."""
    import pandas as pd
    rows = []
    for a, b, label in CHECKS:
        p = ic.paired(boots, a, b)
        q = ic.paired(ll_boots, a, b).set_index("cell") if ll_boots is not None else None
        for _, r in p.iterrows():
            row = dict(check=label, a=pretty(a), b=pretty(b), cell=r.cell, diff=r["diff"], low=r.low, high=r.high,
                       verdict=r.verdict)
            if q is not None and r.cell in q.index:
                l = q.loc[r.cell]
                row.update(ll_diff=l["diff"], ll_low=l.low, ll_high=l.high,
                           ll_verdict="better" if l.high < 0 else "worse" if l.low > 0 else "tied")
            rows.append(row)
    return pd.DataFrame(rows)


def auc_table(pred, names, boot):
    return pi.auc_table(pred, names, boot).assign(model=lambda d: [pretty(n) for n in d.raw])
