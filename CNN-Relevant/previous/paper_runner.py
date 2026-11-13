"""Paper-style chart CNN on the existing 5-minute bundle: new targets, honest baselines.

Targets are rebuilt from the prepared bundle (no WRDS access needed):
  vol    : 1 if |r(t -> t+H)| exceeds that stock's TRAINING median |r| at the same horizon
  xs_dir : 1 if r(t -> t+H) beats the cross-sectional median return at the same decision time
  dir    : 1 if r(t -> t+H) > 0 (raw direction; zero returns dropped)
Inputs are the same 78-bar OHLCV chart images as CNN78u (window-local price axis).
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

from us_pipeline.images import prediction_basis, render_window

LOOKBACK = 78


# ----------------------------------------------------------------------------- data
class Store:
    """Memory-light access to the prepared bundle."""

    def __init__(self, prepared):
        self.root = Path(prepared)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.config = self.manifest["config"]
        assert self.config["lookback"] == LOOKBACK and self.config["bar_minutes"] == 5
        features = np.load(self.root / "features.npy", mmap_mode="r")
        self.close = np.array(features[:, :, 3])          # ~40 MB
        self.volume = np.array(features[:, :, 4])
        self.log_close = np.log(self.close)
        valid = np.load(self.root / "valid.npy")
        self.bad_cumsum = np.concatenate(
            [np.zeros((valid.shape[0], 1), np.int32), np.cumsum(~valid, axis=1, dtype=np.int32)], axis=1)
        clock = pd.to_datetime(self.manifest["decision_clock_ns"], unit="ns", utc=True)
        self.dates = np.array(clock.tz_convert("America/New_York").strftime("%Y-%m-%d"))
        self.T = self.close.shape[1]

    def split_frame(self, split):
        return pd.read_parquet(self.root / f"{split}.parquet",
                               columns=["security_index", "permno", "end_index", "decision_time"])


def build_split(store, split, target, horizon, stride, train_abs_median=None):
    """Return a sample frame with labels for one split. Inputs reuse the audited windows."""
    f = store.split_frame(split)
    f = f[f.end_index % stride == stride - 1].reset_index(drop=True)
    s = f.security_index.to_numpy(np.int64)
    t = f.end_index.to_numpy(np.int64)
    h = int(horizon)
    inside = t + h < store.T
    s, t, f = s[inside], t[inside], f[inside].reset_index(drop=True)
    # every future bar t+1..t+H must be observed
    future_ok = (store.bad_cumsum[s, t + h + 1] - store.bad_cumsum[s, t + 1]) == 0
    # purge: label end must stay inside the same chronological split
    end_dates = store.dates[t + h]
    cfg = store.config
    if split == "train":
        purge_ok = end_dates <= cfg["train_end"]
    elif split == "validation":
        purge_ok = end_dates <= cfg["validation_end"]
    else:
        purge_ok = np.ones(len(t), bool)
    keep = future_ok & purge_ok
    f, s, t = f[keep].reset_index(drop=True), s[keep], t[keep]
    r = store.close[s, t + h] / store.close[s, t] - 1.0
    f["forward_return"] = r
    f["xs_median"] = f.groupby("end_index").forward_return.transform("median")
    f["xs_return"] = r - f.xs_median.to_numpy()
    f["decision_date"] = store.dates[t]
    if target == "dir":
        f = f[f.forward_return != 0]
        f["label"] = (f.forward_return > 0).astype("int64")
    elif target == "xs_dir":
        counts = f.groupby("end_index").end_index.transform("size")
        f = f[(counts >= 20) & (f.xs_return != 0)]
        f["label"] = (f.xs_return > 0).astype("int64")
    elif target == "vol":
        absr = f.forward_return.abs()
        if train_abs_median is None:
            train_abs_median = absr.groupby(f.security_index).median()
        med = f.security_index.map(train_abs_median)
        f = f[med.notna() & (absr != med)]
        f["label"] = (f.forward_return.abs() > f.security_index.map(train_abs_median)).astype("int64")
    else:
        raise ValueError("target must be vol, xs_dir or dir")
    return f.reset_index(drop=True), train_abs_median


class ChartDataset(Dataset):
    def __init__(self, prepared, frame, include_volume=True):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self.y = frame.label.to_numpy(np.int64)
        self.include_volume = include_volume
        self._x = self._m = None

    def __len__(self):
        return len(self.y)

    def __getstate__(self):
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
        x = torch.from_numpy(self.image(i).astype(np.float32)[None] / 255.0)
        return x, int(self.y[i]), int(i)


# ----------------------------------------------------------------------------- model
class PaperCNN(nn.Module):
    """Jiang-Kelly-Xiu style blocks: 5x3 conv, BN, LeakyReLU, vertical-only 2x1 max-pool;
    first layer dilation (2,1). Two deliberate adaptations to the 234x234 / 78-bar image:
      * first-layer stride (3,3): each bar is exactly 3 pixels wide, so every output column
        corresponds to one bar (78 columns kept all the way to the classifier);
      * height-only average pool to `head_height` rows before the linear head, keeping the
        flattened fan-in (~80k) close to the paper's 20-day model (~46k) instead of ~600k.
        With Adam, a huge fan-in makes each step move the logit too far and training noisy."""

    def __init__(self, channels=(64, 128, 256), dropout=0.5, head_height=4, input_hw=(234, 234)):
        super().__init__()
        layers, c_in = [], 1
        for i, c in enumerate(channels):
            if i == 0:
                conv = nn.Conv2d(c_in, c, (5, 3), stride=(3, 3), dilation=(2, 1), padding=(12, 0))
            else:
                conv = nn.Conv2d(c_in, c, (5, 3), padding=(2, 1))
            layers += [conv, nn.BatchNorm2d(c), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
            c_in = c
        layers.append(nn.AdaptiveAvgPool2d((head_height, None)))
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            shape = self.features(torch.zeros(1, 1, *input_hw)).shape
        self.feature_shape = tuple(shape[1:])
        self.flat_features = int(np.prod(self.feature_shape))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.flat_features, 2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero output layer: initial prediction is exactly 50/50, initial loss = ln 2.
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------- helpers
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def make_loader(ds, batch_size, workers, shuffle, seed):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers,
              pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
              generator=torch.Generator().manual_seed(seed))
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


def mirror(path, persistent):
    if persistent is None:
        return
    persistent = Path(persistent); persistent.mkdir(parents=True, exist_ok=True)
    tmp = persistent / (path.name + ".copying")
    shutil.copy2(path, tmp); os.replace(tmp, persistent / path.name)


def save_json(path, value, persistent=None):
    path.write_text(json.dumps(value, indent=2, default=float) + "\n"); mirror(path, persistent)


@torch.no_grad()
def predict(model, loader, device, progress=None):
    model.eval(); out = np.empty(len(loader.dataset), np.float64)
    for x, _, idx in loader:
        p = model(x.to(device, non_blocking=True)).float().softmax(1)[:, 1].cpu().numpy()
        out[idx.numpy()] = p
    return out


def metrics(y, p, prior=None):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7); y = np.asarray(y)
    m = dict(samples=len(y), positive_fraction=float(y.mean()),
             accuracy=float(accuracy_score(y, p >= .5)),
             balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
             roc_auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
             prob_std=float(p.std()))
    if prior is not None:
        base = log_loss(y, np.full(len(y), prior), labels=[0, 1])
        m["log_loss_vs_prior_pct"] = float(100 * (base - m["log_loss"]) / base)
    return m


# ----------------------------------------------------------------------------- baselines
def hand_features(store, frame):
    s = frame.security_index.to_numpy(np.int64); t = frame.end_index.to_numpy(np.int64)
    lc = store.log_close
    ret = lambda k: lc[s, t - k + 1] - lc[s, t - k]
    cols = {f"r{k}": ret(k) for k in range(1, 13)}
    for w in (6, 12, 39, 77):
        cols[f"cum{w}"] = lc[s, t] - lc[s, t - w]
    past = np.stack([ret(k) for k in range(1, 78)])
    cols["rv77"] = np.nanstd(past, axis=0)
    cols["absr12"] = np.nanmean(np.abs(past[:12]), axis=0)
    vol = store.volume
    cols["vrel"] = np.log(vol[s, t] / np.nanmean(np.stack([vol[s, t - k] for k in range(1, 78)]), axis=0))
    cols["tod"] = t % 78
    return pd.DataFrame(cols)


def baselines(store, target, train, valid, seed, max_train=300_000):
    """Return {name: validation score array}; rules use inputs only, HGB is fit on training only."""
    out = {}
    prior = float(train.label.mean())
    out["training_prior"] = np.full(len(valid), prior)
    Fva = hand_features(store, valid)
    if target == "vol":
        out["rule_recent_abs_return"] = Fva.absr12.to_numpy()
    else:
        out["rule_last_bar_momentum"] = Fva.r1.to_numpy()
        out["rule_78bar_momentum"] = Fva.cum77.to_numpy()
    sub = train.sample(min(max_train, len(train)), random_state=seed)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, min_samples_leaf=500,
                                         random_state=seed)
    hgb.fit(hand_features(store, sub), sub.label)
    out["hgb_hand_features"] = hgb.predict_proba(Fva)[:, 1]
    return out, prior


def score_table(y, scores, prior):
    rows = []
    for name, p in scores.items():
        p = np.nan_to_num(np.asarray(p, float), nan=np.nanmedian(p))
        probability = name in ("training_prior", "hgb_hand_features", "CNN")
        if probability:
            rows.append(dict(name=name, **metrics(y, p, prior)))
        else:  # rule score: only ranking is meaningful
            rows.append(dict(name=name, samples=len(y), roc_auc=float(roc_auc_score(y, p))))
    return pd.DataFrame(rows)


def decile_report(frame, p, target):
    """Within each decision time, sort stocks by predicted probability into deciles."""
    f = frame[["end_index", "decision_date", "forward_return", "xs_return"]].copy()
    f["p"] = p
    f["decile"] = f.groupby("end_index").p.transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) if len(x) >= 20 else np.nan)
    f = f.dropna(subset=["decile"])
    value = f.forward_return.abs() if target == "vol" else f.xs_return
    f["value_bps"] = value * 1e4
    table = f.groupby("decile").value_bps.agg(["mean", "count"]).reset_index()
    spread = None
    if target != "vol":
        g = f.groupby(["decision_date", "end_index", "decile"]).value_bps.mean().unstack()
        ls = (g[9] - g[0]).groupby(level=0).mean()            # one number per day
        spread = dict(days=int(len(ls)), mean_bps=float(ls.mean()),
                      t_stat_daily=float(ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls)))),
                      note="Top-minus-bottom decile, gross of costs; per-day averages")
    return table, spread


# ----------------------------------------------------------------------------- training
def run_experiment(prepared, exp, settings, local_dir, persistent_dir, device, smoke=False):
    local_dir = Path(local_dir); local_dir.mkdir(parents=True, exist_ok=True)
    seed = settings["seed"]; seed_all(seed)
    store = Store(prepared)
    stride = settings["stride"]
    train, med = build_split(store, "train", exp["target"], exp["horizon"], stride)
    valid, _ = build_split(store, "validation", exp["target"], exp["horizon"], stride, med)
    if smoke:
        train = train.sample(min(256, len(train)), random_state=seed).reset_index(drop=True)
        valid = valid.groupby("end_index").filter(lambda g: len(g) >= 20)
        valid = valid[valid.end_index.isin(valid.end_index.drop_duplicates().sample(2, random_state=seed))]
        valid = valid.reset_index(drop=True)
    info = dict(experiment=exp, settings=settings, train_samples=len(train), validation_samples=len(valid),
                train_positive=float(train.label.mean()), validation_positive=float(valid.label.mean()))
    print(json.dumps(info, default=str))
    save_json(local_dir / "experiment.json", info, persistent_dir)
    if med is not None:
        med.to_csv(local_dir / "train_abs_return_median.csv"); mirror(local_dir / "train_abs_return_median.csv", persistent_dir)

    print("Fitting baselines (training data only)...")
    base_scores, prior = baselines(store, exp["target"], train, valid, seed,
                                   max_train=2000 if smoke else 300_000)

    model = PaperCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                     head_height=settings["head_height"]).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()), "| feature map:", model.feature_shape)
    opt = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    tr_ds, va_ds = ChartDataset(prepared, train), ChartDataset(prepared, valid)
    tr_loader = make_loader(tr_ds, settings["batch_size"], settings["num_workers"], True, seed)
    va_loader = make_loader(va_ds, settings["eval_batch_size"], settings["num_workers"], False, seed)

    history, best, bad, start_epoch = [], float("inf"), 0, 1
    if (local_dir / "last.pt").exists():
        ck = torch.load(local_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        history, best, bad, start_epoch = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        print(f"Resumed after epoch {ck['epoch']}")
        if ck.get("finished"):
            start_epoch = settings["epochs"] + 1

    log_every = max(1, len(tr_loader) // 10)
    for epoch in range(start_epoch, settings["epochs"] + 1):
        model.train(); tr_loader.generator.manual_seed(seed + epoch)
        t0 = time.perf_counter(); seen = loss_sum = correct = 0
        for b, (x, y, _) in enumerate(tr_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x); loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            seen += len(y); loss_sum += loss.item() * len(y); correct += int((logits.argmax(1) == y).sum())
            if (b + 1) % log_every == 0:
                print(f"  epoch {epoch} batch {b+1}/{len(tr_loader)} train_loss={loss_sum/seen:.5f} "
                      f"acc={correct/seen:.4f} {seen/(time.perf_counter()-t0):.0f} img/s", flush=True)
        train_seconds = time.perf_counter() - t0
        p = predict(model, va_loader, device)
        m = metrics(valid.label.to_numpy(), p, prior)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_accuracy=correct / seen,
                   train_seconds=train_seconds, **{"val_" + k: v for k, v in m.items()})
        history.append(row); print(json.dumps(row))
        improved = m["log_loss"] < best - settings["min_delta"]
        if m["log_loss"] < best:
            best = m["log_loss"]
            torch.save(dict(model=model.state_dict(), epoch=epoch, settings=settings, experiment=exp),
                       local_dir / "best.pt"); mirror(local_dir / "best.pt", persistent_dir)
            np.save(local_dir / "best_validation_probability.npy", p)
        bad = 0 if improved else bad + 1
        finished = bad >= settings["patience"] or epoch == settings["epochs"]
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), history=history, best=best,
                        bad=bad, epoch=epoch, finished=finished), local_dir / "last.pt")
        mirror(local_dir / "last.pt", persistent_dir)
        pd.DataFrame(history).to_csv(local_dir / "history.csv", index=False)
        mirror(local_dir / "history.csv", persistent_dir)
        if bad >= settings["patience"]:
            print(f"Early stop: no validation log-loss improvement for {bad} epochs"); break

    p_best = np.load(local_dir / "best_validation_probability.npy")
    scores = {"CNN": p_best, **base_scores}
    table = score_table(valid.label.to_numpy(), scores, prior)
    table.to_csv(local_dir / "validation_comparison.csv", index=False)
    deciles, spread = decile_report(valid, p_best, exp["target"])
    deciles.to_csv(local_dir / "validation_deciles.csv", index=False)
    hgb_deciles, hgb_spread = decile_report(valid, base_scores["hgb_hand_features"], exp["target"])
    summary = dict(experiment=exp, best_epoch=int(pd.DataFrame(history).sort_values("val_log_loss").epoch.iloc[0]),
                   epochs_run=len(history), cnn_long_short=spread, hgb_long_short=hgb_spread,
                   validation=table.to_dict("records"), smoke_only=smoke, test_evaluated=False)
    save_json(local_dir / "summary.json", summary, persistent_dir)
    for name in ["validation_comparison.csv", "validation_deciles.csv"]:
        mirror(local_dir / name, persistent_dir)
    return dict(summary=summary, history=pd.DataFrame(history), table=table, deciles=deciles,
                store=store, median=med, prior=prior)


def evaluate_test(prepared, exp, settings, local_dir, persistent_dir, device, median, prior, smoke=False):
    """One-shot held-out evaluation of best.pt. Do not tune anything after looking at it."""
    store = Store(prepared)
    test, _ = build_split(store, "test", exp["target"], exp["horizon"], settings["stride"], median)
    if smoke:
        test = test[test.end_index.isin(test.end_index.drop_duplicates().sample(2, random_state=0))].reset_index(drop=True)
    model = PaperCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                     head_height=settings["head_height"]).to(device)
    model.load_state_dict(torch.load(Path(local_dir) / "best.pt", map_location=device, weights_only=False)["model"])
    loader = make_loader(ChartDataset(prepared, test), settings["eval_batch_size"], settings["num_workers"], False, 0)
    p = predict(model, loader, device)
    m = metrics(test.label.to_numpy(), p, prior)
    deciles, spread = decile_report(test, p, exp["target"])
    result = dict(experiment=exp, test=m, long_short=spread, smoke_only=smoke)
    save_json(Path(local_dir) / "test_summary.json", result, persistent_dir)
    return result, deciles
