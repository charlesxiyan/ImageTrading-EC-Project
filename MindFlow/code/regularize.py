"""Step 6g: regularization ablation on Step 6d's pure-image CNN (`pure_all`).

The model shows little overfitting (validation loss is lowest at epoch 2-3, test is not worse than validation), and
distillation (a form of regularization) helped the small student. Does stronger regularization lower log loss or
raise AUC on the same picture and rows? Each variant changes one thing in Step 6d's training rule:

  ema          exponential moving average of the weights (decay 0.9995), used for early stopping and scoring
  adamw_1      AdamW with decoupled weight decay 1 (6d: Adam, no decay)
  adamw_10     AdamW with decoupled weight decay 10 (hard)
  drop_25      head dropout 0.25 (6d: 0.5), the weaker direction
  drop_75      head dropout 0.75 (hard)
  smooth_05    label smoothing 0.05 (expected to hurt log loss: it pulls probabilities toward 0.5)
  seed_1..4    Step 6d's rule with other seeds; averaged with 6d's model they give a 5-model ensemble (`ens5`)

Everything else (picture, CNN, windows, rows, early stopping on the early-stop days, patience) is Step 6d's.
Reported: AUC and log loss with paired day-bootstrap checks against 6d's `pure_all`, and the generalization gap
(eval-mode log loss on a fixed training subset vs validation).
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import indicator_cnn as ic
import pure_image as pi

__version__ = "2026-11-28.1"

REF = "pure_all"
VARIANTS = {
    "ema": dict(ema=0.9995),
    "adamw_1": dict(optimizer="adamw", weight_decay=1.0),
    "adamw_10": dict(optimizer="adamw", weight_decay=10.0),
    "drop_25": dict(dropout=0.25),
    "drop_75": dict(dropout=0.75),
    "smooth_05": dict(label_smoothing=0.05),
    "seed_1": dict(seed_offset=1),
    "seed_2": dict(seed_offset=2),
    "seed_3": dict(seed_offset=3),
    "seed_4": dict(seed_offset=4),
}
SEEDS = [v for v in VARIANTS if v.startswith("seed_")]
LABELS = {
    "ema": "weight EMA (0.9995)",
    "adamw_1": "AdamW, weight decay 1",
    "adamw_10": "AdamW, weight decay 10",
    "drop_25": "dropout 0.25",
    "drop_75": "dropout 0.75",
    "smooth_05": "label smoothing 0.05",
    "seed_1": "6d rule, seed +1", "seed_2": "6d rule, seed +2", "seed_3": "6d rule, seed +3", "seed_4": "6d rule, seed +4",
    "ens5": "ensemble of 5 seeds (6d + 4)",
}


def pretty(name):
    if name.startswith("reg_"):
        return LABELS.get(name[4:], name)
    return {"pure_all": "pure_all, 6d rule (Adam, dropout 0.5)", "hgb_tab43": "HGB 43 (6b)",
            "final": "final model, image + numbers (6b)"}.get(name, name)


def checks():
    rows = [(f"reg_{v}", REF, f"{LABELS[v]} vs pure_all (6d)") for v in VARIANTS if v not in SEEDS]
    rows.append(("reg_ens5", REF, "ensemble of 5 seeds vs pure_all (6d)"))
    rows.append(("reg_ens5", "hgb_tab43", "ensemble of 5 seeds vs HGB 43"))
    return rows


def variant_settings(settings, variant):
    """Step 6d's settings with one change."""
    s = dict(settings)
    r = VARIANTS[variant]
    s.update(optimizer=r.get("optimizer", "adam"), label_smoothing=r.get("label_smoothing", 0.0), ema=r.get("ema"))
    if "weight_decay" in r:
        s["weight_decay"] = r["weight_decay"]
    if "dropout" in r:
        s["dropout"] = r["dropout"]
    s["seed"] = settings["seed"] + r.get("seed_offset", 0)
    return s


def build(settings):
    return pi.PureCNN(settings, REF)


# ----------------------------------------------------------------------------- training

def fit(model, train_loader, es_loader, y_es, settings, device, out_dir, drive_dir=None):
    """Step 6d's loop (ic.fit) with three switches from `settings`: optimizer ("adam" with L2 as in 6d, or "adamw"
    with decoupled decay), label_smoothing, and ema (decay; the averaged weights are early-stopped and saved).
    Resumes from last.pt; last.pt is deleted when the variant is finished."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opt_cls = torch.optim.AdamW if settings["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(settings["ema"]), use_buffers=True)
           if settings.get("ema") else None)
    amp, smooth = settings["amp"], settings["label_smoothing"]
    history, best, bad, first = [], np.inf, 0, 1
    for src in (out_dir, drive_dir):
        if src is not None and (Path(src) / "last.pt").exists():
            ck = torch.load(Path(src) / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            if ema is not None and ck.get("ema") is not None:
                ema.load_state_dict(ck["ema"])
            history, best, bad, first = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
            if drive_dir is not None and (Path(drive_dir) / "best.pt").exists() and not (out_dir / "best.pt").exists():
                shutil.copy2(Path(drive_dir) / "best.pt", out_dir / "best.pt")
            print(f"  resuming after epoch {ck['epoch']}")
            if bad >= settings["patience"]:
                first = settings["epochs"] + 1
            break
    for epoch in range(first, settings["epochs"] + 1):
        train_loader.generator.manual_seed(settings["seed"] + epoch)
        model.train()
        seen = loss_sum = 0
        start = time.perf_counter()
        n_batches = len(train_loader)
        for b, (x, t, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ic.autocast(device, amp):
                logits = model(x, t)
            loss = nn.functional.cross_entropy(logits.float(), y, label_smoothing=smooth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update_parameters(model)
            seen += len(y); loss_sum += loss.item() * len(y)
            if b % max(1, n_batches // 4) == 0:
                print(f"    batch {b}/{n_batches} loss {loss_sum / seen:.4f} "
                      f"{seen / (time.perf_counter() - start):.0f} img/s", flush=True)
        scored = ema.module if ema is not None else model
        p = ic.predict(scored, es_loader, device, amp)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_seconds=time.perf_counter() - start,
                   es_log_loss=log_loss(y_es, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                   es_auc=roc_auc_score(y_es, p) if 0 < y_es.mean() < 1 else np.nan)
        history.append(row)
        print(f"  epoch {epoch}: train loss {row['train_loss']:.4f} | early-stop days: AUC {row['es_auc']:.4f}, "
              f"log loss {row['es_log_loss']:.5f} | {row['train_seconds'] / 60:.1f} min")
        if row["es_log_loss"] < best:
            torch.save(dict(model=scored.state_dict(), epoch=epoch), out_dir / "best.pt")
            ic._mirror(out_dir / "best.pt", drive_dir)
        bad = 0 if row["es_log_loss"] < best - settings["min_delta"] else bad + 1
        best = min(best, row["es_log_loss"])
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        ema=ema.state_dict() if ema is not None else None, history=history, best=best, bad=bad,
                        epoch=epoch), out_dir / "last.pt")
        ic._mirror(out_dir / "last.pt", drive_dir)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        ic._mirror(out_dir / "history.csv", drive_dir)
        if bad >= settings["patience"]:
            print(f"  early stop: {bad} epochs without improvement")
            break
    for d in (out_dir, drive_dir):
        if d is not None and (Path(d) / "last.pt").exists():
            (Path(d) / "last.pt").unlink()
    return pd.DataFrame(history)


def run_variant(variant, prepared, data, settings, device, local_dir, drive_dir=None):
    """Train one variant (skipped if done); returns (validation scores, info)."""
    local_dir = Path(local_dir) / variant
    drive_dir = Path(drive_dir) / variant if drive_dir is not None else None
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    s = variant_settings(settings, variant)
    torch.manual_seed(s["seed"]); np.random.seed(s["seed"])
    model = build(s).to(device)
    n = ic.count(model)
    if (home / "valid_prob.npy").exists() and (home / "variant.json").exists():
        print(f"  {variant}: already trained, loaded")
        return np.load(home / "valid_prob.npy"), json.loads((home / "variant.json").read_text())
    print(f"  --- {LABELS[variant]} ({variant}): {n['total']:,} parameters")
    w, bs, ebs = s["num_workers"], s["batch_size"], s["eval_batch_size"]

    def mk(part, split, shuffle, b):
        ds = pi.dataset(prepared, data[part], REF, data["scales"], s, data["market_close"], data["lrv"][split],
                        data["numbers"][split])
        return ic.make_loader(ds, b, w, shuffle, s["seed"])

    t0 = time.perf_counter()
    history = fit(model, mk("train", "train", True, bs), mk("es", "valid", False, ebs), data["es"].label.to_numpy(),
                  s, device, local_dir, drive_dir)
    model.load_state_dict(torch.load(local_dir / "best.pt", map_location=device, weights_only=False)["model"])
    p = ic.predict(model, mk("valid", "valid", False, ebs), device, s["amp"])
    info = dict(parameters=n, change=VARIANTS[variant], minutes=(time.perf_counter() - t0) / 60,
                best_epoch=int(history.loc[history.es_log_loss.idxmin(), "epoch"]), epochs=int(history.epoch.max()),
                seconds_per_epoch=float(history.train_seconds.mean()))
    np.save(local_dir / "valid_prob.npy", p)
    (local_dir / "variant.json").write_text(json.dumps(info))
    for f in ("valid_prob.npy", "variant.json"):
        ic._mirror(local_dir / f, drive_dir)
    print(f"  {variant}: validation AUC {roc_auc_score(data['valid'].label, p):.4f}, "
          f"log loss {log_loss(data['valid'].label, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]):.4f}, "
          f"best epoch {info['best_epoch']}, {info['minutes']:.1f} min")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, info


@torch.no_grad()
def score(variant, prepared, frame, settings, scales, market_close, lrv, numbers, ckpt, device):
    """Scores of a saved checkpoint (variant None = Step 6d's pure_all)."""
    s = variant_settings(settings, variant) if variant else settings
    model = build(s).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = pi.dataset(prepared, frame, REF, scales, s, market_close, lrv, numbers)
    return ic.predict(model, ic.make_loader(ds, s["eval_batch_size"], s["num_workers"], False), device, s["amp"])


# ----------------------------------------------------------------------------- comparison

def ensemble(pred, members):
    """Mean probability of the member columns (score_<name>)."""
    return pred[[f"score_{m}" for m in members]].mean(axis=1).to_numpy()


def checks_table(boots, ll_boots):
    rows = []
    for a, b, label in checks():
        p = ic.paired(boots, a, b)
        if p.empty:                                      # a variant that was not run
            continue
        q = ic.paired(ll_boots, a, b).set_index("cell")
        for _, r in p.iterrows():
            l = q.loc[r.cell]
            rows.append(dict(check=label, a=pretty(a), b=pretty(b), cell=r.cell, diff=r["diff"], low=r.low,
                             high=r.high, verdict=r.verdict, ll_diff=l["diff"], ll_low=l.low, ll_high=l.high,
                             ll_verdict="better" if l.high < 0 else "worse" if l.low > 0 else "tied"))
    return pd.DataFrame(rows)


def auc_table(pred, names, boot):
    return pi.auc_table(pred, names, boot).assign(model=lambda d: [pretty(n) for n in d.raw])


def gap_row(name, y_train, p_train, y_valid, p_valid):
    """Eval-mode log loss on a fixed training subset vs validation: the generalization gap."""
    tr = log_loss(y_train, np.clip(p_train, 1e-7, 1 - 1e-7), labels=[0, 1])
    va = log_loss(y_valid, np.clip(p_valid, 1e-7, 1 - 1e-7), labels=[0, 1])
    return dict(model=name, label=pretty(name), train_log_loss=tr, valid_log_loss=va, gap=va - tr,
                train_auc=roc_auc_score(y_train, p_train), valid_auc=roc_auc_score(y_valid, p_valid))
