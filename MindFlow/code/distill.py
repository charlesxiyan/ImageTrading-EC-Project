"""Step 7b: knowledge distillation of the final model into small students.

Teacher: the Step 6b final model (CNN on the chart + EWSD band, with the 43 numbers as a second input; 783,556
parameters). Students see exactly the same inputs and rows and use the same training rule (Adam, early stopping on the
log loss of the early-stopping days); only their size and the training target change.

**The loss.** With two classes, the teacher's soft target at temperature T is softmax(z_teacher / T); the student's
loss is

    (1 - alpha) * CE(student, hard label) + alpha * T^2 * KL(softmax(z_teacher / T) || softmax(z_student / T))

(Hinton, Vinyals & Dean 2015). With T = 1 the soft part is the cross-entropy against the teacher's probability. The
teacher's logit difference is recovered from its saved probability (z = log p / (1 - p)), so nothing is re-trained.
alpha = 0 is plain training on the labels: the "hard" twin of every distilled student.

**Students.** A narrower CNN of the same design (channels 16-32-64 instead of 64-128-256), on the full chart or on a
chart halved in height (2x1 max pooling first, so one-pixel lines survive; a bar stays three pixels wide, which the
first convolution needs), and the 43-number MLP without an image. Each image student is trained twice, on labels and
on the teacher, so the paired difference is the effect of distillation alone.

The teacher's soft targets on the training windows come from its best checkpoint (in-sample, as usual for
distillation). Early stopping and every comparison use the true labels of 2023.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

import indicator_cnn as ic
from chartcnn.models import ChartCNN
from chartcnn.training import autocast

__version__ = "2026-11-26.1"

# name: (CNN channels or None for no image, halve the chart's height, alpha)
STUDENTS = {
    "small_hard": ((16, 32, 64), False, 0.0),
    "small_kd": ((16, 32, 64), False, None),
    "small_half_hard": ((16, 32, 64), True, 0.0),
    "small_half_kd": ((16, 32, 64), True, None),
    "mlp_kd": (None, False, None),
}
LABELS = {
    "small_hard": "small CNN 16-32-64, labels",
    "small_kd": "small CNN 16-32-64, distilled",
    "small_half_hard": "small CNN, half-height chart, labels",
    "small_half_kd": "small CNN, half-height chart, distilled",
    "mlp_kd": "43-number MLP, distilled (no image)",
}
CHECKS = [
    ("stu_small_kd", "stu_small_hard", "distillation, small CNN"),
    ("stu_small_half_kd", "stu_small_half_hard", "distillation, small CNN on the half-height chart"),
    ("stu_mlp_kd", "teacher_tab_only_rv", "distillation, 43-number MLP (vs the 6b MLP trained on labels)"),
    ("stu_small_kd", "teacher", "distilled small CNN vs the teacher"),
    ("stu_small_half_kd", "teacher", "distilled half-height CNN vs the teacher"),
    ("stu_mlp_kd", "teacher", "distilled MLP vs the teacher"),
    ("stu_small_kd", "hgb_tab43", "distilled small CNN vs HGB 43"),
    ("stu_small_half_kd", "stu_small_kd", "half-height chart vs full chart (both distilled)"),
]


def pretty(name):
    if name.startswith("stu_"):
        return LABELS.get(name[4:], name)
    return {"teacher": "teacher: final CNN (6b)", "teacher_tab_only_rv": "43-number MLP, labels (6b)",
            "hgb_tab43": "HGB 43 (6b)", "hgb_tab38": "HGB 38 (6b)"}.get(name, name)


class Student(nn.Module):
    """IndicatorCNN with its own channel widths, optionally on a chart halved in height. Same interface as
    indicator_cnn.IndicatorCNN (cnn / tab / n_tab), so indicator_cnn.predict and count work on it."""

    def __init__(self, settings, channels, n_tab, half=False):
        super().__init__()
        self.half = bool(half)
        h = ic.SIZE + ic.BAND
        self.cnn = (ChartCNN(channels=tuple(channels), dropout=settings["dropout"], head_height=settings["head_height"],
                             arch="plain", input_hw=(h // 2 if half else h, ic.SIZE)) if channels else None)
        self.n_tab = n_tab
        if n_tab:
            k = settings["tab_hidden"]
            self.tab = nn.Sequential(nn.Linear(n_tab, k), nn.GELU(), nn.Linear(k, k), nn.GELU(), nn.Linear(k, 2))
            nn.init.zeros_(self.tab[-1].weight); nn.init.zeros_(self.tab[-1].bias)

    def forward(self, x, t=None):
        if self.cnn is None:
            return self.tab(t.to(x.device, non_blocking=True))
        if self.half:
            x = F.max_pool2d(x, (2, 1))
        logits = self.cnn(x)
        if self.n_tab:
            logits = logits + self.tab(t.to(logits.device, non_blocking=True))
        return logits


def build(name, settings, n_tab):
    channels, half, _ = STUDENTS[name]
    return Student(settings, channels, n_tab, half)


def alpha_of(name, settings):
    a = STUDENTS[name][2]
    return settings["kd_alpha"] if a is None else a


def variant_of(name, settings):
    """The picture a student draws (the teacher's, settings["teacher_variant"]), or the image-free placeholder."""
    return "tab_only_rv" if STUDENTS[name][0] is None else settings["teacher_variant"]


def teacher_logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def kd_loss(logits, y, z_teacher, alpha, temperature):
    """Hard + soft loss for two classes; z_teacher is the teacher's logit difference (class 1 minus class 0)."""
    hard = F.cross_entropy(logits, y)
    if alpha <= 0:
        return hard
    t = temperature
    teacher = torch.stack([torch.zeros_like(z_teacher), z_teacher], 1) / t
    soft = F.kl_div(F.log_softmax(logits / t, 1), F.log_softmax(teacher, 1), log_target=True, reduction="batchmean")
    return (1 - alpha) * hard + alpha * t * t * soft


def fit(model, train_loader, z_train, es_loader, y_es, settings, alpha, device, out_dir, drive_dir=None):
    """indicator_cnn.fit with the distillation loss (z_train: teacher logits by training-row position).
    Resumes from last.pt (local, else Drive)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = [dict(params=model.cnn.parameters(), lr=settings["learning_rate"])] if model.cnn is not None else []
    if model.n_tab:
        groups.append(dict(params=model.tab.parameters(), lr=settings["tab_learning_rate"]))
    optimizer = torch.optim.Adam(groups, weight_decay=settings["weight_decay"])
    amp, T = settings["amp"], settings["kd_temperature"]
    z_all = torch.as_tensor(z_train, dtype=torch.float32)
    history, best, bad, first = [], np.inf, 0, 1
    for src in (out_dir, drive_dir):
        if src is not None and (Path(src) / "last.pt").exists():
            ck = torch.load(Path(src) / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            history, best, bad, first = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
            if drive_dir is not None and (Path(drive_dir) / "best.pt").exists() and not (out_dir / "best.pt").exists():
                ic.shutil.copy2(Path(drive_dir) / "best.pt", out_dir / "best.pt")
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
        for b, (x, t, y, idx) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = z_all[idx].to(device, non_blocking=True)
            with autocast(device, amp):
                logits = model(x, t)
            loss = kd_loss(logits.float(), y, z, alpha, T)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(y); loss_sum += loss.item() * len(y)
            if b % max(1, n_batches // 4) == 0:
                print(f"    batch {b}/{n_batches} loss {loss_sum / seen:.4f} "
                      f"{seen / (time.perf_counter() - start):.0f} img/s", flush=True)
        p = ic.predict(model, es_loader, device, amp)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_seconds=time.perf_counter() - start,
                   es_log_loss=log_loss(y_es, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                   es_auc=roc_auc_score(y_es, p) if 0 < y_es.mean() < 1 else np.nan)
        history.append(row)
        print(f"  epoch {epoch}: train loss {row['train_loss']:.4f} | early-stop days: AUC {row['es_auc']:.4f}, "
              f"log loss {row['es_log_loss']:.5f} | {row['train_seconds'] / 60:.1f} min")
        if row["es_log_loss"] < best:
            torch.save(dict(model=model.state_dict(), epoch=epoch), out_dir / "best.pt")
            ic._mirror(out_dir / "best.pt", drive_dir)
        bad = 0 if row["es_log_loss"] < best - settings["min_delta"] else bad + 1
        best = min(best, row["es_log_loss"])
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history, best=best,
                        bad=bad, epoch=epoch), out_dir / "last.pt")
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


def dataset(prepared, frame, name, scale, settings, tab):
    return ic.Charts(prepared, frame, variant_of(name, settings), scale, settings["halflife"], tab[frame.row.to_numpy()])


def run_student(name, prepared, data, settings, device, local_dir, drive_dir=None):
    """Train one student (skipped if done); returns (validation scores, info)."""
    local_dir = Path(local_dir) / name
    drive_dir = Path(drive_dir) / name if drive_dir is not None else None
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings["seed"]); np.random.seed(settings["seed"])
    model = build(name, settings, data["tab_train"].shape[1]).to(device)
    n = ic.count(model)
    if (home / "valid_prob.npy").exists() and (home / "student.json").exists():
        print(f"  {name}: already trained, loaded")
        return np.load(home / "valid_prob.npy"), json.loads((home / "student.json").read_text())
    alpha = alpha_of(name, settings)
    print(f"  --- {LABELS[name]} ({name}): {n['total']:,} parameters (CNN {n['cnn']:,} + MLP {n['tab']:,}); "
          f"alpha {alpha}, T {settings['kd_temperature']}")
    w, bs, ebs = settings["num_workers"], settings["batch_size"], settings["eval_batch_size"]
    t0 = time.perf_counter()
    history = fit(model, ic.make_loader(dataset(prepared, data["train"], name, data["scale"], settings, data["tab_train"]),
                                        bs, w, True, settings["seed"]),
                  data["z_train"], ic.make_loader(dataset(prepared, data["es"], name, data["scale"], settings,
                                                          data["tab_valid"]), ebs, w, False),
                  data["es"].label.to_numpy(), settings, alpha, device, local_dir, drive_dir)
    model.load_state_dict(torch.load(local_dir / "best.pt", map_location=device, weights_only=False)["model"])
    p = ic.predict(model, ic.make_loader(dataset(prepared, data["valid"], name, data["scale"], settings, data["tab_valid"]),
                                         ebs, w, False), device, settings["amp"])
    info = dict(parameters=n, alpha=alpha, temperature=settings["kd_temperature"],
                best_epoch=int(history.loc[history.es_log_loss.idxmin(), "epoch"]),
                minutes=(time.perf_counter() - t0) / 60, epochs=int(history.epoch.max()))
    np.save(local_dir / "valid_prob.npy", p)
    (local_dir / "student.json").write_text(json.dumps(info))
    for f in ("valid_prob.npy", "student.json"):
        ic._mirror(local_dir / f, drive_dir)
    print(f"  {name}: validation AUC {roc_auc_score(data['valid'].label, p):.4f}, best epoch {info['best_epoch']}, "
          f"{info['minutes']:.1f} min")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, info


@torch.no_grad()
def score(name, prepared, frame, settings, scale, tab, ckpt, device):
    model = build(name, settings, tab.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = dataset(prepared, frame, name, scale, settings, tab)
    return ic.predict(model, ic.make_loader(ds, settings["eval_batch_size"], settings["num_workers"], False), device,
                      settings["amp"])


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
