"""Step 7b+: knowledge distillation of the pure-image model (Step 6d).

Teacher: Step 6d's `pure_all` CNN (no numbers input; everything it knows is drawn into one 466 x 234 picture;
776,450 parameters; ties HGB 43 on 2023 at 5 minutes). Students see exactly the same picture and rows and use the
same training rule; only their size and the training target change. Unlike Step 7b, every student is a pure-image
CNN: there is no numbers MLP anywhere, so the whole line stays "the machine only looks at the chart".

Loss, teacher logits and the hard / distilled twins are those of Step 7b (distill.kd_loss, distill.fit).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import distill as kd
import indicator_cnn as ic
import pure_image as pi
from chartcnn.models import ChartCNN

__version__ = "2026-11-27.1"

# name: (CNN channels, halve the picture's height, alpha or None = settings["kd_alpha"])
STUDENTS = {
    "small_hard": ((16, 32, 64), False, 0.0),
    "small_kd": ((16, 32, 64), False, None),
    "small_half_hard": ((16, 32, 64), True, 0.0),
    "small_half_kd": ((16, 32, 64), True, None),
    "tiny_kd": ((8, 16, 32), False, None),
}
LABELS = {
    "small_hard": "small pure-image CNN 16-32-64, labels",
    "small_kd": "small pure-image CNN 16-32-64, distilled",
    "small_half_hard": "small pure-image CNN, half-height picture, labels",
    "small_half_kd": "small pure-image CNN, half-height picture, distilled",
    "tiny_kd": "tiny pure-image CNN 8-16-32, distilled",
}
CHECKS = [
    ("stu_small_kd", "stu_small_hard", "distillation, small CNN"),
    ("stu_small_half_kd", "stu_small_half_hard", "distillation, small CNN on the half-height picture"),
    ("stu_small_kd", "teacher", "distilled small CNN vs the teacher (6d pure_all)"),
    ("stu_small_half_kd", "teacher", "distilled half-height CNN vs the teacher"),
    ("stu_tiny_kd", "teacher", "distilled tiny CNN vs the teacher"),
    ("stu_small_half_kd", "stu_small_kd", "half-height vs full picture (both distilled)"),
    ("stu_tiny_kd", "stu_small_kd", "tiny vs small (both distilled)"),
    ("stu_small_kd", "hgb_tab43", "distilled small CNN vs HGB 43"),
    ("stu_small_half_kd", "hgb_tab43", "distilled half-height CNN vs HGB 43"),
    ("stu_small_kd", "final", "distilled small pure-image CNN vs the 6b final model (image + numbers)"),
]


def pretty(name):
    if name.startswith("stu_"):
        return LABELS.get(name[4:], name)
    return {"teacher": "teacher: pure image, all numbers drawn (6d)", "pure_key": "pure image, key strips (6d)",
            "final": "final model, image + numbers (6b)", "hgb_tab43": "HGB 43 (6b)", "hgb_tab38": "HGB 38 (6b)",
            "tab_only_rv": "43-number MLP (6b)"}.get(name, name)


class PureStudent(nn.Module):
    """A pure-image CNN of its own width, optionally on the picture halved in height (2x1 max pooling first, so
    one-pixel lines survive and a bar stays 3 pixels wide). Same interface as indicator_cnn.IndicatorCNN."""

    def __init__(self, settings, channels, variant, half=False):
        super().__init__()
        self.half = bool(half)
        h = pi.height(variant)
        self.cnn = ChartCNN(channels=tuple(channels), dropout=settings["dropout"], head_height=settings["head_height"],
                            arch="plain", input_hw=(h // 2 if half else h, ic.SIZE))
        self.n_tab = 0

    def forward(self, x, t=None):
        if self.half:
            x = F.max_pool2d(x, (2, 1))
        return self.cnn(x)


def build(name, settings):
    channels, half, _ = STUDENTS[name]
    return PureStudent(settings, channels, settings["teacher_variant"], half)


def alpha_of(name, settings):
    a = STUDENTS[name][2]
    return settings["kd_alpha"] if a is None else a


def loader(prepared, data, part, split, settings, shuffle, batch):
    ds = pi.dataset(prepared, data[part], settings["teacher_variant"], data["scales"], settings, data["market_close"],
                    data["lrv"][split], data["numbers"][split])
    return ic.make_loader(ds, batch, settings["num_workers"], shuffle, settings["seed"])


def run_student(name, prepared, data, settings, device, local_dir, drive_dir=None):
    """Train one student (skipped if done); returns (validation scores, info)."""
    local_dir = Path(local_dir) / name
    drive_dir = Path(drive_dir) / name if drive_dir is not None else None
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings["seed"]); np.random.seed(settings["seed"])
    model = build(name, settings).to(device)
    n = ic.count(model)
    if (home / "valid_prob.npy").exists() and (home / "student.json").exists():
        print(f"  {name}: already trained, loaded")
        return np.load(home / "valid_prob.npy"), json.loads((home / "student.json").read_text())
    alpha = alpha_of(name, settings)
    print(f"  --- {LABELS[name]} ({name}): {n['total']:,} parameters; alpha {alpha}, T {settings['kd_temperature']}")
    bs, ebs = settings["batch_size"], settings["eval_batch_size"]
    t0 = time.perf_counter()
    history = kd.fit(model, loader(prepared, data, "train", "train", settings, True, bs), data["z_train"],
                     loader(prepared, data, "es", "valid", settings, False, ebs), data["es"].label.to_numpy(),
                     settings, alpha, device, local_dir, drive_dir)
    model.load_state_dict(torch.load(local_dir / "best.pt", map_location=device, weights_only=False)["model"])
    p = ic.predict(model, loader(prepared, data, "valid", "valid", settings, False, ebs), device, settings["amp"])
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
def score(name, prepared, frame, settings, scales, market_close, lrv, numbers, ckpt, device):
    model = build(name, settings).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = pi.dataset(prepared, frame, settings["teacher_variant"], scales, settings, market_close, lrv, numbers)
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
    return kd.auc_table(pred, names, boot).assign(model=lambda d: [pretty(n) for n in d.raw])
