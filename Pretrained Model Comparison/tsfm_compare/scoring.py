"""Turn price forecasts into classification scores and compare every model on one table.

Zero-shot scores (no label is ever used by the forecaster):
    vol     expected |return| over the horizon, mean_i |q_i - 1| (q_i = forecast quantiles),
            divided by the stock's training median |return| (the label's own threshold)
    xs_dir  forecast median return over the horizon minus its cross-sectional median at
            the same decision bar (the label is "beats the cross-section")

AUC uses the raw score, so it is a pure zero-shot number. For log loss we need a
probability: a 1-feature logistic regression is fitted on the *training-period*
calibration sample and applied unchanged to validation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from chartcnn import build_split
from chartcnn.evaluation import decile_report, metrics
from chartcnn.baselines import fit_baselines

from .samples import KEY, align


def zero_shot_score(q, frame, target, horizon, median=None):
    """q: (n, H, 9) quantile forecasts aligned to `frame` rows."""
    qh = q[:, horizon - 1, :].astype(np.float64)
    if target == "vol":
        expected_abs = np.abs(qh - 1.0).mean(1)
        scale = frame.security_index.map(median).to_numpy()
        return np.log(expected_abs / scale + 1e-12)
    predicted = qh[:, 4] - 1.0                                         # median forecast return
    if target == "xs_dir":
        cs = pd.Series(predicted).groupby(frame.end_index.to_numpy()).transform("median").to_numpy()
        return predicted - cs
    return predicted


def forecast_skill(q, frame, horizon):
    """MAE of the median return forecast relative to the no-change forecast (< 1 beats a random walk)."""
    r = frame.forward_return.to_numpy()
    return float(np.mean(np.abs(q[:, horizon - 1, 4] - 1.0 - r)) / np.mean(np.abs(r)))


def calibrate(score_calib, y_calib, score_valid):
    """Monotone 1-feature logistic map fitted on training-period data only."""
    mu, sd = np.nanmean(score_calib), np.nanstd(score_calib) + 1e-12
    lr = LogisticRegression(C=1.0).fit(((score_calib - mu) / sd)[:, None], y_calib)
    return lr.predict_proba(((score_valid - mu) / sd)[:, None])[:, 1]


def evaluate_forecaster(name, n_params, q_valid, q_calib, keys_valid, keys_calib, samples, experiments,
                        kind="foundation (zero-shot)", seconds=None):
    """One row per experiment for one foundation model, plus its validation probabilities."""
    rows, probs = {}, {}
    for exp in experiments:
        d = samples[exp["name"]]
        qv = q_valid[align(keys_valid, d["valid"])]
        qc = q_calib[align(keys_calib, d["calib"])]
        sv = zero_shot_score(qv, d["valid"], exp["target"], exp["horizon"], d["median"])
        sc = zero_shot_score(qc, d["calib"], exp["target"], exp["horizon"], d["median"])
        p = calibrate(sc, d["calib"].label.to_numpy(), sv)
        prior = float(d["train"].label.mean())
        row = metrics(d["valid"].label.to_numpy(), p, prior)
        row.update(model=name, parameters=n_params, kind=kind, seconds=seconds,
                   mae_vs_random_walk=forecast_skill(qv, d["valid"], exp["horizon"]))
        _, spread = decile_report(d["valid"], sv, exp["target"])
        if spread:
            row.update(long_short_bps=spread["mean_bps"], long_short_t=spread["t_stat"])
        rows[exp["name"]], probs[exp["name"]] = row, p
    return rows, probs


def baseline_rows(store, samples, experiments, seed=69, max_train=300_000):
    """Training prior, simple rules and HGB on hand features (shared chartcnn code), same samples."""
    out = {}
    for exp in experiments:
        d = samples[exp["name"]]
        scores, prior, hgb = fit_baselines(store, exp["target"], d["train"], d["valid"], seed, max_train)
        y = d["valid"].label.to_numpy()
        rows = []
        for name, s in scores.items():
            s = np.where(np.isfinite(s), s, np.nanmedian(s))
            if name in ("training_prior", "hgb_hand_features"):
                row = metrics(y, s, prior)
            else:                                                        # rules give ranks, not probabilities
                from sklearn.metrics import roc_auc_score
                row = dict(samples=len(y), roc_auc=float(roc_auc_score(y, s)))
            row.update(model=name, kind="baseline")
            if name == "hgb_hand_features":
                row["parameters"] = int(sum(len(p.nodes) for st in hgb._predictors for p in st))
                row["parameters_note"] = "tree nodes"
                _, spread = decile_report(d["valid"], s, exp["target"])
                if spread:
                    row.update(long_short_bps=spread["mean_bps"], long_short_t=spread["t_stat"])
            rows.append(row)
        out[exp["name"]] = rows
    return out


def cnn_rows(store, samples, experiments, run_dir):
    """Validation probabilities saved by a ChartCNN run (CNN-Relevant/runs/<RUN_ID>/).

    Layout: <RUN_ID>/settings.json and <RUN_ID>/<experiment>_<arch>/best_valid_prob.npy
    (older runs: <RUN_ID>/<experiment>/best_valid_prob.npy). The CNN's validation frame is
    rebuilt with the run's own stride and matched to our samples by (stock, bar).
    """
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "settings.json").read_text())
    stride = settings.get("settings", settings).get("stride", 6)
    out = {}
    for exp in experiments:
        d, rows = samples[exp["name"]], []
        folders = sorted(p for p in run_dir.glob(f"{exp['name']}*") if (p / "best_valid_prob.npy").exists())
        if not folders:
            continue
        train, median = build_split(store, "train", exp["target"], exp["horizon"], stride)
        valid, _ = build_split(store, "validation", exp["target"], exp["horizon"], stride, median)
        for folder in folders:
            p = np.load(folder / "best_valid_prob.npy")
            if len(p) != len(valid):
                print(f"skip {folder.name}: {len(p)} probabilities vs {len(valid)} rebuilt samples")
                continue
            arch = folder.name[len(exp["name"]):].strip("_") or "plain"
            merged = d["valid"][KEY].merge(valid[KEY].assign(p=p), on=KEY, how="left")
            ok = merged.p.notna().to_numpy()
            y = d["valid"].label.to_numpy()[ok]
            row = metrics(y, merged.p.to_numpy()[ok], float(d["train"].label.mean()))
            row.update(model=f"ChartCNN_{arch}", kind="CNN (trained)", matched_share=float(ok.mean()))
            ck = folder / "best.pt"
            if ck.exists():
                import torch
                state = torch.load(ck, map_location="cpu", weights_only=False)["model"]
                row["parameters"] = int(sum(v.numel() for k, v in state.items()
                                            if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))))
            if ok.all():
                _, spread = decile_report(d["valid"], merged.p.to_numpy(), exp["target"])
                if spread:
                    row.update(long_short_bps=spread["mean_bps"], long_short_t=spread["t_stat"])
            rows.append(row)
        out[exp["name"]] = rows
    return out


COLUMNS = ["kind", "parameters", "samples", "roc_auc", "log_loss_gain_vs_prior_pct", "log_loss",
           "accuracy", "balanced_accuracy", "prob_std", "mae_vs_random_walk",
           "long_short_bps", "long_short_t", "seconds"]


def comparison_table(exp_name, *row_sets):
    rows = [r for rs in row_sets for r in rs.get(exp_name, [])]
    t = pd.DataFrame(rows).set_index("model")
    t = t[[c for c in COLUMNS if c in t.columns]]
    return t.sort_values("roc_auc", ascending=False)
