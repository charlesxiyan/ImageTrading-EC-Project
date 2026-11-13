"""Metrics that are honest for weak-signal classification."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

PROBABILISTIC = ("CNN", "training_prior", "hgb_hand_features")


def metrics(y, p, prior=None):
    """AUC (ranking), log loss (probabilities) and log-loss gain over the training prior."""
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    out = dict(samples=len(y), positive_fraction=float(y.mean()),
               accuracy=float(accuracy_score(y, p >= 0.5)),
               balanced_accuracy=float(balanced_accuracy_score(y, p >= 0.5)),
               roc_auc=float(roc_auc_score(y, p)),
               log_loss=float(log_loss(y, p, labels=[0, 1])),
               prob_std=float(p.std()))               # ~0 means the model collapsed to a constant
    if prior is not None:
        base = log_loss(y, np.full(len(y), prior), labels=[0, 1])
        out["log_loss_gain_vs_prior_pct"] = float(100 * (base - out["log_loss"]) / base)
    return out


def comparison_table(y, scores, prior):
    """One row per model. Rule scores are not probabilities, so only their AUC is reported."""
    rows = []
    for name, p in scores.items():
        p = np.asarray(p, float)
        p = np.where(np.isfinite(p), p, np.nanmedian(p))
        if name in PROBABILISTIC:
            rows.append(dict(model=name, **metrics(y, p, prior)))
        else:
            rows.append(dict(model=name, samples=len(y), roc_auc=float(roc_auc_score(y, p))))
    return pd.DataFrame(rows).set_index("model")


def decile_report(frame, p, target):
    """Sort stocks into deciles of predicted probability WITHIN each timestamp.

    vol    : mean realised |return| per decile (should rise monotonically)
    others : mean excess return per decile, plus a top-minus-bottom spread averaged
             per day with a t-statistic across days (gross of costs)
    """
    f = frame[["end_index", "decision_date", "forward_return", "xs_return"]].copy()
    f["p"] = p
    f["decile"] = f.groupby("end_index").p.transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) if len(x) >= 20 else np.nan)
    f = f.dropna(subset=["decile"])
    f["bps"] = 1e4 * (f.forward_return.abs() if target == "vol" else f.xs_return)
    table = f.groupby("decile").bps.agg(["mean", "count"]).reset_index()

    spread = None
    if target != "vol":
        by_time = f.groupby(["decision_date", "end_index", "decile"]).bps.mean().unstack()
        daily = (by_time[9] - by_time[0]).groupby(level=0).mean()
        spread = dict(days=int(len(daily)), mean_bps=float(daily.mean()),
                      t_stat=float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))))
    return table, spread
