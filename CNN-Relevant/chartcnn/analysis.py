"""Post-run analysis: plain vs residual, CNN vs baselines, calibration, long-short, takeaways.

Everything here works on the dict returned by `run_experiment`, keyed "<experiment>_<arch>".
Uncertainty comes from a day-level bootstrap on the validation month: frames from the same
day share bars and market moves, so resampling days (not images) gives honest intervals.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluation import PROBABILISTIC

ARCH_ORDER = ("plain", "residual", "series1d")   # series1d = 1D CNN on raw OHLCV, no image


def _split_key(key):
    exp, arch = key.rsplit("_", 1)
    return exp, arch


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def day_bootstrap(valid, score_sets, n_boot=300, seed=0):
    """AUC draws for each score vector when whole validation days are resampled.

    score_sets: {name: probabilities aligned with `valid`}. Returns DataFrame (n_boot x names).
    The same day draw is used for every name, so differences between columns are paired.
    """
    rng = np.random.default_rng(seed)
    y = valid.label.to_numpy()
    day_rows = [np.flatnonzero(m) for m in
                (valid.decision_date.to_numpy() == d for d in valid.decision_date.unique())]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([day_rows[i] for i in rng.integers(0, len(day_rows), len(day_rows))])
        draws.append({k: _auc(y[idx], np.asarray(p)[idx]) for k, p in score_sets.items()})
    return pd.DataFrame(draws)


def _ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def summary_table(results, param_counts, n_boot=300):
    """One row per (experiment, model): AUC with 95% day-bootstrap CI, accuracy, log-loss gain,
    probability spread, parameters, best epoch and (direction targets) the decile long-short."""
    rows, boots = [], {}
    for exp in dict.fromkeys(_split_key(k)[0] for k in results):
        runs = {a: results[f"{exp}_{a}"] for a in ARCH_ORDER if f"{exp}_{a}" in results}
        first = next(iter(runs.values()))
        valid, table = first["valid"], first["table"]
        scores = {f"CNN ({a})": r["scores"]["CNN"] for a, r in runs.items()}
        scores["HGB hand features"] = first["scores"]["hgb_hand_features"]
        rules = [m for m in table.index if m not in PROBABILISTIC]
        best_rule = max(rules, key=lambda m: table.loc[m, "roc_auc"]) if rules else None
        if best_rule:
            scores[f"rule: {best_rule}"] = first["scores"][best_rule]
        boot = day_bootstrap(valid, scores, n_boot)
        boots[exp] = boot

        for name in scores:
            arch = name[5:-1] if name.startswith("CNN") else None
            src = runs[arch]["table"].loc["CNN"] if arch else (
                table.loc["hgb_hand_features"] if name.startswith("HGB") else table.loc[best_rule])
            lo, hi = _ci(boot[name])
            row = dict(experiment=exp, model=name, roc_auc=src["roc_auc"], auc_ci_low=lo, auc_ci_high=hi,
                       accuracy=src.get("accuracy", np.nan),
                       balanced_accuracy=src.get("balanced_accuracy", np.nan),
                       log_loss_gain_vs_prior_pct=src.get("log_loss_gain_vs_prior_pct", np.nan),
                       prob_std=src.get("prob_std", np.nan),
                       parameters=param_counts.get(arch, np.nan) if arch else np.nan,
                       best_epoch=np.nan, long_short_bps=np.nan, long_short_t=np.nan)
            if arch:
                h = runs[arch]["history"]
                row["best_epoch"] = int(h.loc[h.val_log_loss.idxmin(), "epoch"])
                ls = runs[arch]["long_short"]["cnn"]
            elif name.startswith("HGB"):
                ls = first["long_short"]["hgb"]
            else:
                ls = None
            if ls:
                row["long_short_bps"], row["long_short_t"] = ls["mean_bps"], ls["t_stat"]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["experiment", "model"]), boots


def calibration_table(y, p, bins=10):
    """Mean predicted probability vs observed positive rate in quantile bins of p."""
    f = pd.DataFrame(dict(y=np.asarray(y), p=np.asarray(p)))
    f["bin"] = pd.qcut(f.p.rank(method="first"), bins, labels=False)
    return f.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), count=("y", "size"))


def plot_analysis(results, summary, path=None):
    """One row per experiment: AUC with CIs | validation AUC by epoch | calibration | deciles."""
    exps = list(dict.fromkeys(_split_key(k)[0] for k in results))
    fig, axes = plt.subplots(len(exps), 4, figsize=(20, 4 * len(exps)), squeeze=False)
    for ax, exp in zip(axes, exps):
        s = summary.loc[exp]
        y = np.arange(len(s))
        ax[0].errorbar(s.roc_auc, y, xerr=[s.roc_auc - s.auc_ci_low, s.auc_ci_high - s.roc_auc],
                       fmt="o", capsize=4)
        ax[0].set_yticks(y, s.index); ax[0].axvline(0.5, ls=":", c="gray"); ax[0].invert_yaxis()
        ax[0].set(title=f"{exp}: validation AUC (95% day bootstrap)", xlabel="AUC")

        for arch in ARCH_ORDER:
            r = results.get(f"{exp}_{arch}")
            if r is None:
                continue
            h = r["history"]
            ax[1].plot(h.epoch, h.val_roc_auc, "o-", label=f"{arch} val")
            cal = calibration_table(r["valid"].label, r["scores"]["CNN"])
            ax[2].plot(cal.predicted, cal.observed, "o-", label=arch)
            d = r["deciles"]
            ax[3].plot(d.decile + 1, d["mean"], "o-", label=arch)
        first = results[next(k for k in results if k.startswith(exp + "_"))]
        hgb = calibration_table(first["valid"].label, first["scores"]["hgb_hand_features"])
        ax[2].plot(hgb.predicted, hgb.observed, "s--", c="tab:red", label="HGB")
        lim = [min(ax[2].get_xlim()[0], ax[2].get_ylim()[0]), max(ax[2].get_xlim()[1], ax[2].get_ylim()[1])]
        ax[2].plot(lim, lim, ":", c="gray")
        ax[1].axhline(s.loc["HGB hand features", "roc_auc"], ls="--", c="tab:red", label="HGB")
        ax[1].set(title="validation AUC by epoch", xlabel="epoch"); ax[1].legend()
        ax[2].set(title="calibration (deciles of p)", xlabel="mean predicted", ylabel="observed rate")
        ax[2].legend()
        ax[3].set(title="mean |return| by decile (bps)" if exp.startswith("vol") else "mean excess return by decile (bps)",
                  xlabel="CNN probability decile within timestamp"); ax[3].legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.show()


def takeaways(results, summary, boots):
    """Short plain-English conclusions generated from the numbers (no hand-written claims)."""
    lines = []
    for exp, s in summary.groupby(level=0):
        s, boot = s.loc[exp], boots[exp]
        cnn = [m for m in s.index if m in ("CNN (plain)", "CNN (residual)")]   # image models
        best = max(cnn, key=lambda m: s.loc[m, "roc_auc"])
        b = s.loc[best]
        signal = "above" if b.auc_ci_low > 0.5 else "not distinguishable from"
        lines.append(("\n" if lines else "") + f"**{exp}** — best CNN is {best}: AUC {b.roc_auc:.3f} "
                     f"[{b.auc_ci_low:.3f}, {b.auc_ci_high:.3f}], {signal} a coin flip.")

        if {"CNN (plain)", "CNN (residual)"} <= set(s.index):
            d = boot["CNN (residual)"] - boot["CNN (plain)"]
            lo, hi = _ci(d)
            verdict = ("residual is better" if lo > 0 else "plain is better" if hi < 0
                       else "no significant difference")
            ratio = s.loc["CNN (residual)", "parameters"] / s.loc["CNN (plain)", "parameters"]
            lines.append(f"- Residual − plain AUC: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] → {verdict} "
                         f"(residual has {ratio:.1f}× the parameters).")

        if "CNN (series1d)" in s.index:
            d = boot[best] - boot["CNN (series1d)"]
            lo, hi = _ci(d)
            verdict = ("the image helps" if lo > 0 else "the raw sequence is better" if hi < 0
                       else "no significant difference: the picture adds nothing measurable")
            lines.append(f"- {best} − 1D CNN on raw OHLCV (AUC {s.loc['CNN (series1d)', 'roc_auc']:.3f}): "
                         f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] → {verdict}.")

        d = boot[best] - boot["HGB hand features"]
        lo, hi = _ci(d)
        verdict = ("CNN beats HGB" if lo > 0 else "HGB beats CNN" if hi < 0 else "CNN and HGB are tied")
        lines.append(f"- {best} − HGB AUC: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] → {verdict}.")

        gain = b.log_loss_gain_vs_prior_pct
        r = results[f"{exp}_{best[5:-1]}"]
        cal = calibration_table(r["valid"].label, r["scores"]["CNN"])
        bias = float((cal.predicted - cal.observed).mul(cal["count"]).sum() / cal["count"].sum())
        lines.append(f"- Probabilities: log loss {gain:+.1f}% vs training prior; mean predicted − observed "
                     f"= {bias:+.3f}" + (" (calibration shift between train and validation)." if abs(bias) > 0.03 else "."))
        if b.prob_std < 0.01:
            lines.append(f"- Warning: prob std {b.prob_std:.4f}, the CNN has collapsed to a near-constant output.")
        if np.isfinite(b.long_short_bps):
            lines.append(f"- Top-minus-bottom decile: {b.long_short_bps:+.2f} bps per {exp.split('_h')[-1]} bars, "
                         f"t = {b.long_short_t:.2f} (gross; |t| < 2 means no reliable spread).")
    return "\n".join(lines)
