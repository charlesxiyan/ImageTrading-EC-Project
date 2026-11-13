"""Figures used in the notebook."""
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(name, result, path=None):
    """Loss curves, validation AUC vs the HGB baseline, and the decile profile."""
    h, table, d = result["history"], result["table"], result["deciles"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train")
    ax[0].plot(h.epoch, h.val_log_loss, "o-", label="validation")
    ax[0].axhline(np.log(2), ls=":", c="gray", label="ln 2 (coin flip)")
    ax[0].set(title=f"{name}: loss", xlabel="epoch"); ax[0].legend()

    ax[1].plot(h.epoch, h.val_roc_auc, "o-", label="CNN")
    ax[1].axhline(table.loc["hgb_hand_features", "roc_auc"], ls="--", c="tab:red", label="HGB hand features")
    ax[1].axhline(0.5, ls=":", c="gray")
    ax[1].set(title="validation AUC", xlabel="epoch"); ax[1].legend()

    ax[2].bar(d.decile + 1, d["mean"])
    ax[2].set(title="mean |return| (bps)" if name.startswith("vol") else "mean excess return (bps)",
              xlabel="CNN probability decile within timestamp")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.show()


def plot_frame_corr(similarity, label_acf, stride=6, path=None):
    """Left: how similar frames t and t+k look (to us and to the CNN).
    Right: how similar their labels are. Dashed line = training stride."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for col, style in [("pixel_raw", "o-"), ("pixel_aligned", "s-")]:
        ax[0].plot(similarity.lag, similarity[col], style, label=col)
    for col in similarity.columns:
        if col.endswith(("_embed", "_prob")):
            ax[0].plot(similarity.lag, similarity[col], "^--", label=col)
    for name, acf in label_acf.items():
        ax[1].plot(acf.lag, acf.label_autocorr, ".-", label=name)
    for a in ax:
        a.set_xscale("log"); a.axvline(stride, ls="--", c="gray"); a.axvline(78, ls=":", c="gray")
        a.axhline(0, c="black", lw=0.6); a.set_xlabel("lag k (5-min bars; 78 = one session)"); a.legend(fontsize=8)
    ax[0].set(title="frame similarity: corr(frame t, frame t+k)", ylabel="correlation")
    ax[1].set(title="label autocorrelation", ylabel="correlation")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.show()
