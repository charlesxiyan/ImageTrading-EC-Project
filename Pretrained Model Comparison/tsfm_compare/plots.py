"""Figures for the comparison notebook."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"baseline": "#9e9e9e", "CNN (trained)": "#1f77b4",
          "foundation (zero-shot)": "#ff7f0e", "foundation (fine-tuned)": "#d62728"}


def plot_comparison(tables, path=None):
    """Validation AUC per model, one panel per experiment (dotted line = coin flip)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5 * len(tables), 0.45 * max(map(len, tables.values())) + 1.5))
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        t = t.sort_values("roc_auc")
        ax.barh(t.index, t.roc_auc - 0.5, left=0.5, color=[COLORS.get(k, "k") for k in t.kind])
        ax.axvline(0.5, ls=":", c="k")
        for y, v in enumerate(t.roc_auc):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
        ax.set(title=f"{name}: validation ROC-AUC", xlabel="AUC")
        lo = min(0.48, t.roc_auc.min() - 0.01)
        ax.set_xlim(lo, max(0.52, t.roc_auc.max() + 0.03))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, COLORS.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if path:
        fig.savefig(path, dpi=120)
    plt.show()


def plot_forecast_examples(q, close_context, realised, n=4, path=None):
    """A few context windows with the model's 10-90% band, for a visual sanity check."""
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 2.8))
    L, H = close_context.shape[1], q.shape[1]
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(np.arange(L), close_context[k], c="k", lw=0.8)
        x = np.arange(L, L + H)
        ax.fill_between(x, q[k, :, 0], q[k, :, -1], alpha=0.3, label="10-90%")
        ax.plot(x, q[k, :, 4], c="tab:orange", lw=1, label="median")
        if realised is not None:
            ax.plot(x, realised[k], c="tab:green", lw=1, label="realised")
        ax.set_xlim(L - 30, L + H)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show()
