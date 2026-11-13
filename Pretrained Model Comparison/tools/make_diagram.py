"""Draws assets/flow.png, the minimal flow diagram at the top of the notebook."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets" / "flow.png"
fig = plt.figure(figsize=(13, 4.6), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(-0.4, 4.2); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec="#555", lw=1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=8.3, color="#222")


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))


box(0.15, 1.45, 2.3, 1.3, "Data", "cnn78u bundle\n528 stocks, 5-min bars\nJan-Mar | Apr | May-Jun", "#eeeeee")
box(2.95, 1.45, 2.5, 1.3, "Shared samples", "chartcnn.build_split\nvol_h1, xsdir_h12\nsame stocks x bars for all", "#e7e3f4")
rows = [(2.95, "Chart CNN", "images -> trained CNN\n(CNN-Relevant run, read only)", "#dbe9f6"),
        (1.6, "Baselines", "prior, rules,\nHGB on hand features", "#eeeeee"),
        (0.15, "Foundation models", "Chronos-2, TimesFM, Moirai 2, Kronos\nprice context -> quantile forecast", "#fde2cf")]
for y, t, b, c in rows:
    box(6.05, y, 3.0, 1.1, t, b, c)
    arrow(5.45, 2.1, 6.03, y + 0.55)
box(9.65, 0.15, 1.45, 1.1, "Score", "E|r| or rel. return\n+ 1-feature calib.", "#fde2cf")
arrow(9.05, 0.7, 9.63, 0.7)
box(11.45, 1.45, 1.45, 1.3, "One table", "AUC, log-loss\ngain, long-short,\nparameters", "#d9f0d3")
arrow(9.05, 3.5, 11.43, 2.4); arrow(9.05, 2.15, 11.43, 2.1); arrow(11.1, 0.7, 11.6, 1.43)
arrow(2.45, 2.1, 2.93, 2.1)
ax.text(7.55, -0.08, "optional: light Chronos-2 fine-tune (LoRA) on training-period prices only",
        ha="center", fontsize=8.3, va="top", color="#a33")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
