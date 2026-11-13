"""Draws the network-flow figure shown at the top of the notebook (plain + residual).

Usage: python tools/make_diagram.py <CNN-Relevant dir> <prepared data dir> assets/network_flow.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, sys.argv[1])
from chartcnn import ChartCNN, Store, build_split, ChartDataset, parameter_table

P, out = sys.argv[2], sys.argv[3]


def block_params(model):
    t, total, _ = parameter_table(model)
    names = t.layer.str.split(":").str[0]
    p = dict(zip(names, t.parameters))
    stem = sum(p[str(i)] for i in range(4))
    if model.arch == "plain":
        b2, b3 = sum(p[str(i)] for i in range(4, 8)), sum(p[str(i)] for i in range(8, 12))
    else:
        b2, b3 = p["4"], p["5"]
    return [stem, b2, b3, 0, p["head.2"]], total


store = Store(P)
f, _ = build_split(store, "validation", "vol", 1, 6)
img = ChartDataset(P, f.iloc[[1234]]).image(0)

rows = [
    ("plain", "Conv block", "5×3 conv\nBN · LeakyReLU · pool 2×1", "#dbe9f6"),
    ("residual", "Residual block", "[5×3 conv · BN · LReLU · 5×3 conv · BN]\n+ 1×1 skip → LReLU · pool 2×1", "#d9f0e3"),
]
shapes = ["64 × 42 × 78", "128 × 21 × 78", "256 × 10 × 78", "256 × 4 × 78", "2 logits → P(y=1)"]
fig = plt.figure(figsize=(16, 6.4), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.4); ax.axis("off")
iax = fig.add_axes([0.012, 0.30, 0.13, 0.40]); iax.imshow(img, cmap="gray"); iax.set_xticks([]); iax.set_yticks([])
ax.text(1.25, 4.75, "Input chart", ha="center", fontsize=12, weight="bold")
ax.text(1.25, 1.55, "1 × 234 × 234\n78 five-min bars\nOHLC + volume", ha="center", va="center", fontsize=8.5, color="#444")
x0, w, gap = 2.75, 2.3, 0.33
for r, (arch, mid_title, mid_detail, mid_color) in enumerate(rows):
    params, total = block_params(ChartCNN(arch=arch))
    y = 3.35 - r * 3.05
    titles = ["Conv block 1", f"{mid_title} 2", f"{mid_title} 3", "Height pool", "Head"]
    details = ["5×3 conv · stride 3×3\nBN · LeakyReLU · pool 2×1", mid_detail, mid_detail,
               "avg pool rows → 4\nkeep 78 bar columns", "flatten 79,872\ndropout 0.5 · linear"]
    colors = ["#dbe9f6", mid_color, mid_color, "#e7e3f4", "#fde2cf"]
    ax.text(x0, y + 2.62, f"{arch}  ·  {total:,} parameters", fontsize=11.5, weight="bold", color="#1f4e79")
    for i in range(5):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y + 0.45), w, 1.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=colors[i], ec="#555", lw=1))
        ax.text(x + w / 2, y + 2.08, titles[i], ha="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + 1.45, details[i], ha="center", va="center", fontsize=7.6 if i in (1, 2) and arch == "residual" else 8.6)
        ax.text(x + w / 2, y + 0.72, shapes[i], ha="center", fontsize=9, family="monospace", color="#1f4e79")
        ax.text(x + w / 2, y + 0.2, f"{params[i]:,} params", ha="center", fontsize=8, color="#666")
        ax.annotate("", xy=(x - 0.02, y + 1.42), xytext=(x - gap + 0.02, y + 1.42),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("ok")
