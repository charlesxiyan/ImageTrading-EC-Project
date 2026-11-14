"""Builds Multi_Target_Chart_CNN.ipynb.  Usage: python tools/make_notebook.py <output.ipynb>"""
import json, sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Multi-target Chart CNN baseline: horizon × target grid

**Question.** At high frequency, is there any signal in the *direction* of the next move, or only in its *size*?

| | 5 min (h1, reference) | 15 min (h3) | 30 min (h6) | 60 min (h12) |
|---|---|---|---|---|
| `dir` absolute direction: return > 0 (zeros dropped) | `dir_h1` | `dir_h3` | `dir_h6` | `dir_h12` |
| `vol` volatility: \\|return\\| > stock's training median | `vol_h1` | `vol_h3` | `vol_h6` | `vol_h12` |
| `xsdir` relative direction: return > cross-sectional median | `xsdir_h1` | `xsdir_h3` | `xsdir_h6` | `xsdir_h12` |

- **Input**: one trading day (78 five-minute bars), adjusted point in time to its last bar. Same windows for all 12 cells.
- **Models per cell**: `plain` image CNN (Jiang–Kelly–Xiu style), `series1d` 1D CNN on the raw OHLCV numbers (does the picture help?), HGB on 20 hand features, simple rules. The residual CNN is left out to save GPU time (it tied the plain CNN at 2.6× the cost).
- **Stride 12 for every horizon**: one sample per stock per hour, so a stock's labels never overlap (stride ≥ horizon), and all horizons are scored on the same decision times.
- **Resumable**: each finished cell is saved on Drive and skipped on a rerun; an unfinished CNN resumes from its last epoch.
- **Code**: `chartcnn/` next to this notebook (`chartcnn/grid.py` holds the grid logic).
""")

md("## 1. Setup")
code("""
import os, sys, shutil, tempfile
from pathlib import Path

LOCAL_TEST = os.environ.get("CNN78U_LOCAL_NOTEBOOK_TEST") == "1"     # repository QA only
if LOCAL_TEST:
    DEMO = Path(os.environ["CNN78U_LOCAL_DEMO"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DEMO = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo")
HERE = DEMO / "Multi-Target Chart CNN Baseline"   # this notebook, chartcnn/ and runs/
# chartcnn/ is copied into this folder on Drive; in the repository it lives in ../CNN-Relevant.
CODE = HERE if (HERE / "chartcnn").is_dir() else DEMO / "CNN-Relevant"
sys.path.insert(0, str(CODE))

import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
import chartcnn
from chartcnn import grid
from chartcnn.utils import stage_bundle, save_json
assert chartcnn.__version__ >= "2026-11-14.3", f"stale chartcnn {chartcnn.__version__} in {CODE}"

LOCAL = Path(tempfile.mkdtemp(prefix="mtcnn_", dir=None if LOCAL_TEST else "/content"))
PREPARED = stage_bundle(DEMO, LOCAL / "bundle")  # bundle v2 on fast local disk (derived from v1 zip if needed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("data:", PREPARED, "| device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")
""")

md("""
## 2. Configuration
Change settings here only. **To resume after a disconnect**, set `RUN_ID` to the id printed by the first run (for example `"run_20261114_190000"`) and Run all again: finished cells are loaded from Drive, and the cell that was running continues from its last epoch.
""")
code("""
from datetime import datetime, timezone
SMOKE = LOCAL_TEST                                # tiny code check; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"

RUN_ID = None                                     # None = new run; or an old RUN_ID to resume it
SETTINGS = dict(
    learning_rate=1e-5, weight_decay=0.0,         # Adam, as in Jiang-Kelly-Xiu and ChartCNN_Baseline
    batch_size=4 if SMOKE else 128, eval_batch_size=8 if SMOKE else 512,
    epochs=1 if SMOKE else 12, patience=2, min_delta=1e-5,   # early stop on validation log loss
    channels=[64, 128, 256], head_height=4, dropout=0.5,
    amp=True,                                     # bfloat16 autocast on A100/L4/H100; ignored on T4/CPU
    num_workers=0 if SMOKE else min(8, os.cpu_count() or 2),
    seed=69,
)
HORIZONS = [1, 3, 6, 12]                          # bars of 5 minutes: 5, 15, 30, 60 minutes
KINDS = ["dir", "vol", "xsdir"]                   # absolute direction, volatility, relative direction
STRIDE = 12                                       # one sample per stock per hour, for every horizon
ARCHS = ["plain", "series1d"]                     # add "residual" if you have GPU time to spare
CELLS = grid.grid_cells(HORIZONS, KINDS, STRIDE)

RESUME = RUN_ID is not None
RUN_ID = RUN_ID or ("local_smoke" if LOCAL_TEST else datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))
RUN_DRIVE, RUN_LOCAL = HERE / "runs" / RUN_ID, LOCAL / "run"
if RESUME:
    assert RUN_DRIVE.exists(), f"{RUN_DRIVE} not found"
else:
    assert LOCAL_TEST or not RUN_DRIVE.exists(), "RUN_ID exists: set RUN_ID to resume it"
RUN_DRIVE.mkdir(parents=True, exist_ok=True); RUN_LOCAL.mkdir(parents=True, exist_ok=True)
save_json(RUN_LOCAL / ("settings_resume.json" if RESUME else "settings.json"),
          dict(settings=SETTINGS, cells=CELLS, archs=ARCHS, chartcnn=chartcnn.__version__), RUN_DRIVE)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
""")

md("## 3. Models\nLayer-by-layer output shapes and parameter counts. Output layers start at zero, so every initial prediction is exactly 0.5.")
code("""
store = chartcnn.Store(PREPARED)
sample, _ = chartcnn.build_split(store, "train", "vol", 1, STRIDE)
for arch in ARCHS:
    data = chartcnn.make_dataset(PREPARED, sample.head(4), arch)
    x = torch.stack([data[i][0] for i in range(4)])
    model = chartcnn.build_model(dict(SETTINGS, arch=arch))
    table, total, trainable = chartcnn.parameter_table(model)
    print(f"{arch}: {total:,} parameters ({trainable:,} trainable)")
    display(table)
    print("initial P(y=1):", model.eval()(x).softmax(1)[:, 1].tolist(), "\\n")
del model
""")

md("""
## 4. Labels
Sample counts and positive shares per cell. Labels use unadjusted prices; a label whose horizon contains an ex-date is dropped, and `dir` drops zero returns (per cell). The validation month (April 2023) was calmer than the training months, so the `vol` positive share is below 0.5 there; `dir` shares drift with the market. AUC is not affected by either; log loss is.
""")
code("""
OVERVIEW = grid.label_overview(store, CELLS)
display(OVERVIEW.round(3)); OVERVIEW.to_csv(RUN_LOCAL / "labels.csv")
""")

md("""
## 5. Train the grid
Per cell: labels → HGB and rules → one CNN per architecture → validation predictions saved to `runs/<RUN_ID>/<cell>/`. Watch `val AUC`, the log-loss gain vs prior (must be > 0 for a useful probability) and `prob std` (≈ 0 means the CNN collapsed to a constant).

Expected time on an L4 (measured on the earlier ChartCNN run at the same stride, without bfloat16): about 2–2.5 min per plain-CNN epoch and 3–6 epochs per cell, so roughly 10–20 min per cell and **2.5–3.5 hours for all 12 cells**; `series1d` and HGB add about 2 min per cell. bfloat16 should make it shorter.
""")
code("""
import time
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = grid.run_cell(PREPARED, cell, SETTINGS, ARCHS, DEVICE, RUN_LOCAL / cell["name"],
                                          RUN_DRIVE / cell["name"], smoke=SMOKE)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
""")

md("""
## 6. Analysis
AUC per cell and model with 95% intervals from resampling whole validation days (frames on one day share bars and market moves, so resampling images would overstate precision).
""")
code("""
TABLE, BOOTS = grid.grid_table(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_LOCAL / "grid_table.csv")
for model in ["CNN (plain)", "CNN (series1d)", "HGB hand features"]:
    if model in TABLE.index.get_level_values("model"):
        print(model); display(grid.auc_matrix(TABLE, model))
display(TABLE.drop(columns=["kind", "horizon"]).round(4))
""")
code("""
grid.plot_grid(TABLE, RUN_LOCAL / "auc_by_horizon.png")
TAKEAWAYS = grid.takeaways(TABLE, BOOTS)
(RUN_LOCAL / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
for name in ("labels.csv", "grid_table.csv", "auc_by_horizon.png", "takeaways.md"):
    shutil.copy2(RUN_LOCAL / name, RUN_DRIVE / name)
print("saved to", RUN_DRIVE)
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", sys.argv[1])
