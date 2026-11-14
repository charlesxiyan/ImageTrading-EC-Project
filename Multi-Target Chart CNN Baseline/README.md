# Multi-Target Chart CNN Baseline

Is there a signal in the **direction** of the next intraday move, or only in its **size**?
This folder trains the chart CNN on a grid of 3 target kinds × 4 horizons (12 cells) and plots
validation AUC with day-bootstrap intervals against the horizon.

| kind | label | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|---|
| `dir` | return over the horizon strictly positive (zero returns dropped) | `dir_h1` | `dir_h3` | `dir_h6` | `dir_h12` |
| `vol` | \|return\| above the stock's training-period median \|return\| for that horizon | `vol_h1` | `vol_h3` | `vol_h6` | `vol_h12` |
| `xsdir` | return above the cross-sectional median at the same bar | `xsdir_h1` | `xsdir_h3` | `xsdir_h6` | `xsdir_h12` |

5 min (`h1`) is kept as the reference that the earlier ChartCNN runs used.

## Design

- **Data**: bundle v2 (2023 H1, ~528 stocks), train to 2023-03-31, validate on
  April 2023. Test (May–June) is not touched. v2 has the same bars as v1 but keeps every window:
  v1 had dropped windows whose next-bar return was exactly zero (4.6% of them), which selects on
  future data. Direction targets now drop zero returns themselves, per target and horizon.
- **Labels** use unadjusted (traded) prices, as the price-adjustment policy requires. A label whose
  horizon contains an ex-date (only possible across the overnight gap) is dropped, because its
  unadjusted return would include the split or dividend jump: 0.02% of windows at 5 minutes,
  0.2% at 60 minutes.
- **Inputs**: one trading day of 78 five-minute bars, adjusted point in time to the window's last bar
  (`chartcnn.images.point_in_time_window`), drawn as a 234×234 chart or fed as raw OHLCV numbers.
- **Stride 12 for every horizon**: one sample per stock per hour (~143k train / ~42k validation per
  cell). Stride ≥ horizon, so one stock's labels never overlap, and all 12 cells share the same
  decision times, so a change across horizons is not a change in sample.
- **Models per cell**: `plain` image CNN (776,450 parameters), `series1d` 1D CNN on raw OHLCV
  (720,770 parameters), HistGradientBoosting on 20 hand features (tree-node count printed), and
  simple rules. Every model prints its parameter count. The residual CNN is left out to save GPU time
  (in `run_20261114_020923` it tied the plain CNN at 2.6× the cost); add `"residual"` to `ARCHS` to include it.
- **Training**: same as ChartCNN_Baseline (Adam lr 1e-5, batch 128, up to 12 epochs, early stop on
  validation log loss with patience 2, seed 69), plus bfloat16 autocast on A100/L4/H100 (`amp=True`).
- **Analysis**: AUC per cell and model with 95% day-bootstrap CI, log-loss gain vs the training
  prior, decile long-short for the direction kinds, paired image-vs-1D and CNN-vs-HGB comparisons,
  and auto-generated takeaways.

## Run (Google Colab)

1. Drive layout:
   - `Colab Notebooks/EC/Project/Demo/cnn78u_colab_bundle.zip` (the existing data zip). The notebook
     turns it into v2 at start-up (about 30 seconds: same bars, all windows kept). If
     `cnn78u_colab_bundle_v2.zip` is in `Demo/`, that is used instead.
   - `Colab Notebooks/EC/Project/Demo/Multi-Target Chart CNN Baseline/` (this folder: notebook, `chartcnn/`)
2. Open `Multi_Target_Chart_CNN.ipynb`, Runtime → Change runtime type → GPU (L4 or A100), **Run all**.
3. The run id is printed in section 2. Outputs go to `runs/<RUN_ID>/`.
4. **After a disconnect**: set `RUN_ID = "<that id>"` in section 2 and Run all again. Finished cells
   are loaded from Drive; the unfinished one continues from its last saved epoch.

Estimated time on an L4 without bfloat16, from the earlier run at the same stride: about 2–2.5 minutes
per plain-CNN epoch and 3–6 epochs per cell, so 10–20 minutes per cell and about **2.5–3.5 hours** for
all 12 cells. `series1d` and HGB add about 2 minutes per cell. bfloat16 should make it shorter.

## Outputs (`runs/<RUN_ID>/`)

| file | content |
|---|---|
| `settings.json` | settings, cells, architectures, chartcnn version |
| `labels.csv` | samples and positive share per cell and split |
| `<cell>/predictions.parquet` | validation labels, returns and every model's score |
| `<cell>/done.json` | parameters, best epochs, prior, minutes taken |
| `<cell>/<arch>/` | `best.pt`, `last.pt`, `history.csv` |
| `grid_table.csv` | one row per (cell, model): AUC and CI, log-loss gain, long-short, parameters |
| `auc_by_horizon.png` | AUC vs horizon per kind, and the best model's AUC − 0.5 for all kinds |
| `takeaways.md` | generated conclusions |

## Code

The grid logic is `chartcnn/grid.py`. In the repository, `chartcnn/` lives once, in `../CNN-Relevant/`,
and the notebook falls back to it. On Drive, a copy of `chartcnn/` sits in this folder so that the
`CNN-Relevant` folder there is not changed. `tools/make_notebook.py` regenerates the notebook.
