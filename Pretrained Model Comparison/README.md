# Pretrained Model Comparison

Zero-shot (and lightly fine-tuned) time-series foundation models vs the chart CNN on the same
5-minute data, splits and targets as `CNN-Relevant/ChartCNN_Baseline.ipynb`.

![flow](assets/flow.png)

## Run (Google Colab)

1. Drive layout (`Colab Notebooks/EC/Project/Demo/`):
   - `cnn78u_colab_bundle.zip` (data, already there)
   - `CNN-Relevant/chartcnn/` (shared data/eval code, already there)
   - `Pretrained Model Comparison/` (this folder: notebook + `tsfm_compare/`)
2. Optional but recommended: run `CNN-Relevant/ChartCNN_Baseline.ipynb` first. The newest run in
   `CNN-Relevant/runs/` is picked up automatically and added to the table.
3. Open `Pretrained_Model_Comparison.ipynb`, choose a GPU runtime, **Run all**.
4. Results: `Pretrained Model Comparison/runs/<RUN_ID>/comparison.csv`, `comparison.png`, `results.zip`.
   After a disconnect, set `RESUME_RUN_ID` in section 3 and run all again; finished forecasts are reused.

## Models

| name | Hugging Face id | input | notes |
|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` | close | also light LoRA fine-tune (section 6) |
| Chronos-Bolt base | `amazon/chronos-bolt-base` | close | |
| TimesFM 3.0 | `google/timesfm-3.0-pytorch` | close | non-commercial weights; falls back to 2.5 (Apache-2.0) |
| Moirai 2.0 small | `Salesforce/moirai-2.0-R-small` | close | installed with `--no-deps` |
| Kronos small | `NeoQuasar/Kronos-small` | OHLCV candles + timestamps | cloned from GitHub; 8 sampled paths |

Each model prints its total parameter count when loaded, and the table has a `parameters` column.

## How a forecast becomes a score

- Input: the last 78 closes (the same day the CNN sees as a chart), divided by the last close.
- Output: 12-step quantile forecast (10 %, 20 %, …, 90 %).
- `vol_h1`: expected |return| of the next bar = mean over quantiles of |q − 1|, divided by the
  stock's training median |return| (the label threshold).
- `xsdir_h12`: median forecast return over 12 bars minus its cross-sectional median at that bar.
- AUC uses the raw score (pure zero-shot). For log loss, a 1-feature logistic regression is fitted
  on a training-period calibration sample and applied unchanged to validation.

## Layout

| Path | Content |
|---|---|
| `Pretrained_Model_Comparison.ipynb` | install → setup → config → samples → models → fine-tune → baselines/CNN → table → save |
| `tsfm_compare/samples.py` | Shared samples (via `chartcnn.build_split`), timestamp subsampling, model inputs |
| `tsfm_compare/forecasters.py` | One wrapper per model: load, parameter count, quantile forecast, Chronos-2 fine-tune |
| `tsfm_compare/scoring.py` | Forecast → score, calibration, baseline and CNN rows, comparison table |
| `tsfm_compare/runner.py` | Runs a model with forecast caching (Drive) |
| `tsfm_compare/plots.py` | AUC bar chart, forecast examples |
| `tools/` | Regenerate `assets/flow.png` and the notebook |

## Rebuild

```bash
python "Pretrained Model Comparison/tools/make_diagram.py"
python "Pretrained Model Comparison/tools/make_notebook.py"
```

## Results: run `run_20261114_012926` (Colab GPU)

Same 2023 H1 bundle and April 2023 validation month as `CNN-Relevant`, but scored on a subsample of
80 validation timestamps (stride 6; about 27k samples per target) so the large models fit in a
Colab session. Calibration uses 40 training-period timestamps. The CNN rows are read from the
CNN run `run_20261113_185632` and rescored on the same subsample, so their AUCs are not the ones in
`CNN-Relevant/README.md` (that table uses every validation sample). Settings are in
`results/run_20261114_012926/settings.json`.

| target | model | kind | parameters | val AUC | log loss vs prior |
|---|---|---|---|---|---|
| `vol_h1` | HGB hand features | baseline | 18,300 tree nodes | 0.734 | +14.6% |
| `vol_h1` | ChartCNN series1d | CNN (trained) | 720,770 | 0.722 | +12.7% |
| `vol_h1` | ChartCNN plain | CNN (trained) | 776,450 | 0.719 | +13.9% |
| `vol_h1` | ChartCNN residual | CNN (trained) | 2,047,362 | 0.718 | +12.1% |
| `vol_h1` | Kronos small | zero-shot | 28,699,418 | 0.699 | +11.1% |
| `vol_h1` | Chronos-2 | zero-shot | 119,477,664 | 0.615 | +5.8% |
| `vol_h1` | Chronos-2 + LoRA | fine-tuned | 120,684,576 | 0.611 | +5.8% |
| `vol_h1` | rule: recent abs return | baseline | 0 | 0.611 | |
| `vol_h1` | Moirai 2.0 small | zero-shot | 11,387,208 | 0.609 | +5.2% |
| `vol_h1` | TimesFM | zero-shot | 330,710,976 | 0.605 | +5.6% |
| `vol_h1` | Chronos-Bolt base | zero-shot | 205,292,928 | 0.594 | +5.0% |
| `xsdir_h12` | Chronos-2 | zero-shot | 119,477,664 | 0.506 | +0.0% |
| `xsdir_h12` | HGB hand features | baseline | 18,300 tree nodes | 0.506 | -0.2% |
| `xsdir_h12` | TimesFM | zero-shot | 330,710,976 | 0.501 | -0.0% |
| `xsdir_h12` | Chronos-Bolt base | zero-shot | 205,292,928 | 0.498 | -0.0% |
| `xsdir_h12` | Chronos-2 + LoRA | fine-tuned | 120,684,576 | 0.493 | -0.1% |
| `xsdir_h12` | Moirai 2.0 small | zero-shot | 11,387,208 | 0.487 | -0.0% |
| `xsdir_h12` | Kronos small | zero-shot | 28,699,418 | 0.480 | -0.2% |

Takeaways:

- **No foundation model beats the small trained models on `vol_h1`.** Kronos, the only one that
  sees full OHLCV candles, comes closest (0.699). The close-only models (Chronos, TimesFM, Moirai)
  are no better than the one-line rule "recent absolute return" (0.611), despite 11M-331M parameters.
- **LoRA fine-tuning of Chronos-2 did not help** (0.611 vs 0.615 zero-shot).
- **`xsdir_h12`: every model is at 0.5.** Nothing, trained or pretrained, predicts 12-bar
  cross-sectional direction.

Files in `results/run_20261114_012926/`: `comparison.csv` (all metrics, including accuracy,
MAE vs random walk, run time and decile long-short), `comparison.png`, `settings.json`,
`chronos2_ft_params.json`. The executed notebook keeps its outputs.

Not in git (stay on Drive, in `Colab Notebooks/EC/Project/Demo/Pretrained Model Comparison/runs/run_20261114_012926/`):
cached forecasts (`forecasts/*.npy`) and `results.zip` (about 74 MB).
