# CNN78u: five-minute US equity image pipeline

The active pipeline has **one model, CNN78u**, and one five-minute configuration: `configs/us_5min.json`. Legacy A-share names and the thirty-minute model/configuration have been removed from the active pipeline. The original A-share repository and older ignored research artifacts are preserved.

## Input interval count and prediction horizon

A normal US regular session runs from 09:30 to 16:00 New York time: 390 minutes / 5 minutes = **78 intervals**. There is no extra opening observation, so the answer is 78 rather than 79. CNN78u's suffix denotes input bar count, not neurons, feature channels, or network depth.

A rolling input contains the most recent 78 consecutive scheduled trading bars. It has one normal session's worth of observations but need not start at the opening bell; a mid-session prediction can look back across the previous overnight closure. The renderer assigns three columns per bar, producing a 234x234 grayscale image with an optional volume band.

The selected prediction horizon is **H=1**: use the past 78 completed bars to predict the immediately following bar. For input bars 1 through 78, the target compares bar 79's adjusted close against bar 78's adjusted close. The label is 1 for an increase and 0 for a decrease; exact ties are excluded. No later bar may substitute for a missing bar 79.

During a session, the target ends five minutes after the input endpoint. The existing cross-session policy is retained: after the 16:00 closing bar, the next scheduled bar is the following trading session's 09:30-09:35 bar. That target includes the overnight move and is not a five-minute wall-clock return. Prediction length is independent of the name CNN78u, whose suffix describes only the input.

The current adapter verifies 2023 H1 full sessions only. Early-close sessions in another period require calendar changes; do not assume every US session has 78 intervals.

## Workflow

```text
Dividend-adjusted five-minute OHLCV, keyed by PERMNO
  -> verify source checksums and scheduled trading grid
  -> align all securities without compressing missing time slots
  -> require complete 78-bar lookbacks and complete H-bar future intervals
  -> assign chronological splits and purge crossing return horizons
  -> save numeric memmaps and sample endpoint indices
  -> render input-only OHLC/volume images on demand
  -> train CNN78u with unweighted cross-entropy
  -> choose checkpoints using the fixed validation period
  -> explicitly evaluate the held-out test set
```

Missing windows are discarded rather than filled. Days without index membership remain gaps in the dense grid. Ticker changes do not split a security's history because grouping uses PERMNO. No imputation, interpolation, SMOTE, class weighting, balanced sampler, or random 20% retention is used. Normal training shuffling is uniform and does not change class frequencies.

The original `KNNImputer` and `SMOTE` references were imports only; no active calls were found. The A-share fixed daily-limit branches and their special image placement have been removed. US LULD bands and trading halts still exist. Empty-bar screening does not identify every partial-session halt; explicit halt annotations would need a separate source. [Nasdaq LULD information](https://www.nasdaqtrader.com/trader.aspx?id=currentregulatory).

## Architecture and depth

CNN78u defaults to four convolutional blocks:

```text
[batch,1,234,234]
  Conv(1,32,5x3) -> BN -> ReLU -> MaxPool    : 32x117x117
  Conv(32,64,5x3) -> BN -> ReLU -> MaxPool   : 64x59x59
  Conv(64,128,5x3) -> BN -> ReLU -> MaxPool  : 128x30x30
  Conv(128,256,5x3) -> BN -> ReLU -> MaxPool : 256x15x15
  AdaptiveAvgPool(4x4) -> Flatten -> Dropout(0.5) -> Linear(4096,2)
  Output: two raw logits; Softmax is applied only for probabilities
```

There are **655,234 trainable parameters**. Adaptive pooling retains coarse spatial locations while avoiding a large dense layer. Merely expanding the old two-block/flatten model to 234x234 would produce approximately 985,602 parameters; that is a capacity calculation, not an experimental comparison. Both architecture depth and the head have changed, so no predictive improvement can be attributed to depth without an ablation.

The theoretical interior receptive field before the adaptive head is 76x46 pixels with four blocks, compared with 16x10 pixels with two. Thus deeper local features can combine longer patterns. This does **not** mean a two-block model is unable to use the full input: its dense head can already combine features across the entire chart. Greater depth may improve representation but can also lose fine detail, increase optimization cost, or overfit correlated windows. Performance must be established on validation, not assumed from sample count or depth.

Only one model class is active. Its `model.channels` list controls block depth for a later controlled ablation, with at most four blocks. The default is `[32,64,128,256]`. Any comparison should keep the same input, labels, splits, training budget, and evaluation procedure; channel widths/head capacity should also be considered. Test data must not select the depth. Model settings can change without rebuilding the same sample indices, but each training run requires a fresh run output directory.

## Samples, labels, and preprocessing

A sample ending at completed bar t uses input `[t-77,t]`. Its decision timestamp is `bar_end[t]`. Its label is:

```text
return = adjusted_close[t+H] / adjusted_close[t] - 1
label = 1 if return > 0; 0 if return < 0
```

Every input and future interval must be observed, finite, positive, and have valid OHLC geometry and eligible volume. Missing bars are not deleted before constructing windows. Exactly neutral returns are excluded and counted; `zero_return_epsilon` defaults to zero. The return horizon is exactly H, not H+1. This target is a dividend-adjusted chart-price return, not an exact reinvested total return or an executable trading return.

The default permits normal session closures in trading-bar time. A fully same-session 78-bar input plus a positive future horizon cannot fit into one 78-bar session, so `allow_cross_session=false` requires shorter window settings and is not supported by the current one-day input configuration.

Preprocessing uses a shared OHLC low/high coordinate axis and a separate volume maximum from the input window alone. It does not estimate global parameters from validation/test data. Adjusted prices and volumes are first rebased by the input endpoint's respective multiplier to cancel arbitrary end-of-dataset scale. No future target observation enters the image. This cancellation does not turn revised vendor data into an as-published historical archive. BatchNorm running statistics are learned only during training and frozen during evaluation.

The default image includes volume. Flat price windows render at a stable center level without A-share limit-specific rules. Input images are converted to float32 and divided by 255. No rotations, flips, or generic image augmentation are applied.

## Chronological split

All securities use the same boundaries: January–March 2023 for training, April for validation, and May–June for test. Training labels cannot enter validation; validation labels cannot enter test. Validation/test lookbacks can use preceding history that was already available at prediction time. No random window-level train/validation split is used.

Training uses unweighted CrossEntropyLoss on raw logits. Checkpoints minimize validation loss. Validation uses the complete fixed validation set after each epoch by default. A positive `training.validate_every_batches`, such as 500, adds intra-epoch checks and may be expensive. Patience counts validation checks. After every check, the model returns to training mode. Training parameters/seed can change without regenerating the sample indices.

## Commands and outputs

```bash
cd ImageTrading-EC-Project

# Dependency list for a new environment; the local .venv is already configured.
python -m pip install -r requirements-pipeline.txt

# Already prepared in this workspace. Existing prepared outputs are protected.
.venv/bin/python -m us_pipeline prepare --config configs/us_5min.json
.venv/bin/python -m us_pipeline audit --config configs/us_5min.json
.venv/bin/python -m us_pipeline preview --config configs/us_5min.json

# Short connectivity check, not a performance estimate; use a fresh smoke output.
.venv/bin/python -m us_pipeline train --config configs/us_5min.json --smoke

# Full training has not been performed.
.venv/bin/python -m us_pipeline train --config configs/us_5min.json

# Explicit final evaluation only after model/parameter selection.
.venv/bin/python -m us_pipeline test --config configs/us_5min.json \
  --checkpoint artifacts/cnn78u_5min_h1/training/best.pt
```

The default artifact root is `artifacts/cnn78u_5min_h1`, separate from all earlier horizon-50 and legacy artifacts. Numeric memmaps, split Parquets, source fingerprints, screening counts, and audits are in `prepared/`. Examples are in `preview/`, and short validation runs in `smoke/`. Full training writes `training/best.pt` and a report. Final test evaluation writes probability rows and metrics to `test_evaluation/` only when explicitly requested, and rejects smoke checkpoints.

Do not reuse old sample indices or checkpoints from another input length. Change the artifact root when changing data/window/label settings. Existing outputs are not overwritten. Data parameters are verified against prepared metadata; checkpoint and prepared-data fingerprints are checked during test evaluation.

## Current preparation results (L=78, H=1)

| Split | Samples | Label 0 | Label 1 | Positive fraction |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,720,827 | 848,755 | 872,072 | 50.68% |
| Validation | 509,964 | 250,143 | 259,821 | 50.95% |
| Test | 1,187,989 | 582,311 | 605,678 | 50.98% |

A total of 3,418,780 samples remain. Sequential exclusions are 1,478,373 invalid input windows, 4,158 invalid next-bar targets, 755 boundary-crossing labels, and 163,566 neutral returns. These are candidate-window counts on an all-security trading grid, including nonmembership gaps, not counts of distinct missing bars. A missing next bar rejects the sample even when a later bar exists. Mild measured imbalance currently does not motivate class resampling. Overlapping samples and common market moves remain correlated; millions of samples do not imply millions of independent observations.

Tests cover the full-session count, legacy-config rejection, missing-slot preservation, exact horizon indexing, chronological purging, label signs, neutral returns, shared OHLC geometry, scale invariance, flat windows, CNN78u tensor shape, finite gradients, and configurable depth. The real-data audit and smoke report record actual verification status. A smoke run does not establish accuracy, and test data have not selected the architecture. Quote-based outlier cleaning, explicit halt annotations, executable-price assumptions, transaction costs, and strategy evaluation remain separate work.

```bash
.venv/bin/python -m unittest discover -s tests -v
```
