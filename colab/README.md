# CNN78u Colab baseline

Upload these two files directly to Google Drive:

```text
MyDrive/Colab Notebooks/EC/Project/Demo/
  CNN78u_Colab_Baseline.ipynb
  cnn78u_colab_bundle.zip
```

Open the notebook in Colab, choose a GPU runtime, and run the cells in order. The ZIP already includes the current source code and the validated **78-input / 1-next-bar** prepared dataset. Do not upload `.venv`, old artifacts, all pre-generated images, or WRDS credentials. Raw bars are not needed for this baseline run because preparation has already been completed.

The notebook also accepts an expanded bundle with `us_pipeline/` and `prepared/`, or a full project tree with the current `artifacts/cnn78u_5min_h1/prepared/` directory. The provided ZIP is the simplest option and includes upload checksums.

## Frozen baseline

- One CNN78u, four convolutional blocks, 655,234 parameters, input image 234x234 with volume.
- Input: 78 complete scheduled five-minute bars. Target: adjusted close of the immediately following bar versus the current close. Neutral returns excluded.
- All 1,720,827 training samples and 509,964 validation samples; final test has 1,187,989 samples.
- January-March train, April validation, May-June test. Cross-boundary targets are purged.
- Default: 3 epochs maximum, batch size 64, 2 loader workers, Adam, learning rate 0.001, weight decay 0.0001, seed 69, full precision.
- Validation log loss selects the checkpoint; threshold stays 0.5. No balancing, SMOTE, imputation, or class weighting.
- The final test cell is enabled by default after validation selection. Freeze settings first and do not use test outcomes for further selection.
- `SMOKE_MODE=True` is only a tiny code check and cannot produce formal test results.

The next scheduled bar after the closing bar belongs to the following trading session. Overnight/holiday targets are included under the current policy and reported separately from intraday targets.

## Evaluation

The notebook records accuracy, balanced accuracy, up-class precision/recall/F1, macro F1, MCC, ROC-AUC, positive-class average precision (AP), log loss, Brier score, and binary probability calibration/ECE. AP is not trapezoidal PR area.

Baselines are the training-prior constant probability/majority class and input-only last-bar/lookback directional rules. The hard rules do not claim calibrated probabilities. The report includes confusion, ROC/PR, calibration, daily stability, and confidence/coverage plots plus per-month, per-security, and intraday/overnight tables.

Accuracy differences against the training-majority rule include an approximate paired moving-block bootstrap interval using five contiguous decision-day blocks, retaining all stocks/windows in each selected day. It is not an IID-window interval and does not capture model-selection/training uncertainty. Confidence thresholds are descriptive, not test-selected trading policies.

Timing distinguishes setup/staging/audit, per-epoch data-to-optimizer time, validation/test end-to-end inference, checkpoint copying, and warmed model-only latency/throughput. GPU measurements synchronize CUDA. Batch-1 latency and amortized batched latency are different quantities. GPU peak allocated/reserved memory, main-process RSS, hardware, package versions, precision, and data fingerprints are saved. Worker memory is not included in main-process RSS.

## Persistence and recovery

Data are copied from Drive to Colab's local VM disk before training. Best/last checkpoints, history, validation results, and status are mirrored back after completed epochs. Results are saved under:

```text
Demo/runs/<RUN_ID>/
  run_plan.json
  best.pt
  last.pt
  history.csv
  training_report.json
  training_curves.png
  hardware_benchmark.json
  validation/
  test/
  results.zip
  summary.json
```

If interrupted, set `RESUME=True` and reuse that run's ID. Resume is at an epoch boundary; unfinished work within an epoch is repeated. Keep data/code/optimizer/batch/seed settings unchanged. Epoch count may be extended before test evaluation. Once test is evaluated, training for that run is locked. An unchanged repeated test cell returns its existing report; a changed checkpoint cannot overwrite it.

Full training can take substantial time and actual GPU availability varies. The notebook makes no promise about free-tier session duration or model accuracy. Seeded runs are not guaranteed bitwise identical across GPU/library versions. Proportional dividend-adjusted chart returns are not an exact reinvested total-return index; classification scores do not establish a tradable strategy.

## Rebuild and local verification

```bash
cd ImageTrading-EC-Project
.venv/bin/python colab/create_notebook.py
.venv/bin/python colab/build_colab_bundle.py
.venv/bin/python -m unittest discover -s tests -v
```

The unexecuted notebook is the deliverable. Repository QA executes its local-test branch with a tiny training/validation subset and skips the real held-out test. The experiment tests use fictional data to verify resume, final-test reports, and test-result locking. Local CPU QA is not a Colab GPU benchmark.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [scikit-learn evaluation definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).
