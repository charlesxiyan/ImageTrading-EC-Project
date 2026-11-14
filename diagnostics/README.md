# Baseline integrity diagnosis

The original Colab run is preserved. These diagnostics use training and
validation inputs only; they do not produce another test-set evaluation.

Run from the project root:

```bash
.venv/bin/python -m diagnostics.check_baseline \
  --checkpoint artifacts/cnn78u_5min_h1/colab_best.pt \
  --output artifacts/baseline_diagnostics_new
```

Use a new output directory for each run. The checkpoint must be your own trusted
project checkpoint: it contains Python and NumPy RNG state and requires full
PyTorch checkpoint deserialization. The script verifies the prepared-data
fingerprint before evaluation. Test files are hashed for provenance only.

The diagnostic performs:

1. Seeded, uniform sampling of 512 training and 512 validation examples.
2. Independent next-close label checks, image uniqueness checks, and fixed-model
   evaluation with probability range, standard deviation, and unique values.
3. Per-layer activation inspection on 16 training examples.
4. A disposable checkpoint copy with batch statistics and dropout disabled,
   followed by a backward pass to inspect gradient flow.
5. A fresh model fitted to 64 fixed training examples for at most 300 full-batch
   optimizer updates. Dropout and weight decay are disabled for this capacity
   check. The architecture, labels, renderer, and cross-entropy loss are retained.
   It stops when eval-mode accuracy reaches 98% and eval loss is below 0.1.

The memorization result is not out-of-sample performance. Disabling two forms of
regularization together does not identify which caused a full-run failure.
A failed memorization check is inconclusive about ultimate model capacity; it
requires further optimization diagnosis. A passing check demonstrates basic
fit capability, not that the baseline hyperparameters are appropriate.

Reports are written beneath the chosen output directory. `checkpoint_report.json`
is saved before the longer memorization experiment; `report.json` adds its final
history and parameter-update norms. Indices and probabilities are saved for
reproduction. Baseline weights and source files are never updated by this script.

The first real-checkpoint findings are documented in
[BASELINE_DIAGNOSIS.md](BASELINE_DIAGNOSIS.md).
