# CNN78u baseline diagnosis — November 12, 2026

## Verdict

The saved baseline has suffered feature-extractor collapse. Its recorded accuracy
is a valid description of this failed training run, but it should not be treated
as evidence that a functioning CNN has no predictive signal. Preserve it as the
initial failed run. Establish a healthy training baseline before attributing
improvements to a new architecture.

## Evidence from the actual Colab checkpoint

Source run: `cnn78u_h1_baseline_20261112_065607_utc`, `best.pt`, completed epoch 3,
80,664 optimizer steps. The checkpoint is not a smoke checkpoint. Its data
fingerprint exactly matches the local prepared dataset. The original Drive
checkpoint and notebook were not modified.

On 512 uniformly sampled training examples and 512 independently sampled
validation examples (seed 69):

| Check | Training | Validation |
|---|---:|---:|
| Distinct rendered images | 512 | 512 |
| Independent next-close label mismatches | 0 | 0 |
| Distinct predicted probabilities | 1 | 1 |
| Predicted probability of up | 0.5109832286834717 | 0.5109832286834717 |
| Probability standard deviation | 0 | 0 |
| Predicted-up fraction | 100% | 100% |

These sampled accuracies are not additional generalization estimates. The
purpose is to diagnose dependence on input and basic data integrity.

All four convolution weight tensors have maximum absolute values of approximately
6.3e-41 to 5.7e-40. BatchNorm scaling parameters are also approximately 1e-40.
Float64 norm calculations confirm the magnitudes rather than relying on
underflow-prone float32 norms. All inspected weights are finite.

On 16 training examples, early convolution outputs are numerically zero and the
later representations do not vary across examples. Switching a disposable copy
to batch statistics with dropout disabled still produces identical predictions.
The convolution gradient norms are numerically zero, while parts of the final
head still receive gradients. This is not merely a 0.5 threshold issue or a
train/eval BatchNorm discrepancy.

Fourteen existing window, label, image, and model unit tests passed. The sampled
label check independently compares the next adjusted close to the current close.
It does not revalidate all upstream WRDS records, corporate-action inputs, or
historical index membership.

## Cause: established facts versus hypothesis

Feature-extractor collapse is established from the checkpoint. The sequence of
events that caused it is not established: per-step weight and activation histories
were not recorded. Adam uses a nonzero weight-decay coefficient on all parameters,
including BatchNorm scaling and biases. Shrinkage after the supervised gradient
weakens is consistent with the final near-zero tensors. A controlled run is needed
to distinguish the roles of weight decay, learning rate, ReLU dynamics, dropout,
and signal strength. The final checkpoint alone cannot isolate a root cause.

## Fresh-model memorization control

The same four-block architecture, freshly initialized with seed 69, was trained
on 64 fixed real training examples using full-batch Adam at learning rate 0.001.
Dropout and weight decay were disabled specifically for this capacity control.
After 100 updates, eval-mode accuracy on those same 64 examples was **100%** and
cross-entropy was **0.04023**, passing the predeclared 98% / 0.1 criteria. The
experiment took approximately 179 seconds on local CPU (PyTorch 2.15.0).

This demonstrates basic fit capability with the existing renderer, labels, model,
and backward path. It does not validate the original regularization settings or
establish out-of-sample performance. Changing dropout and weight decay together
does not isolate the cause of the full-run collapse.

Machine-readable evidence: `artifacts/baseline_diagnostics/report.json`.
The original Colab checkpoint is stored separately at
`artifacts/cnn78u_5min_h1/colab_best.pt`. No baseline training parameters were edited.

## Next experiments

Keep the four-block architecture initially. Add gradient, weight, activation and
probability-dispersion monitoring. Compare the original optimizer configuration
with a controlled regularization change using the same training subset, seed,
batch size, and validation examples. Change one factor at a time. Do not combine
a new architecture and a new retention rate and attribute the gain to either one.

The original held-out test has already been inspected. Use training and validation
for diagnosis and development, and reserve a new untouched period for final
independent assessment. Passing a memorization control does not establish market
predictability or profitability.
