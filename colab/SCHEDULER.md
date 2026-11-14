# CNN78u scheduler experiment

Upload `CNN78u_Scheduler_20pct.ipynb` beside the existing
`cnn78u_colab_bundle.zip` in `MyDrive/Colab Notebooks/EC/Project/Demo`.
Only the new notebook needs uploading. Select a GPU runtime and run cells in order.
The helper source is embedded in the notebook. Outputs go into
`Demo/scheduler_runs/<EXPERIMENT_ID>`; earlier results are preserved.

## Frozen protocol

- Original four-block CNN78u, Xavier uniform initialization, dropout 0.5.
- Fixed seed-69 20% training subset: 344,165 eligible windows, with no balancing.
  The subset is identical to the earlier repair notebook. Prior index files are
  checked for exact agreement when present with matching sampling/data contracts.
- Fresh initialization; Adam, learning rate 0.001, weight decay 0, batch size 64.
- ReduceLROnPlateau on complete validation log loss: factor 0.5, patience 2,
  absolute threshold 0.00001, cooldown 0, minimum learning rate 0.00001.
- Maximum 15 epochs (80,670 updates if all complete). Early stopping after six
  consecutive epochs without a validation-loss improvement exceeding 0.00001.
- Save the checkpoint with the actual lowest validation log loss. This best-model
  rule is distinct from the tolerance used to reset plateau/early-stop counters.
- No test evaluation, residual network, Kaiming change, or architecture search.

PyTorch scheduler patience 2 permits two bad epochs and reduces the learning rate
on the third consecutive bad epoch. The new rate applies to the next epoch. Early
stopping permits training after the first reduction, but a later reduction can
coincide with stopping. Both the rate used and the next scheduled rate are logged.

This changes both schedule and maximum budget relative to the previous three-epoch
run; it cannot isolate the scheduler's causal contribution. Fifteen epochs are a
cap, not an instruction to train beyond early stopping. Runtime can be several
hours because the complete validation set is evaluated every epoch.

## Monitoring and recovery

Every 500 updates and at epoch end, record fixed train/validation probe loss,
accuracy, probability range/dispersion, feature-weight magnitudes, BatchNorm scales,
ReLU activity, and gradient norms. Probes run in eval mode without changing running
statistics. Their small sample size makes them diagnostic rather than substitutes
for full validation. Timing, throughput, optimizer updates and peak allocated GPU
memory are recorded. Numerical-collapse and nonfinite checks remain enabled.

After a disconnect, set `RESUME=True` and set `EXPERIMENT_ID` to the original ID,
then rerun from the beginning. Keep all settings and code unchanged. The checkpoint
restores optimizer, scheduler, early-stop counters, RNG and epoch history. A partly
finished epoch is repeated. Completed runs return cached results. Use a new ID for
changed settings. Checkpoints must be your own trusted project files.

The output includes a sampling contract, fixed row indices, helper source,
`history.csv`, `scheduler_diagnostics.png`, validation metrics/predictions and
baseline plots, `best.pt`, `last.pt`, health JSON and `scheduler_results.zip`.
A numerical failure additionally writes `stopped.pt`.

## Validation

Local QA executes every notebook code cell using tiny CPU training/validation
subsets. A forced plateau tests a disconnect after learning-rate reduction and
verifies that resumed weights exactly match uninterrupted CPU training. Unit tests
check patience semantics, minimum LR, improvement tolerance and saved controller
state. This is not a full Colab GPU training run and provides no performance claim.
