# CNN78u training repair notebook

Upload `CNN78u_Training_Repair_20pct.ipynb` to the existing Google Drive folder
`Colab Notebooks/EC/Project/Demo`, alongside the existing
`cnn78u_colab_bundle.zip`. No new data archive is required. The diagnostic helper
is embedded in the notebook and written to temporary runtime storage.

Select a GPU runtime and run the cells in order. The notebook writes a new
`Demo/repair_runs/<EXPERIMENT_ID>` directory. It does not overwrite the original
notebook, baseline weights, or baseline data archive.

## Fixed comparison

| Setting | Control | Candidate |
|---|---|---|
| Architecture | Original four-block CNN78u | Same |
| Training subset | Fixed seeded 20%, 344,165 windows | Same indices |
| Training order and initialization | Seed 69 | Same |
| Adam learning rate | 0.001 | 0.001 |
| Adam weight decay | 0.0001, all parameters | 0 |
| Dropout | 0.5 | 0.5 |
| Batch size | 64 | 64 |
| Maximum epochs | 3 | 3 |
| Maximum optimizer updates | 16,134 | 16,134 |
| Validation | Complete validation split | Same |
| Test evaluation | None | None |

Sampling uses uniform random selection without replacement, not stride five.
The selected index array and its SHA-256 are persisted for reuse by later models.
There is no class balancing. Probes are sampled from retained training examples
and validation only. Monitoring does not modify running BatchNorm statistics.

The two arms run sequentially. Three epochs at this sampling rate have a much
smaller update budget than the original full-data run; this protocol may not
reproduce a late failure. A budget extension should be predeclared for both arms
as a new experiment. Do not treat a healthy three-epoch run as proof that the
original weight decay was harmless or uniquely responsible.

## Monitoring and recovery

The notebook logs gradients, convolution/BatchNorm weight magnitudes, ReLU
activity, train/validation probe probability dispersion, losses, full-validation
metrics, optimizer updates, wall time, monitoring overhead, throughput, and peak
allocated GPU memory. It plots health versus optimizer updates.

The conservative collapse stop requires three successive checks with near-zero
feature weights/scales and near-constant outputs on both probes. Constant output
alone is not sufficient. A collapsed arm saves `stopped.pt`. Completed epochs save
`last.pt`; the best completed epoch by validation log loss saves `best.pt`.
The terminal health and best checkpoint metrics are reported separately.

For a disconnect, set `RESUME=True` and replace `EXPERIMENT_ID` with the original
ID, then rerun from the beginning. Keep every setting and the notebook helper code
unchanged. Recovery repeats any unfinished epoch. A completed or terminally
stopped arm is cached; the other arm can still proceed. Use a new ID to change
settings or extend the budget. Checkpoints contain trusted project RNG state and
must not be replaced with arbitrary externally supplied files.

Results include `comparison.csv`, `health_comparison.png`, per-arm history and
health JSON, validation reports and predictions, checkpoints, and
`repair_results.zip`. Completing the budget is not itself a pass verdict.

## Scope and validation

This notebook ends with optimizer diagnosis and validation comparison. It does
not add residual connections, change the prediction horizon, introduce new
architectures, or evaluate the previously inspected test split.

The local QA executes all 12 code cells on a tiny CPU subset and verifies identical
initialization/subsets, cached completion, incompatible-contract rejection, and
recovery after a simulated disconnect at an epoch boundary. Dedicated tests cover
the fixed sample contract and the collapse detector, including nonzero biases and
preservation of model state during monitoring. This is not a full Colab GPU run.
