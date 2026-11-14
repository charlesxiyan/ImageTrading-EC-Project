"""Generate a self-contained optimizer diagnostic using the existing data bundle."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

HERE=Path(__file__).resolve().parent
original=nbf.read(HERE/'CNN78u_Colab_Baseline.ipynb',as_version=4)
cells=original.cells[:7]
cells[0]=nbf.v4.new_markdown_cell('''# CNN78u: fixed 20% sampling and optimizer diagnosis

This notebook ends before residual networks or other architecture comparisons.
It preserves the original four-block CNN and compares Adam weight decay 0.0001
against 0, using exactly the same fixed training subset, initialization and
per-epoch ordering. Learning rate 0.001, dropout 0.5, batch size 64 and three epochs
are identical. This is an optimizer diagnostic, not a claim that removing weight
decay improves generalization.

Upload **this notebook only** beside the existing `cnn78u_colab_bundle.zip` in
`MyDrive/Colab Notebooks/EC/Project/Demo`. The helper source is embedded below;
no replacement of the original notebook or bundle is needed. Select a GPU runtime.

Only training and validation data are evaluated. There is no test-evaluation cell.
The already inspected historical test period is not a fresh holdout.
''')

def md(s): cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(dedent(s).strip()))

md('''## 4. Load the diagnostic runner
The helper is written to the temporary runtime directory. The uploaded bundle and
original run are preserved. GPU runs use full precision and deterministic cuDNN
settings; exact cross-hardware reproducibility is not guaranteed.''')
source=(HERE/'repair_runner.py').read_text()
code('import importlib\nrunner_path = LOCAL_PROJECT / "repair_runner.py"\nrunner_path.write_text('+repr(source)+')\nimportlib.invalidate_caches()\nimport repair_runner as repair\n')
code('''
import numpy as np, pandas as pd, torch
from IPython.display import display, Image
from datetime import datetime, timezone
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_MODE = LOCAL_TEST  # Local QA only; never interpret smoke scores as performance.
if DEVICE != "cuda" and not SMOKE_MODE:
    raise RuntimeError("Select a GPU runtime before running the paired experiment")
torch.set_num_threads(min(4, os.cpu_count() or 1))
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
print(repair.runtime_info(torch.device(DEVICE)))
''')
md('''## 5. Freeze the protocol and output folder
Normal runs retain `floor(0.2 * N)` eligible training windows uniformly without
replacement. The same indices are reused for all epochs and both arms; only their
order changes. No class balancing occurs. Validation is complete at each epoch.
A 20% subset with three epochs has about one fifth of the original update budget.

Set `RESUME=True` and the original `EXPERIMENT_ID` after a disconnect. All settings,
code, data, sampling and probe indices must match. A partly completed epoch is
repeated from the last completed epoch; a terminal collapsed run is not resumed.
Completed arms return cached summaries. Do not reuse this ID for a new experiment.
The two arms run sequentially to keep memory use manageable.''')
code('''
SAMPLING_RATE = 0.2
SAMPLE_SEED = 69
SETTINGS = dict(epochs=1 if SMOKE_MODE else 3,
                batch_size=4 if SMOKE_MODE else 64,
                num_workers=0 if SMOKE_MODE else 2,
                learning_rate=0.001, seed=69,
                monitor_every=1 if SMOKE_MODE else 500, device=DEVICE)
RESUME = False
EXPERIMENT_ID = ("repair_local_qa" if LOCAL_TEST else
                "cnn78u_repair_20pct_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc"))
PERSISTENT = DRIVE_DEMO / "repair_runs" / EXPERIMENT_ID
LOCAL_REPAIR = LOCAL_CACHE / "repair"
if RESUME:
    if not (PERSISTENT / "sampling.json").exists():
        raise FileNotFoundError("No sampling contract for this experiment ID")
    shutil.copytree(PERSISTENT, LOCAL_REPAIR)
else:
    if PERSISTENT.exists():
        raise FileExistsError("Choose a new experiment ID or explicitly resume")
    PERSISTENT.mkdir(parents=True)
    LOCAL_REPAIR.mkdir()
print("Persistent output:", PERSISTENT)
display(pd.Series(SETTINGS))
''')
md('''## 6. Persist the fixed training subset and fixed diagnostic probes
The probes are 128 seeded examples from the selected training subset and 128
seeded validation examples. They detect numerical failure, not statistical alpha.
Full validation remains the source of epoch metrics. Hashes and original row indices
are saved so later models can use exactly the same subset. For the current bundle,
the normal training subset contains 344,165 of 1,720,827 eligible examples.''')
code('''
train = repair.ImageDataset(PREPARED, "train")
validation = repair.ImageDataset(PREPARED, "validation")
indices = repair.fixed_indices(len(train), SAMPLING_RATE, SAMPLE_SEED)
if SMOKE_MODE:
    indices = indices[:8]
probe_rng = np.random.default_rng(SAMPLE_SEED + 1)
probe_ids = {"train":np.sort(probe_rng.choice(indices, min(128, len(indices)), replace=False)),
             "validation":np.sort(probe_rng.choice(len(validation), 8 if SMOKE_MODE else min(128,len(validation)), replace=False))}
contract = dict(requested_sampling_rate=SAMPLING_RATE, actual_fraction=len(indices)/len(train),
                sampling_seed=SAMPLE_SEED, samples=len(indices), eligible_samples=len(train),
                subset_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                train_index_file_sha256=repair.sha(PREPARED / "train.parquet"),
                smoke_only=SMOKE_MODE)
if RESUME:
    assert json.loads((LOCAL_REPAIR / "sampling.json").read_text()) == contract
    assert np.array_equal(np.load(LOCAL_REPAIR / "train_indices.npy"), indices)
else:
    np.save(LOCAL_REPAIR / "train_indices.npy", indices)
    repair.mirror(LOCAL_REPAIR / "train_indices.npy", PERSISTENT)
    repair.save_json(LOCAL_REPAIR / "sampling.json", contract, PERSISTENT)
    np.savez(LOCAL_REPAIR / "probe_indices.npz", **probe_ids)
    repair.mirror(LOCAL_REPAIR / "probe_indices.npz", PERSISTENT)
display(pd.Series(contract))
print("Retained training up fraction:", train.samples.iloc[indices].label.mean())
print("Validation samples:", len(validation) if not SMOKE_MODE else len(probe_ids["validation"]))
''')
md('''## 7. Original optimizer control: weight decay = 0.0001
Every 500 updates and at each epoch end, record probability dispersion, convolution
and BatchNorm weight magnitudes, ReLU activity, and gradient norms from the latest
training batch. The probes run in eval mode and do not update BatchNorm statistics.

Stop after three consecutive checks where **both** probe probability standard
deviations are at most 1e-8 **and** all convolution weights / BatchNorm scales have
maximum absolute magnitude at most 1e-8. Biases are excluded from that magnitude
check because a collapsed network can retain constant offsets. Nonfinite loss or
monitored gradients also stop the arm. These conservative guards do not detect
all possible failures; low discrimination still requires review.

A stopped arm saves `stopped.pt`. Each completed epoch saves `last.pt`; the best
completed epoch by full-validation log loss saves `best.pt`. The comparison flags
terminal health separately from the best pre-collapse checkpoint.''')
code('''
control = repair.run_arm(PREPARED, LOCAL_REPAIR / "wd_1e-4", PERSISTENT / "wd_1e-4",
                         indices, probe_ids, SETTINGS, weight_decay=0.0001, smoke=SMOKE_MODE)
display(pd.DataFrame(control["history"]))
print("Control status:", control["status"], control["stop_reason"])
''')
md('''## 8. Repair candidate: weight decay = 0
Only weight decay changes. Dropout remains 0.5, unlike the earlier small-sample
memorization check. This arm starts from the same fresh initialization as the
control; it does not fine-tune the collapsed checkpoint.''')
code('''
candidate = repair.run_arm(PREPARED, LOCAL_REPAIR / "wd_0", PERSISTENT / "wd_0",
                           indices, probe_ids, SETTINGS, weight_decay=0.0, smoke=SMOKE_MODE)
assert control["initial_parameters_sha256"] == candidate["initial_parameters_sha256"]
assert control["train_subset_sha256"] == candidate["train_subset_sha256"]
display(pd.DataFrame(candidate["history"]))
print("Candidate status:", candidate["status"], candidate["stop_reason"])
''')
md('''## 9. Compare validation and training health
Compare validation log loss, balanced accuracy and AUC against the training-prior
baseline. Accuracy above 50% alone is insufficient. Completing the budget only
means the stop guard did not trigger; it does not certify useful predictive signal.
If one arm stops early, report its smaller update budget explicitly.

The validation report uses the best completed epoch, while terminal health uses
the final monitored model. If no healthy epoch completed, no best report exists.
Baseline tables and reliability/confusion/ROC plots are saved under each arm's
`validation/` directory. No result here is a new held-out test result.''')
code('''
rows=[]
for name, result in [("Original WD 1e-4", control),("Candidate WD 0", candidate)]:
    metrics=result["best_validation_metrics"] or {}
    rows.append(dict(arm=name, status=result["status"], steps=result["steps"],
        validation_accuracy=metrics.get("accuracy"), balanced_accuracy=metrics.get("balanced_accuracy"),
        roc_auc=metrics.get("roc_auc"), log_loss=metrics.get("log_loss"),
        terminal_probe_std=result["final_health"]["validation"]["probability_std"],
        terminal_feature_max_abs=result["final_health"]["feature_parameter_max_abs"],
        training_wall_seconds=sum(r["train_wall_seconds_including_monitor"] for r in result["history"]),
        validation_seconds=sum(r["validation_seconds"] for r in result["history"])))
comparison=pd.DataFrame(rows)
display(comparison)
comparison.to_csv(LOCAL_REPAIR / "comparison.csv", index=False)
repair.mirror(LOCAL_REPAIR / "comparison.csv", PERSISTENT)
for arm in ["wd_1e-4","wd_0"]:
    report = LOCAL_REPAIR / arm / "validation"
    if report.exists():
        print(arm)
        display(pd.read_csv(report / "baselines.csv"))
        display(Image(filename=str(report / "evaluation.png")))
''')
code('''
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for arm in ["wd_1e-4", "wd_0"]:
    health = json.loads((LOCAL_REPAIR / arm / "health.json").read_text())
    steps = [r["step"] for r in health]
    axes[0].plot(steps, [max(r["feature_parameter_max_abs"],1e-45) for r in health], label=arm)
    axes[1].plot(steps, [max(r["validation"]["probability_std"],1e-12) for r in health], label=arm)
    history = json.loads((LOCAL_REPAIR / arm / "history.json").read_text())
    axes[2].plot([r["steps_total"] for r in history], [r.get("val_log_loss",np.nan) for r in history], marker="o", label=arm)
for ax,title in zip(axes,["Feature weights: max absolute value", "Validation probe probability std", "Full-validation log loss"]):
    ax.set_title(title); ax.set_xlabel("Optimizer updates"); ax.legend()
axes[0].set_yscale("log"); axes[1].set_yscale("log")
if SMOKE_MODE:
    axes[2].set_title("Smoke validation log loss")
    fig.suptitle("SMOKE CHECK ONLY: not a performance estimate")
fig.tight_layout()
fig.savefig(LOCAL_REPAIR / "health_comparison.png",dpi=140)
plt.close(fig)
repair.mirror(LOCAL_REPAIR / "health_comparison.png",PERSISTENT)
display(Image(filename=str(LOCAL_REPAIR / "health_comparison.png")))
''')
md('''## 10. Save and decide the next step
- If the control collapses and the candidate remains input-sensitive, the paired
  experiment supports removing this decay configuration as a repair under this
  protocol. It does not isolate effects across all datasets, budgets or seeds.
- If both remain healthy but validation is near chance, investigate signal and
  representation before claiming success.
- If both collapse, investigate learning rate, activations and optimization using
  further controlled experiments.
- If neither collapses within three epochs, the smaller update budget may not
  reproduce the original late failure. Do not claim weight decay was cleared or
  proved causal. Any extended budget should be predeclared for both arms.

Stop here. Residual connections, depth changes and other architectures belong to
a later experiment after establishing a functioning baseline. Preserve this fixed
subset and the original failed run. Use a new untouched period for final evaluation.
''')
code('''
summary = dict(experiment_id=EXPERIMENT_ID, smoke_only=SMOKE_MODE, test_evaluated=False,
               same_initialization=True, same_training_subset=True,
               arms={"wd_1e-4":control["status"],"wd_0":candidate["status"]})
repair.save_json(LOCAL_REPAIR / "experiment_summary.json",summary,PERSISTENT)
archive=Path(shutil.make_archive(str(LOCAL_CACHE / "repair_results"),"zip",root_dir=LOCAL_REPAIR))
shutil.copy2(archive,PERSISTENT / "repair_results.zip")
print("Saved diagnostic results:",PERSISTENT)
display(pd.Series(summary))
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'})
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.execution_count=None;cell.outputs=[]
nbf.validate(nb)
nbf.write(nb,HERE/'CNN78u_Training_Repair_20pct.ipynb')
print('Generated',len(cells),'cells')
