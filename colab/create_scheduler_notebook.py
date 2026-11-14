"""Build the English scheduler-only CNN78u Colab notebook."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

HERE=Path(__file__).resolve().parent
base=nbf.read(HERE/'CNN78u_Colab_Baseline.ipynb',as_version=4)
cells=base.cells[:7]
cells[0]=nbf.v4.new_markdown_cell('''# CNN78u: validation-driven learning-rate schedule

Use the same fixed 20% training subset, original four-block CNN, Xavier uniform
initialization, dropout 0.5, batch size 64 and Adam weight decay 0. Start from fresh
weights with seed 69; do not continue either earlier checkpoint.

The experiment adds ReduceLROnPlateau and increases the maximum budget to 15
epochs. Because both schedule and budget change, comparison with the earlier
three-epoch run cannot isolate the scheduler's causal effect.

Upload **only this notebook** alongside the existing `cnn78u_colab_bundle.zip` in
`MyDrive/Colab Notebooks/EC/Project/Demo`. Choose a GPU runtime. The helper source
is embedded. Outputs go into a new `scheduler_runs` folder. There is no test
inference, residual model, or Kaiming initialization in this experiment.
''')

def md(s): cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(dedent(s).strip()))

md('''## 4. Install the embedded experiment helper
The source is staged locally and copied into the experiment folder for provenance.
Original notebooks, uploaded data, and previous experiment directories are preserved.''')
source=(HERE/'scheduler_runner.py').read_text()
code('import importlib\nhelper = LOCAL_PROJECT / "scheduler_runner.py"\nhelper.write_text('+repr(source)+')\nimportlib.invalidate_caches()\nimport scheduler_runner as runner\n')
md('''## 5. Freeze the training protocol
- Initial LR 0.001; factor 0.5; scheduler patience 2; minimum LR 0.00001.
- Both plateau detection and early stopping use an absolute improvement tolerance
  of 0.00001 in full-validation log loss. A new best checkpoint uses the actual
  lowest loss, even if its change is below this tolerance.
- **PyTorch patience semantics:** patience=2 permits two bad epochs and reduces
  LR after the third consecutive non-improving epoch. The new LR applies to the
  following epoch. It does not reduce LR every two epochs.
- Stop after six consecutive non-improving epochs, or at 15 epochs, whichever
  comes first. This allows training after the first LR reduction; a second
  reduction may coincide with early stopping and therefore never be used.
- Numerical-collapse monitoring remains active every 500 updates and at epoch end.

All epoch metrics use the complete validation split. Fixed train and validation
probes use eval mode, so dropout is disabled and BatchNorm running statistics are
not changed. Probe loss/accuracy are diagnostics, not replacement validation scores.
''')
code('''
import numpy as np, pandas as pd, torch
from datetime import datetime, timezone
from IPython.display import display, Image
SMOKE_MODE = LOCAL_TEST
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda" and not SMOKE_MODE:
    raise RuntimeError("Select a GPU runtime")
torch.set_num_threads(min(4, os.cpu_count() or 1))
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
SETTINGS = dict(epochs=2 if SMOKE_MODE else 15,
                batch_size=4 if SMOKE_MODE else 64,
                num_workers=0 if SMOKE_MODE else 2,
                learning_rate=0.001, seed=69,
                monitor_every=1 if SMOKE_MODE else 500, device=DEVICE,
                lr_factor=0.5, lr_patience=2, min_lr=0.00001,
                min_delta=0.00001, early_stop_patience=6)
SAMPLING_RATE = 0.2
SAMPLE_SEED = 69
RESUME = False
EXPERIMENT_ID = ("scheduler_local_qa" if LOCAL_TEST else
                "cnn78u_scheduler_20pct_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc"))
PERSISTENT = DRIVE_DEMO / "scheduler_runs" / EXPERIMENT_ID
LOCAL_RUN = LOCAL_CACHE / "scheduler_run"
if RESUME:
    if not (PERSISTENT / "sampling.json").exists():
        raise FileNotFoundError("No experiment found for this ID")
    shutil.copytree(PERSISTENT, LOCAL_RUN)
else:
    if PERSISTENT.exists():
        raise FileExistsError("Use a new ID, or explicitly resume the existing experiment")
    PERSISTENT.mkdir(parents=True)
    LOCAL_RUN.mkdir()
display(pd.Series(SETTINGS, name="Frozen protocol"))
print("Results:", PERSISTENT)
print(runner.runtime_info(torch.device(DEVICE)))
''')
md('''## 6. Restore the same fixed 20% sampling contract
The random subset is identical to the previous repair notebook: seed 69, uniform
sampling without replacement, sorted original training-row indices. There is no
class balancing and no re-sampling between epochs. Full validation is retained.
The normal dataset yields 344,165 training examples and 5,378 updates per epoch;
15 complete epochs would give 80,670 updates. This is a budget cap, not a target
that overrides early stopping.

To resume after a disconnect, set RESUME=True and reuse the original ID with all
settings unchanged, then rerun from the beginning. The helper restores model,
optimizer, scheduler, early-stop counters, RNG and completed-epoch history.
A partly finished epoch is repeated. Completed runs return cached results. A
stopped unhealthy run is terminal once its summary is saved.
''')
code('''
training = runner.ImageDataset(PREPARED, "train")
validation = runner.ImageDataset(PREPARED, "validation")
assert training.config["model"]["channels"] == [32,64,128,256]
assert training.config["model"]["dropout"] == 0.5
indices = runner.fixed_indices(len(training), SAMPLING_RATE, SAMPLE_SEED)
if SMOKE_MODE:
    indices = indices[:8]
probe_rng = np.random.default_rng(SAMPLE_SEED + 1)
probe_ids = {"train":np.sort(probe_rng.choice(indices, min(128,len(indices)),replace=False)),
             "validation":np.sort(probe_rng.choice(len(validation),8 if SMOKE_MODE else min(128,len(validation)),replace=False))}
sampling = dict(sampling_rate=SAMPLING_RATE, sampling_seed=SAMPLE_SEED,
    count=len(indices), eligible_count=len(training), smoke_only=SMOKE_MODE,
    indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
    train_index_file_sha256=runner.sha(PREPARED / "train.parquet"))
if RESUME:
    assert json.loads((LOCAL_RUN / "sampling.json").read_text()) == sampling
    assert np.array_equal(np.load(LOCAL_RUN / "train_indices.npy"),indices)
else:
    np.save(LOCAL_RUN / "train_indices.npy",indices)
    runner.mirror(LOCAL_RUN / "train_indices.npy",PERSISTENT)
    np.savez(LOCAL_RUN / "probe_indices.npz",**probe_ids)
    runner.mirror(LOCAL_RUN / "probe_indices.npz",PERSISTENT)
    runner.save_json(LOCAL_RUN / "sampling.json",sampling,PERSISTENT)
    shutil.copy2(helper,LOCAL_RUN / "scheduler_runner.py")
    runner.mirror(LOCAL_RUN / "scheduler_runner.py",PERSISTENT)
# If prior repair indices are present, verify exact equality for the same data contract.
matched_previous = []
if not SMOKE_MODE:
    for previous in (DRIVE_DEMO / "repair_runs").glob("*/sampling.json"):
        previous_contract = json.loads(previous.read_text())
        if (not previous_contract.get("smoke_only") and
            previous_contract.get("train_index_file_sha256") == sampling["train_index_file_sha256"] and
            previous_contract.get("requested_sampling_rate") == SAMPLING_RATE and
            previous_contract.get("sampling_seed") == SAMPLE_SEED):
            assert np.array_equal(np.load(previous.parent / "train_indices.npy"),indices)
            matched_previous.append(previous.parent.name)
display(pd.Series(sampling))
print("Matching prior runs:",matched_previous)
print("Retained training up fraction:",training.samples.iloc[indices].label.mean())
print("Full validation samples:",len(validation))
''')
md('''## 7. Train, validate, schedule and checkpoint
Order: train epoch -> full validation -> update scheduler/early-stop state -> save
checkpoint -> next epoch. History records both LR used and LR scheduled next.
Best checkpoint selection uses validation log loss only.

The original conservative collapse guard requires both near-constant probe
probabilities and near-zero convolution/BatchNorm scales for three consecutive
checks. Not triggering it is not proof of predictive ability. Nonfinite loss or
monitored gradients also stop training.

Checkpoints and reports are mirrored to Drive. Full validation happens every
epoch, so this can take several hours. The final budget may be smaller due to
early stopping. Do not infer a runtime guarantee from the earlier three-epoch run.
''')
code('''
result = runner.run_experiment(PREPARED,LOCAL_RUN / "training",PERSISTENT / "training",
                               indices,probe_ids,SETTINGS,weight_decay=0.0,smoke=SMOKE_MODE)
history = pd.DataFrame(result["history"])
display(history)
print("Status:",result["status"],"Reason:",result["stop_reason"])
print("Best validation epoch:",result["best_epoch"])
print("Test evaluated:",result["test_evaluated"])
history.to_csv(LOCAL_RUN / "history.csv",index=False)
runner.mirror(LOCAL_RUN / "history.csv",PERSISTENT)
''')
md('''## 8. Inspect learning rate, fit and numerical health
The fixed training-probe eval loss is directly comparable in mode to validation
loss, but its small sample size makes it noisy. The in-training loss additionally
contains dropout and changing weights. Inspect all three instead of treating
in-training and eval-mode accuracy as identical measurements.

In the second panel, both curves are fixed probes; full-validation loss is shown
separately. If training-probe fit improves but validation does not, investigate
generalization before increasing model complexity.
''')
code('''
import matplotlib.pyplot as plt
health=json.loads((LOCAL_RUN / "training/health.json").read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].step(history.epoch,history.learning_rate_used,where="mid",label="LR used")
axes[0,0].set_yscale("log");axes[0,0].set_title("Learning rate used per epoch")
axes[0,0].set_xlabel("Epoch");axes[0,0].legend()
for split in ["train","validation"]:
    axes[0,1].plot([h["step"] for h in health],[h[split]["log_loss"] for h in health],label=split+" probe (eval)")
axes[0,1].set_title("Fixed-probe log loss");axes[0,1].set_xlabel("Updates");axes[0,1].legend()
axes[1,0].plot(history.epoch,history.reindex(columns=["val_log_loss"])["val_log_loss"],marker="o",label="Full validation")
axes[1,0].set_title("Validation log loss");axes[1,0].set_xlabel("Epoch");axes[1,0].legend()
for split in ["train","validation"]:
    axes[1,1].plot([h["step"] for h in health],[max(h[split]["probability_std"],1e-12) for h in health],label=split)
axes[1,1].set_yscale("log");axes[1,1].set_title("Probe probability standard deviation")
axes[1,1].set_xlabel("Updates");axes[1,1].legend()
if SMOKE_MODE:
    fig.suptitle("SMOKE CHECK ONLY: not a performance estimate")
fig.tight_layout()
fig.savefig(LOCAL_RUN / "scheduler_diagnostics.png",dpi=140)
plt.close(fig)
runner.mirror(LOCAL_RUN / "scheduler_diagnostics.png",PERSISTENT)
display(Image(filename=str(LOCAL_RUN / "scheduler_diagnostics.png")))
''')
md('''## 9. Best-checkpoint validation and export
The best completed epoch may differ from terminal health. Compare classification,
ranking and probability metrics with the training-prior baseline. A majority-class
prediction can exceed 50% accuracy without discriminating examples. A tiny gain
on this repeatedly used validation period does not establish out-of-sample alpha.

No Kaiming or architecture change is made here. Once this training protocol is
assessed, a separate initialization comparison can hold it fixed. A fresh untouched
time period is required for a final independent test.
''')
code('''
report = LOCAL_RUN / "training/validation"
if result["best_validation_metrics"] is not None:
    display(pd.Series({k:v for k,v in result["best_validation_metrics"].items() if not isinstance(v,list)},name="Best validation"))
    display(pd.read_csv(report / "baselines.csv"))
    display(Image(filename=str(report / "evaluation.png")))
else:
    print("No eligible completed checkpoint. Inspect the stop reason and health records.")
summary = dict(experiment_id=EXPERIMENT_ID, status=result["status"],
    best_epoch=result["best_epoch"], completed_or_partial_epochs=len(history),
    optimizer_updates=result["steps"], smoke_only=SMOKE_MODE,test_evaluated=False,
    train_wall_seconds=float(history.train_wall_seconds_including_monitor.sum()),
    validation_seconds=float(history.validation_seconds.sum()),
    initialization="Original Xavier uniform",weight_decay=0,dropout=0.5)
runner.save_json(LOCAL_RUN / "experiment_summary.json",summary,PERSISTENT)
archive=Path(shutil.make_archive(str(LOCAL_CACHE / "scheduler_results"),"zip",root_dir=LOCAL_RUN))
shutil.copy2(archive,PERSISTENT / "scheduler_results.zip")
display(pd.Series(summary))
print("Saved:",PERSISTENT)
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'})
for c in nb.cells:
    if c.cell_type=='code': c.execution_count=None;c.outputs=[];compile(c.source,'cell','exec')
nbf.validate(nb)
nbf.write(nb,HERE/'CNN78u_Scheduler_20pct.ipynb')
print('Generated',len(cells),'cells')
