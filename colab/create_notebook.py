"""Generate the English Colab baseline notebook; execution is a separate QA step."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

cells=[]
def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

md('''
# CNN78u: next-bar baseline and evaluation

**Task:** 78 completed five-minute OHLCV bars predict whether the immediately following bar's adjusted close rises or falls. Neutral targets and incomplete windows are excluded. The single CNN78u uses four convolutional blocks; no imputation, synthetic samples, class balancing, or class weighting is applied.

**Drive location:** `MyDrive/Colab Notebooks/EC/Project/Demo`.

Upload this notebook and `cnn78u_colab_bundle.zip` directly into that folder. The bundle contains portable source code and the already prepared next-bar dataset; WRDS access is not required in Colab. The source period is January-June 2023, with January-March train, April validation, and May-June test.

Before a real run, select **Runtime > Change runtime type > GPU**. Set hyperparameters before running the test cell. A tiny smoke mode only verifies code and cannot provide formal test results. GPU types/quotas vary; this notebook has no assumed runtime guarantee.
''')
md('''
## 1. Dependencies

Keep Colab's preinstalled CUDA-enabled PyTorch. Install only missing reporting/data packages. Python code and outputs are in English. The local-test branch is for repository QA and does not mount Drive or evaluate the real held-out test set.
''')
code('''
import os, sys, subprocess, importlib.util
from pathlib import Path
LOCAL_TEST = os.environ.get("CNN78U_LOCAL_NOTEBOOK_TEST") == "1"
packages = {"numpy":"numpy", "pandas":"pandas", "pyarrow":"pyarrow", "PIL":"Pillow",
            "sklearn":"scikit-learn", "matplotlib":"matplotlib", "tqdm":"tqdm", "psutil":"psutil"}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if importlib.util.find_spec("torch") is None:
    raise RuntimeError("PyTorch is missing. Select a standard Colab GPU runtime.")
''')
md('''
## 2. Mount Drive

Authentication is performed by Colab. No username, password, token, or service-account key is embedded in this notebook.
''')
code('''
if LOCAL_TEST:
    DRIVE_DEMO = Path(os.environ["CNN78U_LOCAL_DEMO"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DEMO = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo")
assert DRIVE_DEMO.is_dir(), f"Upload files first: {DRIVE_DEMO}"
print("Drive project:", DRIVE_DEMO)
''')
md('''
## 3. Stage data on the runtime disk and verify the upload

Training reads numeric memmaps from the VM's local disk, not thousands of small Drive reads. The preferred upload is the provided ZIP. An expanded bundle or project tree with `us_pipeline/` and the current prepared directory is also accepted. Copy/unpack/hash time is measured separately from training.

[Colab's official FAQ](https://research.google.com/colaboratory/faq.html) recommends avoiding many small Drive I/O operations. Runtime storage is temporary; checkpoints and reports are copied back to Drive.
''')
code('''
import hashlib, json, shutil, tempfile, time, zipfile
staging_start = time.perf_counter()
cache_parent = None if LOCAL_TEST else "/content"
LOCAL_CACHE = Path(tempfile.mkdtemp(prefix="cnn78u_", dir=cache_parent))
LOCAL_PROJECT = LOCAL_CACHE / "project"
LOCAL_PROJECT.mkdir()
archive_candidates = [DRIVE_DEMO / "cnn78u_colab_bundle.zip",
                      DRIVE_DEMO / "colab/cnn78u_colab_bundle.zip",
                      DRIVE_DEMO / "ImageTrading-CV-Project/colab/cnn78u_colab_bundle.zip"]
archive_source = next((p for p in archive_candidates if p.is_file()), None)
if archive_source:
    archive = LOCAL_CACHE / "bundle.zip"
    shutil.copy2(archive_source, archive)
    with zipfile.ZipFile(archive) as z:
        required = sum(info.file_size for info in z.infolist())
        if required > shutil.disk_usage(LOCAL_CACHE).free:
            raise RuntimeError("Not enough runtime disk space to unpack the bundle")
        for info in z.infolist():
            destination = (LOCAL_PROJECT / info.filename).resolve()
            if not destination.is_relative_to(LOCAL_PROJECT.resolve()):
                raise ValueError("Unsafe archive member path")
        z.extractall(LOCAL_PROJECT)
else:
    candidates = [DRIVE_DEMO, DRIVE_DEMO / "cnn78u_colab_bundle", DRIVE_DEMO / "ImageTrading-CV-Project"]
    source = next((p for p in candidates if (p / "us_pipeline/models.py").is_file()), None)
    if source is None:
        raise FileNotFoundError("Upload cnn78u_colab_bundle.zip next to this notebook")
    prepared_source = next((p for p in [source / "prepared", source / "artifacts/cnn78u_5min_h1/prepared"]
                            if (p / "manifest.json").is_file()), None)
    if prepared_source is None:
        raise FileNotFoundError("Missing prepared next-bar dataset; raw bars alone are insufficient here")
    shutil.copytree(source / "us_pipeline", LOCAL_PROJECT / "us_pipeline", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(prepared_source, LOCAL_PROJECT / "prepared")
    if (source / "bundle_manifest.json").exists():
        shutil.copy2(source / "bundle_manifest.json", LOCAL_PROJECT / "bundle_manifest.json")
verified_upload = False
if (LOCAL_PROJECT / "bundle_manifest.json").exists():
    upload = json.loads((LOCAL_PROJECT / "bundle_manifest.json").read_text())
    for relative, details in upload["files"].items():
        path = (LOCAL_PROJECT / relative).resolve()
        assert path.is_relative_to(LOCAL_PROJECT.resolve()), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == details["sha256"], relative
    verified_upload = True
sys.path.insert(0, str(LOCAL_PROJECT))
PREPARED = LOCAL_PROJECT / "prepared"
STAGING_SECONDS = time.perf_counter() - staging_start
print({"local_project": str(LOCAL_PROJECT), "staging_seconds": round(STAGING_SECONDS, 2),
       "upload_checksums_verified": verified_upload})
''')
md('''
## 4. Hardware and reproducibility

CUDA is required for a normal baseline run. CPU is permitted only in smoke/QA mode. Seeds and hardware/software versions are recorded, but bitwise reproducibility across GPU types or library versions is not promised. The default is full precision; changing mixed precision changes the experiment and timing conditions.
''')
code('''
import numpy as np, pandas as pd, torch
from IPython.display import display, Image
from us_pipeline.experiment import runtime_info, train_experiment, final_test, benchmark_checkpoint
from us_pipeline.data import audit_prepared
from us_pipeline.models import build_model
from us_pipeline.dataset import ImageDataset
from us_pipeline.experiment import save_json, mirror

SMOKE_MODE = LOCAL_TEST                     # Set True for a tiny code check only.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda" and not SMOKE_MODE:
    raise RuntimeError("Enable a GPU runtime, or explicitly set SMOKE_MODE=True for a code check")
torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
ENVIRONMENT = runtime_info(DEVICE)
display(pd.Series({k:v for k,v in ENVIRONMENT.items() if k != "versions"}, name="Runtime"))
display(pd.Series(ENVIRONMENT["versions"], name="Package version"))
''')
md('''
## 5. Freeze the experiment settings

The default uses all eligible training and validation samples for up to three epochs, selects the lowest validation log loss, then evaluates the fixed best checkpoint on the entire test set once. No class resampling occurs. Runtime can be substantial: 1.72 million train images are rendered on demand per epoch.

For epoch-boundary recovery, set `RESUME=True` and reuse the original `RUN_ID`. Copying resumes from `last.pt`; a partly completed epoch is repeated. Do not change batch size, seed, optimizer settings, source code, or data while resuming. Epoch count can be extended before test evaluation. Once test has been evaluated, do not continue tuning this run. Test results are cached by checkpoint hash.
''')
code('''
from datetime import datetime, timezone
EPOCHS = 1 if LOCAL_TEST else 3
BATCH_SIZE = 4 if LOCAL_TEST else 64
NUM_WORKERS = 0 if LOCAL_TEST else 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
SEED = 69
MIXED_PRECISION = False
PATIENCE_EPOCHS = 3
RUN_FINAL_TEST = not SMOKE_MODE
RESUME = False
RUN_ID = "local_smoke" if LOCAL_TEST else "cnn78u_h1_baseline_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")

SETTINGS = {"epochs":EPOCHS, "batch_size":BATCH_SIZE, "num_workers":NUM_WORKERS,
            "learning_rate":LEARNING_RATE, "weight_decay":WEIGHT_DECAY, "seed":SEED,
            "amp":MIXED_PRECISION, "patience_epochs":PATIENCE_EPOCHS, "smoke":SMOKE_MODE}
PERSISTENT_RUN = DRIVE_DEMO / "runs" / RUN_ID
LOCAL_RUN = LOCAL_CACHE / "run"
if RESUME:
    assert (PERSISTENT_RUN / "last.pt").exists(), "No epoch checkpoint found for this RUN_ID"
    shutil.copytree(PERSISTENT_RUN, LOCAL_RUN)
else:
    if PERSISTENT_RUN.exists():
        raise FileExistsError("RUN_ID already exists: choose a new ID or explicitly resume")
    PERSISTENT_RUN.mkdir(parents=True)
print("Results:", PERSISTENT_RUN)
display(pd.Series(SETTINGS, name="Frozen settings"))
''')
md('''
## 6. Dataset contract and full index audit

Input: 78 bars, next-bar horizon: 1, one model: CNN78u, class 1: up, class 0: down. Flat returns are excluded. At a session close, the next scheduled bar is the next trading session's opening bar; those targets include an overnight/holiday move and are reported separately.

The audit checks every saved sample's input/future validity, security identity, exact target endpoint, and chronological boundary. This inspects data construction, not model performance or hyperparameter selection.
''')
code('''
metadata = json.loads((PREPARED / "manifest.json").read_text())
config = metadata["config"]
assert config["lookback"] == 78 and config["horizon"] == 1 and config["bar_minutes"] == 5
assert config["model"]["name"] == "CNN78u"
audit_config = dict(config)
audit_config["artifact_root"] = str(LOCAL_PROJECT)
audit_started = time.perf_counter()
audit_report = audit_prepared(audit_config)
AUDIT_SECONDS = time.perf_counter() - audit_started
rows = [{"split":name, "samples":info["samples"], "positive_fraction":info["positive_fraction"],
         "start":info["decision_start"], "end":info["decision_end"]}
        for name, info in metadata["splits"].items()]
display(pd.DataFrame(rows))
model_description = build_model(config)
print("Parameters:", sum(p.numel() for p in model_description.parameters()))
print("Convolutional blocks:", len(model_description.channels))
print("Input image: [1, 234, 234]")
del model_description
save_json(PERSISTENT_RUN / "run_plan.json", {"settings":SETTINGS, "runtime":ENVIRONMENT,
    "data_manifest_sha256":hashlib.sha256((PREPARED / "manifest.json").read_bytes()).hexdigest(),
    "staging_seconds":STAGING_SECONDS, "sample_audit_seconds":AUDIT_SECONDS,
    "upload_checksums_verified":verified_upload, "run_id":RUN_ID})
''')
md('''
## 7. Input examples (training only)

Each image uses a shared OHLC coordinate system and a separate volume band. No future target bar appears in the image. No fitted global normalization is estimated on validation or test.
''')
code('''
import matplotlib.pyplot as plt
training_examples = ImageDataset(PREPARED, "train")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, index in zip(axes, [0, len(training_examples)//2, len(training_examples)-1]):
    row = training_examples.samples.iloc[index]
    axis.imshow(training_examples.image(index), cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"PERMNO {int(row.permno)} | target={int(row.label)}")
    axis.axis("off")
fig.tight_layout()
preview_path = LOCAL_CACHE / "training_inputs.png"
fig.savefig(preview_path, dpi=140);plt.close(fig)
mirror(preview_path, PERSISTENT_RUN)
display(Image(filename=str(preview_path)))
del training_examples
''')
md('''
## 8. Train and select using validation only

Loss: unweighted cross-entropy. Optimizer: Adam. Decision threshold: fixed 0.5. Early stopping monitors full-validation log loss. `best.pt` and `last.pt` are mirrored to Drive after each completed epoch; history and status files are also persisted. A disconnect within an epoch can lose that epoch's unfinished work.

Smoke mode uses a tiny seeded sample for code verification and is explicitly blocked from formal test evaluation. Its accuracy is not an estimate of the baseline's performance.
''')
code('''
training_report_path = LOCAL_RUN / "training_report.json"
if (LOCAL_RUN / "test/metrics.json").exists():
    training_report = json.loads(training_report_path.read_text())
    if training_report["settings"] != SETTINGS:
        raise ValueError("This run already evaluated test; do not change its settings")
    print("Existing completed test run: training is skipped.")
else:
    training_report = train_experiment(PREPARED, LOCAL_RUN, SETTINGS, device=DEVICE,
        persistent_dir=PERSISTENT_RUN, resume=RESUME, progress=not LOCAL_TEST)
    # The runner does not evaluate test during training.
    assert training_report["test_evaluated"] is False
history = pd.DataFrame(training_report["history"])
display(history)
display(Image(filename=str(LOCAL_RUN / "training_curves.png")))
print("Selected epoch:", training_report["best_epoch"])
''')
md('''
## 9. Validation metrics and fixed baselines

- Accuracy and balanced accuracy: directional classification.
- Precision/recall/F1 for up, macro F1, MCC, and confusion matrix: error composition.
- ROC-AUC and positive-class **average precision (AP)**: ranking. AP is not trapezoidal PR area.
- Log loss, Brier score, and reliability/ECE: probability quality.
- Constant probability and majority class are fixed from **training labels only**.
- Last-bar and full-lookback directional rules use only the input window; their probability metrics are omitted because they do not supply calibrated probabilities.

The charts include calibration, daily stability, and fixed confidence/coverage thresholds. These are diagnostics, not a policy selected on test data. [Metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).
''')
code('''
validation_report = json.loads((LOCAL_RUN / "validation/metrics.json").read_text())
display(pd.Series({k:v for k,v in validation_report["metrics"].items() if not isinstance(v, list)}, name="Validation"))
display(pd.read_csv(LOCAL_RUN / "validation/baselines.csv"))
display(Image(filename=str(LOCAL_RUN / "validation/evaluation.png")))
''')
md('''
## 10. Timing, memory, and model-only latency

Epoch train time includes loading, rendering, device transfers, forward/backward passes, and optimization. Validation/test end-to-end time includes prediction collection but excludes report serialization. Setup/staging/audit, checkpoint synchronization, GPU peak allocated/reserved memory, and main-process RSS are separate fields.

The following warmed benchmark uses **validation tensors already loaded on the device**. CUDA events are synchronized. It reports batch-1 latency and batched throughput; amortized batch milliseconds/sample is not single-request latency. Compare times only under recorded hardware, precision, batch size, and loader settings. Main-process RSS excludes worker processes.
''')
code('''
hardware_report = benchmark_checkpoint(PREPARED, LOCAL_RUN, device=DEVICE,
    persistent_dir=PERSISTENT_RUN, warmup=1 if LOCAL_TEST else 10,
    repetitions=3 if LOCAL_TEST else 30)
display(pd.DataFrame(hardware_report["benchmarks"]))
timing_columns = ["epoch", "train_seconds", "train_samples_per_second", "validation_seconds",
                  "checkpoint_sync_seconds", "epoch_wall_seconds", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"]
display(history.reindex(columns=timing_columns))
print("Staging seconds:", STAGING_SECONDS)
print("Sample audit seconds:", AUDIT_SECONDS)
print("Main-process peak RSS bytes:", training_report["main_process_peak_rss_bytes_excludes_workers"])
''')
md('''
## 11. Final held-out test (once the experiment is fixed)

With the default `RUN_FINAL_TEST=True`, this runs **after** selecting the best epoch using validation. Do not tune the architecture, threshold, or hyperparameters after inspecting test results. Re-executing an unchanged test cell returns its cached report; a different checkpoint cannot overwrite the original run's test report.

Accuracy differences against the training-majority classifier have a paired moving-block interval: resample contiguous blocks of five decision days while keeping all stocks/windows within each selected day. This avoids an IID-window/binomial interval, but remains an approximation conditional on one trained model and block length; it does not include training/model-selection uncertainty.
''')
code('''
test_report = None
if RUN_FINAL_TEST and not SMOKE_MODE:
    test_report = final_test(PREPARED, LOCAL_RUN, device=DEVICE,
        persistent_dir=PERSISTENT_RUN, progress=True)
    display(pd.Series({k:v for k,v in test_report["metrics"].items() if not isinstance(v, list)}, name="Held-out test"))
    display(pd.read_csv(LOCAL_RUN / "test/baselines.csv"))
    display(pd.Series(test_report["paired_accuracy_difference_interval"], name="Paired day-block interval"))
    display(Image(filename=str(LOCAL_RUN / "test/evaluation.png")))
else:
    print("Formal test skipped. Smoke results cannot be treated as baseline performance.")
''')
md('''
## 12. Inspect temporal, security, and overnight stability

Group tables include every evaluated security rather than cherry-picked winners. Small groups can be noisy. Confidence/coverage tables use predeclared thresholds and are descriptive. Dividend-adjusted classification results do not establish profitability after spreads, slippage, costs, or portfolio constraints.
''')
code('''
REPORT_SPLIT = "test" if test_report is not None else "validation"
for name in ["by_month", "by_target_session"]:
    print(name)
    display(pd.read_csv(LOCAL_RUN / REPORT_SPLIT / (name + ".csv")))
print("Per-security results (first rows; full table is saved):")
display(pd.read_csv(LOCAL_RUN / REPORT_SPLIT / "by_security.csv").head(10))
print("Confidence / coverage (descriptive):")
display(pd.read_csv(LOCAL_RUN / REPORT_SPLIT / "confidence_coverage.csv"))
''')
md('''
## 13. Persist the result package

Drive receives per-epoch checkpoints and reports throughout training. This final archive additionally packages the local run's metrics, predictions, figures, history, and checkpoints. No WRDS credentials are present. If the VM disconnects before this cell, completed epoch checkpoints remain available in the run folder.
''')
code('''
result_archive = Path(shutil.make_archive(str(LOCAL_CACHE / (RUN_ID + "_results")), "zip", root_dir=LOCAL_RUN))
shutil.copy2(result_archive, PERSISTENT_RUN / "results.zip")
summary = {"run_id":RUN_ID, "mode":"smoke_only" if SMOKE_MODE else "baseline",
           "best_epoch":training_report["best_epoch"], "test_evaluated":test_report is not None,
           "persistent_results":str(PERSISTENT_RUN), "result_zip_bytes":result_archive.stat().st_size,
           "training_seconds_recorded":training_report["training_seconds_recorded"],
           "validation_seconds_recorded":training_report["validation_seconds_recorded"]}
save_json(PERSISTENT_RUN / "summary.json", summary)
display(pd.Series(summary, name="Run summary"))
print("Keep this RUN_ID for epoch-boundary recovery:", RUN_ID)
''')
md('''
## Interpreting the run

Use validation to select the model and report the frozen test result with baselines, confusion matrix, probability quality, stability tables, and timing conditions. A score above 50% alone is insufficient: the empirical class frequency, simple input-only rules, correlated observations, and overnight targets all matter. The current four-block architecture is an unproven baseline, not evidence that four blocks outperform two.

The raw data's conservative trade filters, missing-window selection, proportional dividend chart adjustment, and historical membership definitions still apply. Quote-based outlier filtering, explicit halt annotations, calibrated execution costs, and strategy backtesting remain separate research tasks.
''')

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'accelerator':'GPU',
    'colab':{'name':'CNN78u_Colab_Baseline.ipynb','provenance':[]}})
for i,cell in enumerate(cells):
    if cell.cell_type=='code':
        compile(cell.source,f'cell_{i}','exec')
nbf.validate(notebook)
path=Path(__file__).with_name('CNN78u_Colab_Baseline.ipynb')
nbf.write(notebook,path)
print(f'Wrote {path}: {len(cells)} cells, {sum(c.cell_type=="code" for c in cells)} code cells')
