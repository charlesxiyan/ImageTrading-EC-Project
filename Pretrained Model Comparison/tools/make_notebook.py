"""Builds Pretrained_Model_Comparison.ipynb (run from anywhere; writes next to tsfm_compare/)."""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# Pretrained time-series models vs the chart CNN — 5-minute US equities

<img src="FLOW_DATA_URI" alt="flow" width="100%">

- **Question**: do off-the-shelf forecasting foundation models (Chronos-2, TimesFM, Moirai 2, Kronos) predict our targets better than the chart CNN or simple baselines?
- **Same data, splits and labels** as `CNN-Relevant/ChartCNN_Baseline.ipynb` (shared `chartcnn` package): January–March train, April validation; `vol_h1` (next bar moves more than usual), `xsdir_h12` (beats the cross-section over the next hour), and absolute direction `dir_h1` / `dir_h12` (next bar / hour strictly up; zero returns dropped).
- **Zero-shot**: each model sees the last 78 closes and forecasts 12 bars ahead as quantiles. Its score is the forecast |return| (`vol`), the forecast return relative to other stocks (`xs_dir`), or the forecast return itself (`dir`). Labels are only used to fit a 1-feature calibration on training-period data, so log loss can be compared.
- **Light fine-tune (optional)**: Chronos-2 with LoRA on training-period price series. It learns to forecast prices; it never sees the labels.
- **Code**: all logic lives in `tsfm_compare/` next to this notebook. Cells only configure, run and display.
""")

md("## 1. Install\nRestart is **not** needed on a fresh runtime (if `peft` was already imported in this session, use Runtime > Restart session once after this cell). Moirai's package pins old library versions, so it is installed without dependencies.")
code("""
import os
LOCAL_TEST = os.environ.get("PMC_LOCAL_TEST") == "1"          # repository QA only
if not LOCAL_TEST:
    !pip -q install "chronos-forecasting>=2.2" "timesfm[torch]>=3.0" peft
    !pip -q install --no-deps "uni2ts==2.0.0"
    !pip -q install einops jaxtyping lightning gluonts hydra-core python-dotenv
    # Colab ships torchao 0.10, which recent peft (LoRA) refuses to import. We don't use torchao, so remove it.
    !pip -q uninstall -y torchao
    !git clone -q --depth 1 https://github.com/shiyu-coder/Kronos /content/Kronos 2>/dev/null || echo "Kronos already cloned"
""")

md("## 2. Setup\nMounts Drive, finds the shared `chartcnn` package and the data bundle, and copies the data to the Colab disk.")
code("""
import sys, shutil, tempfile, json
from pathlib import Path
from datetime import datetime, timezone

if LOCAL_TEST:
    DEMO = Path(os.environ["PMC_LOCAL_DEMO"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DEMO = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo")
HERE = DEMO / "Pretrained Model Comparison"                    # this notebook + tsfm_compare/
CNN_HOME = next(p for p in (DEMO / "CNN-Relevant", DEMO) if (p / "chartcnn").is_dir())
sys.path[:0] = [str(HERE), str(CNN_HOME)]

import numpy as np, pandas as pd, torch
from IPython.display import display
import chartcnn
from chartcnn.utils import stage_bundle, save_json, seed_all
from tsfm_compare import build_samples, union_keys, training_series, forecasters, runner, scoring, plots

LOCAL = Path(tempfile.mkdtemp(prefix="pmc_", dir=None if LOCAL_TEST else "/content"))
PREPARED = stage_bundle(DEMO, LOCAL / "bundle")                  # bundle v2 -> local disk (derived from DEMO/cnn78u_colab_bundle.zip if needed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("chartcnn from:", CNN_HOME, "| data:", PREPARED, "| device:", DEVICE)
from IPython.display import Image
if (HERE / "assets" / "flow.png").exists():
    display(Image(filename=str(HERE / "assets" / "flow.png"), width=900))   # same diagram as the top cell
""")

md("""
## 3. Configuration
Change settings here only. Validation is a random set of whole decision bars (all stocks at that bar), identical for every model; 80 bars ≈ 35k samples per target.
""")
code("""
SMOKE = LOCAL_TEST                                              # tiny code check; never a result
assert DEVICE == "cuda" or SMOKE, "Select a GPU runtime (Runtime > Change runtime type)"

EXPERIMENTS = [
    dict(name="vol_h1", target="vol", horizon=1),               # volatility: signal known to exist
    dict(name="dir_h1", target="dir", horizon=1),               # next bar strictly up vs down (zeros dropped)
    dict(name="xsdir_h12", target="xs_dir", horizon=12),        # 1-hour relative direction
    dict(name="dir_h12", target="dir", horizon=12),             # 1-hour absolute direction
]
CONTEXT = 78            # bars the models see = exactly the CNN's chart window (one trading day)
HORIZON = 12            # forecast once to 12 bars ahead, used by every target
STRIDE = 6              # same thinning as the CNN run
N_VALID_TIMES = 2 if SMOKE else 80
N_CALIB_TIMES = 2 if SMOKE else 40                              # training bars used for calibration only
SEED = 69

MODELS = [   # kind, Hugging Face id, inference batch size; add or remove freely
    dict(kind="chronos", name="chronos2", model_id="amazon/chronos-2", batch_size=1024),
    dict(kind="chronos", name="chronos_bolt_base", model_id="amazon/chronos-bolt-base", batch_size=1024),
    dict(kind="timesfm", name="timesfm", model_id="google/timesfm-3.0-pytorch", batch_size=512),
    dict(kind="moirai", name="moirai2_small", model_id="Salesforce/moirai-2.0-R-small", batch_size=1024),
    dict(kind="kronos", name="kronos_small", model_id="NeoQuasar/Kronos-small",
         tokenizer="NeoQuasar/Kronos-Tokenizer-base", repo="/content/Kronos", samples=8, batch_size=256),
]
FINETUNE = dict(enabled=True, base="amazon/chronos-2", mode="lora", learning_rate=1e-5,
                num_steps=3 if SMOKE else 2000, batch_size=8 if SMOKE else 256)

# Trained CNN to compare against: newest CNN-Relevant run with saved validation probabilities (or set a path / None).
runs = sorted((DEMO / "CNN-Relevant" / "runs").glob("*/settings.json"), key=lambda p: p.stat().st_mtime)
CNN_RUN = runs[-1].parent if runs else None

if SMOKE:
    MODELS = json.loads(os.environ["PMC_TEST_MODELS"])
    FINETUNE["base"] = MODELS[0]["model_id"]
RESUME_RUN_ID = None    # after a disconnect: put the old RUN_ID here to reuse its finished forecasts
RUN_ID = RESUME_RUN_ID or ("local_smoke" if SMOKE else datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))
RUN_DRIVE, RUN_LOCAL = HERE / "runs" / RUN_ID, LOCAL / "run"
if RESUME_RUN_ID:
    shutil.copytree(RUN_DRIVE, RUN_LOCAL, dirs_exist_ok=True)
RUN_DRIVE.mkdir(parents=True, exist_ok=True); RUN_LOCAL.mkdir(parents=True, exist_ok=True)
save_json(RUN_LOCAL / "settings.json", dict(experiments=EXPERIMENTS, context=CONTEXT, horizon=HORIZON, stride=STRIDE,
          n_valid_times=N_VALID_TIMES, n_calib_times=N_CALIB_TIMES, models=MODELS, finetune=FINETUNE,
          cnn_run=str(CNN_RUN)), RUN_DRIVE)
print("CNN run:", CNN_RUN, "\\nresults ->", RUN_DRIVE)
""")

md("## 4. Samples\nSame labels as the CNN notebook. Each (stock, bar) pair is forecast once and shared by all targets.")
code("""
seed_all(SEED)
store = chartcnn.Store(PREPARED)
SAMPLES = build_samples(store, EXPERIMENTS, STRIDE, N_VALID_TIMES, N_CALIB_TIMES, SEED)
KEYS = {part: union_keys(SAMPLES, part) for part in ("valid", "calib")}
display(pd.DataFrame({name: {f"{part} samples": len(d[part]) for part in ("train", "calib", "valid")}
                      | {"valid positive share": round(d["valid"].label.mean(), 3)}
                      for name, d in SAMPLES.items()}))
print({part: f"{len(k):,} series to forecast" for part, k in KEYS.items()})
""")

md("""
## 5. Zero-shot foundation models
For each model: load (prints the **total parameter count**), forecast validation and calibration series, score every target. Forecasts are cached in `runs/<RUN_ID>/forecasts/` on Drive; after a disconnect set `RESUME_RUN_ID` in section 3 and run all again, and finished models are not re-forecast. A model that fails to install or load is reported and skipped.
""")
code("""
ROWS, PROBS, FORECASTS = {}, {}, {}

def record(result, kind):
    rows, probs = scoring.evaluate_forecaster(result["name"], result["n_params"], result["q"]["valid"], result["q"]["calib"],
                                              KEYS["valid"], KEYS["calib"], SAMPLES, EXPERIMENTS, kind, result["seconds"])
    for exp, row in rows.items():
        ROWS.setdefault(exp, []).append(row)
        PROBS[(exp, result["name"])] = probs[exp]
    FORECASTS[result["name"]] = result["q"]["valid"]
    display(pd.DataFrame(rows).T[["parameters", "roc_auc", "log_loss_gain_vs_prior_pct", "mae_vs_random_walk"]])

for spec in MODELS:
    print("\\n" + "=" * 20, spec.get("name", spec["kind"]), "=" * 20)
    try:
        result = runner.run_model(spec, store, PREPARED, KEYS, CONTEXT, HORIZON, DEVICE,
                                  RUN_LOCAL / "forecasts", RUN_DRIVE / "forecasts")
    except Exception as err:
        print(f"SKIPPED {spec['model_id']}: {type(err).__name__}: {err}")
        continue
    record(result, "foundation (zero-shot)")
    result["model"].unload()
""")

md("Visual check: context (black), forecast median and 10–90% band, realised path (green) for a few validation series.")
code("""
from tsfm_compare.samples import close_contexts
ctx = close_contexts(store, KEYS["valid"].head(4), CONTEXT)
fut = KEYS["valid"].head(4)
realised = store.close[fut.security_index.to_numpy()[:, None], fut.end_index.to_numpy()[:, None] + np.arange(1, HORIZON + 1)]
realised = realised / store.close[fut.security_index.to_numpy(), fut.end_index.to_numpy()][:, None]
for name, q in FORECASTS.items():
    print(name); plots.plot_forecast_examples(q[:4], ctx, realised)
""")

md("""
## 6. Light fine-tune: Chronos-2 + LoRA (optional)
Trains on the January–March price series of every stock (forecast loss, no labels, nothing after the training end date), then is scored exactly like the zero-shot models.
""")
code("""
cached = all((RUN_LOCAL / "forecasts" / f"chronos2_ft_{part}.npy").exists() for part in ("valid", "calib"))
if FINETUNE["enabled"] and cached:          # resumed run: reuse the fine-tuned forecasts, skip training
    meta = RUN_LOCAL / "forecasts" / "chronos2_ft_params.json"
    n_params = json.loads(meta.read_text())["parameters"] if meta.exists() else None
    q = {part: np.load(RUN_LOCAL / "forecasts" / f"chronos2_ft_{part}.npy") for part in ("valid", "calib")}
    print(f"chronos2_ft: reusing cached forecasts ({n_params} parameters)")
    record(dict(name="chronos2_ft", n_params=n_params, q=q, seconds=None), "foundation (fine-tuned)")
elif FINETUNE["enabled"]:
    base = forecasters.build(dict(kind="chronos", name="chronos2", model_id=FINETUNE["base"]), DEVICE).load()
    series = training_series(store, max_stocks=20 if SMOKE else None, seed=SEED)
    print(f"fine-tune data: {len(series)} stock series, {sum(map(len, series)):,} bars")
    tuned = base.finetune(series, HORIZON, CONTEXT, mode=FINETUNE["mode"], learning_rate=FINETUNE["learning_rate"],
                          num_steps=FINETUNE["num_steps"], batch_size=FINETUNE["batch_size"],
                          output_dir=str(LOCAL / "chronos2_ft"))
    result = runner.run_model(None, store, PREPARED, KEYS, CONTEXT, HORIZON, DEVICE,
                              RUN_LOCAL / "forecasts", RUN_DRIVE / "forecasts", model=tuned)
    record(result, "foundation (fine-tuned)")
    save_json(RUN_LOCAL / "forecasts" / "chronos2_ft_params.json", dict(parameters=tuned.n_params), RUN_DRIVE / "forecasts")
    tuned.unload(); base.unload()
""")

md("## 7. Baselines and the trained CNN on the same samples")
code("""
BASE_ROWS = scoring.baseline_rows(store, SAMPLES, EXPERIMENTS, SEED, max_train=5_000 if SMOKE else 300_000)
CNN_ROWS = scoring.cnn_rows(store, SAMPLES, EXPERIMENTS, CNN_RUN) if CNN_RUN else {}
if not CNN_ROWS:
    print("No CNN validation probabilities found; run CNN-Relevant/ChartCNN_Baseline.ipynb first to add CNN rows.")
""")

md("""
## 8. Comparison
One table per target, sorted by validation AUC. Read it as:
- `roc_auc`: ranking skill, zero-shot for the foundation models (0.5 = none).
- `log_loss_gain_vs_prior_pct`: must be > 0 to beat the constant base rate.
- `mae_vs_random_walk`: < 1 means the median price forecast beats "no change".
- `long_short_bps`, `long_short_t`: top-minus-bottom decile within each bar, gross of costs (direction target only).
- `parameters`: total model parameters (HGB: tree nodes).
""")
code("""
TABLES = {exp["name"]: scoring.comparison_table(exp["name"], BASE_ROWS, CNN_ROWS, ROWS) for exp in EXPERIMENTS}
for name, table in TABLES.items():
    print(name); display(table.round(4))
    table.to_csv(RUN_LOCAL / f"comparison_{name}.csv")
pd.concat(TABLES, names=["experiment"]).to_csv(RUN_LOCAL / "comparison.csv")
plots.plot_comparison(TABLES, RUN_LOCAL / "comparison.png")
""")

md("## 9. Save")
code("""
archive = shutil.make_archive(str(LOCAL / "results"), "zip", root_dir=RUN_LOCAL)
for f in [archive, RUN_LOCAL / "comparison.csv", RUN_LOCAL / "comparison.png"]:
    shutil.copy2(f, RUN_DRIVE / Path(f).name)
print("saved to", RUN_DRIVE, "| RUN_ID:", RUN_ID)
""")

# Colab ignores Jupyter "attachment:" images, so the diagram is inlined as a data URI.
image = base64.b64encode((ROOT / "assets" / "flow.png").read_bytes()).decode()
cells[0]["source"] = cells[0]["source"].replace("FLOW_DATA_URI", "data:image/png;base64," + image)
for c in cells:                       # nbformat stores source as a list of lines
    c["source"] = c["source"].splitlines(keepends=True)
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                        kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Pretrained_Model_Comparison_dir_targets.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(out)
