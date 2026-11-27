"""Cell contents of the MindFlow notebooks (used by build_notebooks.py)."""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


FLOW = """
| step | question | notebooks |
|---|---|---|
| **1** | Does the JKX chart signal exist at the daily frequency on our large caps (also for pretrained models)? | `Step1a_Daily_JKX_CNN`, `Step1b_Daily_Pretrained` |
| **2** | Moved to 5-minute charts: does any model (CNN, ResNet, 1D CNN, ViT, HGB, frozen DINOv2/CLIP, time-series foundation models) find direction? | `Step2a_Intraday_Direction_Models`, `Step2b_Intraday_Direction_Pretrained` |
| **3** | Same models, target = volatility: is there a signal, and who wins? | `Step3a_Intraday_Volatility_Models`, `Step3b_Intraday_Volatility_Pretrained` |
| **4** | Add information to the picture step by step until the image model reaches HGB; is the image still useful beyond the features; which features matter? | `Step4a_Add_Image_Information`, `Step4b_Feature_Selection` |
| **5** | One figure and one parameter table over all steps | `Step5_Summary` |
| **6** | Next questions: does market-regime information (VIX, multi-day realized volatility) help? | `Step6a_VIX_Regime`, `Step6b_Final_Model` (final model + held-out test), `Step6c_Volatility_Use` (IC, long-short, risk sizing), `Step6d_Pure_Image` (the numbers drawn into the picture, no numbers input), `Step6e_Pure_Image_60min` |
| **7** | Efficient computing for the final model: 16-bit / INT8 inference, LoRA fine-tuning of a pretrained ViT | `Step7a_Efficient_Computing`, `Step7b_Distillation` (small students taught by the final CNN), `Step7b_plus_Pure_Image_Distillation` (small pure-image students taught by the 6d pure-image CNN) |
"""


def header(step, title, question, body=""):
    return md(f"""
# MindFlow Step {step} · {title}

**Question.** {question}

{body}

<details><summary>Where this notebook sits in the MindFlow</summary>

{FLOW}
</details>

**Data.** `alldata_bundle_v4.zip` (TAQ filter v4, point-in-time price adjustment): S&P 500 ∪ NASDAQ 100 ∪ DJIA 30
members by date, 2,514 sessions 2016-01-04 … 2025-12-31, five-minute bars plus SPY on the same clock.
**Split.** Train 2016–2022 · validation 2023 (every choice) · test 2024–2025 (last section, run once at the end).
**Advisor rule.** Every model prints its parameter count; for HGB the count is tree nodes.
""")


def setup(imports, gpu=True):
    """Shared setup: Drive, bundle (unpacked once per runtime), code from MindFlow/code."""
    device = ('DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
              'print("device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")')
    if not gpu:
        device = 'DEVICE = torch.device("cpu")'
    return [md("""
## 1. Setup
Put `alldata_bundle_v4.zip` and the `code/` folder in `Demo/MindFlow` on Drive. The bundle (1.3 GB) is copied to the
Colab disk and unzipped once per runtime; the next notebook in the same runtime reuses it.
"""), code(f"""
import os, sys, json, shutil, time
from pathlib import Path

LOCAL_TEST = os.environ.get("MINDFLOW_LOCAL_TEST") == "1"          # repository QA only (synthetic data)
if LOCAL_TEST:
    HERE, LOCAL_ROOT = Path(os.environ["MINDFLOW_HERE"]), Path(os.environ["MINDFLOW_LOCAL"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    HERE, LOCAL_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo/MindFlow"), Path("/content/mindflow")
CODE = HERE / "code"
assert (CODE / "mindflow.py").is_file(), f"Put the code/ folder in {{HERE}}"
sys.path[:0] = [str(CODE)]
import mindflow as mf
PREPARED = mf.stage_bundle(HERE / mf.BUNDLE_NAME, LOCAL_ROOT)
PANELS_ADDED = mf.stage_panels(HERE, PREPARED)          # e.g. vixy_panel_v4.zip, if uploaded
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
{imports.strip()}
{device}
""")]


# ============================================================================= Step 1a

def daily_cells():
    c = [header("1a", "Daily JKX replication", "The chart-image signal of Jiang, Kelly & Xiu (2023) is daily "
                "(images of 5 or 20 days, labels over the next 5 or 20 days). Is it there on our stocks and years?", """
Same images, network and training rule as JKX; only the universe (large caps) and years differ.

| | JKX (2023) | here |
|---|---|---|
| data | CRSP daily, all US stocks, 1993–2019 | daily bars built from our 5-minute bars (open of the first bar, high, low, 16:00 close, summed volume) |
| images | I5 (32×15), I20 (64×60): OHLC, moving average, volume | same |
| labels | sign of the next 5 / 20-day return (`dir`) | same, plus the cross-sectional version (`xsdir`) |
| CNN | 2 / 3 blocks, 64-128-256, 5×3 conv, dropout 0.5, Adam 1e-5, batch 128 | same (I5 155,138 · I20 708,866 parameters) |
| training | random 70/30 split of the training years, early stop after 2 bad epochs, 5 seeds averaged | same |

Benchmarks on identical rows: HGB on daily hand features and the classic rules (1-day and 1-week reversal, 1- and
3-month momentum). **Leakage controls**: each window is divided by its anchor day's multipliers (point in time);
labels are h-day total returns; training labels end inside the training years (purge); validation and test anchors
start h sessions after a boundary (embargo); evaluation uses one anchor every h days (non-overlapping labels).
""")]
    c += setup("import daily_jkx as dj\nprint('daily_jkx', dj.__version__)")
    c += [md("""
## 2. Configuration
**To resume after a disconnect**, set `RUN_ID` to the id printed by the first run and Run all again.
`QUICK = True`: one CNN instead of five, at most 60,000 training images, I20/R5 only (~10–15 min).
The full run (5 CNNs × 6 cells) takes roughly 1–2 hours on an L4 (images are small and stay on the GPU).
The JKX design is fixed in advance (nothing tuned on 2023), so `RUN_TEST = True` scores 2024–25 in the same run.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"
RUN_ID = None                                       # None = new run; or an old RUN_ID to resume it
QUICK = False
RUN_TEST = True
CONFIGS = [(5, 5), (20, 5), (20, 20)]               # (image days, label horizon days): I5/R5, I20/R5, I20/R20
TARGETS = ["dir", "xsdir"]
SETTINGS = dict(train_stride=5, min_bars=60, seeds=5, learning_rate=1e-5, batch_size=128, max_epochs=30,
                patience=2, min_delta=0.0, dropout=0.5, holdout_share=0.3, max_train=None, seed=69)
if QUICK:
    SETTINGS.update(seeds=1, max_train=60_000); CONFIGS = [(20, 5)]
if SMOKE:
    SETTINGS.update(seeds=2, max_epochs=2, batch_size=64)
CELLS = [dict(name=f"{t}_I{L}R{h}", image_days=L, horizon=h, target=t) for L, h in CONFIGS for t in TARGETS]
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "1a", RUN_ID, "daily", LOCAL_TEST)
mf.save_settings(RUN_DIR, RESUME, settings=SETTINGS, cells=CELLS, quick=QUICK, run_test=RUN_TEST, daily_jkx=dj.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
"""), md("""
## 3. Daily bars from the 5-minute bars
Sessions, share of stock-days present, and a check that the daily close equals the 16:00 bar's close.
"""), code("""
t0 = time.perf_counter()
store = dj.DailyStore(PREPARED, min_bars=SETTINGS["min_bars"])
cfg = store.config
print(f"{store.S} stocks x {store.D} sessions ({store.dates[0]} – {store.dates[-1]}) in {time.perf_counter() - t0:.0f}s")
years = pd.Series(store.dates).str[:4]
print("stock-days present by year:", (pd.Series(store.valid.mean(0)).groupby(years.values).mean().round(3)).to_dict())
print("splits: train ≤", cfg["train_end"], "| validation ≤", cfg["validation_end"], "| test after")
s0 = int(np.argmax(store.valid.sum(1)))
feats = np.load(PREPARED / "features.npy", mmap_mode="r")
last_bar_close = np.asarray(feats[s0, dj.SLOTS - 1::dj.SLOTS, 3], float)
ok = store.valid[s0]
assert np.allclose(store.ohlcv[s0, ok, 3], last_bar_close[ok]), "daily close is not the 16:00 bar's close"
print(f"stock {s0}: daily close equals the 16:00 bar close on all {ok.sum()} sessions")
"""), md("""
## 4. Anchors and images
One anchor every 5 sessions for every stock; each image length is rendered once and shared by every cell.
"""), code("""
import matplotlib.pyplot as plt
FRAMES, IMAGES = {}, {}
for L in sorted({L for L, _ in CONFIGS}):
    t0 = time.perf_counter()
    FRAMES[L] = dj.anchors(store, L, SETTINGS["train_stride"])
    imgs = dj.render_all(store, FRAMES[L], L)
    IMAGES[L] = torch.from_numpy(imgs).to(DEVICE)
    print(f"I{L}: {len(FRAMES[L]):,} images {imgs.shape[1:]} in {time.perf_counter() - t0:.0f}s")
    ex = FRAMES[L].sample(4, random_state=1)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, (i, r) in zip(axes, ex.iterrows()):
        ax.imshow(imgs[i], cmap="gray", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"I{L}  stock {r.security_index}  {r.date}", fontsize=8)
    fig.tight_layout(); fig.savefig(RUN_DIR / f"examples_I{L}.png", dpi=110); plt.show()
for L in IMAGES:
    m = dj.JKXCNN(L)
    print(f"JKX CNN I{L}: {dj.count(m):,} parameters (fully connected fan-in {m.flat:,})")
"""), md("""
## 5. Train (CNN × seeds, HGB, rules) per cell
Finished cells are saved to `runs/step1a_daily_jkx/<RUN_ID>/<cell>/` and skipped on a rerun.
"""), code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = dj.run_cell(store, cell, FRAMES[cell["image_days"]], IMAGES[cell["image_days"]],
                                        SETTINGS, DEVICE, RUN_DIR / cell["name"])
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
if not RUN_TEST:
    for r in RESULTS.values():
        r["predictions"] = r["predictions"][r["predictions"].split != "test"]
"""), md("""
## 6. Results
AUC with 95% intervals from resampling whole rebalance dates; accuracy and log-loss gain for the CNN and HGB;
`ls_*` = equal-weighted top-minus-bottom decile return over the next h days (bps per period, t-statistic, Sharpe).
"""), code("""
TABLE = dj.evaluate(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DIR / "daily_table.csv", index=False)
cols = ["image", "horizon_days", "model", "auc", "auc_low", "auc_high", "accuracy", "log_loss_gain_pct",
        "ls_mean_bps", "ls_t", "ls_sharpe_ann", "dates", "rows"]
for split in ("validation", "test"):
    for tgt in TARGETS:
        t = TABLE[(TABLE.split == split) & (TABLE.target == tgt)]
        if len(t):
            print(f"{split} — {tgt}: {dj.TARGETS[tgt]}")
            display(t[[c for c in cols if c in t]].round(4))
PARAMS = pd.DataFrame([dict(cell=k, **v["info"]["parameters"]) for k, v in RESULTS.items()]).set_index("cell")
print("Parameters (CNN: weights; HGB: tree nodes):"); display(PARAMS)
TAKEAWAYS = dj.takeaways(TABLE)
(RUN_DIR / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **CNN AUC clearly above 0.5 with a positive long-short**: the JKX signal exists here at the daily frequency.
- **≈ 0.50 everywhere** (what the v1 data gave): no daily image signal in these large caps. JKX find most of their
  effect in small stocks (equal-weighted Sharpe ≫ value-weighted), which this universe does not contain.
- Step 1b runs pretrained models on exactly these rows; Step 5 puts the daily and 5-minute AUCs side by side.
""")]
    return c


# ============================================================================= Step 1b

def daily_pretrained_cells():
    c = [header("1b", "Daily data, pretrained models", "Do models pretrained on other data find the daily "
                "direction signal that the JKX CNN does not?", """
Two families, both scored on the **same validation and test rows as Step 1a** (its CNN, HGB and rule scores are joined
row by row):

- **Frozen image encoders**: DINOv2-S (self-supervised on natural images), CLIP-B/16 (image–text) and a randomly
  initialised ViT-S (the floor that says whether pretraining matters) read Step 1a's JKX images, resized to 224×224.
  Their features ([CLS, mean patch]) feed a linear or a 2-layer MLP probe trained on 2016–2022. Probe regularisation and
  early stopping use a random 30% of the training sample, as for the JKX CNN; 2023 is never used for a choice.
- **Time-series foundation models**, zero-shot: Chronos-2, TimesFM, Moirai 2 and Kronos (trained on candlesticks)
  forecast the next h daily closes from the last 60 sessions; the score is the median forecast return (for `xsdir`
  minus its cross-sectional median).
""")]
    c += [md("""
### Install the foundation-model packages (once per runtime)
Moirai needs `uni2ts` without its pinned dependencies; Kronos is cloned from GitHub. Restart is not needed.
"""), code("""
import os, subprocess, sys
if os.environ.get("MINDFLOW_LOCAL_TEST") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting>=2.0", "timesfm[torch]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "uni2ts"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gluonts", "lightning", "einops", "jaxtyping",
                    "hydra-core"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # breaks peft on Colab
    if not os.path.isdir("/content/Kronos"):
        subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos", "/content/Kronos"], check=False)
""")]
    c += setup("""
import daily_jkx as dj, frozen_probe as fp, pretrained_ts as pt, daily_pretrained as dp
print("daily_pretrained", dp.__version__, "| pretrained_ts", pt.__version__)
""")
    c += [md("""
## 2. Configuration
**To resume**, set `RUN_ID`. `STEP1A_RUN = None` uses the newest Step 1a run; this notebook must use the same
`CONFIGS`, `TARGETS` and `train_stride` as that run (checked below).

Time on an L4 (estimate): images ~2 min per length; encoders ~10 min for DINOv2-S + random ViT-S + CLIP-B/16;
probes ~1–3 min per cell. Chronos-2 and TimesFM take minutes; Kronos samples paths autoregressively and is the slow one
(roughly 1 hour per horizon for the full validation + test rows), so it uses `tsfm_max_rows`.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"
RUN_ID = None
STEP1A_RUN = None                                   # None = newest run of Step 1a
QUICK = False
CONFIGS = [(5, 5), (20, 5), (20, 20)]
TARGETS = ["dir", "xsdir"]
ENCODER_NAMES = ["dinov2_s", "dinov2_s_random", "clip_b16"]
PROBES = ["linear", "mlp"]
TSFM = ["randomwalk", "chronos2", "timesfm", "moirai2", "kronos"]  # randomwalk = yardstick; a model that fails to install is skipped
TSFM_SPLITS = ("validation", "test")
SETTINGS = dict(train_stride=5, min_bars=60, holdout_share=0.3, max_train=None, seed=69,
                probe_lr=1e-3, probe_batch=4096, probe_epochs=30, probe_patience=3,
                linear_l2_grid=[1e-5, 1e-4, 1e-3], mlp_hidden=512, mlp_dropout=0.2, mlp_weight_decay_grid=[1e-2, 1e-1],
                encode_batch=512, amp=True, tsfm_context=60, tsfm_max_rows=None)
pt.MODELS["kronos"]["repo"] = "/content/Kronos"
if QUICK:
    CONFIGS, ENCODER_NAMES, TSFM = [(20, 5)], ["dinov2_s", "dinov2_s_random"], ["randomwalk", "chronos2"]
    SETTINGS.update(max_train=60_000)
if SMOKE:
    ENCODER_NAMES, TSFM = ["dinov2_s", "dinov2_s_random"], ["randomwalk"]
    SETTINGS.update(probe_epochs=3, probe_batch=64, linear_l2_grid=[1e-4], mlp_weight_decay_grid=[1e-2], encode_batch=64)
CELLS = [dict(name=f"{t}_I{L}R{h}", image_days=L, horizon=h, target=t) for L, h in CONFIGS for t in TARGETS]
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "1b", RUN_ID, "dailypre", LOCAL_TEST)
STEP1A = mf.latest_run(HERE, "1a", STEP1A_RUN)
if STEP1A is not None:
    s1a = json.loads((STEP1A / "settings.json").read_text())["settings"]
    assert s1a["train_stride"] == SETTINGS["train_stride"] and s1a["min_bars"] == SETTINGS["min_bars"], "Step 1a used other anchors"
mf.save_settings(RUN_DIR, RESUME, settings=SETTINGS, cells=CELLS, encoders=ENCODER_NAMES, probes=PROBES, tsfm=TSFM,
                 step1a=str(STEP1A), quick=QUICK)
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
print("Step 1a run:", STEP1A if STEP1A else "none found: pretrained models are scored alone")
"""), md("## 3. Daily bars, anchors and images (as Step 1a)"), code("""
store = dj.DailyStore(PREPARED, min_bars=SETTINGS["min_bars"])
FRAMES, IMAGES = {}, {}
for L in sorted({L for L, _ in CONFIGS}):
    t0 = time.perf_counter()
    FRAMES[L] = dj.anchors(store, L, SETTINGS["train_stride"])
    IMAGES[L] = dj.render_all(store, FRAMES[L], L)
    print(f"I{L}: {len(FRAMES[L]):,} images {IMAGES[L].shape[1:]} in {time.perf_counter() - t0:.0f}s")
"""), md("""
## 4. Frozen encoders: features of every image (cached on the Colab disk)
Each encoder prints its parameter count (all frozen).
"""), code("""
ENCODERS = fp.build_encoders(ENCODER_NAMES, DEVICE, tiny=SMOKE, seed=SETTINGS["seed"])
ENCODER_PARAMS = {e: enc.parameters_count()[0] for e, enc in ENCODERS.items()}
FEATS = {L: dp.encode(IMAGES[L], ENCODERS, LOCAL_ROOT / f"daily_features_{RUN_ID}", f"I{L}", DEVICE,
                      SETTINGS["encode_batch"], SETTINGS["amp"]) for L in IMAGES}
del ENCODERS
if DEVICE.type == "cuda": torch.cuda.empty_cache()
"""), md("""
## 5. Time-series foundation models (zero-shot forecasts, cached on Drive)
One forecast per evaluation anchor and horizon; each model prints its parameter count.
"""), code("""
TS, TS_PARAMS = {}, {}
for name in list(TSFM):
    try:
        q, p = dp.forecast_daily(store, FRAMES, CELLS, [name], SETTINGS, DEVICE, RUN_DIR / "tsfm", TSFM_SPLITS)
        TS.update(q); TS_PARAMS.update(p)
    except Exception as err:                      # a package that failed to install: report and continue
        print(f"{name} skipped: {type(err).__name__}: {err}")
_ = (RUN_DIR / "parameters.json").write_text(json.dumps({**{f"frozen_{k}": v for k, v in ENCODER_PARAMS.items()}, **TS_PARAMS}, indent=2))
"""), md("## 6. Probes per cell, joined with Step 1a"), code("""
RESULTS = {}
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = dp.run_cell(store, cell, FRAMES[cell["image_days"]], FEATS[cell["image_days"]],
                                        ENCODER_NAMES, PROBES, TS, SETTINGS, DEVICE, RUN_DIR / cell["name"], STEP1A)
"""), md("""
## 7. Results on identical rows
Same evaluation as Step 1a (AUC with date-bootstrap CI, decile long-short). Time-series models are scored on the rows
they forecast (all of them unless `tsfm_max_rows` thinned the dates).
"""), code("""
TABLE = dj.evaluate(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DIR / "daily_pretrained_table.csv", index=False)
cols = ["image", "horizon_days", "model", "auc", "auc_low", "auc_high", "ls_mean_bps", "ls_t", "dates", "rows"]
for split in ("validation", "test"):
    for tgt in TARGETS:
        t = TABLE[(TABLE.split == split) & (TABLE.target == tgt)]
        if len(t):
            print(f"{split} — {tgt}"); display(t[[c for c in cols if c in t]].round(4))
rows = [dict(model=f"frozen {k} (encoder)", parameters=v) for k, v in ENCODER_PARAMS.items()]
rows += [dict(model=pt.label(k[5:]), parameters=v) for k, v in TS_PARAMS.items()]
first = next(iter(RESULTS.values()))["info"]
rows += [dict(model=dp.MODEL_LABEL.get(k, k) + " (trainable head)", parameters=v["parameters"]) for k, v in first["probes"].items()]
print("Parameters:"); display(pd.DataFrame(rows).set_index("model"))
"""), md("""
### How to read it
- **Pretrained ≈ 0.50 like the JKX CNN**: no daily direction signal in these stocks, whatever the model.
- **Pretrained image probe > random ViT-S but ≈ 0.5**: pretraining changes the features, not the conclusion.
- **A foundation model clearly above 0.5 on validation and test with a positive long-short**: worth a closer look.
""")]
    return c


from cells_intraday import grid_cells, probe_cells, indicator_cells, feature_cells, summary_cells, regime_cells  # noqa: E402,F401
from cells_efficiency import efficiency_cells  # noqa: E402,F401
from cells_final import final_cells  # noqa: E402,F401
from cells_vol_use import vol_use_cells  # noqa: E402,F401
from cells_distill import distill_cells  # noqa: E402,F401
from cells_pure import pure_cells  # noqa: E402,F401
from cells_pure_distill import pure_distill_cells  # noqa: E402,F401
