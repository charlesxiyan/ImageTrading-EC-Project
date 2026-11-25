"""Cells of the 5-minute MindFlow notebooks (Steps 2-5)."""
from cells import code, header, md, setup

ALL_ARCHS = ["plain", "plain_spy", "residual", "series1d", "series1d_spy", "vit", "vit_spy"]


# ============================================================================= Steps 2a / 3a

def grid_cells(step):
    direction = step == "2a"
    if direction:
        h = header("2a", "5-minute charts, direction, every model",
                   "Moved from daily to 5-minute charts of one session (78 bars): does any architecture predict the "
                   "direction of the next 5–60 minutes?", """
Targets: `dir` (return > 0) and `xsdir` (return > the cross-sectional median at that time), at 5 and 60 minutes by
default (add 15 and 30). One sample per stock per hour (stride 12) for every horizon, so labels of one stock never
overlap.
""")
    else:
        h = header("3a", "5-minute charts, volatility target",
                   "Same charts and the same models as Step 2, but the target is the size of the move: is "
                   "|r(next h)| above the stock's own training-period median? Is there a signal, and which model wins?", """
Why volatility: returns are r = μ + σ·ε with ε unpredictable, so direction is a coin flip, but σ is persistent
(volatility clustering, ARCH/GARCH) and has an intraday U-shape, so the size of the next move is predictable. The
threshold is each stock's own median, so volatile stocks do not win by construction.
""")
    models = """
**Models** (same rows for every model; parameter counts printed in section 5 and saved per cell):

| model | input | parameters |
|---|---|---|
| `plain` image CNN | 234×234 chart of the stock's 78 bars | 776,450 |
| `plain_spy` image CNN + SPY | 468×234: stock chart above SPY's chart, bar columns aligned | 936,194 |
| `residual` ResNet-style image CNN | 234×234 chart | 2,047,362 |
| `series1d` 1D CNN | the 5 OHLCV series, scaled exactly as the chart is | 720,770 |
| `series1d_spy` 1D CNN + SPY | the stock's 5 series and SPY's 5 | 723,970 |
| `vit` image ViT (Tokens-to-Token style, from scratch) | 234×234 chart, 169 tokens, 8 blocks of width 96 | 864,002 |
| `vit_spy` image ViT + SPY | 468×234 (338 tokens) | 880,226 |
| HGB hand features | 20 numbers of the same 78 bars (returns, realized vol, volume, time of day) | tree nodes |
| HGB hand + SPY | the 20 plus 18 SPY / stock-vs-SPY numbers | tree nodes |
| rules | last-bar return, 78-bar momentum (direction); mean recent |r| (volatility) | 0 |

The CNNs use Adam 1e-5 as in Jiang–Kelly–Xiu; the ViTs need their own rate (1e-4). Pretrained models (frozen DINOv2 /
CLIP, time-series foundation models) are in the `b` notebook of this step, on the same rows.
"""
    c = [h, md(models)]
    c += setup("""
import chartcnn
from chartcnn import grid
from chartcnn.utils import save_json
assert chartcnn.__version__ >= "2026-11-23.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
torch.backends.cudnn.benchmark = True
""")
    kinds = '["xsdir", "dir"]' if direction else '["vol"]'
    horizons = "[12, 1]" if direction else "[1, 12]"
    c += [md(f"""
## 2. Configuration
**To resume after a disconnect**, set `RUN_ID` to the id printed by the first run and Run all again: finished cells
are skipped and an unfinished network resumes from its last epoch. To add the 15- and 30-minute cells later, set
`HORIZONS = {horizons[:-1]}, 3, 6]` with the same `RUN_ID`.

**To retrain only some networks of a finished run** (e.g. the ViTs after the 2026-11-23 fix), set `RUN_ID` to that run
and `RETRAIN_ARCHS = ["vit", "vit_spy"]`, keep every other setting, and Run all. Only those networks train again, on
exactly the same rows (checked); HGB, rules and the CNNs are kept, the old networks are moved to
`<cell>/<arch>_before_<version>/`, and the tables below are rebuilt with the new scores.

**To add the VIXY panel (intraday VIX proxy) to a finished run**: upload `vixy_panel_v4.zip` (built on the Mac with
`MindFlow/tools/build_panel_bundle.py`) next to the bundle in `Demo/MindFlow`, set `RUN_ID` to that run and
`RETRAIN_ARCHS = VIXY_ARCHS`, and Run all. The networks in `VIXY_ARCHS` (default: CNN and ViT with SPY + VIXY panels, three
stacked charts, 702 × 234) are trained on the same rows and added to the tables, next to their `_spy` versions.

**`QUICK = True`** is a small first pass (40,000 training windows, 3 epochs, every 5th validation day) to check that
everything runs; its numbers are rough.

**Time (estimate on an L4, not measured on v4):** 10–20 min per image CNN or ViT and 3–6 min per 1D CNN per cell,
so roughly 1–1.5 hours per cell for the seven networks, {"4 cells ≈ 5 hours" if direction else "2 cells ≈ 2.5 hours"}.
The networks train on a random 300,000 training windows (`max_train_nn`); HGB uses up to 1,000,000. Early stopping
watches every 4th validation day; the reported validation scores use all of 2023.
"""), code(f"""
SMOKE = LOCAL_TEST                                # tiny code check on synthetic data; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"
RUN_ID = None                                     # None = new run; or an old RUN_ID to resume / extend it
QUICK = False
SETTINGS = dict(
    learning_rate=1e-5, weight_decay=0.0,         # Adam, as in Jiang-Kelly-Xiu
    batch_size=4 if SMOKE else 128, eval_batch_size=8 if SMOKE else 1024,
    epochs=1 if SMOKE else 8, patience=2, min_delta=1e-5,       # early stop on validation log loss
    channels=[64, 128, 256], head_height=4, dropout=0.5,
    vit_learning_rate=3e-4, vit_weight_decay=0.05, vit_warmup_steps=1000, vit_grad_clip=1.0,   # AdamW
    vit_pool="mean", vit_dim=96, vit_depth=8, vit_heads=4, vit_mlp_ratio=3.0, vit_drop_path=0.1, vit_dropout=0.1,
    max_train_nn=300_000, max_train_hgb=1_000_000, early_stop_every_kth_day=4,
    amp=True,                                     # bfloat16 autocast on A100/L4/H100; ignored on T4/CPU
    num_workers=0 if SMOKE else min(8, os.cpu_count() or 2),
    seed=69,
)
if QUICK:
    SETTINGS.update(max_train_nn=40_000, max_train_hgb=200_000, validation_every_kth_day=5, epochs=3)
HORIZONS = {horizons}                             # bars of 5 minutes (1 = 5 min, 12 = 60 min); add 3, 6 later
KINDS = {kinds}
STRIDE = 12                                       # one sample per stock per hour, for every horizon
ARCHS = {ALL_ARCHS!r}
RETRAIN_ARCHS = []                                # e.g. ["vit", "vit_spy"] with an old RUN_ID: retrain only these
VIXY_ARCHS = ["plain_spy_vixy", "vit_spy_vixy"]   # also possible: "plain_vixy", "vit_vixy"
if SMOKE:
    ARCHS = ["plain", "plain_spy", "series1d", "vit"]
    VIXY_ARCHS = ["plain_spy_vixy", "vit_vixy"]
if "vixy" in PANELS_ADDED:                        # vixy_panel_v4.zip is in Demo/MindFlow
    ARCHS = ARCHS + [a for a in VIXY_ARCHS if a not in ARCHS]
CELLS = grid.grid_cells(HORIZONS, KINDS, STRIDE)
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "{step}", RUN_ID, "grid", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, cells=CELLS, archs=ARCHS, retrain=RETRAIN_ARCHS, quick=QUICK,
                 chartcnn=chartcnn.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS), "|", len(ARCHS), "networks:", ", ".join(ARCHS))
"""), md("""
## 3. Data overview
Sessions, securities and windows per split as built and audited locally (`manifest.json`).
"""), code("""
store = chartcnn.Store(PREPARED)
M = json.loads((PREPARED / "manifest.json").read_text())
print("stocks:", len(M["permnos"]), "| slots:", store.T, "| sessions:", store.T // 78,
      "| early-close sessions excluded:", len(M["config"].get("early_close_sessions_excluded", [])))
print("SPY in bundle:", store.has_market, "| SPY valid slots:", M["bundle"]["market_valid_slots"])
print("windows in bundle:", M["bundle"]["windows"])
display(pd.DataFrame(M["splits"]).T[["samples", "decision_start", "decision_end"]])
"""), md("""
## 4. What the models see
Training samples drawn by the same dataset code the models use: the stock chart alone, and with SPY's panel below.
"""), code("""
import matplotlib.pyplot as plt
sample, _ = chartcnn.build_split(store, "train", "vol", 1, STRIDE)
show = sample.sample(4, random_state=1).reset_index(drop=True)
plain, with_spy = chartcnn.make_dataset(PREPARED, show, "plain"), chartcnn.make_dataset(PREPARED, show, "plain_spy")
fig, axes = plt.subplots(2, 4, figsize=(16, 9), gridspec_kw=dict(height_ratios=[1, 2]))
for i in range(4):
    axes[0, i].imshow(plain.images(i), cmap="gray", aspect="auto")
    axes[1, i].imshow(with_spy.images(i), cmap="gray", aspect="auto")
    axes[0, i].set_title(f"PERMNO {show.permno[i]}  {store.dates[show.end_index[i]]}")
    for ax in axes[:, i]: ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("plain (234x234)"); axes[1, 0].set_ylabel("plain_spy (468x234): stock / SPY")
fig.tight_layout(); fig.savefig(RUN_LOCAL / "what_models_see.png", dpi=110); plt.show()
shutil.copy2(RUN_LOCAL / "what_models_see.png", RUN_DRIVE / "what_models_see.png")
"""), md("""
## 5. Models and parameter counts
Layer-by-layer output shapes and parameters. Output layers start at zero, so every initial prediction is exactly 0.5.
"""), code("""
PARAM_ROWS = []
for arch in ARCHS:
    data = chartcnn.make_dataset(PREPARED, show, arch)
    x = torch.stack([data[i][0] for i in range(4)])
    model = chartcnn.build_model(grid.arch_settings(SETTINGS, arch))
    table, total, trainable = chartcnn.parameter_table(model)
    print(f"{arch} ({grid.ARCH_LABEL[arch]}): input {tuple(x.shape[1:])}, {total:,} parameters ({trainable:,} trainable)")
    display(table)
    print("initial P(y=1):", model.eval()(x).softmax(1)[:, 1].tolist(), "\\n")
    PARAM_ROWS.append(dict(model=grid.ARCH_LABEL[arch], arch=arch, input=str(tuple(x.shape[1:])), parameters=total))
del model
PARAMS = pd.DataFrame(PARAM_ROWS).set_index("arch")
PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 6. Labels
Sample counts and positive shares per cell (train 2016–2022, validation 2023).
"""), code("""
OVERVIEW = grid.label_overview(store, CELLS)
display(OVERVIEW.round(3)); OVERVIEW.to_csv(RUN_LOCAL / "labels.csv")
"""), md("""
## 7. Train the grid (validation 2023)
Per cell: labels → rules and both HGBs → each network in `ARCHS` → validation predictions in
`runs/<step>/<RUN_ID>/<cell>/`. Watch `val AUC`, the log-loss gain vs the prior and `prob std` (≈ 0 = collapsed).
"""), code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = grid.run_cell(PREPARED, cell, SETTINGS, ARCHS, DEVICE, RUN_LOCAL / cell["name"],
                                          RUN_DRIVE / cell["name"], smoke=SMOKE, retrain=RETRAIN_ARCHS)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
### Validation analysis
AUC per cell and model with 95% intervals from resampling whole validation days; paired comparisons (with vs without
SPY, ViT vs CNN, image vs 1D, CNN vs HGB) use the same resampled days for both models.
"""), code("""
TABLE, BOOTS = grid.grid_table(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_LOCAL / "grid_table.csv")
for model in ["CNN (plain)", "CNN (plain_spy)", "CNN (residual)", "ViT (vit)", "ViT (vit_spy)", "CNN (series1d)",
              "CNN (series1d_spy)", "HGB hand features", "HGB hand + SPY"]:
    if model in TABLE.index.get_level_values("model"):
        print(model); display(grid.auc_matrix(TABLE, model))
display(TABLE.drop(columns=["kind", "horizon"]).round(4))
grid.plot_grid(TABLE, RUN_LOCAL / "auc_by_horizon.png")
TAKEAWAYS = grid.takeaways(TABLE, BOOTS)
(RUN_LOCAL / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
for name in ("labels.csv", "grid_table.csv", "auc_by_horizon.png", "takeaways.md"):
    shutil.copy2(RUN_LOCAL / name, RUN_DRIVE / name)
print("saved to", RUN_DRIVE)
"""), md(("""
### How to read it
- **Every model's 95% interval contains 0.5 and no decile long-short has |t| ≥ 2**: no direction signal at 5–60
  minutes in these stocks, whatever the architecture. The problem is the target, not the model.
- **`prob std` ≈ 0 for a network**: it collapsed to the training prior (a constant prediction), which on a coin-flip
  target is the right answer.
""" if direction else """
### How to read it
- **All AUCs clearly above 0.5**: the size of the next move is predictable from the chart (volatility clustering
  plus the intraday U-shape). This is also the positive control: the pipeline can learn.
- **HGB vs image CNN** (paired line in the takeaways): on v1 data HGB won by 0.02–0.03 in every cell. Step 4 asks why.
- **ViT vs CNN**: the first v4 run's scratch ViT collapsed (constant output, AUC 0.500) because its tokens carried
  almost no image information at initialisation; fixed in chartcnn 2026-11-23.2 (fan-in init and LayerNorm in the
  tokenizer, mean pooling, AdamW with warmup and clipping, float32 head). Frozen pretrained ViTs are in Step 3b.
""")), md("""
## 8. Held-out test 2024–2025 (run once, at the end)
Only after every choice above is final. Each network is scored from its best validation checkpoint; HGB is refit on the
same training rows. Set `RUN_TEST = True` and run this cell. Do not change anything after looking at it.
"""), code("""
RUN_TEST = SMOKE                                  # set True once, at the very end
if RUN_TEST:
    TEST = {}
    for cell in CELLS:
        folder = (RUN_LOCAL if SMOKE else RUN_DRIVE) / cell["name"]
        TEST[cell["name"]] = grid.evaluate_cell_test(PREPARED, cell, SETTINGS, ARCHS, DEVICE, folder, smoke=SMOKE)
    TEST_TABLE, TEST_BOOTS = grid.grid_table(TEST, n_boot=50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_LOCAL / "test_grid_table.csv")
    display(TEST_TABLE.drop(columns=["kind", "horizon"]).round(4))
    display(Markdown(grid.takeaways(TEST_TABLE, TEST_BOOTS)))
    shutil.copy2(RUN_LOCAL / "test_grid_table.csv", RUN_DRIVE / "test_grid_table.csv")
""")]
    return c


# ============================================================================= Steps 2b / 3b

def probe_cells(step):
    direction = step == "2b"
    ref_step = "2a" if direction else "3a"
    target_text = "direction (`dir`, `xsdir`)" if direction else "volatility (`vol`)"
    c = [header(step, f"5-minute charts, {'direction' if direction else 'volatility'}, pretrained models",
                f"Do pretrained models (frozen vision encoders, time-series foundation models) find more than the "
                f"models trained from scratch in Step {ref_step}? Target: {target_text}.", f"""
**A. Frozen vision encoders.** DINOv2-S, CLIP-B/16 and a random ViT-S (floor) read every 234×234 chart resized to
224×224; features [CLS, mean patch] → linear or 2-layer MLP probe, with and without SPY's chart features. Probe
regularisation and early stopping use every 4th validation day, as the Step {ref_step} networks did.

**B. Time-series foundation models**, zero-shot: Chronos-2, TimesFM, Moirai 2 and Kronos forecast the next h
five-minute closes from the last 78 bars (Kronos also reads OHLCV and timestamps). Score:
{"median forecast return (`dir`), minus its cross-sectional median (`xsdir`)" if direction else
 "mean absolute forecast return over the 9 quantiles, divided by the stock's training median |r| (the label threshold)"}.
They run on every `tsfm_every_kth_day`-th validation day (sampling paths is slow).

**Identical rows.** Step {ref_step}'s scores (image CNN, ViT, 1D CNN, HGB) are joined on (stock, decision bar); only rows
present in both are kept. Inputs here span one session, and sessions next to a split boundary are removed (embargo).
""")]
    c += [md("### Install the foundation-model packages (once per runtime)"), code("""
import os, subprocess, sys
if os.environ.get("MINDFLOW_LOCAL_TEST") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting>=2.0", "timesfm[torch]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "uni2ts"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gluonts", "lightning", "einops", "jaxtyping",
                    "hydra-core"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    if not os.path.isdir("/content/Kronos"):
        subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos", "/content/Kronos"], check=False)
""")]
    c += setup("""
import chartcnn
import frozen_probe as fp, pretrained_ts as pt
assert chartcnn.__version__ >= "2026-11-23.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
fp.REFERENCE_MODELS.update({f"tsfm_{k}": pt.label(k) for k in pt.MODELS})
print("frozen_probe", fp.__version__, "| pretrained_ts", pt.__version__)
""")
    kinds = '["xsdir", "dir"]' if direction else '["vol"]'
    horizons = "[12, 1]" if direction else "[1, 12]"
    c += [md(f"""
## 2. Configuration
**To resume**, set `RUN_ID`. `REFERENCE_RUN = None` uses the newest Step {ref_step} run; keep `HORIZONS` and `KINDS`
equal to that run's cells. `QUICK = True`: 40,000 training windows, every 5th validation day, DINOv2 + random only,
Chronos-2 only.

**Time on an L4 (estimate):** the full run draws ~300,000 training and ~550,000 validation charts once and feeds each to
every encoder: 20–40 min for DINOv2-S + random ViT-S, 30–60 min more for CLIP-B/16; probes 1–3 min per cell. Chronos-2
and TimesFM: minutes. Kronos: roughly 30–60 min per horizon on every 10th validation day.
Features stay on the Colab disk (`FEATURES_ON_DRIVE = True` also keeps them on Drive, ~5 GB, for resuming).
"""), code(f"""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"
RUN_ID = None
REFERENCE_RUN = None                               # None = newest Step {ref_step} run
QUICK = False
ENCODER_NAMES = ["dinov2_s", "dinov2_s_random", "clip_b16"]   # add "dinov2_b" for DINOv2 ViT-B/14
PROBES = ["linear", "mlp"]
TSFM = ["randomwalk", "chronos2", "timesfm", "moirai2", "kronos"]   # randomwalk = yardstick, not pretrained
FEATURES_ON_DRIVE = False
SETTINGS = dict(
    stride=12, embargo_sessions=2, max_train_windows=300_000, validation_every_kth_day=1, early_stop_every_kth_day=4,
    extract_batch=256, chunk=50_000, amp=True, num_workers=0 if SMOKE else min(12, os.cpu_count() or 2),
    probe_lr=1e-3, probe_batch=4096, probe_epochs=30, probe_patience=3, linear_l2_grid=[1e-5, 1e-4, 1e-3],
    mlp_hidden=512, mlp_dropout=0.2, mlp_weight_decay_grid=[1e-2, 1e-1], seed=69,
    tsfm_context=78, tsfm_every_kth_day=5, kronos_every_kth_day=10,
)
pt.MODELS["kronos"]["repo"] = "/content/Kronos"
HORIZONS = {horizons}
KINDS = {kinds}
if QUICK:
    SETTINGS.update(max_train_windows=40_000, validation_every_kth_day=5, tsfm_every_kth_day=2)
    ENCODER_NAMES, TSFM = ["dinov2_s", "dinov2_s_random"], ["randomwalk", "chronos2"]
if SMOKE:
    SETTINGS.update(max_train_windows=300, extract_batch=16, chunk=128, probe_epochs=3, probe_batch=64,
                    embargo_sessions=1, early_stop_every_kth_day=1, linear_l2_grid=[1e-4], mlp_weight_decay_grid=[1e-2],
                    tsfm_every_kth_day=1)
    ENCODER_NAMES, TSFM = ["dinov2_s", "dinov2_s_random"], ["randomwalk"]
CELLS = [dict(name=f"{{k}}_h{{h}}", kind=k, target=fp.KINDS[k], horizon=h, minutes=5 * h, stride=SETTINGS["stride"])
         for h in HORIZONS for k in KINDS]
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "{step}", RUN_ID, "probe", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
REF_DIR = mf.latest_run(HERE, "{ref_step}", REFERENCE_RUN)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, encoders=ENCODER_NAMES, probes=PROBES, tsfm=TSFM, cells=CELLS,
                 quick=QUICK, reference=str(REF_DIR), chartcnn=chartcnn.__version__, frozen_probe=fp.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step {ref_step} reference run:", REF_DIR if REF_DIR else "none found: pretrained models are compared with each other only")
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
"""), md("""
## 3. Windows, purge and embargo
One fixed set of windows per split, shared by every cell, encoder, probe and forecaster (saved on Drive). The checks fail
loudly if a validation input reaches into the training period or a training label ends after it.
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
EMBARGO = fp.embargo_dates(store, SETTINGS["embargo_sessions"])
W = {}
for split, kw in (("train", dict(max_rows=SETTINGS["max_train_windows"])),
                  ("valid", dict(every_kth_day=SETTINGS["validation_every_kth_day"]))):
    path = RUN_DRIVE / f"windows_{split}.parquet"
    if path.exists():
        W[split] = pd.read_parquet(path)
    else:
        W[split] = fp.decision_windows(store, "train" if split == "train" else "validation", SETTINGS["stride"],
                                       EMBARGO, seed=SETTINGS["seed"], **kw)
        W[split].to_parquet(path, index=False)
first_valid_input = store.dates[W["valid"].end_index.min() - fp.LOOKBACK + 1]
last_train_label = store.dates[W["train"].end_index.max() + max(HORIZONS)]
assert first_valid_input > cfg["train_end"], "a validation input window reaches into the training period"
assert last_train_label <= cfg["train_end"], "a training label ends after the training period"
print(f"train windows {len(W['train']):,}, last training label bar on {last_train_label}")
print(f"validation windows {len(W['valid']):,} on {W['valid'].decision_date.nunique()} days, first input bar {first_valid_input}")
SLOTS = np.unique(np.concatenate([W["train"].end_index, W["valid"].end_index]))
SLOT_POS = np.full(store.T, -1, np.int64); SLOT_POS[SLOTS] = np.arange(len(SLOTS))
"""), md("""
## 4. A · Frozen encoders: features (parameter counts printed; all frozen)
"""), code("""
ENCODERS = fp.build_encoders(ENCODER_NAMES, DEVICE, tiny=SMOKE, seed=SETTINGS["seed"])
ENCODER_PARAMS = {e: enc.parameters_count()[0] for e, enc in ENCODERS.items()}
FEAT_LOCAL, FEAT_DRIVE = LOCAL_ROOT / f"features_{RUN_DRIVE.parent.name}_{RUN_ID}", RUN_DRIVE / "features"
FEATS = {e: {} for e in ENCODER_NAMES}
kw = dict(batch_size=SETTINGS["extract_batch"], workers=SETTINGS["num_workers"], chunk=SETTINGS["chunk"],
          amp=SETTINGS["amp"], drive_dir=FEAT_DRIVE if FEATURES_ON_DRIVE else None)
t0 = time.perf_counter()
for split in ("train", "valid"):
    out = fp.extract(fp.StockCharts(PREPARED, W[split]), ENCODERS, FEAT_LOCAL, split, DEVICE, **kw)
    for e in out: FEATS[e][split] = out[e]
SPY = fp.extract(fp.SpyCharts(PREPARED, SLOTS), ENCODERS, FEAT_LOCAL, "spy", DEVICE, **kw)
print(f"features ready in {(time.perf_counter() - t0) / 60:.1f} min")
del ENCODERS
if DEVICE.type == "cuda": torch.cuda.empty_cache()
"""), md("## 5. A · Probes per cell"), code("""
RESULTS, MEDIANS = {}, {}
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    tr, median = fp.cell_rows(store, cell, "train", W["train"])
    va, _ = fp.cell_rows(store, cell, "validation", W["valid"], median)
    MEDIANS[cell["name"]] = median
    assert (store.dates[tr.end_index + cell["horizon"]] <= cfg["train_end"]).all()          # purge
    assert (store.dates[va.end_index + cell["horizon"]] <= cfg["validation_end"]).all()
    data = dict(train=tr, valid=va, feats=FEATS, spy=SPY, slot_pos=SLOT_POS)
    RESULTS[cell["name"]] = fp.run_cell(cell, data, ENCODER_NAMES, PROBES, SETTINGS, DEVICE,
                                        RUN_LOCAL / cell["name"], RUN_DRIVE / cell["name"], smoke=SMOKE)
    if median is not None:
        median.to_csv(RUN_DRIVE / cell["name"] / "train_abs_return_median.csv")
"""), md("""
## 6. A · Compare with Step {ref} on identical rows
AUC with 95% intervals from resampling whole validation days; paired comparisons use the same days for both models.
""".replace("{ref}", ref_step)), code("""
JOINED = {}
for name, r in RESULTS.items():
    pred, share = fp.join_reference(r["predictions"], REF_DIR / name) if REF_DIR else (r["predictions"], None)
    print(f"{name}: {len(pred):,} rows" + ("" if share is None else f", {share:.1%} matched in the reference run"))
    JOINED[name] = dict(info=r["info"], predictions=pred)
TABLE, BOOTS = fp.table(JOINED, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "probe_table.csv")
MAIN = ["DINOv2-S linear probe", "DINOv2-S MLP probe", "random ViT-S MLP probe", "CLIP-B/16 MLP probe",
        "image CNN (scratch)", "image ViT (scratch)", "HGB hand + SPY"]
for m in MAIN + ["DINOv2-S + SPY MLP probe", "image CNN + SPY panel (scratch)", "1D CNN (scratch)"]:
    if m in set(TABLE.index.get_level_values("model")):
        print(m); display(fp.auc_matrix(TABLE, m))
fp.plot(TABLE, MAIN, RUN_LOCAL / "probe_auc_by_horizon.png")
TAKEAWAYS = fp.takeaways(TABLE, BOOTS)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
shutil.copy2(RUN_LOCAL / "probe_auc_by_horizon.png", RUN_DRIVE)
"""), md("""
## 7. B · Time-series foundation models (zero-shot)
Forecasts on every `tsfm_every_kth_day`-th validation day (Kronos: every `kronos_every_kth_day`-th), cached on Drive.
Each model prints its parameter count. The table compares every model on the rows the foundation model forecast.
"""), code("""
days = np.sort(W["valid"].decision_date.unique())
TS_ROWS, TS_Q, TS_PARAMS = {}, {}, {}
for name in TSFM:
    k = SETTINGS["kronos_every_kth_day"] if name == "kronos" else SETTINGS["tsfm_every_kth_day"]
    keys = W["valid"][W["valid"].decision_date.isin(days[::k])][["security_index", "end_index", "decision_date"]]
    keys = keys.reset_index(drop=True)
    try:
        q, p = pt.forecast_all([name], lambda h, keys=keys: pt.intraday_contexts(PREPARED, store, keys, SETTINGS["tsfm_context"], h),
                               keys, HORIZONS, RUN_DRIVE / "tsfm" / name, DEVICE)
    except Exception as err:
        print(f"{name} skipped: {type(err).__name__}: {err}"); continue
    TS_ROWS[name], TS_Q[name] = keys, q[name]; TS_PARAMS.update(p)
_ = (RUN_DRIVE / "parameters.json").write_text(json.dumps({**{f"frozen_{k}": v for k, v in ENCODER_PARAMS.items()}, **TS_PARAMS}, indent=2))
TS_TABLES = []
for name in TS_ROWS:
    TS_RESULTS = {}
    for cell in CELLS:
        pred = pt.attach(JOINED[cell["name"]]["predictions"], TS_ROWS[name], {name: TS_Q[name][cell["horizon"]]},
                         cell["kind"], ["security_index", "end_index"], "end_index", MEDIANS.get(cell["name"]))
        TS_RESULTS[cell["name"]] = dict(info=JOINED[cell["name"]]["info"], predictions=pred)
    t, _ = fp.table(TS_RESULTS, n_boot=50 if SMOKE else 300)
    t = t.assign(rows_of=pt.label(name))
    TS_TABLES.append(t)
if TS_TABLES:
    TS_TABLE = pd.concat(TS_TABLES)
    TS_TABLE.to_csv(RUN_DRIVE / "tsfm_table.csv")
    keep = ["tsfm_" + n for n in TS_ROWS] + ["dinov2_s_mlp", "cnn_plain", "cnn_vit", "hgb_hand_spy"]
    show = TS_TABLE[TS_TABLE.raw.isin(keep)].reset_index()
    pd.set_option("display.width", 200)
    display(show[["rows_of", "cell", "model", "roc_auc", "auc_ci_low", "auc_ci_high", "rows", "days"]].round(4))
"""), md("## 8. Parameter counts (advisor rule)"), code("""
rows = [dict(model=f"{fp.ENCODERS[e]['label']} (frozen encoder)", parameters=n) for e, n in ENCODER_PARAMS.items()]
rows += [dict(model=pt.label(k[5:]), parameters=v) for k, v in TS_PARAMS.items()]
first = next(iter(RESULTS.values()))["info"]
rows += [dict(model=fp.pretty(k) + " (trainable head)", parameters=v["parameters"]) for k, v in first["probes"].items()]
if REF_DIR and (REF_DIR / "network_parameters.csv").exists():
    ref = pd.read_csv(REF_DIR / "network_parameters.csv")
    rows += [dict(model=f"{r.model} (Step reference, trained)", parameters=r.parameters) for r in ref.itertuples()]
PARAMS = pd.DataFrame(rows).set_index("model"); PARAMS.to_csv(RUN_DRIVE / "parameter_table.csv"); display(PARAMS)
"""), md(("""
### How to read it
- **Every pretrained model ≈ 0.50**: general visual or time-series pretraining does not create a direction signal.
- **Pretrained probe > random ViT-S**: pretraining changes the features; if both are ≈ 0.5 it does not matter here.
""" if direction else """
### How to read it
- **Pretrained probe > random ViT-S** (v1: +0.04): general visual features do read volatility off a chart.
- **Probe ≈ image CNN < HGB** (v1): a frozen pretrained ViT reaches the scratch CNN, not HGB. Fine-tuning (LoRA) is the
  next step for the ViT line.
- **Foundation models**: on v1-era data, close-only models ≈ the |recent r| rule; Kronos (reads OHLCV) did better.
"""))]
    return c


# ============================================================================= Step 4a

def indicator_cells():
    c = [header("4a", "Add information to the picture, step by step",
                "The image CNN trails HGB on volatility. Which missing information explains the gap, and can the image "
                "model reach HGB when it gets it? Is the image still useful once the numbers are in?", """
**Diagnosis.** Every chart is scaled to its own low/high and its own maximum volume, so a quiet and a wild stock draw
almost the same picture: the *level* of volatility is not in the image, while HGB gets `realized_vol` as a number.

**The ladder** (all on the same canvas, the same CNN, the same 300,000 training windows and identical validation rows):

| rung | model | what is added |
|---|---|---|
| 0 | `plain` | the original chart (control: should match Step 3a's image CNN) |
| 1 | image CNN + SPY panel | the market's chart below the stock's (from Step 3a, joined on the same rows) |
| 2 | `ewsd` | an exponentially weighted volatility band under the chart on **one fixed log scale** (fitted on train) |
| 3 | `ma_ewsd` | + a moving average of the previous session drawn over the bars |
| 4 | `ma_ewsd_tab` | + the 38 hand + SPY numbers HGB uses, through a small MLP whose logit is added to the CNN's |
| control | `tab_only` | the same MLP on the 38 numbers, **no image**: is the image still useful? |
| control | `plain_tab` | the original chart + the 38 numbers |
| benchmark | HGB hand + SPY | from Step 3a |

**Image validity test.** If `ma_ewsd_tab` (or `plain_tab`) is significantly above `tab_only` in a paired day bootstrap,
the picture carries information that the 38 numbers do not. The feature-only MLP is the fairest control because it is
exactly the feature branch of the hybrid. Step 4b then asks which of the 38 numbers matter.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
assert chartcnn.__version__ >= "2026-11-23.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
print("indicator_cnn", ic.__version__)
""")
    c += [md("""
## 2. Configuration
**To resume**, set `RUN_ID`. `REFERENCE_RUN = None` uses the newest Step 3a run (vol cells at the same horizons).
`QUICK = True`: 40,000 training windows, 3 epochs, every 5th validation day, 5 minutes only (~25 min for 7 variants;
prints images per second so you can judge the full run).

**Full run (estimate on an L4):** about 20–35 min per CNN variant per cell, `tab_only` a few minutes, so
roughly 3–4 hours per cell and 6–8 hours for 5 and 60 minutes. Reruns resume per variant.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"
RUN_ID = None
REFERENCE_RUN = None                               # None = newest Step 3a run
QUICK = False
VARIANTS = ["plain", "ewsd", "ma_ewsd", "ma_ewsd_tab", "tab_only", "plain_tab", "ma"]   # run order: the ladder first
HORIZONS = [1, 12]                                 # bars of 5 minutes
KINDS = ["vol"]
SETTINGS = dict(
    stride=12, embargo_sessions=2, max_train_windows=300_000, validation_every_kth_day=1, early_stop_every_kth_day=4,
    halflife=39, learning_rate=1e-5, weight_decay=0.0, batch_size=128, eval_batch_size=1024, epochs=8, patience=2,
    min_delta=1e-5, channels=[64, 128, 256], head_height=4, dropout=0.5, amp=True,
    tab_hidden=64, tab_learning_rate=1e-3,
    num_workers=0 if SMOKE else min(12, os.cpu_count() or 2), seed=69,
)
if QUICK:
    SETTINGS.update(max_train_windows=40_000, validation_every_kth_day=5, epochs=3); HORIZONS = [1]
if SMOKE:
    SETTINGS.update(max_train_windows=400, batch_size=32, eval_batch_size=64, epochs=2, early_stop_every_kth_day=1,
                    validation_every_kth_day=8)
    HORIZONS = [1]
CELLS = [dict(name=f"{k}_h{h}", kind=k, target=ic.KINDS[k], horizon=h, minutes=5 * h, stride=SETTINGS["stride"])
         for h in HORIZONS for k in KINDS]
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "4a", RUN_ID, "indicator", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
REF_DIR = mf.latest_run(HERE, "3a", REFERENCE_RUN)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, variants=VARIANTS, cells=CELLS, quick=QUICK, reference=str(REF_DIR),
                 chartcnn=chartcnn.__version__, indicator_cnn=ic.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 3a reference run:", REF_DIR)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS), "|", len(VARIANTS), "variants:", ", ".join(VARIANTS))
"""), md("""
## 3. Windows, purge and embargo
An input spans two sessions (the previous one feeds MA and EWSD), so two sessions are removed on each side of a boundary.
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
EMBARGO = ic.embargo_dates(store, SETTINGS["embargo_sessions"])
W = {}
for split, kw in (("train", dict(max_rows=SETTINGS["max_train_windows"])),
                  ("valid", dict(every_kth_day=SETTINGS["validation_every_kth_day"]))):
    path = RUN_DRIVE / f"windows_{split}.parquet"
    if path.exists():
        W[split] = pd.read_parquet(path)
    else:
        W[split] = ic.decision_windows(store, "train" if split == "train" else "validation", SETTINGS["stride"],
                                       EMBARGO, seed=SETTINGS["seed"], **kw)
        W[split].to_parquet(path, index=False)
first_valid_input = store.dates[W["valid"].end_index.min() - ic.LOOKBACK - ic.HISTORY + 1]
last_train_label = store.dates[W["train"].end_index.max() + max(HORIZONS)]
assert first_valid_input > cfg["train_end"], "a validation input reaches into the training period"
assert last_train_label <= cfg["train_end"], "a training label ends after the training period"
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,} on {W['valid'].decision_date.nunique()} days")
"""), md("""
## 4. Preprocessing fitted on train only, what the CNN sees, parameter counts
"""), code("""
import matplotlib.pyplot as plt
SCALE_PATH = RUN_DRIVE / "preprocessing.json"
if SCALE_PATH.exists():
    SCALE = tuple(json.loads(SCALE_PATH.read_text())["ewsd_log_scale"])
else:
    SCALE = ic.fit_ewsd_scale(PREPARED, W["train"], SETTINGS["halflife"], n=500 if SMOKE else 20_000, seed=SETTINGS["seed"])
TAB_RAW = {s: ic.tab_features(store, W[s]) for s in W}
TAB_SCALER = ic.TabScaler(TAB_RAW["train"])                      # train rows only
TAB = {s: TAB_SCALER(TAB_RAW[s]) for s in W}
if not SCALE_PATH.exists():
    SCALE_PATH.write_text(json.dumps(dict(ewsd_log_scale=SCALE, halflife=SETTINGS["halflife"], tab=TAB_SCALER.state())))
print(f"EWSD band: log scale {SCALE[0]:.2f} .. {SCALE[1]:.2f}; {TAB['train'].shape[1]} hand/SPY numbers:", ", ".join(TAB_SCALER.columns))
rv = TAB_RAW["train"]["realized_vol"].to_numpy()
pick = W["train"].iloc[[int(np.nanargmin(np.abs(rv - np.nanpercentile(rv, q)))) for q in (5, 95)]].reset_index(drop=True)
show = ["plain", "ma", "ewsd", "ma_ewsd"]
fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 7.2))
for r in range(2):
    for c_, v in enumerate(show):
        img = ic.Charts(PREPARED, pick, v, SCALE, SETTINGS["halflife"]).image(r)
        axes[r, c_].imshow(img, cmap="gray", aspect="auto"); axes[r, c_].set_xticks([]); axes[r, c_].set_yticks([])
        axes[r, c_].set_title(f"{v}\\n{'quiet' if r == 0 else 'volatile'}", fontsize=9)
fig.tight_layout(); fig.savefig(RUN_DRIVE / "what_the_cnn_sees.png", dpi=110); plt.show()
w0 = ic.Charts(PREPARED, pick, "plain", SCALE, SETTINGS["halflife"]).long_window(0)
assert (ic.draw_chart(w0[-ic.LOOKBACK:]) == chartcnn.images.render_window(w0[-ic.LOOKBACK:])).all()
PARAM_ROWS = []
for v in VARIANTS:
    m = ic.IndicatorCNN(SETTINGS, TAB["train"].shape[1] if ic.VARIANTS[v][2] else 0, image=v not in ic.IMAGE_FREE)
    n = ic.count(m)
    PARAM_ROWS.append(dict(variant=v, model=ic.VARIANT_LABEL[v], total=n["total"], cnn=n["cnn"], feature_mlp=n["tab"]))
PARAMS = pd.DataFrame(PARAM_ROWS).set_index("variant"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
del m
"""), md("""
## 5. Train every variant (validation 2023)
"""), code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    assert (store.dates[tr.end_index + cell["horizon"]] <= cfg["train_end"]).all()          # purge
    assert (store.dates[va.end_index + cell["horizon"]] <= cfg["validation_end"]).all()
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    data = dict(train=tr, es=es, valid=va, tab_train=TAB["train"], tab_valid=TAB["valid"], scale=SCALE,
                prior=float(tr.label.mean()))
    RESULTS[cell["name"]] = ic.run_cell(PREPARED, cell, data, VARIANTS, SETTINGS, DEVICE,
                                        RUN_LOCAL / cell["name"], RUN_DRIVE / cell["name"])
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 6. Compare on identical rows (Step 3a scores joined)
"""), code("""
JOINED = {}
for name, r in RESULTS.items():
    pred, share = ic.join_reference(r["predictions"], REF_DIR / name) if REF_DIR else (r["predictions"], None)
    print(f"{name}: {len(pred):,} rows" + ("  (no Step 3a run)" if share is None else f", {share:.1%} matched in Step 3a"))
    JOINED[name] = dict(info=r["info"], predictions=pred)
TABLE, BOOTS = ic.table(JOINED, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "indicator_table.csv")
MAIN = [ic.pretty("ind_" + v) for v in VARIANTS] + ["image CNN (ref run)", "image CNN + SPY panel (ref run)", "HGB hand + SPY (ref)"]
for m in MAIN:
    if m in set(TABLE.index.get_level_values("model")):
        print(m); display(ic.auc_matrix(TABLE, m))
ic.plot(TABLE, MAIN, RUN_LOCAL / "indicator_auc_by_horizon.png")
"""), md("""
## 7. The ladder and the image-validity test
Each rung's AUC with its 95% interval; the dashed line is HGB hand + SPY. Paired differences (same resampled days):
"better"/"worse" when the 95% interval of the difference excludes zero.
"""), code("""
LADDER = [("ind_plain", "0  original chart"), ("cnn_plain_spy", "1  + SPY panel (Step 3a)"),
          ("ind_ewsd", "2  + EW volatility band"), ("ind_ma_ewsd", "3  + moving average"),
          ("ind_ma_ewsd_tab", "4  + 38 hand/SPY numbers"), ("ind_tab_only", "control: numbers only, no image"),
          ("hgb_hand_spy", "benchmark: HGB hand + SPY")]
t = TABLE.reset_index()
rows = []
for cell in RESULTS:
    for raw, label in LADDER:
        r = t[(t.cell == cell) & (t.raw == raw)]
        if len(r):
            r = r.iloc[0]
            rows.append(dict(cell=cell, rung=label, raw=raw, auc=r.roc_auc, low=r.auc_ci_low, high=r.auc_ci_high))
LADDER_TABLE = pd.DataFrame(rows)
LADDER_TABLE.to_csv(RUN_DRIVE / "ladder.csv", index=False)
display(LADDER_TABLE.pivot(index="rung", columns="cell", values="auc").round(4))
fig, axes = plt.subplots(1, len(RESULTS), figsize=(6.5 * len(RESULTS), 4.2), squeeze=False)
for ax, cell in zip(axes[0], RESULTS):
    d = LADDER_TABLE[(LADDER_TABLE.cell == cell) & ~LADDER_TABLE.raw.eq("hgb_hand_spy")].reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.errorbar(d.auc, y, xerr=[d.auc - d.low, d.high - d.auc], fmt="o", capsize=3)
    ax.set_yticks(y, d.rung, fontsize=8)
    h = LADDER_TABLE[(LADDER_TABLE.cell == cell) & LADDER_TABLE.raw.eq("hgb_hand_spy")]
    if len(h):
        ax.axvline(h.auc.iloc[0], ls="--", c="tab:orange", label=f"HGB hand + SPY {h.auc.iloc[0]:.3f}")
        ax.axvspan(h.low.iloc[0], h.high.iloc[0], color="tab:orange", alpha=0.12)
        ax.legend(fontsize=8, loc="lower right")
    ax.set(title=f"{cell}: validation AUC", xlabel="AUC")
fig.tight_layout(); fig.savefig(RUN_DRIVE / "ladder.png", dpi=130); plt.show()
for a, b, label in ic.CHECKS:
    p = ic.paired(BOOTS, a, b)
    if len(p):
        print(label, f"({ic.pretty(a)} − {ic.pretty(b)})"); display(p.round(4))
TAKEAWAYS = ic.takeaways(TABLE, BOOTS)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **Rung 2 > rung 0**: the picture was missing the volatility level; a fixed-scale band gives it to the CNN.
- **Rung 4 tied with HGB**: with the same information the image model reaches HGB.
- **Rung 4 > `tab_only` (paired, excluding zero)**: the image adds something the 38 numbers do not. **Tied**: the
  hybrid's gain comes from the numbers; the image is redundant once they are in.
- **`tab_only` < HGB**: the MLP uses the same 38 numbers less well than the trees (a fair caveat for the test above).

## 8. Full results table
"""), code("""
display(TABLE.drop(columns=["raw"]).round(4))
""")]
    return c


# ============================================================================= Step 4b

def feature_cells():
    c = [header("4b", "Which hand features matter?", "Of the 38 hand + SPY numbers that HGB (and the Step 4 hybrid) use, "
                "which carry the volatility signal and which are redundant?", """
Every choice is made on a **selection year** (2022, the last training year) with HGB fitted on 2016–2021; 2023 is used
once, to confirm the chosen subsets next to the full model (paired day bootstrap). The test years are not used.

1. **Permutation importance**: AUC drop on 2022 when one feature is shuffled (5 repeats).
2. **Feature groups**: stock returns (16), stock volatility (2), stock volume and time of day (2), SPY (12),
   stock vs SPY (6): retrain without each group, and with each group alone.
3. **Forward selection**: add the feature that raises 2022 AUC most, until within 0.002 of all 38.
4. **Confirmation on 2023**: full, forward-selected, top-k by importance and each group-drop model.

CPU is enough for this notebook (no GPU needed).
""")]
    c += setup("""
import chartcnn
import feature_select as fs
print("feature_select", fs.__version__)
""", gpu=False)
    c += [md("""
## 2. Configuration
**Time (estimate, Colab CPU):** permutation importance ~5 min per cell; group ablation ~5 min; forward selection over the
top `forward_candidates` features ~30–60 min per cell. `QUICK = True` uses fewer rows and 10 candidates.
"""), code("""
SMOKE = LOCAL_TEST
RUN_ID = None
QUICK = False
HORIZONS = [1, 12]                                 # bars of 5 minutes
SETTINGS = dict(stride=12, selection_start="2022-01-01", embargo_sessions=2,
                max_inner=300_000, max_selection=200_000, max_train=1_000_000,
                n_repeats=5, tolerance=0.002, max_features=15, forward_candidates=20, top_k=(5, 10), seed=69)
if QUICK:
    SETTINGS.update(max_inner=100_000, max_selection=60_000, max_train=300_000, forward_candidates=10, n_repeats=3)
if SMOKE:
    SETTINGS.update(selection_start=None, max_inner=2_000, max_selection=2_000, max_train=3_000, n_repeats=1,
                    forward_candidates=4, max_features=3)
CELLS = [dict(name=f"vol_h{h}", kind="vol", target="vol", horizon=h, minutes=5 * h, stride=SETTINGS["stride"]) for h in HORIZONS]
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "4b", RUN_ID, "features", LOCAL_TEST)
store = chartcnn.Store(PREPARED)
if SETTINGS["selection_start"] is None:             # synthetic data: last ~third of the training period
    tr_days = np.unique(store.dates[store.dates <= store.config["train_end"]])
    SETTINGS["selection_start"] = str(tr_days[int(len(tr_days) * 0.65)])
mf.save_settings(RUN_DIR, RESUME, settings=SETTINGS, cells=CELLS, quick=QUICK, feature_select=fs.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DIR, "| selection year starts", SETTINGS["selection_start"])
pd.DataFrame([dict(group=g, n=len(c), features=", ".join(c)) for g, c in fs.GROUPS.items()]).set_index("group")
"""), md("## 3. Selection on 2022, confirmation on 2023, per cell"), code("""
ALL = {}
for cell in CELLS:
    out = RUN_DIR / cell["name"]; out.mkdir(exist_ok=True)
    if (out / "confirmation.csv").exists():
        print(cell["name"], "already done"); ALL[cell["name"]] = {k: pd.read_csv(out / f"{k}.csv") for k in
                                                                  ("importance", "groups", "forward", "confirmation")}
        continue
    print("\\n" + "=" * 20, cell["name"], "=" * 20)
    t0 = time.perf_counter()
    d, X = fs.build_data(store, cell, SETTINGS)
    yi, ys = d["inner"].label.to_numpy(), d["sel"].label.to_numpy()
    full, full_auc = fs.fit_score(list(X["inner"].columns), X["inner"], yi, X["sel"], ys, SETTINGS["seed"])
    print(f"all 38 features: selection AUC {full_auc:.4f} ({fs.tree_size(full):,} tree nodes)")
    base, imp = fs.permutation_importance(full, X["sel"], ys, SETTINGS["n_repeats"], SETTINGS["seed"])
    fs.plot_importance(imp, out / "importance.png", f"{cell['name']}: permutation importance (selection year)")
    print("groups (retrained on inner train, scored on the selection year):")
    groups = fs.group_ablation(X["inner"], yi, X["sel"], ys, SETTINGS["seed"], full_auc)
    cand = list(imp.feature[:SETTINGS["forward_candidates"]])
    print(f"forward selection over the top {len(cand)} features by importance:")
    chosen, fwd = fs.forward_selection(X["inner"], yi, X["sel"], ys, SETTINGS["seed"], full_auc, SETTINGS["tolerance"],
                                       SETTINGS["max_features"], cand)
    subsets = {"all 38 features": list(X["train"].columns), f"forward-selected ({len(chosen)})": chosen}
    for k in SETTINGS["top_k"]:
        subsets[f"top {k} by importance"] = list(imp.feature[:k])
    for g, gcols in fs.GROUPS.items():
        subsets[f"without {g}"] = [c for c in X["train"].columns if c not in gcols]
        subsets[f"only {g}"] = [c for c in gcols if c in X["train"].columns]
    conf, _ = fs.confirm(subsets, d, X, SETTINGS["seed"], n_boot=50 if SMOKE else 300)
    res = dict(importance=imp, groups=groups, forward=fwd, confirmation=conf)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    ALL[cell["name"]] = res
    print(f"{cell['name']} done in {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 4. Results
Confirmation on 2023: AUC with 95% day-bootstrap interval, and the paired difference to the full 38-feature model
("tied" = the subset loses nothing measurable).
"""), code("""
for name, r in ALL.items():
    print(f"=== {name}: top 15 by permutation importance (selection year)"); display(r["importance"].head(15).round(5))
    print("groups"); display(r["groups"].round(4))
    print("forward selection path"); display(r["forward"].round(4))
    print("confirmation on validation 2023"); display(r["confirmation"].round(4))
useless = {name: set(r["importance"].loc[r["importance"].auc_drop <= 0, "feature"]) for name, r in ALL.items()}
common = set.intersection(*useless.values()) if useless else set()
print("features with no importance (AUC drop ≤ 0) in every cell:", ", ".join(sorted(common)) or "none")
"""), md("""
### How to read it
- **Forward-selected subset tied with all 38 on 2023**: the rest are redundant for HGB; a smaller feature branch can be
  used in the Step 4 hybrid.
- **Without a group, AUC falls (worse)**: that group carries information the others do not.
- **Only a group ≈ all 38**: one group (likely the stock's own volatility) carries most of the signal.
- Permutation importance understates correlated features (shuffling one leaves its twin); the group retraining and the
  forward path are the checks for that.
""")]
    return c


# ============================================================================= Step 5

def summary_cells():
    c = [header("5", "Summary across the steps", "What does the whole line of reasoning show, in one figure and one "
                "parameter table?", """
Reads the newest finished run of every step from `MindFlow/runs/` (or the `RUNS` set below) and puts the headline cell
of each step side by side. Nothing is trained here; it runs in a minute on CPU.
""")]
    c += setup("", gpu=False)
    c += [md("## 2. Which runs, which headline cells"), code("""
RUNS = {s: None for s in ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b")}   # None = newest run of that step
HEADLINE = {"1": "dir_I20R5", "2": "xsdir_h12", "3": "vol_h1", "4": "vol_h1"}  # one cell shown per step
FOUND = {s: mf.latest_run(HERE, s, r) for s, r in RUNS.items()}
for s, p in FOUND.items():
    print(f"Step {s}: {p if p else '— not run yet'}")
OUT = mf.run_folder(HERE, "5"); OUT.mkdir(parents=True, exist_ok=True)

# Steps 2b / 3b written before frozen_probe 2026-11-24.1 took the foundation models' AUC from scores clipped
# to [0, 1] (ties pulled it towards 0.5; the intervals were right). Rescore them once from the saved forecasts.
import pretrained_ts as pt
for s, ref in (("2b", "2a"), ("3b", "3a")):
    if FOUND[s] is not None and not (FOUND[s] / "tsfm_table_before_2026-11-24.csv").exists():
        pt.rebuild_tsfm_table(FOUND[s], FOUND[ref], n_boot=50 if LOCAL_TEST else 300)
"""), md("## 3. Headline AUCs per step"), code("""
def family(label):
    l = label.lower()
    if "hgb" in l: return "tree"
    if "vit" in l and "probe" not in l: return "ViT"
    if "random walk" in l: return "series"
    if any(k in l for k in ("probe", "chronos", "timesfm", "moirai", "kronos", "zero-shot")): return "pretrained"
    if "1d" in l or "series" in l: return "series"
    if "numbers only" in l or "features only" in l: return "features"
    if "rule" in l or "prior" in l: return None
    return "image"

rows = []
def add(step, frame, label_col, auc, low, high):
    for _, r in frame.iterrows():
        f = family(str(r[label_col]))
        if f: rows.append(dict(step=step, label=str(r[label_col]), family=f, auc=r[auc], low=r[low], high=r[high]))

for s in ("1a", "1b"):
    t = mf.read_csv(FOUND[s] / ("daily_table.csv" if s == "1a" else "daily_pretrained_table.csv")) if FOUND[s] else None
    if t is not None:
        t = t[(t.cell == HEADLINE["1"]) & (t.split == "validation")]
        if s == "1b": t = t[~t.raw.isin(["cnn", "hgb"]) & ~t.raw.str.startswith("rule")]
        add("1", t, "model", "auc", "auc_low", "auc_high")
for s, step in (("2a", "2"), ("2b", "2"), ("3a", "3"), ("3b", "3")):
    if FOUND[s] is None: continue
    for name in (("grid_table.csv",) if s.endswith("a") else ("probe_table.csv", "tsfm_table.csv")):
        t = mf.read_csv(FOUND[s] / name)
        if t is None: continue
        t = t[t.cell == HEADLINE[step]]
        if s.endswith("b"):
            t = t[~t.model.str.contains("scratch|HGB|prior", regex=True)]
            if name == "tsfm_table.csv": t = t[t.raw.str.startswith("tsfm_")]
        add(step, t, "model", "roc_auc", "auc_ci_low", "auc_ci_high")
if FOUND["4a"] is not None and (FOUND["4a"] / "ladder.csv").exists():
    t = pd.read_csv(FOUND["4a"] / "ladder.csv"); t = t[t.cell == HEADLINE["4"]]
    for _, r in t.iterrows():
        rows.append(dict(step="4", label=r.rung, family="tree" if "HGB" in r.rung else "features" if "numbers only" in r.rung
                         else "image", auc=r.auc, low=r.low, high=r.high))
SUMMARY = pd.DataFrame(rows).drop_duplicates(["step", "label"])
SUMMARY.to_csv(OUT / "summary.csv", index=False)
display(SUMMARY.round(4))
if len(SUMMARY):
    mf.mindflow_figure(SUMMARY, OUT / "mindflow.png")
"""), md("""
## 4. Parameter counts of every model (advisor rule)
Networks: weights; HGB: tree nodes (largest cell); probes: trainable head (encoder frozen, listed separately);
foundation models: all weights (frozen, zero-shot).
"""), code("""
prow = []
for s, p in FOUND.items():
    prow += mf.parameter_rows(s, p)
PARAMS = mf.parameter_table(prow)
PARAMS.to_csv(OUT / "parameters.csv", index=False)
display(PARAMS)
"""), md("""
## 5. Feature selection (Step 4b)
"""), code("""
if FOUND["4b"] is not None:
    for f in sorted(FOUND["4b"].glob("*/confirmation.csv")):
        print(f.parent.name); display(pd.read_csv(f).round(4))
""")]
    return c


# ============================================================================= Step 6a
def regime_cells():
    c = [header("6a", "Market regime (VIX) for volatility", "The vol label uses each stock's fixed training-period median "
                "|r|, so the volatility regime is part of what it measures. Do the VIX and multi-day realized volatility "
                "raise HGB's AUC beyond the 38 hand + SPY numbers?", """
HGB only (the strongest model so far, Step 3a: 0.689 / 0.679), same rows for every feature set, scored on 2023 with a
paired day bootstrap against the 38-feature model. The test years are not used.

| feature set | what is added to the 38 hand + SPY numbers |
|---|---|
| `+ VIX` | previous session's VIX close (log), its 1- and 5-session change, distance to its 20-session mean |
| `+ multi-day RV` | stock realized vol over the last 5 and 20 completed sessions, their ratio, and SPY's 5/20-session RV |
| `+ VIX + multi-day RV` | both |
| `regime only` | the 9 regime features alone (no 38) |
| `+ VIXY intraday` | VIXY over the same 78-bar window as SPY: returns, cumulative returns, realized vol (needs `vixy_panel_v4.zip`) |

VIX comes from Cboe's public daily file (downloaded once, cached in `MindFlow/data/`). Only closes of sessions **before**
the decision day are used. Realized vol uses intraday 5-minute returns of completed sessions only (no overnight gap,
so no ex-date is ever inside a return). CPU is enough; turn on High-RAM (the bundle's price arrays need ~6 GB).
""")]
    c += setup("""
import chartcnn
import feature_select as fs
import regime as rg
print("regime", rg.__version__, "| feature_select", fs.__version__)
""", gpu=False)
    c += [md("""
## 2. Configuration
**Time (measured, Colab CPU High-RAM):** ~45 min for the 5 feature sets; with the VIXY panel (7 sets) ~60 min. `QUICK = True`: 300,000 rows, ~15 min.
"""), code("""
SMOKE = LOCAL_TEST
RUN_ID = None
QUICK = False
HORIZONS = [1, 12]                                 # bars of 5 minutes
SETTINGS = dict(stride=12, max_train=1_000_000, min_bars=60, seed=69)
if QUICK:
    SETTINGS.update(max_train=300_000)
if SMOKE:
    SETTINGS.update(max_train=3_000, min_bars=10)
CELLS = [dict(name=f"vol_h{h}", kind="vol", target="vol", horizon=h, minutes=5 * h, stride=SETTINGS["stride"]) for h in HORIZONS]
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "6a", RUN_ID, "regime", LOCAL_TEST)
mf.save_settings(RUN_DIR, RESUME, settings=SETTINGS, cells=CELLS, quick=QUICK, regime=rg.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
"""), md("## 3. Regime features (VIX and multi-day realized volatility)"), code("""
store = chartcnn.Store(PREPARED)
VIX_CACHE = HERE / "data" / "VIX_History.csv"
vix = rg.load_vix(VIX_CACHE)
print(f"VIX: {len(vix):,} closes, {vix.index[0]} … {vix.index[-1]} (cached at {VIX_CACHE})")
sessions = np.unique(store.dates)
missing = [d for d in sessions if d not in vix.index]
print(f"bundle sessions without a VIX close: {len(missing)}", missing[:5])
VIXF = rg.vix_table(vix, sessions)
VIXY = rg.load_vixy(PREPARED)                     # VIXY intraday features if vixy_panel_v4.zip is in Demo/MindFlow
print("VIXY intraday features:", "yes" if VIXY is not None else "no (vixy_panel_v4.zip not found)")
t0 = time.perf_counter()
RV = rg.rv_tables(store, SETTINGS["min_bars"])
print(f"realized-vol tables in {(time.perf_counter() - t0) / 60:.1f} min:", {k: v.shape for k, v in RV.items()})
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
days = pd.to_datetime(sessions)
ax[0].plot(days, np.exp(VIXF.vix_log.to_numpy()), lw=0.8); ax[0].set_ylabel("VIX (prev. close)")
if "spy_rv20d" in RV:
    ax[1].plot(days, RV["spy_rv20d"][:len(days)], lw=0.8, c="tab:orange"); ax[1].set_ylabel("SPY RV, 20 sessions")
for a in ax:
    a.axvspan(pd.Timestamp(store.config["train_end"]), pd.Timestamp(store.config["validation_end"]), color="gray", alpha=0.15)
ax[0].set_title("regime features (grey = validation 2023)")
fig.tight_layout(); fig.savefig(RUN_DIR / "regime_series.png", dpi=120); plt.show()
"""), md("""
## 4. HGB with and without regime features, per cell
"""), code("""
ALL = {}
for cell in CELLS:
    out = RUN_DIR / cell["name"]; out.mkdir(exist_ok=True)
    if (out / "comparison.csv").exists():
        print(cell["name"], "already done"); ALL[cell["name"]] = {k: pd.read_csv(out / f"{k}.csv") for k in ("comparison", "by_regime", "importance")}
        continue
    print("\\n" + "=" * 20, cell["name"], "=" * 20)
    t0 = time.perf_counter()
    d, X = rg.build(store, cell, SETTINGS, RV, VIXF, VIXY)
    base = [c for c in X["train"].columns if c not in sum(rg.GROUPS.values(), []) + rg.VIXY_COLS]
    print(f"train {len(d['train']):,} | validation {len(d['valid']):,} on {d['valid'].decision_date.nunique()} days | "
          f"{len(base)} base + {X['train'].shape[1] - len(base)} regime features")
    print("missing share of regime features (validation):", X["valid"][sum(rg.GROUPS.values(), [])].isna().mean().round(3).to_dict())
    subsets = {"38 hand + SPY": base,
               "+ VIX": base + rg.VIX_COLS,
               "+ multi-day RV": base + rg.GROUPS["multi-day RV"],
               "+ VIX + multi-day RV": base + sum(rg.GROUPS.values(), []),
               "regime only": sum(rg.GROUPS.values(), [])}
    if VIXY is not None:
        subsets["+ VIXY intraday"] = base + rg.VIXY_COLS
        subsets["+ VIXY intraday + multi-day RV"] = base + rg.VIXY_COLS + rg.GROUPS["multi-day RV"]
    comp, scores, models = rg.compare(d, X, subsets, SETTINGS["seed"], n_boot=50 if SMOKE else 300)
    display(comp.round(4))
    reg = rg.by_regime(d["valid"], scores, X["valid"].vix_log.to_numpy(), list(subsets))
    print("AUC within VIX terciles of 2023 (previous close):"); display(reg.round(4))
    full = "+ VIXY intraday + multi-day RV" if VIXY is not None else "+ VIX + multi-day RV"
    _, imp = fs.permutation_importance(models[full], X["valid"][subsets[full]], d["valid"].label.to_numpy(),
                                       1 if SMOKE else 3, SETTINGS["seed"])
    res = dict(comparison=comp, by_regime=reg, importance=imp)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    ALL[cell["name"]] = res
    print(f"{cell['name']} done in {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("## 5. Summary"), code("""
rows = []
for name, r in ALL.items():
    t = r["comparison"].assign(cell=name)
    rows.append(t)
SUMMARY = pd.concat(rows)[["cell", "model", "n_features", "tree_nodes", "auc", "auc_low", "auc_high", "diff_vs_ref",
                           "diff_low", "diff_high", "verdict"]]
SUMMARY.to_csv(RUN_DIR / "summary.csv", index=False)
display(SUMMARY.round(4))
for name, r in ALL.items():
    imp = r["importance"]
    reg = imp[imp.feature.isin(sum(rg.GROUPS.values(), []) + rg.VIXY_COLS)]
    print(f"{name}: regime features by permutation importance (2023, full model):")
    display(reg.round(5))
lines = []
for name, r in ALL.items():
    t = r["comparison"].set_index("model")
    for m in ("+ VIX", "+ multi-day RV", "+ VIX + multi-day RV", "+ VIXY intraday", "+ VIXY intraday + multi-day RV"):
        if m not in t.index:
            continue
        x = t.loc[m]
        lines.append(f"- {name} {m}: AUC {x.auc:.4f} vs {t.loc['38 hand + SPY'].auc:.4f}, "
                     f"diff {x.diff_vs_ref:+.4f} [{x.diff_low:+.4f}, {x.diff_high:+.4f}] -> {x.verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DIR / "takeaways.md").write_text(TAKEAWAYS + "\\n"); print(TAKEAWAYS)
"""), md("""
### How to read it
- **`+ VIX` better**: the market's expected volatility adds what one session of bars cannot see; worth adding to the
  CNN's feature branch (Step 4) and trying the VIXY intraday panel.
- **`+ multi-day RV` better but `+ VIX` tied**: the stock's own slower volatility is what matters; VIX adds nothing.
- **All tied**: the 78-bar window already captures the regime; VIX is not worth the extra data.
- **By VIX tercile**: the share of positive labels should rise with VIX (fixed threshold); a model whose AUC drops in
  the high tercile is mis-calibrated across regimes.
""")]
    return c
