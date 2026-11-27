"""Cells of MindFlow Step 6f (compact pictures: the same information with less black)."""
from cells import code, header, md, setup


def compact_cells():
    c = [header("6f", "Compact pictures", "Step 6d's pure-image picture is 466 × 234 pixels and about 90% of it is "
                "black. Can the same information be drawn more densely, and is a denser picture as accurate and "
                "faster?", """
Every variant shows exactly what Step 6d's `pure_all` shows (chart with the session-open line, stock EWSD band, SPY
EWSD band, volume band, the four multi-day RV bars, the 43 numbers as bars), on Step 6d's strip scales. Only the layout
changes:

| variant | CNN input | what changes |
|---|---|---|
| `pure_all` (6d, reference) | 1 × 466 × 234 | strips stacked under the chart |
| `quarter` | 1 × 116 × 234 | 4 × 1 max pool in height first (max keeps one-pixel lines); the 7b+ half-height student lost nothing at 2 × 1 |
| `channels` | 6 × 234 × 234 | each strip drawn on the full 234 rows and stacked as a channel: one pixel column holds price, volatility and volume of the same bar |
| `filled` | 1 × 466 × 234 | grey under each bar's close, band bars 3 pixels wide: more lit pixels, same information |
| `compact` | 6 × 117 × 234 | `channels` with a 2 × 1 max pool in height: the smallest input |

**Width is not changed on purpose.** The CNN's first layer reads each bar's 3 columns (open tick, high–low line, close
tick) with a 3-wide, stride-3 filter, so 234 columns become 78 bar columns at once; empty width costs nothing.

Same CNN (64-128-256, plain), windows, rows, training rule and seed as Step 6d; only the first layer's input channels
and the input size change, so parameters stay within 1%. Reported for every variant: lit share of the picture,
parameters, epochs, training time, GPU inference speed, AUC and log loss with paired day-bootstrap checks against 6d.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import pure_image as pi
import compact_image as ci
import efficiency as ef
import matplotlib.pyplot as plt
fm.register()
print("compact_image", ci.__version__, "| pure_image", pi.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP6D_RUN = None` takes the newest Step 6d run (which names its Step 6b run). **To resume after a disconnect**, set
`RUN_ID` to the printed id and Run all: finished variants are skipped. `HORIZONS = [1]` (5 minutes).

**Time (estimate, not measured):** Step 6d's `pure_all` took about 1–1.5 hours on an L4. `filled` costs the same;
`channels` about the same (same rows as the chart, 6 channels); `quarter` and `compact` are faster on the GPU, but the
pictures are drawn on the CPU, which may set the pace. About 3–5 hours for all four on an L4, less on an A100. Each
variant is saved as it finishes, so the run can be split over sessions. A T4 works but is about 2–3x slower.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None
STEP6D_RUN = None
HORIZONS = [1]
VARIANTS = ["quarter", "compact", "channels", "filled"]   # run order: the cheap ones first
REF = "pure_all"
STEP6D_DIR = mf.latest_run(HERE, "6d", STEP6D_RUN)
assert STEP6D_DIR is not None, "run Step 6d first"
S6D = json.loads((STEP6D_DIR / "settings.json").read_text())
STEP6B_DIR = mf.run_folder(HERE, "6b") / S6D["step6b"]
FINAL = S6D["final"]
SETTINGS = dict(S6D["settings"])                     # identical training rule
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELLS = [c for c in S6D["cells"] if c["horizon"] in HORIZONS and (STEP6D_DIR / c["name"] / "predictions.parquet").exists()]
assert CELLS, f"no finished Step 6d cell for horizons {HORIZONS}"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "6f", RUN_ID, "compact", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, variants=VARIANTS, cells=CELLS, step6d=STEP6D_DIR.name,
                 step6b=STEP6B_DIR.name, compact_image=ci.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 6d run:", STEP6D_DIR.name, "| Step 6b run:", STEP6B_DIR.name, "| cells:", [c["name"] for c in CELLS])
"""), md("""
## 3. Step 6d's windows, numbers and strip scales; parameter counts
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
W = {s: pd.read_parquet(STEP6B_DIR / f"windows_{s}.parquet") for s in ("train", "valid")}
PRE = json.loads((STEP6B_DIR / "preprocessing.json").read_text())
SC43 = fm.scaler_from_state(PRE["tab"])
SCALES = {k: tuple(v) for k, v in json.loads((STEP6D_DIR / "strip_scales.json").read_text()).items()}
t0 = time.perf_counter()
RVT = rg.rv_tables(store, SETTINGS["min_bars"])
print(f"realized-vol tables in {(time.perf_counter() - t0) / 60:.1f} min")
RAW43 = {s: fm.numbers(store, W[s], RVT)[1] for s in W}
LRV = {s: pi.log_rv(RAW43[s]) for s in W}
NUM = {s: SC43(RAW43[s]) for s in W}
MC = store.market_close
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,}")
PARAMS = [dict(model=REF, label=ci.pretty(REF), input=f"1x{pi.H_ALL}x{ic.SIZE}", total=ic.count(pi.PureCNN(SETTINGS, REF))["total"])]
for v in VARIANTS:
    PARAMS.append(dict(model=f"cmp_{v}", label=ci.pretty(f"cmp_{v}"), input="x".join(map(str, ci.input_shape(v))),
                       total=ic.count(ci.CompactCNN(SETTINGS, v))["total"]))
PARAMS = pd.DataFrame(PARAMS).set_index("model"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 4. What each variant sees, and how much of it is black
One real validation window (the most volatile stock at a busy decision bar), drawn in every layout, as the CNN
receives it (after the height pooling). Channels are shown as a 2 × 3 grid: chart, stock EWSD, SPY EWSD / volume,
multi-day RV, numbers. The lit share (pixels > 0) is measured on 500
validation windows. Saved as PNG in the run folder.
"""), code("""
va0 = W["valid"]
bar = va0.end_index.value_counts().index[0]
pick = va0[va0.end_index == bar]
ewsd = np.array([ic.indicators(ic.Charts(PREPARED, pick.reset_index(drop=True), "plain", (0, 1), SETTINGS["halflife"])
                               .long_window(i), SETTINGS["halflife"])[1][-1] for i in range(len(pick))])
r = pick.index[int(np.argmax(ewsd))]
one = W["valid"].loc[[r]].assign(row=r).reset_index(drop=True)
shown = {REF: pi.dataset(PREPARED, one, REF, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]).image(0)[None]}
for v in VARIANTS:
    shown[v] = ci.pooled(ci.dataset(PREPARED, one, v, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]).picture(0), v)
pad = lambda ch: np.pad(ch, ((0, 6), (0, 6)), constant_values=128)       # grey gutter between channels
tiles = {v: np.vstack([np.hstack([pad(ch) for ch in x[k:k + 3]]) for k in range(0, len(x), 3)]) if len(x) > 1 else x[0]
         for v, x in shown.items()}                                       # channels as a 2 x 3 grid
fig, axes = plt.subplots(1, len(tiles), figsize=(18, 7), gridspec_kw=dict(width_ratios=[t.shape[1] for t in tiles.values()]))
for ax, (v, tile) in zip(axes, tiles.items()):
    x = shown[v]
    pi.save_png(tile, RUN_DRIVE / f"sample_{v}.png")
    ax.imshow(tile, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{v}\\n{' x '.join(map(str, x.shape))}", fontsize=8)
fig.tight_layout(); fig.savefig(RUN_DRIVE / "samples.png", dpi=130); plt.show()

sample = va0.sample(min(500, len(va0)), random_state=0)
sample = sample.assign(row=sample.index).reset_index(drop=True)
LIT, DRAW = {}, {}
for v in [REF] + VARIANTS:
    if v == REF:
        ds = pi.dataset(PREPARED, sample, REF, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]); get = lambda i: ds.image(i)[None]
    else:
        ds = ci.dataset(PREPARED, sample, v, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]); get = lambda i, v=v: ci.pooled(ds.picture(i), v)
    t1 = time.perf_counter()
    pics = [get(i) for i in range(len(sample))]
    DRAW[v] = (time.perf_counter() - t1) * 1000 / len(sample)
    LIT[v] = float(np.mean([ci.lit_share(p) for p in pics]))
LOOK = pd.DataFrame(dict(lit_share=LIT, draw_ms_per_picture_cpu=DRAW)).rename(index=lambda v: v if v == REF else f"cmp_{v}")
LOOK = PARAMS.join(LOOK); LOOK.to_csv(RUN_DRIVE / "pictures.csv"); display(LOOK.round(4))
"""), md("""
## 5. Train (validation 2023)
"""), code("""
RESULTS, VA, INFO, t0 = {}, {}, {}, time.perf_counter()
REN = {"score_pure_pure_all": "score_pure_all", "score_pure_pure_key": "score_pure_key"}
ckpt = lambda name, v: next(p for p in (RUN_LOCAL / name / v / "best.pt", RUN_DRIVE / name / v / "best.pt") if p.exists())
for i, cell in enumerate(CELLS, 1):
    name = cell["name"]
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", name, "=" * 20)
    (RUN_DRIVE / name).mkdir(exist_ok=True)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    VA[name] = va
    data = dict(train=tr, es=es, valid=va, scales=SCALES, market_close=MC, lrv=LRV, numbers=NUM)
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    pred = va[keep].copy()
    info = dict(cell=cell, prior=float(tr.label.mean()), parameters={}, best_epoch={}, minutes_trained={}, seconds_per_epoch={})
    for v in VARIANTS:
        p, vi = ci.run_variant(v, PREPARED, data, SETTINGS, DEVICE, RUN_LOCAL / name, RUN_DRIVE / name)
        pred[f"score_cmp_{v}"] = p
        for k, key in (("parameters", "parameters"), ("best_epoch", "best_epoch"), ("minutes_trained", "minutes"),
                       ("seconds_per_epoch", "seconds_per_epoch")):
            info[k][f"cmp_{v}"] = vi.get(key)
    ref = pd.read_parquet(STEP6D_DIR / name / "predictions.parquet").rename(columns=REN)
    cols = [c for c in ref.columns if c.startswith("score_")]
    ref = ref[["security_index", "end_index", "label"] + cols].rename(columns={"label": "ref_label"})
    pred = pred.merge(ref, on=["security_index", "end_index"], how="left")
    assert pred.ref_label.notna().all() and (pred.ref_label == pred.label).all(), "Step 6d rows differ"
    pred = pred.drop(columns="ref_label")
    pred.to_parquet(RUN_DRIVE / name / "predictions.parquet", index=False)
    (RUN_DRIVE / name / "done.json").write_text(json.dumps(info, default=float))
    RESULTS[name], INFO[name] = dict(info=info, predictions=pred), info
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 6. Compare on identical rows: AUC and log loss
AUC with 95% intervals (whole days resampled), log loss, and the paired checks ("better"/"worse" when the 95% interval
of the difference excludes zero). Step 6d's `pure_all` and Step 6b's scores are for exactly these rows.
"""), code("""
SHOW = ["pure_all", "pure_key", "final", "hgb_tab43", "tab_only_rv"]
def compare(results, n_boot):
    tables, boots = [], {}
    for name, r in results.items():
        pred = r["predictions"]
        names = [c[6:] for c in pred.columns if c.startswith("score_") and (c[6:] in SHOW or c.startswith("score_cmp_"))]
        boots[name] = ic.day_bootstrap(pred, names, n_boot)
        tables.append(ci.auc_table(pred, names, boots[name]).assign(cell=name))
    return pd.concat(tables), ci.checks_table(boots)

TABLE, CHECK = compare(RESULTS, 50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "compact_table.csv", index=False)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False)
display(TABLE.set_index("model")[["roc_auc", "auc_ci_low", "auc_ci_high", "log_loss", "cell"]].sort_values("roc_auc", ascending=False).round(4))
display(CHECK.round(4))
"""), md("""
## 7. Learning curves
Training loss (dashed; running mean with dropout on) and log loss on the early-stopping days (solid) per epoch; the
star marks the epoch that was kept. Step 6d's `pure_all` is drawn for reference.
"""), code("""
cell0 = CELLS[0]["name"]
fig, ax = plt.subplots(figsize=(8, 4.5))
curves = {REF: STEP6D_DIR / cell0 / REF / "history.csv"}
curves |= {v: next((p for p in (RUN_DRIVE / cell0 / v / "history.csv", RUN_LOCAL / cell0 / v / "history.csv") if p.exists()), None) for v in VARIANTS}
for k, (v, path) in enumerate(curves.items()):
    if path is None or not Path(path).exists():
        continue
    h = pd.read_csv(path); col = f"C{k}"
    ax.plot(h.epoch, h.train_loss, ls="--", color=col, lw=1)
    ax.plot(h.epoch, h.es_log_loss, color=col, marker="o", ms=3, label=v)
    b = h.loc[h.es_log_loss.idxmin()]; ax.plot(b.epoch, b.es_log_loss, "*", color=col, ms=12)
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8); ax.set_title(f"{cell0}: dashed = train, solid = early-stop days")
fig.tight_layout(); fig.savefig(RUN_DRIVE / "learning_curves.png", dpi=130); plt.show()
"""), md("""
## 8. Speed: GPU inference and training time
GPU milliseconds per batch of real pictures (fp32 and 16-bit), and the measured training seconds per epoch (drawing
on the CPU included). Drawing cost per picture is in section 4.
"""), code("""
BATCHES = [1, 256] if not SMOKE else [1, 8]
sample = VA[cell0].sample(min(512, len(VA[cell0])), random_state=0).reset_index(drop=True)
SPEED = []
models = {REF: (pi.PureCNN(SETTINGS, REF), STEP6D_DIR / cell0 / REF / "best.pt",
                pi.dataset(PREPARED, sample, REF, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]))}
models |= {f"cmp_{v}": (ci.CompactCNN(SETTINGS, v), ckpt(cell0, v),
                        ci.dataset(PREPARED, sample, v, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"])) for v in VARIANTS}
for name, (m, ck, ds) in models.items():
    X = torch.stack([ds[i][0] for i in range(len(sample))]); Tb = torch.zeros(len(sample), 0)
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); m = m.to(DEVICE).eval()
    for precision in (["fp32", "bf16"] if DEVICE.type == "cuda" else ["fp32"]):
        for r in ef.benchmark(name, m, DEVICE, precision, X, Tb, BATCHES, reps=10 if SMOKE else 30):
            SPEED.append(r | dict(precision=precision))
    del m, X
SPEED = pd.DataFrame(SPEED); SPEED.to_csv(RUN_DRIVE / "speed.csv", index=False)
big = SPEED[(SPEED.batch == max(BATCHES)) & (SPEED.precision == SPEED.precision.iloc[-1])].set_index("variant")
t6d = json.loads((STEP6D_DIR / cell0 / REF / "variant.json").read_text()) if (STEP6D_DIR / cell0 / REF / "variant.json").exists() else {}
h6d = pd.read_csv(STEP6D_DIR / cell0 / REF / "history.csv") if (STEP6D_DIR / cell0 / REF / "history.csv").exists() else None
epoch_s = {REF: float(h6d.train_seconds.mean()) if h6d is not None else np.nan} | INFO[cell0]["seconds_per_epoch"]
t0 = TABLE[TABLE.cell == cell0].set_index("raw")
SUMMARY = LOOK.join(big[["ms_per_batch", "images_per_s"]]).join(pd.Series(epoch_s, name="train_s_per_epoch"))
SUMMARY = SUMMARY.join(t0[["roc_auc", "log_loss"]])
SUMMARY["gpu_speedup_vs_6d"] = SUMMARY.loc[REF, "ms_per_batch"] / SUMMARY.ms_per_batch
SUMMARY.to_csv(RUN_DRIVE / "summary.csv"); display(SUMMARY.round(4))
"""), md("""
## 9. Takeaways
"""), code("""
lines = [f"Step 6d run {STEP6D_DIR.name} (reference `pure_all`), Step 6b run {STEP6B_DIR.name}.", ""]
for name, r in SUMMARY.iterrows():
    lines.append(f"- {ci.pretty(name)}: input {r['input']}, lit {100 * r.lit_share:.1f}%, {int(r.total):,} parameters, "
                 f"AUC {r.roc_auc:.4f}, log loss {r.log_loss:.4f}, GPU {r.gpu_speedup_vs_6d:.2f}x vs 6d, "
                 f"{r.train_s_per_epoch / 60:.1f} min per epoch")
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **`quarter` or `compact` tied with `pure_all`**: most of the picture's rows are not needed; the smaller input trains
  and runs faster for the same accuracy.
- **`channels` better than `pure_all`**: putting price, volatility and volume of one bar in one pixel column helps the
  CNN combine them.
- **`filled` better than `pure_all`**: the CNN reads levels more easily from filled areas than from thin lines.
- **Everything tied**: the black space was never the problem; the information content sets the ceiling (≈ HGB 43).

## 10. Held-out test 2024–2025 (run once, at the end)
Every variant from its 2023 checkpoint on Step 6b's test windows; Step 6d's and Step 6b's test scores are joined when
those runs have them. Set `RUN_TEST = True` once, after looking at 2023 only.
"""), code("""
RUN_TEST = SMOKE
if RUN_TEST:
    W["test"] = pd.read_parquet(STEP6B_DIR / "windows_test.parquet")
    RAW43["test"] = fm.numbers(store, W["test"], RVT)[1]
    LRV["test"], NUM["test"] = pi.log_rv(RAW43["test"]), SC43(RAW43["test"])
    TEST = {}
    for cell in CELLS:
        name = cell["name"]
        _, median = ic.cell_rows(store, cell, "train", W["train"])
        te, _ = ic.cell_rows(store, cell, "test", W["test"], median)
        out = te[["security_index", "permno", "end_index", "decision_date", "label"]].copy()
        out["score_pure_all"] = pi.score(REF, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"],
                                         STEP6D_DIR / name / REF / "best.pt", DEVICE)
        for v in VARIANTS:
            out[f"score_cmp_{v}"] = ci.score(v, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"], ckpt(name, v), DEVICE)
        out = pi.join_6b(out, STEP6B_DIR / name / "test_predictions.parquet", FINAL)
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(predictions=out)
        print(f"{name}: {len(out):,} test rows on {out.decision_date.nunique()} days")
    TEST_TABLE, TEST_CHECK = compare(TEST, 50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv", index=False); TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False)
    display(TEST_TABLE.set_index("model")[["roc_auc", "log_loss", "cell"]].round(4)); display(TEST_CHECK.round(4))
""")]
    return c
