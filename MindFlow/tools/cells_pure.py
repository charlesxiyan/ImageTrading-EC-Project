"""Cells of MindFlow Step 6d (a pure-image model: the numbers drawn into the picture)."""
from cells import code, header, md, setup


def pure_cells(step="6d", horizons="[1]", variants='["pure_key", "pure_all"]'):
    intro = "" if step == "6d" else (
        "**Step 6e = Step 6d at 60 minutes.** Same code and pictures; the strip scales are fitted again on the same "
        "training windows (identical to 6d's), the comparison is with Step 6b's and 4a's 60-minute scores. Run it after "
        "Step 7b+. By default only `pure_all` (the full picture); add `pure_key` to `VARIANTS` for the key strips.\n\n")
    c = [header(step, "A pure-image model" + (" at 60 minutes" if step != "6d" else ""), "The project started from one idea: can a machine read the chart the way a "
                "trader does? The final model (6b) reaches HGB only with 43 numbers as a second input. If what those "
                "numbers know is drawn into the picture instead, can a CNN that sees nothing but the picture do as well?", """
**Why it might work.** The chart is stretched to each window's own low and high, so no level is visible in it. The one
level already drawn on a fixed scale, the EWSD band, gave the largest gain inside the image (Step 4a, +0.0175). Here the
other levels the numbers carry are drawn the same way, and the numbers input is removed.

| picture | rows | what is added under Step 6b's chart + EWSD band (312 rows) |
|---|---|---|
| `pure_key` | 423 | a grey line at the first bar of the decision session (time of day); SPY's EWSD band; a volume band (each bar's volume ÷ the mean of the 78 bars before the window, log scale); four bars for the multi-day RV (stock 5 and 20 sessions, SPY 5 and 20 sessions) |
| `pure_all` | 466 | `pure_key` + all 43 standardised numbers of Step 6b as small bars (up if positive, down if negative): the upper bound, every number is in the picture |

Every strip uses a fixed scale fitted on training windows only (0.5–99.5% of the log values; the stock EWSD band keeps
Step 6b's scale). Same CNN (64-128-256, plain), windows, rows, training rule as Step 6b; no numbers input.

**What it answers** (all paired on identical rows of 2023):
- `pure_key` vs 4a's chart + EWSD CNN (0.678): do the key levels help a model that sees only pictures?
- `pure_key` / `pure_all` vs the final model (0.688) and HGB 43 (0.689): can a pure-image model reach them?
- `pure_all` vs `pure_key`: is anything left in the numbers that the key strips do not show?
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import pure_image as pi
import matplotlib.pyplot as plt
fm.register()
print("pure_image", pi.__version__, "| final_model", fm.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP6B_RUN = None` takes the newest Step 6b run (its windows, rows, scaling and scores are reused). **To resume after
a disconnect**, set `RUN_ID` to the printed id and Run all: finished variants are skipped.

**Time (estimate on an L4, from Step 6b's ~45–60 min per image variant and cell):** the canvas is 35–50% taller, so
~60–90 min per variant and cell: ~2.5–3 hours for both variants at 5 minutes, ~5–6 hours with 60 minutes too.
`HORIZONS`: 1 = 5 minutes, 12 = 60 minutes (one 60-minute cell takes about as long as one 5-minute cell). High-RAM is needed for the realized-volatility tables.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None
STEP6B_RUN = None
HORIZONS = {horizons}
VARIANTS = {variants}                # run order
if SMOKE:
    HORIZONS = [1]                                 # the synthetic smoke run of Step 6b has 5 minutes only
STEP6B_DIR = mf.latest_run(HERE, "6b", STEP6B_RUN)
assert STEP6B_DIR is not None, "run Step 6b first"
S6 = json.loads((STEP6B_DIR / "settings.json").read_text())
FINAL = json.loads((STEP6B_DIR / "final_choice.json").read_text())["final"]
SETTINGS = dict(S6["settings"])                    # identical training rule
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELLS = [c for c in S6["cells"] if c["horizon"] in HORIZONS
         and (STEP6B_DIR / c["name"] / "predictions_with_references.parquet").exists()]
assert CELLS, f"no finished Step 6b cell for horizons {HORIZONS}"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "{step}", RUN_ID, "pure", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, variants=VARIANTS, cells=CELLS, step6b=STEP6B_DIR.name,
                 final=FINAL, pure_image=pi.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 6b run:", STEP6B_DIR.name, "| final model:", FINAL, "| cells:", [c["name"] for c in CELLS], "| variants:", VARIANTS)
"""), md("""
## 3. Step 6b's windows and numbers, the strip scales (training windows only), parameter counts
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
W = {s: pd.read_parquet(STEP6B_DIR / f"windows_{s}.parquet") for s in ("train", "valid")}
PRE = json.loads((STEP6B_DIR / "preprocessing.json").read_text())
SC43 = fm.scaler_from_state(PRE["tab"])            # Step 6b's scaling of the 43 numbers (training rows only)
t0 = time.perf_counter()
RVT = rg.rv_tables(store, SETTINGS["min_bars"])
print(f"realized-vol tables in {(time.perf_counter() - t0) / 60:.1f} min")
RAW43 = {s: fm.numbers(store, W[s], RVT)[1] for s in W}
LRV = {s: pi.log_rv(RAW43[s]) for s in W}
NUM = {s: SC43(RAW43[s]) for s in W}
MC = store.market_close
spath = RUN_DRIVE / "strip_scales.json"
if spath.exists():
    SCALES = {k: tuple(v) for k, v in json.loads(spath.read_text()).items()}
else:
    SCALES = pi.fit_scales(PREPARED, W["train"], LRV["train"], MC, SETTINGS["halflife"], PRE["ewsd_log_scale"],
                           n=2_000 if SMOKE else 20_000, seed=SETTINGS["seed"])
    spath.write_text(json.dumps(SCALES, indent=2) + "\\n")
print("strip scales (log units):", json.dumps(SCALES))
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,}")
print("missing share of the multi-day RV (validation):", np.isnan(LRV["valid"]).mean(0).round(4).tolist())
PARAMS = []
for v in VARIANTS:
    n = ic.count(pi.PureCNN(SETTINGS, v))
    PARAMS.append(dict(variant=v, model=pi.LABELS[v], canvas=f"{pi.height(v)}x{ic.SIZE}", total=n["total"]))
PARAMS = pd.DataFrame(PARAMS).set_index("variant"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 4. What the pure-image model sees
Two real validation windows at the same decision bar: the calmest and the most volatile stock of that bar (by the
5-minute EWSD at the decision bar). From top to bottom: chart (grey line = first bar of the decision session), stock
EWSD band, SPY EWSD band, volume band, the four multi-day RV bars, and (`pure_all` only) the 43 numbers. Saved as PNG
in the run folder for the report.
"""), code("""
va0 = W["valid"]
bar = va0.end_index.value_counts().index[0]                  # a decision bar with many stocks
pick = va0[va0.end_index == bar]
ewsd = np.array([ic.indicators(ic.Charts(PREPARED, pick.reset_index(drop=True), "plain", (0, 1), SETTINGS["halflife"])
                               .long_window(i), SETTINGS["halflife"])[1][-1] for i in range(len(pick))])
rows = [pick.index[int(np.argmin(ewsd))], pick.index[int(np.argmax(ewsd))]]
fig, axes = plt.subplots(1, 4, figsize=(14, 7))
for j, r in enumerate(rows):
    f = W["valid"].loc[[r]].assign(row=r).reset_index(drop=True)
    for k, v in enumerate(VARIANTS):
        img = pi.dataset(PREPARED, f, v, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"]).image(0)
        tag = f"{'calm' if j == 0 else 'volatile'}_{v}"
        pi.save_png(img, RUN_DRIVE / f"sample_{tag}.png")
        ax = axes[2 * j + k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"permno {int(f.permno[0])}, {f.decision_date[0]}\\n{v} ({img.shape[0]}x{img.shape[1]})", fontsize=8)
fig.tight_layout(); fig.savefig(RUN_DRIVE / "samples.png", dpi=130); plt.show()
"""), md("""
## 5. Train (validation 2023)
"""), code("""
RESULTS, VA, t0 = {}, {}, time.perf_counter()
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
    info = dict(cell=cell, prior=float(tr.label.mean()), parameters={}, best_epoch={}, minutes_trained={})
    for v in VARIANTS:
        p, vi = pi.run_variant(v, PREPARED, data, SETTINGS, DEVICE, RUN_LOCAL / name, RUN_DRIVE / name)
        pred[f"score_pure_{v}"] = p
        info["parameters"][f"pure_{v}"] = vi["parameters"]
        info["best_epoch"][f"pure_{v}"], info["minutes_trained"][f"pure_{v}"] = vi["best_epoch"], vi["minutes"]
    pred = pi.join_6b(pred, STEP6B_DIR / name / "predictions_with_references.parquet", FINAL)
    pred.to_parquet(RUN_DRIVE / name / "predictions.parquet", index=False)
    (RUN_DRIVE / name / "done.json").write_text(json.dumps(info, default=float))
    RESULTS[name] = dict(info=info, predictions=pred)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 6. Compare on identical rows
AUC with 95% intervals (whole days resampled) and the paired checks ("better"/"worse" when the 95% interval of the
difference excludes zero). Step 6b's and Step 4a's scores are for exactly these rows.
"""), code("""
def compare(results, n_boot):
    tables, boots = [], {}
    for name, r in results.items():
        pred = r["predictions"]
        names = [c[6:] for c in pred.columns if c.startswith("score_")]
        boots[name] = ic.day_bootstrap(pred, names, n_boot)
        tables.append(pi.auc_table(pred, names, boots[name]).assign(cell=name))
    return pd.concat(tables), pi.checks_table(boots)

TABLE, CHECK = compare(RESULTS, 50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "pure_table.csv", index=False)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False)
display(TABLE.pivot_table(index="model", columns="cell", values="roc_auc").sort_values(CELLS[0]["name"], ascending=False).round(4))
display(CHECK.round(4))
"""), md("""
## 7. Takeaways
"""), code("""
lines = [f"Step 6b run {STEP6B_DIR.name}, final model `{FINAL}`; strip scales fitted on training windows.", ""]
for name in RESULTS:
    t = TABLE[TABLE.cell == name].set_index("raw")
    for k in ["final", "hgb_tab43", "tab_only_rv", "s4a_ewsd", "s4a_plain"] + [f"pure_{v}" for v in VARIANTS]:
        if k in t.index:
            n = PARAMS.total.get(k[5:]) if k.startswith("pure_") else None
            lines.append(f"- {name} {pi.pretty(k)}: AUC {t.loc[k, 'roc_auc']:.4f} [{t.loc[k, 'auc_ci_low']:.4f}, "
                         f"{t.loc[k, 'auc_ci_high']:.4f}]" + (f", {int(n):,} parameters" if n else ""))
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **`pure_key` > chart + EWSD (4a)**: the key levels drawn on fixed scales help a model that sees only pictures.
- **`pure_key` tied with the final model**: the picture alone is enough when it is drawn right; the numbers input is not needed.
- **`pure_all` > `pure_key`**: something the key strips do not show is still in the 43 numbers.
- **`pure_all` < the final model**: the CNN reads numbers from bars less well than an MLP reads them directly.

## 8. Held-out test 2024–2025 (run once, at the end)
Both variants from their 2023 checkpoints on Step 6b's test windows; the final model's and HGB's test scores come from
the 6b test section. Set `RUN_TEST = True` once, after looking at 2023 only.
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
        for v in VARIANTS:
            ck = next(p for p in (RUN_LOCAL / name / v / "best.pt", RUN_DRIVE / name / v / "best.pt") if p.exists())
            out[f"score_pure_{v}"] = pi.score(v, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"], ck, DEVICE)
        out = pi.join_6b(out, STEP6B_DIR / name / "test_predictions.parquet", FINAL)
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(predictions=out)
        print(f"{name}: {len(out):,} test rows on {out.decision_date.nunique()} days")
    TEST_TABLE, TEST_CHECK = compare(TEST, 50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv", index=False); TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False)
    display(TEST_TABLE.pivot_table(index="model", columns="cell", values="roc_auc").round(4)); display(TEST_CHECK.round(4))
""")]
    for cell in c:
        cell["source"] = (cell["source"].replace("{horizons}", horizons).replace("{variants}", variants)
                          .replace('"{step}"', f'"{step}"'))
    return c
