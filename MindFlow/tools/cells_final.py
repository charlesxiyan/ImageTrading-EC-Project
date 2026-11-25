"""Cells of MindFlow Step 6b (the final model)."""
from cells import code, header, md, setup


def final_cells():
    c = [header("6b", "The final model", "Step 4a's best image model plus the one regime input that helped in Step 6a "
                "(multi-day realized volatility): how good is it on 2023, is the image still needed, and how does it do "
                "once on the held-out years 2024–2025?", """
**What changes and what does not.** Only the numbers given to the CNN's second input grow from 38 to 43 (the stock's
realized volatility over the last 5 and 20 completed sessions, their log ratio, SPY's 5- and 20-session RV; Step 6a
found +0.0025 AUC for HGB, significant). Windows, purge/embargo, the EWSD scale, the CNN and its training rule are
Step 4a's. VIX and the VIXY panel are left out: both tied in Steps 6a and 3a.

| model | input | role |
|---|---|---|
| `ma_ewsd_tab_rv` | chart + MA + EWSD band, 38 + 5 numbers | Step 4a's top rung + RV (paired against Step 4a's own `ma_ewsd_tab`) |
| `ewsd_tab_rv` | chart + EWSD band, 38 + 5 numbers | the same without the MA line (Step 4a: MA added nothing) |
| `tab_only_rv` | 38 + 5 numbers, no image | control: is the picture still needed? |
| HGB 38 / HGB 43 | the numbers | benchmark, fitted here on the same training windows |

**Choosing the final model** uses 2023 only. The test section scores every model above once on 2024–2025.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import feature_select as fs
import regime as rg
import final_model as fm
fm.register()
print("final_model", fm.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP4A_RUN = None` takes the newest Step 4a run; its windows and EWSD scale are reused, so rows are identical.
**To resume after a disconnect**, set `RUN_ID` to the printed id and Run all: finished variants are skipped.

**Time (estimate on an L4, from Step 4a):** ~45–60 min per image variant per cell and ~5 min for the features-only MLP,
so ~4 hours for 2 image variants × 2 cells. `VARIANTS = ["ma_ewsd_tab_rv", "tab_only_rv"]` halves it. The test section
adds ~30 min. Realized-vol tables need ~6 GB of RAM (High-RAM).
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None
STEP4A_RUN = None                                  # None = newest Step 4a run (its windows and EWSD scale are reused)
VARIANTS = ["ma_ewsd_tab_rv", "tab_only_rv", "ewsd_tab_rv"]   # run order
HORIZONS = [1, 12]
STEP4A_DIR = mf.latest_run(HERE, "4a", STEP4A_RUN)
assert STEP4A_DIR is not None, "run Step 4a first"
S4 = json.loads((STEP4A_DIR / "settings.json").read_text())
SETTINGS = dict(S4["settings"])                    # identical training rule
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2), min_bars=10 if SMOKE else 60)
if SMOKE:
    HORIZONS = [1]
CELLS = [c for c in S4["cells"] if c["horizon"] in HORIZONS]
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "6b", RUN_ID, "final", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, variants=VARIANTS, cells=CELLS, step4a=str(STEP4A_DIR.name),
                 final_model=fm.__version__, indicator_cnn=ic.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 4a run:", STEP4A_DIR.name, "| cells:", ", ".join(c["name"] for c in CELLS), "| variants:", ", ".join(VARIANTS))
"""), md("""
## 3. Windows (Step 4a's), the 43 numbers, parameter counts
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
W = {s: pd.read_parquet(STEP4A_DIR / f"windows_{s}.parquet") for s in ("train", "valid")}
for s in W:                                       # a copy, so this run is self-contained (Step 7a reads it)
    if not (RUN_DRIVE / f"windows_{s}.parquet").exists():
        shutil.copy2(STEP4A_DIR / f"windows_{s}.parquet", RUN_DRIVE / f"windows_{s}.parquet")
PRE4 = json.loads((STEP4A_DIR / "preprocessing.json").read_text())
SCALE = tuple(PRE4["ewsd_log_scale"])
t0 = time.perf_counter()
RVT = rg.rv_tables(store, SETTINGS["min_bars"])
print(f"realized-vol tables in {(time.perf_counter() - t0) / 60:.1f} min")
RAW38, RAW43 = {}, {}
for s in W:
    RAW38[s], RAW43[s] = fm.numbers(store, W[s], RVT)
SC43 = ic.TabScaler(RAW43["train"])                              # training rows only
TAB43 = {s: SC43(RAW43[s]) for s in W}
fm.save_json(RUN_DRIVE / "preprocessing.json", dict(ewsd_log_scale=SCALE, halflife=SETTINGS["halflife"], tab=SC43.state(),
                                                    step4a=STEP4A_DIR.name))
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,} on {W['valid'].decision_date.nunique()} days")
print("missing share of the RV numbers (validation):", RAW43["valid"][fm.RV].isna().mean().round(4).to_dict())
display(RAW43["train"][fm.RV].describe().T.round(4))
PARAMS = []
for v in VARIANTS:
    n = ic.count(ic.IndicatorCNN(SETTINGS, TAB43["train"].shape[1], image=v not in ic.IMAGE_FREE))
    PARAMS.append(dict(variant=v, model=ic.VARIANT_LABEL[v], total=n["total"], cnn=n["cnn"], feature_mlp=n["tab"]))
PARAMS = pd.DataFrame(PARAMS).set_index("variant"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 4. Train (validation 2023) and fit HGB on the same rows
"""), code("""
RESULTS, HGB, t0 = {}, {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    data = dict(train=tr, es=es, valid=va, tab_train=TAB43["train"], tab_valid=TAB43["valid"], scale=SCALE,
                prior=float(tr.label.mean()))
    r = ic.run_cell(PREPARED, cell, data, VARIANTS, SETTINGS, DEVICE, RUN_LOCAL / cell["name"], RUN_DRIVE / cell["name"])
    pred = r["predictions"]
    for name, raw in (("hgb_tab38", RAW38), ("hgb_tab43", RAW43)):
        m = fm.fit_hgb(raw["train"], tr.row.to_numpy(), tr.label.to_numpy(), SETTINGS["seed"])
        HGB[(cell["name"], name)] = m
        pred[f"score_{name}"] = m.predict_proba(raw["valid"].iloc[va.row.to_numpy()])[:, 1]
        r["info"]["parameters"][name] = dict(total=fs.tree_size(m))
    pred = fm.join_step4a(pred, STEP4A_DIR / cell["name"])
    pred.to_parquet(RUN_DRIVE / cell["name"] / "predictions_with_references.parquet", index=False)
    RESULTS[cell["name"]] = dict(info=r["info"], predictions=pred)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 5. Compare on identical rows
AUC with 95% intervals (whole days resampled) and the paired checks: "better"/"worse" when the 95% interval of the
difference excludes zero.
"""), code("""
TABLE, BOOTS = ic.table(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "final_table.csv")
display(TABLE.reset_index().pivot_table(index="model", columns="cell", values="roc_auc").sort_values(CELLS[0]["name"],
        ascending=False).round(4))
CHECK = fm.checks_table(BOOTS)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False); display(CHECK.round(4))
"""), md("""
## 6. The final model
The image variant with the highest 2023 AUC at 5 minutes, unless `FINAL` is set by hand. This choice is made before
the test section is run, and it is what Step 7a (efficient computing) should load: `SOURCE = dict(step="6b", ...)`.
"""), code("""
FINAL = None                                       # e.g. "ewsd_tab_rv"; None = best image variant at 5 minutes
t = TABLE.reset_index()
img = [v for v in VARIANTS if v not in ic.IMAGE_FREE]
c5 = CELLS[0]["name"]
auto = max(img, key=lambda v: float(t[(t.cell == c5) & (t.raw == f"ind_{v}")].roc_auc.iloc[0])) if img else None
FINAL = FINAL or auto
fm.save_json(RUN_DRIVE / "final_choice.json", dict(final=FINAL, label=ic.VARIANT_LABEL.get(FINAL), chosen_on="validation 2023",
                                                   cell=c5, run=RUN_ID, step4a=STEP4A_DIR.name))
lines = [f"**Final model: `{FINAL}`** = {ic.VARIANT_LABEL.get(FINAL)} ({int(PARAMS.loc[FINAL, 'total']):,} parameters)", ""]
for cname in RESULTS:
    s = t[t.cell == cname].set_index("raw")
    for k in (f"ind_{FINAL}", "hgb_tab43", "hgb_tab38", "s4a_ma_ewsd_tab", "ind_tab_only_rv"):
        if k in s.index:
            lines.append(f"- {cname} {ic.pretty(k)}: AUC {s.loc[k, 'roc_auc']:.4f} [{s.loc[k, 'auc_ci_low']:.4f}, {s.loc[k, 'auc_ci_high']:.4f}]")
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **`ma_ewsd_tab_rv` > Step 4a `ma_ewsd_tab`**: the CNN uses the slower volatility as HGB does (Step 6a).
- **Final CNN tied with HGB 43**: with the same information the image model matches the trees.
- **Final CNN > `tab_only_rv`**: the picture adds something the 43 numbers do not; tied means it is redundant.

## 7. Held-out test 2024–2025 (run once, at the end)
Every model above, from its 2023 checkpoint (HGB: the models fitted in section 4), on test windows built with the same
rule (stride, complete SPY window, two embargo sessions at the 2023/2024 boundary). Set `RUN_TEST = True` once the
final model is chosen. Do not change anything after looking at it.
"""), code("""
RUN_TEST = SMOKE
if RUN_TEST:
    EMBARGO = ic.embargo_dates(store, SETTINGS["embargo_sessions"])
    path = RUN_DRIVE / "windows_test.parquet"
    if path.exists():
        W["test"] = pd.read_parquet(path)
    else:
        W["test"] = ic.decision_windows(store, "test", SETTINGS["stride"], EMBARGO, seed=SETTINGS["seed"])
        W["test"].to_parquet(path, index=False)
    assert store.dates[W["test"].end_index.min() - ic.LOOKBACK - ic.HISTORY + 1] > cfg["validation_end"]
    RAW38["test"], RAW43["test"] = fm.numbers(store, W["test"], RVT)
    TAB43["test"] = SC43(RAW43["test"])
    TAB38_4A = fm.scaler_from_state(PRE4["tab"])(RAW38["test"])          # Step 4a's own scaling of its 38 numbers
    TEST = {}
    for cell in CELLS:
        name = cell["name"]
        tr, median = ic.cell_rows(store, cell, "train", W["train"])
        te, _ = ic.cell_rows(store, cell, "test", W["test"], median)
        print(f"{name}: {len(te):,} test rows on {te.decision_date.nunique()} days")
        keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
        out = te[keep].copy()
        out["score_training_prior"] = float(tr.label.mean())
        for v in VARIANTS:
            ck = RUN_DRIVE / name / v / "best.pt"
            out[f"score_ind_{v}"] = fm.score_checkpoint(PREPARED, te, v, SETTINGS, SCALE, TAB43["test"], ck, DEVICE,
                                                        SETTINGS["num_workers"])
        s4 = json.loads((STEP4A_DIR / "settings.json").read_text())["settings"]
        s4.update(num_workers=SETTINGS["num_workers"])
        ck = STEP4A_DIR / name / "ma_ewsd_tab" / "best.pt"
        if ck.exists():
            out["score_s4a_ma_ewsd_tab"] = fm.score_checkpoint(PREPARED, te, "ma_ewsd_tab", s4, SCALE, TAB38_4A, ck, DEVICE,
                                                               SETTINGS["num_workers"])
        for hname, raw in (("hgb_tab38", RAW38), ("hgb_tab43", RAW43)):
            m = HGB.get((name, hname)) or fm.fit_hgb(raw["train"], tr.row.to_numpy(), tr.label.to_numpy(), SETTINGS["seed"])
            out[f"score_{hname}"] = m.predict_proba(raw["test"].iloc[te.row.to_numpy()])[:, 1]
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(info=RESULTS[name]["info"], predictions=out)
    TEST_TABLE, TEST_BOOTS = ic.table(TEST, n_boot=50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv")
    display(TEST_TABLE.reset_index().pivot_table(index="model", columns="cell", values="roc_auc").round(4))
    TEST_CHECK = fm.checks_table(TEST_BOOTS)
    TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False); display(TEST_CHECK.round(4))
""")]
    return c
