"""Cells of MindFlow Step 7b (knowledge distillation of the final model)."""
from cells import code, header, md, setup


def distill_cells():
    c = [header("7b", "Distilling the final model", "Step 6b found that a 7,106-parameter MLP on the 43 numbers is "
                "within 0.001 AUC of the 783,556-parameter CNN. How small can the model with the picture be, and does "
                "learning from the teacher's probabilities (distillation) let a small model keep what the big one has?", """
**Teacher:** the Step 6b final CNN (`ewsd_tab_rv`: chart + EWSD band, 43 numbers as a second input). Its checkpoint
and its 2023 / 2024–25 scores are read from the Step 6b run, nothing is re-trained.

| student | image | parameters (approx.) | trained on |
|---|---|---|---|
| `small_hard` / `small_kd` | full chart, CNN 16-32-64 | 86,020 (1/9) | labels / teacher |
| `small_half_hard` / `small_half_kd` | chart halved in height (2×1 max pool), CNN 16-32-64 | 86,020, half the convolution work | labels / teacher |
| `mlp_kd` | none, 43 numbers | 7,106 | teacher (the labels twin is 6b's `tab_only_rv`) |

**Loss:** (1 − α)·CE(label) + α·T²·KL(teacher ‖ student) at temperature T (Hinton et al. 2015), α = 0.7, T = 1.
Same rows, windows, EWSD scale, scaling of the numbers, optimizer, learning rates and early-stopping rule as 6b, so
each paired difference is the effect of size or of distillation alone. Every model prints its parameter count, size
on disk and GPU latency, so the result is a size / speed / AUC frontier.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import efficiency as ef
import distill as kd
import matplotlib.pyplot as plt
fm.register()
print("distill", kd.__version__, "| final_model", fm.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP6B_RUN = None` takes the newest Step 6b run. **To resume after a disconnect**, set `RUN_ID` to the printed id and
Run all: finished students are skipped. `HORIZONS = [1]` (5 minutes) by default; add 12 for 60 minutes.

**Time (estimate on an L4):** teacher scores on the training windows ~5 min; each small-CNN student ~30–45 min (the
same number of charts is drawn as for the teacher, so drawing on the CPU may set the pace); the MLP ~5 min. About
2.5–3 hours for 5 students at 5 minutes. High-RAM is needed for the realized-volatility tables (~6 GB).
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None
STEP6B_RUN = None
HORIZONS = [1]
STUDENTS = ["mlp_kd", "small_kd", "small_hard", "small_half_kd", "small_half_hard"]   # run order
KD = dict(kd_alpha=0.7, kd_temperature=1.0)
STEP6B_DIR = mf.latest_run(HERE, "6b", STEP6B_RUN)
assert STEP6B_DIR is not None, "run Step 6b first"
S6 = json.loads((STEP6B_DIR / "settings.json").read_text())
FINAL = json.loads((STEP6B_DIR / "final_choice.json").read_text())["final"]
SETTINGS = dict(S6["settings"]) | KD | dict(teacher_variant=FINAL)    # identical training rule, the teacher's picture
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELLS = [c for c in S6["cells"] if c["horizon"] in HORIZONS and (STEP6B_DIR / c["name"] / FINAL / "best.pt").exists()]
assert CELLS, f"no Step 6b checkpoint of {FINAL} for horizons {HORIZONS}"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "7b", RUN_ID, "distill", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, students=STUDENTS, cells=CELLS, step6b=STEP6B_DIR.name,
                 teacher=FINAL, distill=kd.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 6b run:", STEP6B_DIR.name, "| teacher:", FINAL, "| cells:", [c["name"] for c in CELLS], "| students:", STUDENTS)
"""), md("""
## 3. Step 6b's windows and numbers, the teacher's soft targets, parameter counts
"""), code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
W = {s: pd.read_parquet(STEP6B_DIR / f"windows_{s}.parquet") for s in ("train", "valid")}
PRE = json.loads((STEP6B_DIR / "preprocessing.json").read_text())
SCALE = tuple(PRE["ewsd_log_scale"])
SC43 = fm.scaler_from_state(PRE["tab"])            # Step 6b's scaling (training rows only)
t0 = time.perf_counter()
RVT = rg.rv_tables(store, SETTINGS["min_bars"])
print(f"realized-vol tables in {(time.perf_counter() - t0) / 60:.1f} min")
RAW43 = {s: fm.numbers(store, W[s], RVT)[1] for s in W}
TAB43 = {s: SC43(RAW43[s]) for s in W}
N_TAB = TAB43["train"].shape[1]
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,}; {N_TAB} numbers")

PARAMS = []
teacher = ic.IndicatorCNN(SETTINGS, N_TAB, image=True)
for name, m in [("teacher", teacher)] + [(f"stu_{s}", kd.build(s, SETTINGS, N_TAB)) for s in STUDENTS]:
    n = ic.count(m)
    PARAMS.append(dict(model=name, label=kd.pretty(name), total=n["total"], cnn=n["cnn"], numbers_mlp=n["tab"],
                       megabytes=ef.model_bytes(m) / 2**20))
PARAMS = pd.DataFrame(PARAMS).set_index("model"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 4. Train the students (validation 2023)
The teacher scores every training window once (saved). Students early-stop on the true labels of the early-stopping
days, as in 6b.
"""), code("""
RESULTS, VA, t0 = {}, {}, time.perf_counter()
ckpt = lambda name, s: next(p for p in (RUN_LOCAL / name / s / "best.pt", RUN_DRIVE / name / s / "best.pt") if p.exists())
for i, cell in enumerate(CELLS, 1):
    name = cell["name"]
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", name, "=" * 20)
    (RUN_DRIVE / name).mkdir(exist_ok=True)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    VA[name] = va
    ck = STEP6B_DIR / name / FINAL / "best.pt"
    zpath = RUN_DRIVE / name / "teacher_train_prob.npy"
    if zpath.exists():
        p_train = np.load(zpath)
    else:
        t1 = time.perf_counter()
        p_train = fm.score_checkpoint(PREPARED, tr, FINAL, SETTINGS, SCALE, TAB43["train"], ck, DEVICE, SETTINGS["num_workers"])
        np.save(zpath, p_train)
        print(f"teacher scored {len(tr):,} training windows in {(time.perf_counter() - t1) / 60:.1f} min")
    print(f"teacher on training windows: AUC {ic._auc(tr.label.to_numpy(), p_train):.4f} (in-sample), "
          f"mean {p_train.mean():.3f}, std {p_train.std():.3f}")
    data = dict(train=tr, es=es, valid=va, tab_train=TAB43["train"], tab_valid=TAB43["valid"], scale=SCALE,
                z_train=kd.teacher_logit(p_train))
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    pred = va[keep].copy()
    info = dict(cell=cell, prior=float(tr.label.mean()), parameters={}, best_epoch={}, minutes_trained={})
    for s in STUDENTS:
        p, si = kd.run_student(s, PREPARED, data, SETTINGS, DEVICE, RUN_LOCAL / name, RUN_DRIVE / name)
        pred[f"score_stu_{s}"] = p
        info["parameters"][f"stu_{s}"] = si["parameters"]
        info["best_epoch"][f"stu_{s}"], info["minutes_trained"][f"stu_{s}"] = si["best_epoch"], si["minutes"]
    ref = pd.read_parquet(STEP6B_DIR / name / "predictions_with_references.parquet")
    cols = {f"score_ind_{FINAL}": "score_teacher", "score_ind_tab_only_rv": "score_teacher_tab_only_rv",
            "score_hgb_tab43": "score_hgb_tab43", "score_hgb_tab38": "score_hgb_tab38"}
    ref = ref[["security_index", "end_index", "label"] + [c for c in cols if c in ref]].rename(columns=cols | {"label": "ref_label"})
    pred = pred.merge(ref, on=["security_index", "end_index"], how="left")
    assert pred.ref_label.notna().all() and (pred.ref_label == pred.label).all(), "Step 6b rows differ"
    pred = pred.drop(columns="ref_label")
    pred.to_parquet(RUN_DRIVE / name / "predictions.parquet", index=False)
    (RUN_DRIVE / name / "done.json").write_text(json.dumps(info, default=float))
    RESULTS[name] = dict(info=info, predictions=pred)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 5. Compare on identical rows
AUC with 95% intervals (whole days resampled) and the paired checks ("better"/"worse" when the 95% interval of the
difference excludes zero).
"""), code("""
def compare(results, n_boot):
    tables, checks, boots = [], [], {}
    for name, r in results.items():
        pred = r["predictions"]
        names = [c[6:] for c in pred.columns if c.startswith("score_")]
        boots[name] = ic.day_bootstrap(pred, names, n_boot)
        tables.append(kd.auc_table(pred, names, boots[name]).assign(cell=name))
    return pd.concat(tables), kd.checks_table(boots)

TABLE, CHECK = compare(RESULTS, 50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "distill_table.csv", index=False)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False)
display(TABLE.pivot_table(index="model", columns="cell", values="roc_auc").sort_values(CELLS[0]["name"], ascending=False).round(4))
display(CHECK.round(4))
"""), md("""
## 6. Size, speed and accuracy
Inference only: parameters, size of the saved weights, and GPU milliseconds per batch of real charts (fp32 and 16-bit),
for the teacher and every student. Drawing the chart is not included (it is the same for every image model).
"""), code("""
BATCHES = [1, 256] if not SMOKE else [1, 8]
cell0 = CELLS[0]["name"]
sample = VA[cell0].sample(min(512, len(VA[cell0])), random_state=0).reset_index(drop=True)
X = torch.stack([ic.Charts(PREPARED, sample, FINAL, SCALE, SETTINGS["halflife"], TAB43["valid"][sample.row.to_numpy()])[i][0]
                 for i in range(len(sample))])
Tb = torch.as_tensor(TAB43["valid"][sample.row.to_numpy()], dtype=torch.float32)
SPEED = []
models = {"teacher": (ic.IndicatorCNN(SETTINGS, N_TAB, image=True), STEP6B_DIR / cell0 / FINAL / "best.pt")}
models |= {f"stu_{s}": (kd.build(s, SETTINGS, N_TAB), ckpt(cell0, s)) for s in STUDENTS}
for name, (m, ck) in models.items():
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); m = m.to(DEVICE).eval()
    for precision in (["fp32", "bf16"] if DEVICE.type == "cuda" else ["fp32"]):
        for r in ef.benchmark(name, m, DEVICE, precision, X, Tb, BATCHES, reps=10 if SMOKE else 30):
            SPEED.append(r | dict(precision=precision))
    del m
SPEED = pd.DataFrame(SPEED); SPEED.to_csv(RUN_DRIVE / "speed.csv", index=False)
big = SPEED[(SPEED.batch == max(BATCHES)) & (SPEED.precision == SPEED.precision.iloc[-1])].set_index("variant")
auc0 = TABLE[TABLE.cell == cell0].set_index("raw").roc_auc
FRONTIER = PARAMS.join(big[["ms_per_batch", "images_per_s"]]).join(auc0.rename("auc_5m_2023"))
FRONTIER["parameters_vs_teacher"] = FRONTIER.total / FRONTIER.loc["teacher", "total"]
FRONTIER["speedup_vs_teacher"] = FRONTIER.loc["teacher", "ms_per_batch"] / FRONTIER.ms_per_batch
FRONTIER.to_csv(RUN_DRIVE / "frontier.csv"); display(FRONTIER.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, r in FRONTIER.iterrows():
    for a, x in zip(ax, (r.total, r.ms_per_batch)):
        a.scatter(x, r.auc_5m_2023, s=40, marker="*" if name == "teacher" else "o")
        a.annotate(kd.pretty(name), (x, r.auc_5m_2023), fontsize=7, xytext=(4, 2), textcoords="offset points")
for name in ("hgb_tab43",):
    if name in auc0:
        for a in ax:
            a.axhline(auc0[name], ls="--", lw=0.8, color="grey"); a.text(a.get_xlim()[0], auc0[name], " HGB 43", fontsize=7, va="bottom")
ax[0].set_xscale("log"); ax[0].set_xlabel("parameters"); ax[1].set_xlabel(f"GPU ms per batch of {max(BATCHES)}")
ax[1].set_xscale("log")
for a in ax:
    a.set_ylabel(f"AUC, {cell0}, 2023")
fig.suptitle("Size / speed vs AUC: teacher and distilled students"); fig.tight_layout()
fig.savefig(RUN_DRIVE / "frontier.png", dpi=130); plt.show()
"""), md("""
## 7. Takeaways
"""), code("""
lines = [f"Teacher `{FINAL}` (Step 6b run {STEP6B_DIR.name}); alpha {SETTINGS['kd_alpha']}, T {SETTINGS['kd_temperature']}.", ""]
for name in RESULTS:
    t = TABLE[TABLE.cell == name].set_index("raw")
    for k in ["teacher", "hgb_tab43", "teacher_tab_only_rv"] + [f"stu_{s}" for s in STUDENTS]:
        if k in t.index:
            n = PARAMS.total.get(k)
            lines.append(f"- {name} {kd.pretty(k)}: AUC {t.loc[k, 'roc_auc']:.4f} [{t.loc[k, 'auc_ci_low']:.4f}, "
                         f"{t.loc[k, 'auc_ci_high']:.4f}]" + (f", {int(n):,} parameters" if n == n and n is not None else ""))
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
for name, r in FRONTIER.iterrows():
    if name != "teacher" and r.ms_per_batch == r.ms_per_batch:
        lines.append(f"- {kd.pretty(name)}: {r.parameters_vs_teacher:.3f} of the teacher's parameters, "
                     f"{r.speedup_vs_teacher:.1f}x faster per batch of {max(BATCHES)} on the GPU")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **`*_kd` better than `*_hard`**: the teacher's probabilities carry something the 0/1 labels do not (distillation helps).
- **Distilled student tied with the teacher**: the small model keeps the big one's accuracy at a fraction of its size.
- **Half-height tied with full**: half of the convolution work is not needed.
- **Distilled MLP tied with the teacher**: the picture's small gain does not need a CNN at inference.

## 8. Held-out test 2024–2025 (run once, at the end)
Students from their 2023 checkpoints on Step 6b's test windows; the teacher's and HGB's test scores come from the 6b
test section (run it first). Set `RUN_TEST = True` once, after the students are chosen.
"""), code("""
RUN_TEST = SMOKE
if RUN_TEST:
    W["test"] = pd.read_parquet(STEP6B_DIR / "windows_test.parquet")
    TAB43["test"] = SC43(fm.numbers(store, W["test"], RVT)[1])
    TEST = {}
    for cell in CELLS:
        name = cell["name"]
        ref = pd.read_parquet(STEP6B_DIR / name / "test_predictions.parquet")
        _, median = ic.cell_rows(store, cell, "train", W["train"])
        te, _ = ic.cell_rows(store, cell, "test", W["test"], median)
        keep = ["security_index", "permno", "end_index", "decision_date", "label"]
        out = te[keep].copy()
        for s in STUDENTS:
            out[f"score_stu_{s}"] = kd.score(s, PREPARED, te, SETTINGS, SCALE, TAB43["test"], ckpt(name, s), DEVICE)
        cols = {f"score_ind_{FINAL}": "score_teacher", "score_ind_tab_only_rv": "score_teacher_tab_only_rv",
                "score_hgb_tab43": "score_hgb_tab43", "score_hgb_tab38": "score_hgb_tab38"}
        ref = ref[["security_index", "end_index", "label"] + [c for c in cols if c in ref]].rename(columns=cols | {"label": "ref_label"})
        out = out.merge(ref, on=["security_index", "end_index"], how="left")
        assert out.ref_label.notna().all() and (out.ref_label == out.label).all(), "Step 6b test rows differ"
        out = out.drop(columns="ref_label")
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(predictions=out)
        print(f"{name}: {len(out):,} test rows on {out.decision_date.nunique()} days")
    TEST_TABLE, TEST_CHECK = compare(TEST, 50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv", index=False); TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False)
    display(TEST_TABLE.pivot_table(index="model", columns="cell", values="roc_auc").round(4)); display(TEST_CHECK.round(4))
""")]
    return c
