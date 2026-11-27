"""Cells of MindFlow Step 7b+ (distillation of the pure-image model)."""
from cells import code, header, md, setup


def pure_distill_cells():
    c = [header("7b+", "Distilling the pure-image model", "Step 6d's pure-image CNN (no numbers input; the numbers "
                "are drawn into the picture) ties HGB 43. How small can a model that only looks at the picture be, "
                "and does learning from the teacher's probabilities let it keep the teacher's accuracy?", """
**Teacher:** Step 6d's `pure_all` (466 × 234 picture: chart + EWSD band + time line, SPY band, volume band, multi-day RV
bars, all 43 numbers drawn as bars; 776,450 parameters). Its checkpoint and 2023 scores are read from the Step 6d run.

| student | picture | parameters (approx.) | trained on |
|---|---|---|---|
| `small_hard` / `small_kd` | full picture, CNN 16-32-64 | 78,914 (1/10) | labels / teacher |
| `small_half_hard` / `small_half_kd` | picture halved in height (2×1 max pool), CNN 16-32-64 | 78,914, half the convolution work | labels / teacher |
| `tiny_kd` | full picture, CNN 8-16-32 | 29,858 (1/26) | teacher |

Every student is a pure-image CNN: no numbers input anywhere (unlike Step 7b, whose smallest student was a numbers MLP).
**Loss:** (1 − α)·CE(label) + α·T²·KL(teacher ‖ student), α = 0.7, T = 1 (as Step 7b). Same rows, windows, strip
scales, optimizer, learning rate and early-stopping rule as Step 6d, so each paired difference is the effect of size or
of distillation alone. Every model prints its parameter count, size on disk and GPU latency.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import pure_image as pi
import distill as kd
import pure_distill as pd_
import efficiency as ef
import matplotlib.pyplot as plt
fm.register()
print("pure_distill", pd_.__version__, "| pure_image", pi.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP6D_RUN = None` takes the newest Step 6d run (which names its Step 6b run). **To resume after a disconnect**, set
`RUN_ID` to the printed id and Run all: finished students are skipped. `HORIZONS = [1]` (5 minutes).

**Time (estimate on an A100, from Step 6d's ~3 min per teacher epoch; drawing on the CPU may set the pace):** teacher
scores on the training windows ~5 min; each student ~10–20 min; about 1–1.5 hours for 5 students.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None
STEP6D_RUN = None
HORIZONS = [1]
STUDENTS = ["small_kd", "small_hard", "small_half_kd", "small_half_hard", "tiny_kd"]   # run order
KD = dict(kd_alpha=0.7, kd_temperature=1.0)
TEACHER = "pure_all"
STEP6D_DIR = mf.latest_run(HERE, "6d", STEP6D_RUN)
assert STEP6D_DIR is not None, "run Step 6d first"
S6D = json.loads((STEP6D_DIR / "settings.json").read_text())
STEP6B_DIR = mf.run_folder(HERE, "6b") / S6D["step6b"]
FINAL = S6D["final"]
SETTINGS = dict(S6D["settings"]) | KD | dict(teacher_variant=TEACHER)
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELLS = [c for c in S6D["cells"] if c["horizon"] in HORIZONS and (STEP6D_DIR / c["name"] / TEACHER / "best.pt").exists()]
assert CELLS, f"no Step 6d checkpoint of {TEACHER} for horizons {HORIZONS}"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "7b+", RUN_ID, "puredistill", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, students=STUDENTS, cells=CELLS, step6d=STEP6D_DIR.name,
                 step6b=STEP6B_DIR.name, teacher=TEACHER, pure_distill=pd_.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 6d run:", STEP6D_DIR.name, "| Step 6b run:", STEP6B_DIR.name, "| cells:", [c["name"] for c in CELLS])
"""), md("""
## 3. Step 6d's windows, numbers and strip scales; the teacher's soft targets; parameter counts
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
print(f"train windows {len(W['train']):,}; validation windows {len(W['valid']):,}; picture {pi.height(TEACHER)}x{ic.SIZE}")

PARAMS = []
for name, m in [("teacher", pi.PureCNN(SETTINGS, TEACHER))] + [(f"stu_{s}", pd_.build(s, SETTINGS)) for s in STUDENTS]:
    PARAMS.append(dict(model=name, label=pd_.pretty(name), total=ic.count(m)["total"], megabytes=ef.model_bytes(m) / 2**20))
PARAMS = pd.DataFrame(PARAMS).set_index("model"); PARAMS.to_csv(RUN_DRIVE / "network_parameters.csv"); display(PARAMS)
"""), md("""
## 4. Train the students (validation 2023)
The teacher scores every training window once (saved). Students early-stop on the true labels of the early-stopping
days, as in Step 6d.
"""), code("""
RESULTS, VA, t0 = {}, {}, time.perf_counter()
ckpt = lambda name, s: next(p for p in (RUN_LOCAL / name / s / "best.pt", RUN_DRIVE / name / s / "best.pt") if p.exists())
REN = {"score_pure_pure_all": "score_teacher", "score_pure_pure_key": "score_pure_key"}
for i, cell in enumerate(CELLS, 1):
    name = cell["name"]
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", name, "=" * 20)
    (RUN_DRIVE / name).mkdir(exist_ok=True)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    VA[name] = va
    zpath = RUN_DRIVE / name / "teacher_train_prob.npy"
    if zpath.exists():
        p_train = np.load(zpath)
    else:
        t1 = time.perf_counter()
        p_train = pi.score(TEACHER, PREPARED, tr, SETTINGS, SCALES, MC, LRV["train"], NUM["train"],
                           STEP6D_DIR / name / TEACHER / "best.pt", DEVICE)
        np.save(zpath, p_train)
        print(f"teacher scored {len(tr):,} training windows in {(time.perf_counter() - t1) / 60:.1f} min")
    print(f"teacher on training windows: AUC {ic._auc(tr.label.to_numpy(), p_train):.4f} (in-sample), "
          f"mean {p_train.mean():.3f}, std {p_train.std():.3f}")
    data = dict(train=tr, es=es, valid=va, scales=SCALES, market_close=MC, lrv=LRV, numbers=NUM,
                z_train=kd.teacher_logit(p_train))
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    pred = va[keep].copy()
    info = dict(cell=cell, prior=float(tr.label.mean()), parameters={}, best_epoch={}, minutes_trained={})
    for s in STUDENTS:
        p, si = pd_.run_student(s, PREPARED, data, SETTINGS, DEVICE, RUN_LOCAL / name, RUN_DRIVE / name)
        pred[f"score_stu_{s}"] = p
        info["parameters"][f"stu_{s}"] = si["parameters"]
        info["best_epoch"][f"stu_{s}"], info["minutes_trained"][f"stu_{s}"] = si["best_epoch"], si["minutes"]
    ref = pd.read_parquet(STEP6D_DIR / name / "predictions.parquet").rename(columns=REN)
    cols = [c for c in ref.columns if c.startswith("score_")]
    ref = ref[["security_index", "end_index", "label"] + cols].rename(columns={"label": "ref_label"})
    pred = pred.merge(ref, on=["security_index", "end_index"], how="left")
    assert pred.ref_label.notna().all() and (pred.ref_label == pred.label).all(), "Step 6d rows differ"
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
SHOW = ["teacher", "pure_key", "final", "hgb_tab43", "tab_only_rv"]
def compare(results, n_boot):
    tables, boots = [], {}
    for name, r in results.items():
        pred = r["predictions"]
        names = [c[6:] for c in pred.columns if c.startswith("score_") and (c[6:] in SHOW or c.startswith("score_stu_"))]
        boots[name] = ic.day_bootstrap(pred, names, n_boot)
        tables.append(pd_.auc_table(pred, names, boots[name]).assign(cell=name))
    return pd.concat(tables), pd_.checks_table(boots)

TABLE, CHECK = compare(RESULTS, 50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "distill_table.csv", index=False)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False)
display(TABLE.pivot_table(index="model", columns="cell", values="roc_auc").sort_values(CELLS[0]["name"], ascending=False).round(4))
display(CHECK.round(4))
"""), md("""
## 6. Size, speed and accuracy
Inference only: parameters, size of the saved weights, and GPU milliseconds per batch of real pictures (fp32 and
16-bit). Drawing the picture is not included (it is the same for every model here).
"""), code("""
BATCHES = [1, 256] if not SMOKE else [1, 8]
cell0 = CELLS[0]["name"]
sample = VA[cell0].sample(min(512, len(VA[cell0])), random_state=0).reset_index(drop=True)
ds = pi.dataset(PREPARED, sample, TEACHER, SCALES, SETTINGS, MC, LRV["valid"], NUM["valid"])
X = torch.stack([ds[i][0] for i in range(len(sample))])
Tb = torch.zeros(len(sample), 0)
SPEED = []
models = {"teacher": (pi.PureCNN(SETTINGS, TEACHER), STEP6D_DIR / cell0 / TEACHER / "best.pt")}
models |= {f"stu_{s}": (pd_.build(s, SETTINGS), ckpt(cell0, s)) for s in STUDENTS}
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
        a.annotate(pd_.pretty(name), (x, r.auc_5m_2023), fontsize=7, xytext=(4, 2), textcoords="offset points")
if "hgb_tab43" in auc0:
    for a in ax:
        a.axhline(auc0["hgb_tab43"], ls="--", lw=0.8, color="grey"); a.text(a.get_xlim()[0], auc0["hgb_tab43"], " HGB 43", fontsize=7, va="bottom")
ax[0].set_xscale("log"); ax[0].set_xlabel("parameters"); ax[1].set_xlabel(f"GPU ms per batch of {max(BATCHES)}")
ax[1].set_xscale("log")
for a in ax:
    a.set_ylabel(f"AUC, {cell0}, 2023")
fig.suptitle("Pure-image models: size / speed vs AUC"); fig.tight_layout()
fig.savefig(RUN_DRIVE / "frontier.png", dpi=130); plt.show()
"""), md("""
## 7. Takeaways
"""), code("""
lines = [f"Teacher `{TEACHER}` (Step 6d run {STEP6D_DIR.name}); alpha {SETTINGS['kd_alpha']}, T {SETTINGS['kd_temperature']}.", ""]
for name in RESULTS:
    t = TABLE[TABLE.cell == name].set_index("raw")
    for k in SHOW + [f"stu_{s}" for s in STUDENTS]:
        if k in t.index:
            n = PARAMS.total.get(k)
            lines.append(f"- {name} {pd_.pretty(k)}: AUC {t.loc[k, 'roc_auc']:.4f} [{t.loc[k, 'auc_ci_low']:.4f}, "
                         f"{t.loc[k, 'auc_ci_high']:.4f}]" + (f", {int(n):,} parameters" if n == n and n is not None else ""))
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
for name, r in FRONTIER.iterrows():
    if name != "teacher" and r.ms_per_batch == r.ms_per_batch:
        lines.append(f"- {pd_.pretty(name)}: {r.parameters_vs_teacher:.3f} of the teacher's parameters, "
                     f"{r.speedup_vs_teacher:.1f}x faster per batch of {max(BATCHES)} on the GPU")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **`*_kd` better than `*_hard`**: the teacher's probabilities carry something the 0/1 labels do not.
- **Distilled student tied with the teacher / HGB 43**: a small model that only looks at the picture keeps the accuracy.
- **Half-height tied with full**: half of the convolution work is not needed.

## 8. Held-out test 2024–2025 (run once, at the end)
Teacher and students from their 2023 checkpoints on Step 6b's test windows; HGB's and the final model's test scores
come from the Step 6b test section. Set `RUN_TEST = True` once.
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
        out["score_teacher"] = pi.score(TEACHER, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"],
                                        STEP6D_DIR / name / TEACHER / "best.pt", DEVICE)
        for s in STUDENTS:
            out[f"score_stu_{s}"] = pd_.score(s, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"], ckpt(name, s), DEVICE)
        out = pi.join_6b(out, STEP6B_DIR / name / "test_predictions.parquet", FINAL)
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(predictions=out)
        print(f"{name}: {len(out):,} test rows on {out.decision_date.nunique()} days")
    TEST_TABLE, TEST_CHECK = compare(TEST, 50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv", index=False); TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False)
    display(TEST_TABLE.pivot_table(index="model", columns="cell", values="roc_auc").round(4)); display(TEST_CHECK.round(4))
""")]
    return c
