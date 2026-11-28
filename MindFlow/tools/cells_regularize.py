"""Cells of MindFlow Step 6g (regularization ablation on the pure-image CNN)."""
from cells import code, header, md, setup


def regularize_cells():
    c = [header("6g", "Regularization ablation", "The pure-image CNN shows little overfitting. Does stronger "
                "regularization still lower its log loss or raise its AUC?", """
Step 6d's `pure_all` CNN (776,450 parameters, Adam 1e-5, no weight decay, head dropout 0.5, early stopping on
log loss) reaches AUC 0.6894 and log loss 0.6278 on validation 2023, the same as HGB 43. Its validation loss is lowest
at epoch 2–3 and the held-out years are not worse, so there is little overfitting to remove. Distillation, itself a
strong regularizer, did help the small student. Each variant changes one thing in Step 6d's training rule:

| variant | change | expectation |
|---|---|---|
| `ema` | exponential moving average of the weights (0.9995), used for early stopping and scoring | smoother weights, often lower log loss |
| `adamw_1`, `adamw_10` | AdamW with decoupled weight decay 1 / 10 (6d: none) | `10` is the hard one |
| `drop_25`, `drop_75` | head dropout 0.25 / 0.75 (6d: 0.5) | one weaker, one harder |
| `smooth_05` | label smoothing 0.05 | pulls probabilities toward 0.5: likely worse log loss |
| `seed_1`–`seed_4` → `ens5` | Step 6d's rule with four other seeds, averaged with 6d's model | an ensemble is not regularization, but it is the usual reference for lowering log loss |

Same picture, CNN, windows, rows and early-stopping rule as Step 6d. Reported for every variant: AUC and log loss with
paired day-bootstrap checks against 6d's `pure_all` (a result is "tied" only when both are tied), learning curves,
and the **generalization gap**: eval-mode log loss (dropout off) on a fixed 50,000-window training subset vs validation.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import pure_image as pi
import compact_image as ci
import regularize as rz
import matplotlib.pyplot as plt
fm.register()
print("regularize", rz.__version__, "| pure_image", pi.__version__, "| indicator_cnn", ic.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
`STEP6D_RUN = None` takes the newest Step 6d run (which names its Step 6b run). **To resume after a disconnect**, set
`RUN_ID` to the printed id and Run all: finished variants are skipped.

**Time (estimate, not measured):** one variant costs about one Step 6d training: about 15 minutes on an A100 (5 epochs
of about 3 minutes in Step 6f), 1–1.5 hours on an L4. Ten variants: about 2.5–3 hours on an A100. The order puts
the regularizers first and the four seeds last, so the run can be split over sessions.
"""), code("""
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (A100 or L4)"
RUN_ID = None
STEP6D_RUN = None
HORIZONS = [1]
VARIANTS = list(rz.VARIANTS)          # regularizers first, then seed_1..seed_4
if SMOKE:
    VARIANTS = ["ema", "adamw_10", "smooth_05", "seed_1"]
GAP_ROWS = 300 if SMOKE else 50_000   # fixed training subset for the eval-mode train loss
REF = rz.REF
ENS = [REF] + [f"reg_{s}" for s in rz.SEEDS if s in VARIANTS]   # members of the seed ensemble
STEP6D_DIR = mf.latest_run(HERE, "6d", STEP6D_RUN)
assert STEP6D_DIR is not None, "run Step 6d first"
S6D = json.loads((STEP6D_DIR / "settings.json").read_text())
STEP6B_DIR = mf.run_folder(HERE, "6b") / S6D["step6b"]
FINAL = S6D["final"]
SETTINGS = dict(S6D["settings"])                     # Step 6d's training rule; each variant changes one thing
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELLS = [c for c in S6D["cells"] if c["horizon"] in HORIZONS and (STEP6D_DIR / c["name"] / "predictions.parquet").exists()]
assert CELLS, f"no finished Step 6d cell for horizons {HORIZONS}"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "6g", RUN_ID, "regularize", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DRIVE.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, variants={v: rz.VARIANTS[v] for v in VARIANTS}, cells=CELLS,
                 step6d=STEP6D_DIR.name, step6b=STEP6B_DIR.name, regularize=rz.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("Step 6d run:", STEP6D_DIR.name, "| Step 6b run:", STEP6B_DIR.name, "| cells:", [c["name"] for c in CELLS])
display(pd.DataFrame([dict(variant=v, label=rz.LABELS[v], **{k: rz.variant_settings(SETTINGS, v)[k] for k in
                      ("optimizer", "weight_decay", "dropout", "label_smoothing", "ema", "seed")}) for v in VARIANTS]))
"""), md("""
## 3. Step 6d's windows, numbers and strip scales
"""), code("""
store = chartcnn.Store(PREPARED)
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
print("parameters (every variant):", f"{ic.count(rz.build(SETTINGS))['total']:,}")
"""), md("""
## 4. Train (validation 2023)
Each variant is saved as it finishes. Step 6d's own `pure_all` scores are joined on the same rows, and the 5-seed
ensemble `ens5` is the mean probability of 6d's model and `seed_1`–`seed_4`.
"""), code("""
RESULTS, VA, TR, INFO, t0 = {}, {}, {}, {}, time.perf_counter()
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
    VA[name], TR[name] = va, tr
    data = dict(train=tr, es=es, valid=va, scales=SCALES, market_close=MC, lrv=LRV, numbers=NUM)
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    pred = va[keep].copy()
    info = dict(cell=cell, prior=float(tr.label.mean()), parameters={}, best_epoch={}, epochs={}, minutes_trained={},
                seconds_per_epoch={})
    for v in VARIANTS:
        p, vi = rz.run_variant(v, PREPARED, data, SETTINGS, DEVICE, RUN_LOCAL / name, RUN_DRIVE / name)
        pred[f"score_reg_{v}"] = p
        for k, key in (("parameters", "parameters"), ("best_epoch", "best_epoch"), ("epochs", "epochs"),
                       ("minutes_trained", "minutes"), ("seconds_per_epoch", "seconds_per_epoch")):
            info[k][f"reg_{v}"] = vi.get(key)
    ref = pd.read_parquet(STEP6D_DIR / name / "predictions.parquet").rename(columns=REN)
    cols = [c for c in ref.columns if c.startswith("score_")]
    ref = ref[["security_index", "end_index", "label"] + cols].rename(columns={"label": "ref_label"})
    pred = pred.merge(ref, on=["security_index", "end_index"], how="left")
    assert pred.ref_label.notna().all() and (pred.ref_label == pred.label).all(), "Step 6d rows differ"
    pred = pred.drop(columns="ref_label")
    pred["score_reg_ens5"] = rz.ensemble(pred, ENS)
    pred.to_parquet(RUN_DRIVE / name / "predictions.parquet", index=False)
    (RUN_DRIVE / name / "done.json").write_text(json.dumps(info, default=float))
    RESULTS[name], INFO[name] = dict(info=info, predictions=pred), info
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
"""), md("""
## 5. Compare on identical rows: AUC and log loss
AUC with 95% intervals (whole days resampled), log loss, and paired checks on both metrics ("better"/"worse" when the
95% interval of the difference excludes zero; for log loss, lower is better).
"""), code("""
SHOW = ["pure_all", "final", "hgb_tab43"]
def compare(results, n_boot):
    tables, boots, ll_boots = [], {}, {}
    for name, r in results.items():
        pred = r["predictions"]
        names = [c[6:] for c in pred.columns if c.startswith("score_") and (c[6:] in SHOW or c.startswith("score_reg_"))]
        boots[name] = ic.day_bootstrap(pred, names, n_boot)
        ll_boots[name] = ci.ll_day_bootstrap(pred, names, n_boot)
        tables.append(rz.auc_table(pred, names, boots[name]).assign(cell=name))
    return pd.concat(tables), rz.checks_table(boots, ll_boots)

TABLE, CHECK = compare(RESULTS, 50 if SMOKE else 300)
TABLE.to_csv(RUN_DRIVE / "regularize_table.csv", index=False)
CHECK.to_csv(RUN_DRIVE / "checks.csv", index=False)
display(TABLE.set_index("model")[["roc_auc", "auc_ci_low", "auc_ci_high", "log_loss", "cell"]].sort_values("log_loss").round(4))
display(CHECK.round(4))
"""), md("""
## 6. Generalization gap: eval-mode train loss vs validation
The training loss in `history.csv` is a running mean with dropout on, so it is not comparable with validation. Here
every model is scored in eval mode (dropout off, BatchNorm frozen) on the same fixed training subset. The gap is
validation log loss minus training log loss; a regularizer that works should shrink it without raising validation
log loss.
"""), code("""
cell0 = CELLS[0]["name"]
sub = TR[cell0].sample(min(GAP_ROWS, len(TR[cell0])), random_state=0).reset_index(drop=True)
y_sub, y_va = sub.label.to_numpy(), RESULTS[cell0]["predictions"].label.to_numpy()
pv = RESULTS[cell0]["predictions"]
GAP = [rz.gap_row(REF, y_sub, rz.score(None, PREPARED, sub, SETTINGS, SCALES, MC, LRV["train"], NUM["train"],
                                       STEP6D_DIR / cell0 / REF / "best.pt", DEVICE), y_va, pv[f"score_{REF}"].to_numpy())]
TRAIN_P = {REF: None}
for v in VARIANTS:
    p_tr = rz.score(v, PREPARED, sub, SETTINGS, SCALES, MC, LRV["train"], NUM["train"], ckpt(cell0, v), DEVICE)
    TRAIN_P[v] = p_tr
    GAP.append(rz.gap_row(f"reg_{v}", y_sub, p_tr, y_va, pv[f"score_reg_{v}"].to_numpy()))
GAP = pd.DataFrame(GAP).set_index("model")
GAP.to_csv(RUN_DRIVE / "generalization_gap.csv"); display(GAP.round(4))
"""), md("""
## 7. Learning curves
Training loss (dashed; running mean with dropout on, and with smoothing for `smooth_05`) and log loss on the
early-stopping days (solid) per epoch; the star marks the epoch that was kept. Seeds are left out for readability.
"""), code("""
fig, ax = plt.subplots(figsize=(9, 5))
curves = {REF: STEP6D_DIR / cell0 / REF / "history.csv"}
curves |= {v: next((p for p in (RUN_DRIVE / cell0 / v / "history.csv", RUN_LOCAL / cell0 / v / "history.csv") if p.exists()), None)
           for v in VARIANTS if v not in rz.SEEDS}
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
## 8. Takeaways
"""), code("""
t0_ = TABLE[TABLE.cell == cell0].set_index("raw")
SUMMARY = t0_[["model", "roc_auc", "log_loss"]].join(GAP[["train_log_loss", "gap"]], how="left")
SUMMARY["best_epoch"] = pd.Series(INFO[cell0]["best_epoch"]); SUMMARY["min_per_epoch"] = pd.Series(INFO[cell0]["seconds_per_epoch"]) / 60
SUMMARY.to_csv(RUN_DRIVE / "summary.csv"); display(SUMMARY.sort_values("log_loss").round(4))
lines = [f"Step 6d run {STEP6D_DIR.name} (reference `pure_all`), Step 6b run {STEP6B_DIR.name}.", ""]
for name, r in SUMMARY.sort_values("log_loss").iterrows():
    gap = "" if pd.isna(r.get("gap")) else f", generalization gap {r.gap:+.4f}"
    ep = "" if pd.isna(r.get("best_epoch")) else f", best epoch {int(r.best_epoch)}"
    lines.append(f"- {r.model}: AUC {r.roc_auc:.4f}, log loss {r.log_loss:.4f}{gap}{ep}")
lines.append("")
for _, r in CHECK.iterrows():
    lines.append(f"- {r.cell} {r.check}: AUC {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}; "
                 f"log loss {r.ll_diff:+.4f} [{r.ll_low:+.4f}, {r.ll_high:+.4f}] -> {r.ll_verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
Read AUC and log loss together: a variant only "helps" when log loss is lower (better) and AUC is not worse.
- **Regularizers tied with `pure_all`, gap already small**: there was no overfitting for them to remove; the ceiling
  is the information in the picture.
- **A regularizer better on log loss**: the 6d model's probabilities were slightly overconfident or noisy; keep it.
- **`adamw_10`, `drop_75` or `smooth_05` worse**: too much regularization underfits (or, for smoothing, biases the
  probabilities toward 0.5).
- **`ens5` better while single regularizers tie**: the remaining error is seed noise, which averaging removes and
  regularization does not.

## 9. Held-out test 2024–2025 (run once, at the end)
Set `RUN_TEST = True` once, after looking at 2023 only. Every variant from its 2023 checkpoint on Step 6b's test
windows, with Step 6d's `pure_all` scored on the same rows.
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
        out[f"score_{REF}"] = rz.score(None, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"],
                                        STEP6D_DIR / name / REF / "best.pt", DEVICE)
        for v in VARIANTS:
            out[f"score_reg_{v}"] = rz.score(v, PREPARED, te, SETTINGS, SCALES, MC, LRV["test"], NUM["test"], ckpt(name, v), DEVICE)
        out["score_reg_ens5"] = rz.ensemble(out, ENS)
        out = pi.join_6b(out, STEP6B_DIR / name / "test_predictions.parquet", FINAL)
        out.to_parquet(RUN_DRIVE / name / "test_predictions.parquet", index=False)
        TEST[name] = dict(predictions=out)
        print(f"{name}: {len(out):,} test rows on {out.decision_date.nunique()} days")
    TEST_TABLE, TEST_CHECK = compare(TEST, 50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_DRIVE / "test_table.csv", index=False); TEST_CHECK.to_csv(RUN_DRIVE / "test_checks.csv", index=False)
    display(TEST_TABLE.set_index("model")[["roc_auc", "log_loss", "cell"]].round(4)); display(TEST_CHECK.round(4))
""")]
    return c
