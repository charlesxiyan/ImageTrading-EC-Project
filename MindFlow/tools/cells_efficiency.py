"""Cells of MindFlow Step 7a (efficient computing)."""
from cells import code, header, md, setup


def efficiency_cells():
    c = [header("7a", "Efficient computing", "Once the model is settled: how much smaller and faster does it get "
                "(16-bit on the GPU, INT8 on the CPU) and at what cost in AUC? And can a large pretrained ViT be "
                "adapted cheaply with LoRA instead of training a ViT from scratch?", """
**Run this after the final model is chosen**: set `SOURCE` in section 2 to that model's run and name. Any network of a
Step 3a grid run (`plain_spy`, `vit_spy`, `plain_spy_vixy`, …) or of a Step 4a indicator run (`ewsd`, `ma_ewsd_tab`, …)
can be loaded. Everything is scored on the source run's own validation rows (2023); the test years are not used.

| part | what is measured | how |
|---|---|---|
| inference | AUC, model size, latency (batch 1 and large batches), GPU memory | the trained network in float32 and bfloat16 on the GPU, float32 and **INT8** on the CPU |
| INT8 | post-training **static** quantization for CNNs (weights and activations int8, BatchNorm folded, ranges calibrated on 2,048 *training* windows); **dynamic** quantization for transformers | PyTorch FX graph mode, x86 backend |
| training | trainable parameters, GPU memory and time per step, AUC | the final CNN, the scratch T2T-ViT, and DINOv2 ViT-S/14 (22M parameters) with **LoRA** (rank 8 on the attention query/value projections, ~0.15M trainable), full fine-tuning and a frozen backbone |

The LoRA ViT gets exactly the CNN's inputs (the same picture, plus the 38 hand/SPY numbers if the final model uses them)
and the same training rows. After training the adapters are merged into the weights, so inference costs the same as
the plain ViT. AUC differences are paired over the same rows and resampled by whole days.
""")]
    c += setup("""
import chartcnn
import efficiency as ef
print("efficiency", ef.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
**`SOURCE`**: the model to study. `step` is `"3a"` (grid) or `"4a"` (indicator); `run_id=None` takes the newest run of
that step; `cell` is `vol_h1` (5 minutes) or `vol_h12` (60 minutes); `model` is the network name in that run.
**To resume after a disconnect**, set `RUN_ID` to the printed id and Run all: saved scores and the LoRA checkpoint are reused.

**Time (estimate on an L4, not measured):** inference part 15–25 min; LoRA fine-tuning ~20 min per epoch on 300,000
windows (up to 3 epochs) plus ~10 min of scoring; training-cost table 5 min. About 1.5–2 hours in all.
"""), code("""
SMOKE = LOCAL_TEST                                # tiny code check on synthetic data; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4)"
RUN_ID = None                                     # None = new run; or an old RUN_ID to resume it
SOURCE = dict(step="3a", run_id=None, cell="vol_h1", model="plain_spy")
# after the final choice, e.g. SOURCE = dict(step="4a", run_id=None, cell="vol_h1", model="ma_ewsd_tab")
WORKERS = 0 if SMOKE else min(12, os.cpu_count() or 2)
EVAL = dict(gpu_every_kth_day=1,                  # GPU variants: every validation day (all rows)
            cpu_every_kth_day=8, cpu_max_rows=30_000,   # CPU variants: a subset of whole days
            vit_cpu_max_rows=4_000,               # the 22M-parameter ViT on the CPU is slow: fewer rows
            calibration_windows=2_048, calib_batch=128, eval_batch=512, n_boot=300, seed=69)
BENCH = dict(gpu_batches=[1, 128, 1024], cpu_batches=[1, 64], reps=20)
RUN_LORA = True
LORA = dict(hub="facebook/dinov2-small", r=8, alpha=16, targets=["query", "value", "q_proj", "v_proj"], lora_dropout=0.05,
            lr=3e-4, weight_decay=0.01, warmup_steps=300, grad_clip=1.0, batch_size=128, eval_batch_size=512,
            epochs=3, patience=1, min_delta=1e-5, max_train=300_000, early_stop_every_kth_day=4,
            num_workers=WORKERS, seed=69)
TRAIN_COST = dict(batch=128, steps=20)
if SMOKE:
    EVAL.update(cpu_every_kth_day=1, cpu_max_rows=200, vit_cpu_max_rows=50, calibration_windows=32, calib_batch=16,
                eval_batch=32, n_boot=30)
    BENCH.update(gpu_batches=[1, 8], cpu_batches=[1, 8], reps=3)
    LORA.update(batch_size=16, eval_batch_size=32, epochs=1, max_train=128, warmup_steps=2, early_stop_every_kth_day=1)
    TRAIN_COST.update(batch=8, steps=2)
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "7a", RUN_ID, "efficiency", LOCAL_TEST)
RUN_LOCAL = LOCAL_ROOT / "work" / RUN_DIR.parent.name / RUN_ID
RUN_LOCAL.mkdir(parents=True, exist_ok=True)
mf.save_settings(RUN_DIR, RESUME, source=SOURCE, eval=EVAL, bench=BENCH, lora=LORA, run_lora=RUN_LORA,
                 train_cost=TRAIN_COST, efficiency=ef.__version__, torch=torch.__version__,
                 gpu=torch.cuda.get_device_name() if DEVICE.type == "cuda" else "cpu", cpu_threads=torch.get_num_threads())
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
print("CPU threads:", torch.get_num_threads(), "| torch", torch.__version__)
"""), md("""
## 3. The model
Loaded from its best validation checkpoint, with the source run's own rows (checked row by row). The first check is
that this notebook reproduces the source run's saved validation scores.
"""), code("""
SRC_DIR = mf.latest_run(HERE, SOURCE["step"], SOURCE["run_id"])
assert SRC_DIR is not None, f"no finished Step {SOURCE['step']} run"
store = chartcnn.Store(PREPARED)
if SOURCE["step"] in ("2a", "3a"):
    SRC = ef.load_grid_source(PREPARED, SRC_DIR, SOURCE["cell"], SOURCE["model"], smoke=SMOKE)
else:
    import indicator_cnn as ic
    SRC = ef.load_indicator_source(PREPARED, store, SRC_DIR, SOURCE["cell"], SOURCE["model"])
TOTAL, _ = ef.count(SRC.model)
print(f"source: Step {SOURCE['step']} run {SRC_DIR.name}, cell {SOURCE['cell']}, model {SRC.name} = {SRC.label}")
print(f"input {(1, *SRC.input_hw)}" + (f" + {SRC.n_tab} hand/SPY numbers" if SRC.n_tab else "") +
      f" | {TOTAL:,} parameters | float32 size {ef.model_bytes(SRC.model) / 2**20:.2f} MB")
print(f"validation rows {len(SRC.valid):,} on {SRC.valid.decision_date.nunique()} days | training rows {len(SRC.train):,}")
V = SRC.valid.iloc[ef.every_kth_day(SRC.valid, EVAL["gpu_every_kth_day"])].reset_index(drop=True)
V_POS = ef.every_kth_day(SRC.valid, EVAL["gpu_every_kth_day"])
CPU_POS = ef.every_kth_day(V, EVAL["cpu_every_kth_day"], EVAL["cpu_max_rows"], EVAL["seed"])
print(f"scored on {len(V):,} rows (GPU variants) and {len(CPU_POS):,} rows on {V.iloc[CPU_POS].decision_date.nunique()} days (CPU variants)")
"""), md("""
## 4. Inference: 16-bit and INT8
Each variant scores the same rows. Scores are saved, so a rerun skips finished variants.
"""), code("""
SCORES_DIR = RUN_DIR / "scores"; SCORES_DIR.mkdir(exist_ok=True)
CALIB = ef.calibration_batches(SRC, EVAL["calibration_windows"], EVAL["calib_batch"], EVAL["seed"])
print(f"calibration: {sum(len(x) for x, _ in CALIB):,} training windows")
t0 = time.perf_counter()
VARIANTS = ef.variants(SRC.model, CALIB, DEVICE)
print(f"variants built in {time.perf_counter() - t0:.0f} s:", ", ".join(VARIANTS))


def scored(tag, model, device, precision, positions):
    path = SCORES_DIR / f"{tag}.npy"
    if path.exists():
        return np.load(path)
    out = np.full(len(V), np.nan)
    t0 = time.perf_counter()
    out[positions] = ef.score(model, SRC.dataset(V.iloc[positions].reset_index(drop=True), "valid"), device, precision,
                              EVAL["eval_batch"], WORKERS)
    np.save(path, out)
    print(f"  {tag}: {len(positions):,} rows in {(time.perf_counter() - t0) / 60:.1f} min")
    return out


ALL_POS = np.arange(len(V))
CNN_SCORES = {}
for name, (m, dev, prec, desc) in VARIANTS.items():
    CNN_SCORES[name] = scored(f"model_{name}", m, dev, prec, ALL_POS if dev.type == "cuda" or DEVICE.type == "cpu" else CPU_POS)
saved = SRC.saved[V_POS]
gpu16 = next((n for n in CNN_SCORES if n.endswith("GPU") and not n.startswith("fp32")), None)
for n in [x for x in ("fp32 GPU", gpu16, "fp32 CPU") if x in CNN_SCORES and not np.isnan(CNN_SCORES[x]).any()]:
    print(f"reproduction: corr({n}, saved scores of the source run) = {np.corrcoef(CNN_SCORES[n], saved)[0, 1]:.6f}, "
          f"max |diff| {np.abs(CNN_SCORES[n] - saved).max():.2e}")
"""), code("""
REF = "fp32 GPU" if "fp32 GPU" in CNN_SCORES else "fp32 CPU"
show_refs = {f"saved: {k}": SRC.references[k][V_POS] for k in ("hgb_hand_spy", "hgb_hand_features") if k in SRC.references}
GPU_CMP = ef.compare(V, {n: s for n, s in CNN_SCORES.items() if not np.isnan(s).any()} | show_refs, REF, EVAL["n_boot"], EVAL["seed"])
print("all rows:"); display(GPU_CMP.round(4))
CPU_CMP = ef.compare(V.iloc[CPU_POS].reset_index(drop=True), {n: s[CPU_POS] for n, s in CNN_SCORES.items()}, REF,
                     EVAL["n_boot"], EVAL["seed"])
print("CPU rows (whole days):"); display(CPU_CMP.round(4))
GPU_CMP.to_csv(RUN_DIR / "inference_auc_all_rows.csv", index=False); CPU_CMP.to_csv(RUN_DIR / "inference_auc_cpu_rows.csv", index=False)
"""), md("""
### Size and speed
A fixed batch of real charts (drawing excluded). GPU numbers include the weights in the memory figure; CPU uses every
core of the runtime (printed above). Batch 1 is the latency of one decision; large batches give throughput.
"""), code("""
bmax = max(BENCH["gpu_batches"] + BENCH["cpu_batches"])
X, T = [], []
for x, t, _, _ in ef.loader(SRC.dataset(V.iloc[:bmax].reset_index(drop=True), "valid"), bmax, WORKERS):
    X.append(x); T.append(t)
X, T = torch.cat(X).float(), torch.cat(T).float()


def bench_rows(model_name, variants):
    rows = []
    for name, (m, dev, prec, desc) in variants.items():
        batches = BENCH["gpu_batches"] if dev.type == "cuda" else BENCH["cpu_batches"]
        for r in ef.benchmark(name, m, dev, prec, X, T, batches, BENCH["reps"]):
            rows.append(dict(model=model_name, size_mb=ef.model_bytes(m) / 2**20, how=desc, **r))
    return rows


BENCH_ROWS = bench_rows(SRC.label, VARIANTS)
display(pd.DataFrame(BENCH_ROWS).round(3))
"""), md("""
## 5. Training: DINOv2 ViT-S/14 with LoRA
Same training rows as the source model (at most `LORA["max_train"]`), same inputs, early stopping on every 4th validation
day as in Steps 3–4. Only the adapters and the head are trained and saved (`lora/best.pt` holds just those weights).
"""), code("""
LORA_SCORES, LORA_ROWS, LORA_HISTORY, LORA_INFO = {}, [], None, {}
if RUN_LORA:
    torch.manual_seed(LORA["seed"])
    VIT = ef.vit_model("lora", SRC.input_hw, SRC.n_tab, hub=LORA["hub"], r=LORA["r"], alpha=LORA["alpha"],
                       targets=LORA["targets"], lora_dropout=LORA["lora_dropout"], tiny=SMOKE, seed=LORA["seed"]).to(DEVICE)
    total, trainable = ef.count(VIT)
    print(f"DINOv2 ViT-S/14 + LoRA (r={LORA['r']}): {total:,} parameters, {trainable:,} trainable ({trainable / total:.2%}); "
          f"chart resized to {VIT.features.size}")
    LORA_DIR, LORA_LOCAL = RUN_DIR / "lora", RUN_LOCAL / "lora"
    if (LORA_DIR / "valid_prob.npy").exists():
        VIT.load_state_dict(torch.load(LORA_DIR / "best.pt", map_location="cpu", weights_only=False)["trainable"], strict=False)
        LORA_HISTORY = pd.read_csv(LORA_DIR / "history.csv"); LORA_INFO = json.loads((LORA_DIR / "lora.json").read_text())
        p = np.load(LORA_DIR / "valid_prob.npy")
        print("already trained: loaded")
    else:
        tr = SRC.train.sample(min(LORA["max_train"], len(SRC.train)), random_state=LORA["seed"]).reset_index(drop=True)
        es_days = np.sort(V.decision_date.unique())[::LORA["early_stop_every_kth_day"]]
        es = V[V.decision_date.isin(es_days)].reset_index(drop=True)
        print(f"training rows {len(tr):,} | early-stopping rows {len(es):,} on {len(es_days)} days")
        t0 = time.perf_counter()
        LORA_HISTORY, best_epoch = ef.fit_vit(VIT, SRC.dataset(tr, "train"), SRC.dataset(es, "valid"), es.label.to_numpy(),
                                              LORA, DEVICE, LORA_LOCAL, LORA_DIR)
        minutes = (time.perf_counter() - t0) / 60
        p = ef.score(VIT, SRC.dataset(V, "valid"), DEVICE, "bf16", LORA["eval_batch_size"], WORKERS)
        LORA_INFO = dict(parameters=total, trainable=trainable, best_epoch=best_epoch, minutes=minutes, train_rows=len(tr),
                         adapter_mb=sum(v.numel() * v.element_size() for v in ef.trainable_state(VIT).values()) / 2**20)
        np.save(LORA_LOCAL / "valid_prob.npy", p); (LORA_LOCAL / "lora.json").write_text(json.dumps(LORA_INFO, indent=2))
        for f in ("valid_prob.npy", "lora.json"):
            ef._mirror(LORA_LOCAL / f, LORA_DIR)
    display(LORA_HISTORY.round(4)); print(LORA_INFO)
    LORA_SCORES["ViT-S LoRA, bf16 GPU"] = p
    MERGED = ef.merge_lora(VIT)
    LORA_VARIANTS = {k: v for k, v in ef.variants(MERGED, None, DEVICE, kind="dynamic").items()}
    vit_cpu = ef.every_kth_day(V, EVAL["cpu_every_kth_day"], EVAL["vit_cpu_max_rows"], EVAL["seed"])
    for name, (m, dev, prec, desc) in LORA_VARIANTS.items():
        if dev.type == "cpu":
            LORA_SCORES[f"ViT-S LoRA, {name}"] = scored(f"lora_{name}", m, dev, prec, vit_cpu)
    LORA_ROWS = bench_rows("DINOv2 ViT-S/14 + LoRA (merged)", LORA_VARIANTS)
    display(pd.DataFrame(LORA_ROWS).round(3))
"""), code("""
if RUN_LORA:
    cands = {f"{SRC.label} ({REF})": CNN_SCORES[REF], "ViT-S LoRA, bf16 GPU": LORA_SCORES["ViT-S LoRA, bf16 GPU"]}
    for k, label in (("cnn_vit_spy", "saved: scratch ViT + SPY"), ("cnn_vit", "saved: scratch ViT"),
                     ("cnn_plain_spy", "saved: CNN + SPY"), ("hgb_hand_spy", "saved: HGB hand + SPY")):
        if k in SRC.references and k != f"cnn_{SRC.name}":
            cands[label] = SRC.references[k][V_POS]
    LORA_CMP = ef.compare(V, cands, f"{SRC.label} ({REF})", EVAL["n_boot"], EVAL["seed"])
    print("LoRA ViT vs the source model, all rows:"); display(LORA_CMP.round(4))
    LORA_CMP.to_csv(RUN_DIR / "lora_auc.csv", index=False)
    q = [k for k in LORA_SCORES if "INT8" in k]
    if q:
        pos = np.flatnonzero(~np.isnan(LORA_SCORES[q[0]]))
        LORA_Q = ef.compare(V.iloc[pos].reset_index(drop=True), {k: v[pos] for k, v in LORA_SCORES.items()},
                            "ViT-S LoRA, bf16 GPU", EVAL["n_boot"], EVAL["seed"])
        print(f"LoRA ViT, INT8 on the CPU vs bf16 on the GPU ({len(pos):,} rows):"); display(LORA_Q.round(4))
"""), md("""
## 6. Training cost per step
One fixed batch of real charts, AdamW, bfloat16 autocast, `TRAIN_COST["batch"]` images per step (drawing excluded).
Every model gets the same inputs as the final model. "frozen" trains only the head on top of DINOv2 (like Step 3b's
probes, but end to end).
"""), code("""
xb, tb, yb = [], [], []
for x, t, y, _ in ef.loader(SRC.dataset(SRC.train.sample(TRAIN_COST["batch"], random_state=0).reset_index(drop=True), "train"),
                            TRAIN_COST["batch"], WORKERS):
    xb.append(x); tb.append(t); yb.append(y)
BATCH = (torch.cat(xb).float(), torch.cat(tb).float(), torch.cat(yb))
from chartcnn.vit import ChartViT


def scratch_vit():
    v = ChartViT(input_hw=SRC.input_hw, arch="vit")
    tab = None
    if SRC.n_tab:
        tab = torch.nn.Sequential(torch.nn.Linear(SRC.n_tab, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(),
                                  torch.nn.Linear(64, 2))
    return ef.Net(v.features, v.head, tab, fp32_head=True)


kw = dict(hub=LORA["hub"], r=LORA["r"], alpha=LORA["alpha"], targets=LORA["targets"], tiny=SMOKE)
COST_MODELS = {f"{SRC.label} (from scratch)": SRC.fresh, "T2T-ViT (from scratch)": scratch_vit,
               "DINOv2 ViT-S + LoRA": lambda: ef.vit_model("lora", SRC.input_hw, SRC.n_tab, **kw),
               "DINOv2 ViT-S, full fine-tune": lambda: ef.vit_model("full", SRC.input_hw, SRC.n_tab, **kw),
               "DINOv2 ViT-S, frozen (head only)": lambda: ef.vit_model("frozen", SRC.input_hw, SRC.n_tab, **kw)}
COST = []
for name, make in COST_MODELS.items():
    try:
        COST.append(ef.train_step_cost(name, make(), BATCH, DEVICE, steps=TRAIN_COST["steps"]))
    except torch.cuda.OutOfMemoryError:
        COST.append(dict(model=name, note="out of GPU memory at this batch"))
        torch.cuda.empty_cache()
COST = pd.DataFrame(COST)
COST.to_csv(RUN_DIR / "training_cost.csv", index=False); display(COST.round(3))
"""), md("""
## 7. Summary
One row per deployable form of each model: validation AUC (on the rows it was scored on, with the paired difference to
the float32 model), size on disk and speed.
"""), code("""
bench = pd.DataFrame(BENCH_ROWS + LORA_ROWS)
first = lambda d, b: d[d.batch == b].set_index(["model", "variant"])
b1 = first(bench, 1)
big = bench.sort_values("batch").groupby(["model", "variant"]).tail(1).set_index(["model", "variant"])
rows = []
G, C = GPU_CMP.set_index("model"), CPU_CMP.set_index("model")
for name, (m, dev, prec, desc) in VARIANTS.items():
    r = (G if dev.type == "cuda" else C).loc[name]
    rows.append(dict(model=SRC.label, variant=name, auc=r.auc, diff_vs_fp32=r.diff_vs_ref, diff_low=r.diff_low,
                     diff_high=r.diff_high, auc_rows=r.rows))
if RUN_LORA:                                      # LoRA: GPU row vs nothing (see section 5), CPU rows vs its bf16 GPU scores
    LG = LORA_CMP.set_index("model")
    LQ = LORA_Q.set_index("model") if "LORA_Q" in globals() else None
    for v in LORA_VARIANTS:
        key = f"ViT-S LoRA, {v}"
        r = LG.loc[key] if key in LG.index else (LQ.loc[key] if LQ is not None and key in LQ.index else None)
        rows.append(dict(model="DINOv2 ViT-S/14 + LoRA (merged)", variant=v, auc=np.nan if r is None else r.auc,
                         diff_vs_fp32=np.nan if r is None or key in LG.index else r.diff_vs_ref,
                         diff_low=np.nan if r is None or key in LG.index else r.diff_low,
                         diff_high=np.nan if r is None or key in LG.index else r.diff_high,
                         auc_rows=np.nan if r is None else r.rows))
SUMMARY = pd.DataFrame(rows).set_index(["model", "variant"])
SUMMARY["size_mb"] = b1.size_mb
SUMMARY["ms_batch1"] = b1.ms_per_batch
SUMMARY["images_per_s_large_batch"] = big.images_per_s
SUMMARY["gpu_memory_mb"] = big.peak_memory_mb
SUMMARY = SUMMARY.reset_index()
SUMMARY.to_csv(RUN_DIR / "summary.csv", index=False)
display(SUMMARY.round(4))
"""), code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for (model, d), mk in zip(SUMMARY.groupby("model"), "os^D"):
    ax[0].scatter(d.ms_batch1, d.auc, marker=mk, s=60, label=model)
    for _, r in d.iterrows():
        ax[0].annotate(r.variant, (r.ms_batch1, r.auc), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].barh(f"{model[:28]} | {r.variant}", r.size_mb, color="tab:blue" if "INT8" not in r.variant else "tab:orange")
ax[0].set_xscale("log"); ax[0].set_xlabel("latency of one chart (ms, log scale)"); ax[0].set_ylabel("validation AUC")
ax[0].legend(fontsize=7); ax[0].set_title("AUC vs latency (batch 1)")
ax[1].set_xlabel("size on disk (MB)"); ax[1].set_title("model size"); ax[1].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(RUN_DIR / "efficiency.png", dpi=130); plt.show()
"""), code("""
lines = [f"# Step 7a: efficient computing ({SRC.label}, {SOURCE['cell']}, Step {SOURCE['step']} run {SRC_DIR.name})", ""]
for _, r in SUMMARY.iterrows():
    d = "" if pd.isna(r.diff_vs_fp32) else f", vs float32 {r.diff_vs_fp32:+.4f} [{r.diff_low:+.4f}, {r.diff_high:+.4f}]"
    lines.append(f"- {r.model} / {r.variant}: AUC {r.auc:.4f}{d}; {r.size_mb:.2f} MB; {r.ms_batch1:.2f} ms per chart; "
                 f"{r.images_per_s_large_batch:,.0f} charts/s at the largest batch")
if RUN_LORA:
    lines.append(f"- LoRA: {LORA_INFO['trainable']:,} of {LORA_INFO['parameters']:,} parameters trained "
                 f"({LORA_INFO['trainable'] / LORA_INFO['parameters']:.2%}); adapter + head {LORA_INFO.get('adapter_mb', float('nan')):.2f} MB; "
                 f"best epoch {LORA_INFO['best_epoch']}; {LORA_INFO['minutes']:.0f} min")
for _, r in COST.iterrows():
    if "ms_per_step" in r and not pd.isna(r.get("ms_per_step")):
        lines.append(f"- training {r.model}: {int(r.trainable):,} trainable, {r.ms_per_step:.0f} ms/step at batch {int(r.batch)}, "
                     f"peak GPU memory {r.peak_train_memory_mb:,.0f} MB")
TAKEAWAYS = "\\n".join(lines)
(RUN_DIR / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **bf16 GPU vs fp32 GPU**: the difference should be ~0 (bf16 is how the models were trained); it is the free speed-up.
- **INT8 CPU vs fp32**: a drop inside ±0.002 AUC with a ~4× smaller file and a faster CPU means the model can run
  without a GPU. A larger drop points at activation ranges (more calibration windows, or keep the first layer in float).
- **LoRA ViT vs CNN**: if it ties the CNN while training ~1% of its parameters, pretrained vision features transfer to
  charts when adapted (frozen probes in Step 3b did not); if it stays below, the CNN's inductive bias (bar columns)
  matters more than pretraining. Either way the cost table shows what each training run needs.
""")]
    return c
