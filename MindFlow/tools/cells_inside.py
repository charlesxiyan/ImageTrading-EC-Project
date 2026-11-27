"""Cells of MindFlow Step 8a (what the pure-image CNN learned, with data)."""
from cells import code, header, md, setup


def inside_cells():
    c = [header("8a", "What did the pure-image CNN learn?", "Step 6d's pure-image CNN ties HGB 43 while seeing only "
                "a picture. Which parts of the picture does it actually use, which moments in time, which of the drawn "
                "numbers, and what do its features encode?", """
The weights alone say little: training moved them only about 5% from their random start (lr 1e-5, best epoch 2–3),
and a weight's size ignores how large the activation it multiplies is. So everything here runs the **trained, fixed**
Step 6d models (`pure_all` and `pure_key`) on real 2023 validation pictures. Nothing is trained.

| analysis | what is changed or measured | answers |
|---|---|---|
| region occlusion | one part of the picture replaced by the average picture there (also: by black) | which parts are used |
| erase | the session-open line; the open ticks / high-low bars / close ticks of the chart | time of day? which price marks? |
| time occlusion | 6-bar blocks of the time-aligned rows (chart, EWSD, SPY, volume bands) | which moments matter |
| number strip | one drawn number set to 0 (its training mean), 43 times (`pure_all`) | which numbers the CNN reads |
| activation × weight | the head's input × its weights, per position and per channel; top channels set to 0 | where the prediction changes |
| saliency | Grad-CAM (last convolution) and gradient × input, averaged; region shares | where the gradient points |
| probing | ridge from pooled CNN features to the 43 numbers and the time of day (folds by day) | what the features encode |

AUC drops are on the same rows with 95% intervals from resampling whole days ("matters" = interval above 0).
Occlusion is a counterfactual: an average-filled strip is a picture the model never saw, so a drop measures reliance,
not a clean causal effect; erase, black fill and the per-number zero give second views.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import regime as rg
import final_model as fm
import pure_image as pi
import inside_cnn as icn
import matplotlib.pyplot as plt
fm.register()
print("inside_cnn", icn.__version__, "| pure_image", pi.__version__, "| indicator_cnn", ic.__version__)
""")
    c += [md("""
## 2. Configuration
`STEP6D_RUN = None` takes the newest Step 6d run. `N_ROWS` validation rows are sampled evenly over the 2023 days
(about 80 per day). Inference only: a T4 is enough.

**Time (estimate):** drawing 20,000 pictures 2–5 min; each model ~80 forward passes over them (A100/L4 ~10 s each,
T4 ~30–60 s each); saliency on `N_SALIENCY` pictures 1–3 min; probing on the CPU ~2 min. About 20–40 min on an L4.
"""), code("""
SMOKE = LOCAL_TEST
RUN_ID = None
STEP6D_RUN = None
MODELS = ["pure_all", "pure_key"]
N_ROWS = 400 if SMOKE else 20_000
N_SALIENCY = 64 if SMOKE else 2_000
N_BOOT = 30 if SMOKE else 300
BATCH = 16 if SMOKE else 256
STEP6D_DIR = mf.latest_run(HERE, "6d", STEP6D_RUN)
assert STEP6D_DIR is not None, "run Step 6d first"
S6D = json.loads((STEP6D_DIR / "settings.json").read_text())
STEP6B_DIR = mf.run_folder(HERE, "6b") / S6D["step6b"]
SETTINGS = dict(S6D["settings"])
SETTINGS.update(num_workers=0 if SMOKE else min(12, os.cpu_count() or 2))
CELL = next(c for c in S6D["cells"] if c["horizon"] == 1)
MODELS = [v for v in MODELS if (STEP6D_DIR / CELL["name"] / v / "best.pt").exists()]
assert MODELS, "no Step 6d checkpoints"
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "8a", RUN_ID, "inside", LOCAL_TEST)
mf.save_settings(RUN_DRIVE, RESUME, settings=SETTINGS, models=MODELS, cell=CELL, step6d=STEP6D_DIR.name,
                 step6b=STEP6B_DIR.name, n_rows=N_ROWS, n_saliency=N_SALIENCY, inside_cnn=icn.__version__)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE, "| Step 6d run:", STEP6D_DIR.name, "| cell:", CELL["name"], "| models:", MODELS)
"""), md("""
## 3. Validation rows, their numbers, and the pictures (drawn once)
"""), code("""
store = chartcnn.Store(PREPARED)
W = {s: pd.read_parquet(STEP6B_DIR / f"windows_{s}.parquet") for s in ("train", "valid")}
PRE = json.loads((STEP6B_DIR / "preprocessing.json").read_text())
SC43 = fm.scaler_from_state(PRE["tab"])
SCALES = {k: tuple(v) for k, v in json.loads((STEP6D_DIR / "strip_scales.json").read_text()).items()}
t0 = time.perf_counter()
RVT = rg.rv_tables(store, SETTINGS["min_bars"])
RAW43 = fm.numbers(store, W["valid"], RVT)[1]
LRV, NUM = pi.log_rv(RAW43), SC43(RAW43)
NAMES = list(RAW43.columns)
_, median = ic.cell_rows(store, CELL, "train", W["train"])
va, _ = ic.cell_rows(store, CELL, "validation", W["valid"], median)
per_day = max(1, N_ROWS // va.decision_date.nunique())
ROWS = (va.sample(frac=1.0, random_state=69).groupby("decision_date").head(per_day)
          .sort_values(["decision_date", "security_index"]).reset_index(drop=True))
Y = ROWS.label.to_numpy()
ds = pi.dataset(PREPARED, ROWS, "pure_all", SCALES, SETTINGS, store.market_close, LRV, NUM)
X = icn.render(ds, SETTINGS["num_workers"], BATCH)
print(f"{len(ROWS):,} rows on {ROWS.decision_date.nunique()} days; pictures {tuple(X.shape)} uint8 "
      f"({X.numel() / 2**30:.2f} GB); {(time.perf_counter() - t0) / 60:.1f} min")
NUMBERS = pd.DataFrame(NUM[ROWS.row.to_numpy()], columns=NAMES)

ref = pd.read_parquet(STEP6D_DIR / CELL["name"] / "predictions.parquet")
REF = ROWS[["security_index", "end_index"]].merge(ref, on=["security_index", "end_index"], how="left")
MODEL, MEAN = {}, {}
for v in MODELS:
    MODEL[v] = icn.load(v, SETTINGS, STEP6D_DIR / CELL["name"] / v / "best.pt", DEVICE)
    MEAN[v] = icn.mean_picture(X, v)
    print(f"{v}: {ic.count(MODEL[v])['total']:,} parameters, picture {pi.height(v)}x{ic.SIZE}")
"""), md("""
### The average picture and its parts
The regions every analysis below refers to (rows of the picture; `pure_key` stops after the RV bars).
"""), code("""
R = icn.regions("pure_all")
fig, ax = plt.subplots(1, 2, figsize=(11, 6.5))
ax[0].imshow(MEAN["pure_all" if "pure_all" in MEAN else MODELS[0]][0, 0].cpu(), cmap="gray", aspect="auto")
ax[0].set_title("average picture (2023 sample)")
ax[1].imshow(X[0].numpy(), cmap="gray", aspect="auto"); ax[1].set_title("one picture")
for a in ax:
    for k, (r0, r1, lab) in R.items():
        a.axhspan(r0, r1, xmin=0.985, xmax=1.0, color="tab:orange", alpha=0.8)
        a.text(238, (r0 + r1) / 2, lab, fontsize=7, va="center")
    a.set_xticks([0, 117, 231]); a.set_xticklabels(["t-77", "t-38", "t"])
fig.tight_layout(); fig.savefig(RUN_DRIVE / "regions.png", dpi=120); plt.show()
"""), md("""
## 4. Occlusion: which parts, which moments, which numbers
"""), code("""
OCC, BASE = {}, {}
for v in MODELS:
    t1 = time.perf_counter()
    ops = icn.ops_for(v)
    scores = icn.run_ops(MODEL[v], X, v, ops, MEAN[v], DEVICE, BATCH, SETTINGS["amp"])
    if f"score_pure_{v}" in REF:
        r = np.corrcoef(scores["base"], REF[f"score_pure_{v}"])[0, 1]
        print(f"{v}: our scores vs Step 6d's saved scores on the same rows: correlation {r:.5f}")
    BASE[v], OCC[v] = icn.drop_table(ROWS, scores, ops, N_BOOT)
    OCC[v].assign(model=v).to_csv(RUN_DRIVE / f"occlusion_{v}.csv", index=False)
    print(f"{v}: AUC on these rows {BASE[v]:.4f}; {len(ops)} occlusions in {(time.perf_counter() - t1) / 60:.1f} min")

def show(group_prefix):
    t = pd.concat([OCC[v].assign(model=v) for v in MODELS])
    t = t[t.group.str.startswith(group_prefix)]
    return t.pivot_table(index="part", columns="model", values="drop").round(4).sort_values(MODELS[0], ascending=False)
display(show("region")); display(show("erase"))
for v in MODELS:
    display(OCC[v][OCC[v].group.str.startswith(("region", "erase"))][["part", "group", "auc", "drop", "low", "high", "verdict"]].round(4))
"""), code("""
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.2, 1, 1.4]))
for j, v in enumerate(MODELS):
    t = OCC[v][OCC[v].group.isin(["region (mean fill)", "erase"])].sort_values("drop")
    y = np.arange(len(t)) + (j - 0.5) * 0.38
    axes[0].barh(y, t["drop"], 0.36, xerr=[t["drop"] - t.low, t.high - t["drop"]], label=v, alpha=0.85)
    if j == 0:
        axes[0].set_yticks(np.arange(len(t))); axes[0].set_yticklabels(t.part, fontsize=8)
axes[0].axvline(0, color="k", lw=0.6); axes[0].set_xlabel("AUC drop"); axes[0].set_title("region (mean fill) and erase"); axes[0].legend()
for v in MODELS:
    t = OCC[v][OCC[v].group.str.startswith("time")]
    x = np.arange(len(t))
    axes[1].errorbar(x, t["drop"], yerr=[t["drop"] - t.low, t.high - t["drop"]], marker="o", capsize=2, label=v)
    axes[1].set_xticks(x); axes[1].set_xticklabels([p.split("..")[0].replace("bars ", "") for p in t.part], rotation=60, fontsize=7)
axes[1].axhline(0, color="k", lw=0.6); axes[1].set_title("6-bar blocks (left = oldest)"); axes[1].set_ylabel("AUC drop"); axes[1].legend()
if "pure_all" in OCC:
    t = OCC["pure_all"][OCC["pure_all"].group.str.startswith("number")].copy()
    t["name"] = [NAMES[int(o.split(":")[1])] for o in t.op]
    t = t.sort_values("drop", ascending=False).head(20)[::-1]
    axes[2].barh(t.name, t["drop"], xerr=[t["drop"] - t.low, t.high - t["drop"]], color="tab:green")
    axes[2].axvline(0, color="k", lw=0.6); axes[2].set_title("number strip: top 20 (pure_all)"); axes[2].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(RUN_DRIVE / "occlusion.png", dpi=130); plt.show()
if "pure_all" in OCC:
    NUMBER_DROP = OCC["pure_all"][OCC["pure_all"].group.str.startswith("number")].assign(
        name=lambda d: [NAMES[int(o.split(":")[1])] for o in d.op]).sort_values("drop", ascending=False)
    NUMBER_DROP.to_csv(RUN_DRIVE / "number_strip_drop.csv", index=False)
    display(NUMBER_DROP[["name", "auc", "drop", "low", "high", "verdict"]].round(4))
"""), md("""
## 5. Activation × weight: where the prediction changes, and which channels carry it
The logit difference (high − low volatility) is Σ features × head weights + bias. Its part at each of the 4 × 78
head positions and from each of the 256 channels varies from picture to picture; the **standard deviation over
pictures** shows what moves the prediction (a part that is the same on every picture only acts as a bias).
Then the channels that move it most are set to zero, against the same number of random channels.
"""), code("""
CON, ABL = {}, {}
fig, axes = plt.subplots(2, len(MODELS), figsize=(8 * len(MODELS), 6.5), squeeze=False)
bins = ["rows 1/4 (chart top)", "rows 2/4", "rows 3/4", "rows 4/4 (bottom strips)"]
for j, v in enumerate(MODELS):
    CON[v] = c = icn.contributions(MODEL[v], X, v, DEVICE, BATCH, SETTINGS["amp"])
    im = axes[0, j].imshow(c["pos_std"], aspect="auto", cmap="viridis"); plt.colorbar(im, ax=axes[0, j])
    axes[0, j].set_yticks(range(4)); axes[0, j].set_yticklabels(bins, fontsize=8)
    axes[0, j].set_xticks([0, 38, 77]); axes[0, j].set_xticklabels(["t-77", "t-39", "t"])
    axes[0, j].set_title(f"{v}: std over pictures of the contribution at each head position")
    prof = c["pos_std"].mean(0)
    axes[1, j].plot(prof / prof.mean()); axes[1, j].axhline(1, color="grey", lw=0.6, ls="--")
    axes[1, j].set_title("by bar (mean over rows), relative to the average"); axes[1, j].set_xticks([0, 38, 77]); axes[1, j].set_xticklabels(["t-77", "t-39", "t"])
    order = np.argsort(-c["ch_std"])
    base, ABL[v] = icn.channel_ablation(MODEL[v], X, v, Y, ROWS, order, [1, 5, 20, 64], DEVICE, BATCH, SETTINGS["amp"])
    share = np.cumsum(np.sort(c["ch_std"] ** 2)[::-1]) / (c["ch_std"] ** 2).sum()
    print(f"{v}: per-row std of contribution {np.round(c['bin_std'], 3)}; top 10% of channels hold "
          f"{share[25]:.0%} of the channel variance; base AUC {base:.4f}")
    display(ABL[v].round(4)); ABL[v].assign(model=v).to_csv(RUN_DRIVE / f"channel_ablation_{v}.csv", index=False)
fig.tight_layout(); fig.savefig(RUN_DRIVE / "activation_x_weight.png", dpi=130); plt.show()
"""), md("""
## 6. Saliency: Grad-CAM and gradient × input
Both for the logit difference, on `N_SALIENCY` pictures. Grad-CAM is coarse (one cell ≈ 12 rows × 1 bar); gradient ×
input is per pixel and only non-zero on lit pixels. The table gives each region's share of gradient × input next to
its share of the picture's rows.
"""), code("""
SAL = {}
sub = np.linspace(0, len(X) - 1, min(N_SALIENCY, len(X))).astype(int)
for v in MODELS:
    t1 = time.perf_counter()
    cams, gx, cam_mean = icn.saliency(MODEL[v], X[sub], v, DEVICE, max(4, BATCH // 4))
    SAL[v] = dict(cams=cams, gx=gx, cam=cam_mean, shares=icn.region_shares(gx, v))
    SAL[v]["shares"].assign(model=v).to_csv(RUN_DRIVE / f"saliency_shares_{v}.csv", index=False)
    print(f"{v}: saliency on {len(sub):,} pictures in {(time.perf_counter() - t1) / 60:.1f} min"); display(SAL[v]["shares"].round(3))
fig, axes = plt.subplots(1, 2 * len(MODELS), figsize=(6 * len(MODELS), 6.5))
for j, v in enumerate(MODELS):
    h = pi.height(v)
    axes[2 * j].imshow(SAL[v]["cam"], aspect="auto", cmap="inferno", extent=(0, ic.SIZE, h, 0))
    axes[2 * j].set_title(f"{v}: mean Grad-CAM", fontsize=9)
    axes[2 * j + 1].imshow(np.log1p(SAL[v]["gx"] / SAL[v]["gx"].mean()), aspect="auto", cmap="inferno")
    axes[2 * j + 1].set_title(f"{v}: mean |gradient x input| (log)", fontsize=9)
    for a in axes[2 * j: 2 * j + 2]:
        for k, (r0, r1, lab) in icn.regions(v).items():
            a.axhline(r0, color="w", lw=0.4, alpha=0.5)
        a.set_xticks([0, 117, 231]); a.set_xticklabels(["t-77", "t-38", "t"])
fig.tight_layout(); fig.savefig(RUN_DRIVE / "saliency.png", dpi=120); plt.show()

v = MODELS[0]
p = icn.run_ops(MODEL[v], X[sub], v, [], MEAN[v], DEVICE, BATCH, SETTINGS["amp"])["base"]
pick = {"lowest predicted": np.argmin(p), "highest predicted": np.argmax(p), "median predicted": np.argsort(p)[len(p) // 2]}
fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 6.5))
for a, (lab, i) in zip(axes, pick.items()):
    img = X[sub[i], :pi.height(v)].numpy()
    cam = np.kron(SAL[v]["cams"][i].astype(np.float32), np.ones((1, 3)))
    a.imshow(img, cmap="gray", aspect="auto")
    a.imshow(cam, cmap="jet", alpha=0.35, aspect="auto", extent=(0, ic.SIZE, pi.height(v), 0))
    a.set_title(f"{lab}: p = {p[i]:.2f}, label {Y[sub[i]]}", fontsize=9); a.set_xticks([])
fig.tight_layout(); fig.savefig(RUN_DRIVE / "gradcam_examples.png", dpi=120); plt.show()
"""), md("""
## 7. Probing: what the features encode
A ridge regression from the CNN's pooled features (per channel: mean over the head's 4 × 78 positions, and the latest
bar's column) to each of the 43 numbers and the time of day, cross-validated by day. For `pure_key` the 43 numbers are
**not** drawn, so a high R² means the CNN computes that quantity from the chart and strips itself.
"""), code("""
TARGETS = NUMBERS.copy()
TARGETS["time_of_day"] = (ROWS.end_index.to_numpy() % ic.LOOKBACK) / (ic.LOOKBACK - 1)
TARGETS["label"] = Y.astype(float)
groups = pd.factorize(ROWS.decision_date)[0]
PROBE = pd.concat([icn.probe(CON[v]["probe"], TARGETS, groups).rename(columns={"r2": v}).set_index("target")[[v]]
                   for v in MODELS], axis=1)
PROBE.to_csv(RUN_DRIVE / "probe_r2.csv")
display(PROBE.sort_values(MODELS[-1], ascending=False).round(3))
fig, ax = plt.subplots(figsize=(12, 4))
PROBE.sort_values(MODELS[-1], ascending=False).plot.bar(ax=ax, width=0.8)
ax.set_ylabel("cross-validated R²"); ax.set_ylim(min(-0.05, PROBE.min().min()), 1); ax.tick_params(axis="x", labelsize=7)
ax.set_title("linear read-out of each number from the CNN's pooled features")
fig.tight_layout(); fig.savefig(RUN_DRIVE / "probe.png", dpi=130); plt.show()
"""), md("""
## 8. Takeaways
"""), code("""
lines = [f"Step 6d run {STEP6D_DIR.name}, cell {CELL['name']}, {len(ROWS):,} validation rows on "
         f"{ROWS.decision_date.nunique()} days of 2023. Drops are AUC(unchanged) - AUC(occluded), 95% day-bootstrap.", ""]
for v in MODELS:
    t = OCC[v]
    lines.append(f"**{v}** (AUC on these rows {BASE[v]:.4f})")
    for g in ["region (mean fill)", "erase"]:
        for _, r in t[t.group == g].sort_values("drop", ascending=False).iterrows():
            lines.append(f"- {g}: {r.part}: drop {r['drop']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] {r.verdict}")
    tt = t[t.group.str.startswith("time")]
    top = tt.sort_values("drop", ascending=False).head(3)
    lines.append("- time blocks with the largest drop: " + "; ".join(f"{r.part} {r['drop']:+.4f}" for _, r in top.iterrows()))
    sh = SAL[v]["shares"].sort_values("share", ascending=False)
    lines.append("- gradient x input shares: " + ", ".join(f"{r.region} {r.share:.0%} (rows {r.share_of_rows:.0%})" for _, r in sh.iterrows()))
    a = ABL[v]
    lines.append("- channels zeroed: " + "; ".join(f"top {int(r.k)} {r.drop_top:+.4f} vs random {r.drop_random_mean:+.4f}" for _, r in a.iterrows()))
    lines.append("")
if "pure_all" in OCC:
    lines.append("Numbers the CNN reads most (pure_all, set to 0): " + ", ".join(
        f"{r['name']} {r['drop']:+.4f}" for _, r in NUMBER_DROP.head(8).iterrows()))
if "pure_key" in PROBE:
    best = PROBE["pure_key"].drop(["label"]).sort_values(ascending=False)
    lines.append("Best linear read-outs from pure_key (numbers not drawn): " + ", ".join(f"{k} {r:.2f}" for k, r in best.head(8).items()))
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
""")]
    return c
