"""Cells of MindFlow Step 6c (what the volatility forecast is worth when used)."""
from cells import code, header, md, setup


def vol_use_cells():
    c = [header("6c", "Using the volatility forecast", "The final model predicts how much a stock will move, not which "
                "way. What is that worth as a factor: how good is the forecast (IC, ICIR), does it earn anything as a "
                "long-short factor (Sharpe, max drawdown), and does it make a book's risk steadier when used for sizing?", """
No training: this step reads the Step 6b predictions (2023, and 2024–2025 once the 6b test section has been run) and
compares, on identical rows, the final CNN, HGB with the same 43 numbers, HGB with 38, Step 4a's hybrid CNN and a
no-forecast baseline (each stock's training-median |return|).

| use | what is measured | expected |
|---|---|---|
| forecast quality | cross-sectional IC and ICIR vs the next |return| | clearly positive (AUC 0.69) |
| return factor | long low / short high predicted volatility: Sharpe, max drawdown, after 0 / 1 / 2 bp | near zero (no direction signal in Steps 2a/2b) |
| risk control | long-only book sized by predicted volatility: Sharpe, drawdown, how steady realized risk is | steadier risk than equal weight or the no-forecast baseline |

**From a probability to a volatility level.** A score is p = P(|r| > m), m = the stock's training median |r| over the
horizon. With an exponential |r| (Laplace returns), P(|r| > m) = exp(−m/s), so the implied mean |r| is s = m / (−ln p).
**Not a trading backtest:** the book holds only for the label horizon after each decision time (one per hour), with
no overnight returns; Sharpe ratios are annualised with the decision times actually traded.
""")]
    c += setup("""
import chartcnn
import indicator_cnn as ic
import vol_use as vu
import matplotlib.pyplot as plt
print("vol_use", vu.__version__)
""", gpu=False)
    c += [md("""
## 2. Configuration
`STEP6B_RUN = None` takes the newest Step 6b run. The test years are read only if the 6b test section has written
`test_predictions.parquet`. CPU is enough (a few minutes).
"""), code("""
SMOKE = LOCAL_TEST
RUN_ID = None
STEP6B_RUN = None
COSTS_BP = (0.0, 1.0, 2.0)                          # one-way cost per trade, in basis points
STEP6B_DIR = mf.latest_run(HERE, "6b", STEP6B_RUN)
assert STEP6B_DIR is not None, "run Step 6b first"
S6 = json.loads((STEP6B_DIR / "settings.json").read_text())
CHOICE = json.loads((STEP6B_DIR / "final_choice.json").read_text())
FINAL = CHOICE["final"]
CELLS = [c for c in S6["cells"] if (STEP6B_DIR / c["name"] / "predictions_with_references.parquet").exists()]
RUN_ID, RUN_DRIVE, RESUME = mf.new_run(HERE, "6c", RUN_ID, "voluse", LOCAL_TEST)
mf.save_settings(RUN_DRIVE, RESUME, step6b=STEP6B_DIR.name, final=FINAL, cells=CELLS, costs_bp=COSTS_BP,
                 vol_use=vu.__version__)
print("RUN_ID:", RUN_ID, "| Step 6b run:", STEP6B_DIR.name, "| final model:", FINAL, "| cells:", [c["name"] for c in CELLS])
"""), md("""
## 3. Predictions and each stock's training median |return|
The medians are the ones that defined the labels (training windows only), recomputed from the bundle.
"""), code("""
store = chartcnn.Store(PREPARED)
W_TRAIN = pd.read_parquet(STEP6B_DIR / "windows_train.parquet")
NAMES = {f"score_ind_{FINAL}": f"final CNN ({FINAL})", "score_hgb_tab43": "HGB 43 (38 + multi-day RV)",
         "score_hgb_tab38": "HGB 38", "score_s4a_ma_ewsd_tab": "Step 4a CNN (38 numbers)",
         "score_static_median": "no forecast (training median |r|)"}
DATA = {}
for cell in CELLS:
    _, median = ic.cell_rows(store, cell, "train", W_TRAIN)
    files = {"validation 2023": "predictions_with_references.parquet", "test 2024-25": "test_predictions.parquet"}
    for split, fname in files.items():
        path = STEP6B_DIR / cell["name"] / fname
        if path.exists():
            p = pd.read_parquet(path)
            models = [m for m in NAMES if m in p.columns]
            DATA[(cell["name"], split)] = (vu.attach(p, median), models)
            print(f"{cell['name']} {split}: {len(p):,} rows, {p.end_index.nunique():,} decision times, "
                  f"{p.groupby('end_index').size().median():.0f} stocks per time; models: {models}")
pretty = lambda m: NAMES.get(m, m)
"""), md("""
## 4. Forecast quality: IC and ICIR
- **|r| / own median**: what the models were trained to rank (is this stock going to move more than it usually does?).
- **|r| (level)**: the implied volatility against the raw next |return|. This also contains each stock's usual
  volatility, so the no-forecast baseline is already high here; the gain over it is what the model adds.
- **signed return**: does predicted volatility say anything about direction? Expected ≈ 0.
"""), code("""
IC_ROWS, IC_SERIES = [], {}
for (cell, split), (f, models) in DATA.items():
    t, s = vu.ic_table(f, models)
    IC_ROWS.append(t.assign(cell=cell, split=split))
    IC_SERIES[(cell, split)] = s
IC = pd.concat(IC_ROWS, ignore_index=True)
IC["model"] = IC.model.map(pretty)
IC.to_csv(RUN_DRIVE / "ic.csv", index=False)
display(IC.pivot_table(index=["cell", "split", "model"], columns="target", values=["mean_ic", "icir"]).round(3))
"""), md("""
## 5. Volatility as a return factor: long low, short high predicted volatility
Each decision time: sort on implied volatility, long the lowest fifth, short the highest fifth, equal weights, hold
for the label horizon. Costs charge four one-way trades of a leg's notional per period (both legs opened and closed).
"""), code("""
LS_ROWS, LS_CURVES = [], {}
for (cell, split), (f, models) in DATA.items():
    t, curves = vu.long_short_table(f, models, COSTS_BP)
    LS_ROWS.append(t.assign(cell=cell, split=split))
    LS_CURVES[(cell, split)] = curves
LS = pd.concat(LS_ROWS, ignore_index=True)
LS["model"] = LS.model.map(pretty)
LS.to_csv(RUN_DRIVE / "long_short.csv", index=False)
display(LS.set_index(["cell", "split", "model", "cost_bp"])[["ann_return", "ann_vol", "sharpe", "max_drawdown",
                                                             "hit_rate", "periods"]].round(4))
"""), md("""
## 6. Volatility for risk control: sizing a long-only book
Same stocks, same periods, four ways to size them. `risk parity` keeps gross exposure at 1 and gives each stock weight
∝ 1/predicted volatility; `volatility target` gives each stock a fixed risk budget, so exposure shrinks when the model
expects large moves and grows when it expects calm (scaled to an average gross of 1). **vol_of_daily_vol** is the
dispersion across days of each day's realised volatility (std / mean): lower means steadier risk.
"""), code("""
SZ_ROWS, SZ_CURVES, RISK_ROWS = [], {}, []
for (cell, split), (f, models) in DATA.items():
    t, curves = vu.sizing_table(f, models)
    SZ_ROWS.append(t.assign(cell=cell, split=split))
    SZ_CURVES[(cell, split)] = curves
    RISK_ROWS.append(vu.risk_forecast_ic(f, models).assign(cell=cell, split=split))
SZ = pd.concat(SZ_ROWS, ignore_index=True)
SZ["model"] = SZ.model.map(lambda m: pretty(m) if m != "-" else "-")
SZ.to_csv(RUN_DRIVE / "sizing.csv", index=False)
display(SZ.set_index(["cell", "split", "scheme", "model"])[["ann_return", "ann_vol", "sharpe", "max_drawdown",
                                                            "vol_of_daily_vol", "mean_gross", "min_gross", "max_gross"]].round(4))
RISK = pd.concat(RISK_ROWS, ignore_index=True)
RISK["model"] = RISK.model.map(pretty)
RISK.to_csv(RUN_DRIVE / "risk_forecast.csv", index=False)
print("Does the model know when the whole book will move? (time-series rank correlation)")
display(RISK.round(3))
"""), md("""
## 7. Figures
"""), code("""
for (cell, split) in DATA:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    s = IC_SERIES[(cell, split)]
    for m in [f"score_ind_{FINAL}", "score_hgb_tab43"]:
        if (m, "|r| / own median") in s:
            r = s[(m, "|r| / own median")]
            ax[0].plot(np.arange(len(r)), r.rolling(60, min_periods=10).mean().to_numpy(), label=pretty(m))
    ax[0].set_title("IC vs |r| / own median (rolling 60 times)"); ax[0].legend(fontsize=7)
    for m, r in LS_CURVES[(cell, split)].items():
        if m in (f"score_ind_{FINAL}", "score_hgb_tab43", "score_static_median"):
            ax[1].plot(np.arange(len(r)), r.cumsum().to_numpy() * 1e4, label=pretty(m))
    ax[1].set_title("long low / short high predicted vol, cumulative bp (no costs)"); ax[1].legend(fontsize=7)
    for name, r in SZ_CURVES[(cell, split)].items():
        if name in ("equal weight", "inverse median vol (no forecast)") or name.endswith(f"score_ind_{FINAL}"):
            ax[2].plot(np.arange(len(r)), r.cumsum().to_numpy() * 1e4, label=name.replace(f"score_ind_{FINAL}", "final CNN"))
    ax[2].set_title("long-only book, cumulative bp"); ax[2].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("decision time")
    fig.suptitle(f"{cell}, {split}")
    fig.tight_layout(); fig.savefig(RUN_DRIVE / f"curves_{cell}_{split.split()[0]}.png", dpi=120); plt.show()
"""), md("""
## 8. Takeaways
"""), code("""
lines = [f"Final model `{FINAL}` (Step 6b run {STEP6B_DIR.name}). Sharpe ratios annualised with the decision times traded.", ""]
fm_name = pretty(f"score_ind_{FINAL}")
for (cell, split) in DATA:
    ic_ = IC[(IC.cell == cell) & (IC.split == split)].set_index(["model", "target"])
    ls = LS[(LS.cell == cell) & (LS.split == split)].set_index(["model", "cost_bp"])
    sz = SZ[(SZ.cell == cell) & (SZ.split == split)].set_index("book")
    lines.append(f"**{cell}, {split}**")
    for m in (fm_name, pretty("score_hgb_tab43")):
        if (m, "|r| / own median") in ic_.index:
            r = ic_.loc[(m, "|r| / own median")]
            d = ic_.loc[(m, "signed return")]
            lines.append(f"- IC {m}: {r.mean_ic:.3f} (ICIR {r.icir:.2f}, t {r.t_stat:.1f}, positive {r.share_positive:.0%}); "
                         f"vs signed return {d.mean_ic:+.3f} (t {d.t_stat:.1f})")
    for m in (fm_name, pretty("score_hgb_tab43"), pretty("score_static_median")):
        if (m, 0.0) in ls.index:
            a, b = ls.loc[(m, 0.0)], ls.loc[(m, COSTS_BP[1])]
            lines.append(f"- long-short {m}: Sharpe {a.sharpe:+.2f} gross, {b.sharpe:+.2f} at {COSTS_BP[1]:g} bp; "
                         f"max drawdown {a.max_drawdown * 1e4:.0f} bp")
    for name in ("equal weight", "inverse median vol (no forecast)", f"risk parity | score_ind_{FINAL}",
                 f"volatility target | score_ind_{FINAL}"):
        if name in sz.index:
            r = sz.loc[name]
            lines.append(f"- {name.replace(f'score_ind_{FINAL}', 'final CNN')}: Sharpe {r.sharpe:+.2f}, ann vol {r.ann_vol:.3f}, "
                         f"max drawdown {r.max_drawdown * 1e4:.0f} bp, vol of daily vol {r.vol_of_daily_vol:.2f}")
    lines.append("")
TAKEAWAYS = "\\n".join(lines)
(RUN_DRIVE / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
""")]
    return c
