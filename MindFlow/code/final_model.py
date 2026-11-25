"""Step 6b: the final model.

Step 4a's best image model (CNN on the chart + MA + fixed-scale EWSD band, with the 38 hand + SPY numbers as a second
input) plus the one regime input that helped in Step 6a: multi-day realized volatility of the stock and of SPY
(rv5d, rv20d, their log ratio, SPY's rv5d and rv20d; intraday 5-minute returns of completed sessions only).

Everything that is not the new input is kept fixed: the Step 4a run's windows (train, validation; same purge and
embargo), its EWSD scale, the CNN, its settings and its training rule. The 43 numbers are standardised with training
rows only. HGB with 38 and 43 numbers is fitted here on the same training windows, so every model is compared on
identical rows; Step 4a's own scores for the same rows are joined in.

The held-out test years (2024-2025) are scored once, at the end, with the checkpoints chosen on 2023.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import feature_select as fs
import indicator_cnn as ic
import regime as rg

__version__ = "2026-11-25.1"

RV = rg.RV_COLS + rg.SPY_RV_COLS
NEW_VARIANTS = {   # name: (MA, EWSD band, numbers) -- the numbers are 38 hand/SPY + 5 multi-day RV
    "ma_ewsd_tab_rv": (True, True, True),
    "ewsd_tab_rv": (False, True, True),
    "tab_only_rv": (False, False, True),
}
NEW_LABELS = {
    "ma_ewsd_tab_rv": "CNN, chart + MA + EWSD + 38 numbers + multi-day RV",
    "ewsd_tab_rv": "CNN, chart + EWSD + 38 numbers + multi-day RV",
    "tab_only_rv": "features only (MLP, 38 + multi-day RV, no image)",
}
REFS = {"hgb_tab38": "HGB 38 hand + SPY (same rows)", "hgb_tab43": "HGB 38 + multi-day RV (same rows)",
        "s4a_ma_ewsd_tab": "Step 4a CNN, chart + MA + EWSD + 38 numbers", "s4a_tab_only": "Step 4a features only (38)",
        "s4a_ewsd": "Step 4a CNN, chart + EWSD (no numbers)", "s4a_plain": "Step 4a CNN, original chart"}
CHECKS = [
    ("ind_ma_ewsd_tab_rv", "s4a_ma_ewsd_tab", "multi-day RV added to the Step 4a hybrid CNN"),
    ("ind_ewsd_tab_rv", "ind_ma_ewsd_tab_rv", "dropping the MA line"),
    ("hgb_tab43", "hgb_tab38", "multi-day RV added to HGB (same rows)"),
    ("ind_ma_ewsd_tab_rv", "hgb_tab43", "final CNN (with MA) vs HGB with the same 43 numbers"),
    ("ind_ewsd_tab_rv", "hgb_tab43", "final CNN (no MA) vs HGB with the same 43 numbers"),
    ("ind_ma_ewsd_tab_rv", "ind_tab_only_rv", "image validity: chart + numbers vs the numbers alone (with MA)"),
    ("ind_ewsd_tab_rv", "ind_tab_only_rv", "image validity: chart + numbers vs the numbers alone (no MA)"),
    ("ind_tab_only_rv", "hgb_tab43", "features-only MLP vs HGB on the same 43 numbers"),
]


def register():
    """Make the new variants known to indicator_cnn (drawing, model, labels)."""
    ic.VARIANTS.update(NEW_VARIANTS)
    ic.VARIANT_LABEL.update(NEW_LABELS)
    ic.IMAGE_FREE.add("tab_only_rv")
    ic.REFERENCE_MODELS.update(REFS)


def rv_features(frame, rv):
    """The 5 multi-day RV numbers of each window (sessions completed before the decision session)."""
    s = frame.security_index.to_numpy(np.int64)
    d = frame.end_index.to_numpy(np.int64) // rg.BARS
    out = pd.DataFrame({"rv5d": rv["rv5d"][s, d], "rv20d": rv["rv20d"][s, d]})
    out["rv_ratio"] = np.log(out.rv5d / out.rv20d)
    out["spy_rv5d"], out["spy_rv20d"] = rv["spy_rv5d"][d], rv["spy_rv20d"][d]
    return out


def numbers(store, windows, rv):
    """(38 hand + SPY, 43 = 38 + RV) raw feature frames for `windows`, in window order."""
    t38 = ic.tab_features(store, windows).reset_index(drop=True)
    return t38, pd.concat([t38, rv_features(windows, rv).reset_index(drop=True)], axis=1)


def scaler_from_state(state):
    s = ic.TabScaler.__new__(ic.TabScaler)
    s.columns, s.mean, s.std = state["columns"], np.asarray(state["mean"]), np.asarray(state["std"])
    return s


def fit_hgb(raw_train, rows_train, y_train, seed):
    """HGB (the reference settings) on the training rows; `rows_train` index `raw_train`."""
    return fs.hgb(seed).fit(raw_train.iloc[rows_train], y_train)


def join_step4a(pred, step4a_cell_dir, names=("ma_ewsd_tab", "tab_only", "ewsd", "plain")):
    """Step 4a's validation scores for the same rows (they must be identical rows)."""
    path = Path(step4a_cell_dir) / "predictions.parquet"
    if not path.exists():
        return pred
    ref = pd.read_parquet(path)
    keys = ["security_index", "end_index"]
    cols = [f"score_ind_{n}" for n in names if f"score_ind_{n}" in ref]
    ref = ref[keys + ["label"] + cols].rename(columns={c: "score_s4a_" + c[10:] for c in cols} | {"label": "ref_label"})
    out = pred.merge(ref, on=keys, how="left")
    assert out.ref_label.notna().all(), "Step 4a rows differ: use the same Step 4a run as the windows"
    assert (out.ref_label == out.label).all(), "Step 4a labels differ"
    return out.drop(columns="ref_label")


def checks_table(boots):
    rows = []
    for a, b, label in CHECKS:
        p = ic.paired(boots, a, b)
        for _, r in p.iterrows():
            rows.append(dict(check=label, a=ic.pretty(a), b=ic.pretty(b), cell=r.cell, diff=r["diff"], low=r.low,
                             high=r.high, verdict=r.verdict))
    return pd.DataFrame(rows)


@torch.no_grad()
def score_checkpoint(prepared, frame, variant, settings, scale, tab, ckpt, device, workers):
    """Validation/test scores of one trained variant from its best checkpoint."""
    uses_tab = ic.VARIANTS[variant][2]
    model = ic.IndicatorCNN(settings, tab.shape[1] if uses_tab else 0, image=variant not in ic.IMAGE_FREE).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    ds = ic.Charts(prepared, frame, variant, scale, settings["halflife"], tab[frame.row.to_numpy()] if uses_tab else None)
    return ic.predict(model, ic.make_loader(ds, settings["eval_batch_size"], workers, False), device, settings["amp"])


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float) + "\n")
