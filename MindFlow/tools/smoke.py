"""Code check of every MindFlow notebook on a tiny synthetic bundle (no WRDS data, no downloads).

Builds a fake `alldata_bundle_v4.zip` (30 stocks + SPY, 280 sessions), copies code/ next to it as on Drive,
and executes the notebooks in MindFlow order with tiny budgets and untrained tiny encoders. Numbers are meaningless.

    python MindFlow/tools/smoke.py <notebook folder> <work dir> [notebook name ...]
"""
import hashlib, json, os, shutil, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MF = Path(__file__).resolve().parents[1]
NB_DIR, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
ONLY = sys.argv[3:]
STOCKS, DAYS, SLOTS = 30, 280, 78
ORDER = ["Step1a_Daily_JKX_CNN", "Step1b_Daily_Pretrained", "Step2a_Intraday_Direction_Models",
         "Step2b_Intraday_Direction_Pretrained", "Step3a_Intraday_Volatility_Models",
         "Step3b_Intraday_Volatility_Pretrained", "Step4a_Add_Image_Information", "Step4b_Feature_Selection",
         "Step5_Summary", "Step6a_VIX_Regime", "Step6b_Final_Model", "Step6c_Volatility_Use", "Step7a_Efficient_Computing",
         "Step7b_Distillation"]


def fake_prepared(folder):
    rng = np.random.default_rng(0)
    T = DAYS * SLOTS
    def bars(n):
        sig = np.exp(rng.normal(-6.2, 0.5, (n, 1)))                   # stock-specific volatility
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1, (n, T)) * sig, axis=1))
        f = np.repeat(close[:, :, None], 5, axis=2)
        f[:, :, 0] *= np.exp(rng.normal(0, 5e-4, (n, T)))
        f[:, :, 1] = f[:, :, :4].max(axis=2) * 1.001; f[:, :, 2] = f[:, :, :4].min(axis=2) * 0.999
        f[:, :, 4] = rng.integers(500, 5000, (n, T))
        return f
    features = bars(STOCKS)
    mult = np.ones((STOCKS, T, 2)); mult[0, :70 * SLOTS, 0] = 0.5          # a 2:1 split on stock 0 at day 70
    valid = np.ones((STOCKS, T), bool); valid[1, 300:310] = False; valid[2, 40 * SLOTS:41 * SLOTS] = False
    market = bars(1)[0]
    days = pd.bdate_range("2022-12-20", periods=DAYS)
    clock = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(f"{d.date()} 09:35", tz="America/New_York"), periods=SLOTS, freq="5min") for d in days]))
    dates = clock.tz_convert("America/New_York").strftime("%Y-%m-%d").to_numpy()
    train_end, valid_end = str(days[150].date()), str(days[215].date())
    folder.mkdir(parents=True)
    np.save(folder / "features.npy", features); np.save(folder / "multipliers.npy", mult)
    np.save(folder / "valid.npy", valid)
    np.save(folder / "market_features.npy", market); np.save(folder / "market_multipliers.npy", np.ones((T, 2)))
    np.save(folder / "market_valid.npy", np.ones(T, bool))
    rows = []
    for s in range(STOCKS):
        for t in range(SLOTS - 1, T):
            if valid[s, t - SLOTS + 1:t + 1].all():
                rows.append(dict(security_index=s, permno=10001 + s, end_index=t, decision_time=clock[t]))
    w = pd.DataFrame(rows)
    d = dates[w.end_index]
    splits = {"train": w[d <= train_end], "validation": w[(d > train_end) & (d <= valid_end)], "test": w[d > valid_end]}
    for name, f in splits.items():
        f.reset_index(drop=True).to_parquet(folder / f"{name}.parquet", index=False)
    manifest = dict(config=dict(lookback=78, bar_minutes=5, train_end=train_end, validation_end=valid_end),
                    permnos=[10001 + s for s in range(STOCKS)], decision_clock_ns=clock.as_unit("ns").asi8.tolist(),
                    splits={k: dict(samples=len(v), decision_start="", decision_end="") for k, v in splits.items()},
                    bundle=dict(market_valid_slots=T, windows=len(w)))
    (folder / "manifest.json").write_text(json.dumps(manifest))


def main():
    here = WORK / "drive" / "MindFlow"
    if not ONLY and WORK.exists():
        shutil.rmtree(WORK)
    if not (here / "alldata_bundle_v4.zip").exists():
        here.mkdir(parents=True)
        staging = WORK / "staging"
        fake_prepared(staging / "prepared")
        files = {str(p.relative_to(staging)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in (staging / "prepared").iterdir()}
        (staging / "bundle_manifest.json").write_text(json.dumps(dict(files=files)))
        with zipfile.ZipFile(here / "alldata_bundle_v4.zip", "w") as z:
            for p in staging.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(staging))
    if (here / "code").exists():
        shutil.rmtree(here / "code")
    shutil.copytree(MF / "code", here / "code", ignore=shutil.ignore_patterns("__pycache__"))
    panel = here / "vixy_panel_v4.zip"                     # synthetic VIXY add-on panel (a few missing bars filled)
    if not panel.exists():
        sys.path.insert(0, str(MF / "tools"))
        from build_panel_bundle import fill_missing
        rng = np.random.default_rng(2)
        T = DAYS * SLOTS
        c = 20 * np.exp(np.cumsum(rng.normal(0, 2e-3, T)))
        x = np.stack([c, c * 1.001, c * 0.999, c, rng.integers(100, 1000, T)], 1)
        ok = rng.random(T) > 0.01
        x[~ok] = np.nan
        xf, mf_, filled = fill_missing(x, np.ones((T, 2)), ok, np.ones(T, bool))
        files = {"prepared/vixy_features.npy": xf.astype(np.float32), "prepared/vixy_multipliers.npy": mf_.astype(np.float32),
                 "prepared/vixy_valid.npy": ok}
        import io
        blobs = {}
        for rel, arr in files.items():
            b = io.BytesIO(); np.save(b, arr); blobs[rel] = b.getvalue()
        blobs["prepared/vixy_panel.json"] = json.dumps(dict(observed=int(ok.sum()), filled_share=filled / T)).encode()
        with zipfile.ZipFile(panel, "w") as z:
            for rel, data in blobs.items():
                z.writestr(rel, data)
            z.writestr("panel_manifest.json", json.dumps(dict(name="vixy", files={
                rel: dict(sha256=hashlib.sha256(d).hexdigest()) for rel, d in blobs.items()})))
    vix = here / "data" / "VIX_History.csv"                     # synthetic VIX (the notebook downloads Cboe's)
    if not vix.exists():
        vix.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        days = pd.bdate_range("2022-11-01", periods=DAYS + 60)
        close = 18 * np.exp(np.cumsum(rng.normal(0, 0.05, len(days))))
        pd.DataFrame(dict(DATE=days.strftime("%m/%d/%Y"), OPEN=close, HIGH=close, LOW=close, CLOSE=close)).to_csv(vix, index=False)

    import nbformat
    from nbclient import NotebookClient
    os.environ.update(MINDFLOW_LOCAL_TEST="1", MINDFLOW_HERE=str(here), MINDFLOW_LOCAL=str(WORK / "local"),
                      MPLBACKEND="Agg")
    (WORK / "executed").mkdir(exist_ok=True)
    for name in ORDER:
        if ONLY and name not in ONLY:
            continue
        nb = nbformat.read(NB_DIR / f"{name}.ipynb", as_version=4)
        print("running", name, flush=True)
        try:
            NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(WORK)}}).execute()
        finally:
            nbformat.write(nb, WORK / "executed" / f"{name}.ipynb")
        print("  ok", flush=True)
    if not ONLY or "Step3a_Intraday_Volatility_Models" in ONLY:       # retrain only the ViT of a finished run
        nb = nbformat.read(NB_DIR / "Step3a_Intraday_Volatility_Models.ipynb", as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and "RETRAIN_ARCHS = []" in cell.source:
                cell.source = cell.source.replace("RUN_ID = None", 'RUN_ID = "local_smoke"').replace(
                    "RETRAIN_ARCHS = []", 'RETRAIN_ARCHS = ["vit"]')
        print("running Step3a again with RETRAIN_ARCHS = ['vit']", flush=True)
        try:
            NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(WORK)}}).execute()
        finally:
            nbformat.write(nb, WORK / "executed" / "Step3a_retrain_vit.ipynb")
        done = json.loads((here / "runs" / "step3a_volatility_models" / "local_smoke" / "vol_h1" / "done.json").read_text())
        assert "vit" in done.get("retrained", {}), done
        print("  ok", flush=True)
    if not ONLY or "Step7a_Efficient_Computing" in ONLY:       # the same notebook on the Step 6b final model
        nb = nbformat.read(NB_DIR / "Step7a_Efficient_Computing.ipynb", as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and 'SOURCE = dict(step="3a"' in cell.source:
                cell.source = cell.source.replace('SOURCE = dict(step="3a", run_id=None, cell="vol_h1", model="plain_spy")',
                                                  'SOURCE = dict(step="6b", run_id=None, cell="vol_h1", model="ewsd_tab_rv")')
                cell.source = cell.source.replace("RUN_ID = None", 'RUN_ID = "local_smoke_6b"')
        out = here / "runs" / "step7a_efficiency" / "local_smoke_6b"
        out.mkdir(parents=True, exist_ok=True)
        print("running Step7a again on the Step 6b model ewsd_tab_rv", flush=True)
        try:
            NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(WORK)}}).execute()
        finally:
            nbformat.write(nb, WORK / "executed" / "Step7a_final_source.ipynb")
        print("  ok", flush=True)
    print("smoke run OK ->", WORK / "executed")


if __name__ == "__main__":
    main()
