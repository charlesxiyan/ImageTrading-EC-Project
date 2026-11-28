"""Shared helpers for the MindFlow notebooks (Drive: Colab Notebooks/EC/Project/Demo/MindFlow).

The notebooks follow the project's line of reasoning, one step per notebook:

    Step 1  daily charts (JKX replication) and pretrained models: no direction signal
    Step 2  5-minute charts, direction targets, every architecture (CNN, ResNet, 1D CNN, ViT, HGB,
            frozen DINOv2/CLIP, time-series foundation models): no direction signal
    Step 3  same models, volatility target: clear signal, HGB ahead of the image models
    Step 4  add information to the picture step by step until the image model reaches HGB,
            with a features-only control and a feature-selection study
    Step 5  summary figure and parameter table across all steps

Every run of every step lives in MindFlow/runs/<step folder>/<RUN_ID>/. Later steps find the
newest finished run of an earlier step with `latest_run`, or use the RUN_ID set in their config.
"""
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "2026-11-25.1"

DRIVE_HOME = "/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo/MindFlow"
BUNDLE_NAME = "alldata_bundle_v4.zip"
STEP_FOLDERS = {
    "1a": "step1a_daily_jkx", "1b": "step1b_daily_pretrained",
    "2a": "step2a_direction_models", "2b": "step2b_direction_pretrained",
    "3a": "step3a_volatility_models", "3b": "step3b_volatility_pretrained",
    "4a": "step4a_image_information", "4b": "step4b_feature_selection",
    "5": "step5_summary", "6a": "step6a_vix_regime", "6b": "step6b_final_model", "6c": "step6c_volatility_use", "6d": "step6d_pure_image", "6f": "step6f_compact_images", "6g": "step6g_regularization", "7b": "step7b_distillation", "7b+": "step7b_plus_pure_distillation", "8a": "step8a_inside_pure_cnn",
}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def stage_bundle(bundle, local_root=None, verify=True):
    """Copy the bundle zip to local disk, unzip it, check every file's sha256; returns prepared/.

    In one Colab runtime the unzipped bundle is reused by later notebooks (same zip size and
    modification time), so running several steps in a row copies the 1.3 GB zip only once."""
    bundle = Path(bundle)
    assert bundle.is_file(), f"{bundle.name} not found in {bundle.parent}"
    root = Path(local_root) if local_root else Path(tempfile.mkdtemp(prefix="mindflow_"))
    stage = root / "bundle"
    stamp = dict(name=bundle.name, size=bundle.stat().st_size, mtime=int(bundle.stat().st_mtime))
    marker = root / "bundle_ok.json"
    if marker.exists() and json.loads(marker.read_text()) == stamp and (stage / "prepared").is_dir():
        print(f"bundle already unpacked in {stage} (same zip), reusing it")
        return stage / "prepared"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    local_zip = root / "bundle.zip"
    shutil.copy2(bundle, local_zip)
    with zipfile.ZipFile(local_zip) as z:
        for info in z.infolist():                       # refuse paths escaping the folder
            assert (stage / info.filename).resolve().is_relative_to(stage.resolve()), info.filename
        z.extractall(stage)
    local_zip.unlink()
    if verify:
        for rel, d in json.loads((stage / "bundle_manifest.json").read_text())["files"].items():
            assert _sha(stage / rel) == d["sha256"], f"checksum mismatch: {rel}"
    marker.write_text(json.dumps(stamp))
    print(f"bundle {bundle.name} unpacked and verified in {stage}")
    return stage / "prepared"


def stage_panels(home, prepared, pattern="*_panel_v4.zip"):
    """Unpack add-on market panels (tools/build_panel_bundle.py, e.g. vixy_panel_v4.zip) found in `home`
    into prepared/, checking every file's sha256. Returns the names of the panels added."""
    names = []
    for z in sorted(Path(home).glob(pattern)):
        with zipfile.ZipFile(z) as f:
            man = json.loads(f.read("panel_manifest.json"))
            for rel, d in man["files"].items():
                target = Path(prepared).parent / rel
                assert target.resolve().is_relative_to(Path(prepared).parent.resolve()), rel
                data = f.read(rel)
                assert hashlib.sha256(data).hexdigest() == d["sha256"], f"checksum mismatch: {rel}"
                target.write_bytes(data)
        info = json.loads((Path(prepared) / f"{man['name']}_panel.json").read_text())
        print(f"panel {man['name']} from {z.name}: {info['observed']:,} observed bars, "
              f"{info['filled_share']:.2%} of full-session slots filled flat")
        names.append(man["name"])
    return names


def run_folder(home, step):
    return Path(home) / "runs" / STEP_FOLDERS[step]


def latest_run(home, step, run_id=None, need_done=True):
    """The folder of run `run_id` of a step, or its newest run with at least one finished cell.
    Returns None when the step has not been run yet."""
    base = run_folder(home, step)
    if run_id:
        path = base / run_id
        assert path.is_dir(), f"{path} not found"
        return path
    if not base.is_dir():
        return None
    runs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    for p in runs:
        if not need_done or any(p.glob("*/done.json")) or any(p.glob("*/confirmation.csv")):
            return p
    return None


def new_run(home, step, run_id, prefix, local_test=False):
    """(run_id, folder, resume) for a step; a new id is the UTC time."""
    from datetime import datetime, timezone
    resume = run_id is not None
    run_id = run_id or ("local_smoke" if local_test else datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%d_%H%M%S"))
    folder = run_folder(home, step) / run_id
    if resume:
        assert folder.exists(), f"{folder} not found"
    else:
        assert local_test or not folder.exists(), "RUN_ID exists: set RUN_ID to resume it"
    folder.mkdir(parents=True, exist_ok=True)
    return run_id, folder, resume


def save_settings(folder, resume, **kw):
    name = "settings_resume.json" if resume else "settings.json"
    (Path(folder) / name).write_text(json.dumps(kw, indent=2, default=str))


# ----------------------------------------------------------------------------- parameter counts

def parameter_rows(step, run_dir):
    """Parameter counts recorded in a run's done.json files (advisor rule). HGB: tree nodes."""
    rows = []
    if run_dir is None:
        return rows
    for done in sorted(Path(run_dir).glob("*/done.json")):
        info = json.loads(done.read_text())
        for model, n in (info.get("parameters") or {}).items():
            if isinstance(n, dict):                     # indicator CNN: {total, cnn, tab}
                n = n.get("total")
            rows.append(dict(step=step, cell=done.parent.name, model=model, parameters=n))
        for model, p in (info.get("probes") or {}).items():
            rows.append(dict(step=step, cell=done.parent.name, model=model, parameters=p.get("parameters"),
                             note="trainable probe head; encoder frozen"))
    extra = Path(run_dir) / "parameters.json"          # frozen encoders, foundation models
    if extra.exists():
        for model, n in json.loads(extra.read_text()).items():
            rows.append(dict(step=step, cell="all", model=model, parameters=n))
    return rows


def parameter_table(rows):
    """One row per (step, model): parameter count (the same in every cell for networks)."""
    if not rows:
        return pd.DataFrame(columns=["step", "model", "parameters"])
    t = pd.DataFrame(rows)
    t["parameters"] = pd.to_numeric(t.parameters, errors="coerce")
    if "note" not in t:
        t["note"] = ""
    g = t.groupby(["step", "model"]).agg(parameters=("parameters", "max"), min_parameters=("parameters", "min"),
                                         note=("note", "first"), cells=("cell", "nunique"))
    g.loc[g.min_parameters != g.parameters, "note"] = g.note.fillna("") + " varies by cell (max shown; HGB tree nodes)"
    return g.drop(columns="min_parameters").reset_index()


# ----------------------------------------------------------------------------- summary

def read_csv(path, **kw):
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


def grid_summary(table_csv, step):
    """Validation AUC by (cell, model) from a step 2a / 3a grid_table.csv."""
    t = read_csv(table_csv)
    if t is None:
        return None
    t = t.rename(columns={"roc_auc": "auc"})
    keep = [c for c in ("cell", "kind", "minutes", "model", "auc", "auc_ci_low", "auc_ci_high") if c in t]
    out = t[keep].copy()
    out["step"] = step
    return out


def mindflow_figure(rows, path=None):
    """One panel per step: validation AUC (95% CI) of each model family, 0.5 = coin flip."""
    import matplotlib.pyplot as plt
    steps = [s for s in ("1", "2", "3", "4") if s in set(rows.step)]
    titles = {"1": "Step 1  daily charts\n(direction, I5/I20 → 1 week / 1 month)",
              "2": "Step 2  5-minute charts\n(direction, every model)",
              "3": "Step 3  5-minute charts\n(volatility target)",
              "4": "Step 4  add information to the picture\n(volatility)"}
    fig, axes = plt.subplots(1, len(steps), figsize=(6.2 * len(steps), 5.6), squeeze=False)
    for ax, s in zip(axes[0], steps):
        d = rows[rows.step == s].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        colours = d.family.map({"image": "tab:blue", "ViT": "tab:purple", "pretrained": "tab:green",
                                "tree": "tab:orange", "series": "tab:gray", "features": "tab:red"}).fillna("black")
        for yi, (_, r), col in zip(y, d.iterrows(), colours):
            ax.errorbar([r.auc], [yi], xerr=[[r.auc - r.low], [r.high - r.auc]], fmt="o", color=col, capsize=2)
        for fam, col in (("image CNN", "tab:blue"), ("ViT", "tab:purple"), ("pretrained", "tab:green"),
                         ("HGB", "tab:orange"), ("1D CNN", "tab:gray"), ("features only", "tab:red")):
            ax.plot([], [], "o", color=col, label=fam)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_yticks(y, d.label, fontsize=7)
        ax.axvline(0.5, ls=":", c="gray")
        ax.set_title(titles[s], fontsize=9)
        ax.set_xlabel("validation AUC (2023)")
        ax.tick_params(axis="y", labelleft=True)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()
