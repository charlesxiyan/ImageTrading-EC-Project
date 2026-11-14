"""Small I/O and reproducibility helpers."""
import hashlib
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


BUNDLE = "cnn78u_colab_bundle_v2.zip"   # v2: all windows kept (no next-bar zero-return filter)
V1 = "cnn78u_colab_bundle.zip"


def rescreen_all_windows(prepared):
    """Rewrite train/validation/test.parquet of an unpacked v1 bundle as v2: every window with 78
    complete input bars and a complete next bar, inside its split, with no zero-return filter.
    Mirrors us_pipeline.data.sample_indices with drop_zero_returns=False (the v2 zip was built by
    that function; tests/test_bundle_v2.py checks the two agree). Returns sample counts."""
    import pandas as pd
    prepared = Path(prepared)
    m = json.loads((prepared / "manifest.json").read_text())
    c = m["config"]
    if c["horizon"] != 1 or c["stride"] != 1 or not c["allow_cross_session"]:
        raise ValueError("expected the v1 next-bar bundle (horizon 1, stride 1, cross-session allowed)")
    if not c.get("drop_zero_returns", True):
        return {s: m["splits"][s]["samples"] for s in m["splits"]}         # already v2
    lookback, h = c["lookback"], 1
    valid = np.load(prepared / "valid.npy")
    close = np.load(prepared / "features.npy", mmap_mode="r")
    clock = pd.to_datetime(m["decision_clock_ns"], unit="ns", utc=True).tz_convert("America/New_York")
    dates = np.array(clock.strftime("%Y-%m-%d"))
    cut1 = int(np.flatnonzero(dates <= c["train_end"])[-1])
    cut2 = int(np.flatnonzero(dates <= c["validation_end"])[-1])
    ends = np.arange(lookback - 1, len(clock) - h, dtype=np.int64)
    parts = {"train": [], "validation": [], "test": []}
    for stock, permno in enumerate(m["permnos"]):
        bad = np.r_[0, np.cumsum(~valid[stock])]
        e = ends[(bad[ends + 1] - bad[ends - lookback + 1]) == 0]
        e = e[(bad[e + h + 1] - bad[e + 1]) == 0]
        split = np.where(e <= cut1, 0, np.where(e <= cut2, 1, 2))
        inside = e + h <= np.where(split == 0, cut1, np.where(split == 1, cut2, len(clock) - 1))
        e, split = e[inside], split[inside]
        r = close[stock, e + h, 3] / close[stock, e, 3] - 1.0
        for k, name in enumerate(parts):
            t = e[split == k].astype(np.int32)
            if len(t):
                parts[name].append(pd.DataFrame({
                    "security_index": np.full(len(t), stock, dtype=np.int32),
                    "permno": np.full(len(t), int(permno), dtype=np.int64), "end_index": t,
                    "label": (r[split == k] > 0).astype("int8"), "forward_return": r[split == k],
                    "decision_time": clock[t], "label_end_time": clock[t + h]}))
    counts = {}
    for name, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(prepared / f"{name}.parquet", index=False, compression="zstd")
        m["splits"][name]["samples"] = counts[name] = len(frame)
    m["config"]["drop_zero_returns"] = False
    m["rescreened_in_colab"] = "chartcnn.utils.rescreen_all_windows (v1 zip -> v2 windows)"
    (prepared / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print("v2 windows:", counts)
    return counts


def stage_bundle(drive_dir, local_dir, name=BUNDLE):
    """Copy + unzip the data bundle to fast local disk; verify checksums.
    Returns the path of the `prepared/` data folder.

    v1 `cnn78u_colab_bundle.zip` dropped every window whose next-bar return was exactly zero, a
    selection on future data. v2 keeps them (same bars, more windows); pass the v1 name only to
    reproduce runs made before 2026-11-14."""
    drive_dir, local_dir = Path(drive_dir), Path(local_dir)
    bundle = drive_dir / name
    if name == BUNDLE and not bundle.is_file() and (drive_dir / V1).is_file():
        # No v2 zip on Drive: unpack v1 and rebuild its window lists without the zero filter.
        print(f"{name} not found; deriving it from {V1} (same bars, all windows kept)")
        prepared = stage_bundle(drive_dir, local_dir, V1)
        rescreen_all_windows(prepared)
        return prepared
    if not bundle.is_file():
        raise FileNotFoundError(f"Put {name} in {drive_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    if not (local_dir / "prepared" / "manifest.json").exists():
        tmp = local_dir / "bundle.zip"
        shutil.copy2(bundle, tmp)
        with zipfile.ZipFile(tmp) as z:
            for info in z.infolist():                       # refuse paths escaping local_dir
                assert (local_dir / info.filename).resolve().is_relative_to(local_dir.resolve())
            z.extractall(local_dir)
        tmp.unlink()
        manifest = local_dir / "bundle_manifest.json"
        if manifest.exists():
            for rel, d in json.loads(manifest.read_text())["files"].items():
                assert hashlib.sha256((local_dir / rel).read_bytes()).hexdigest() == d["sha256"], rel
    return local_dir / "prepared"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def make_loader(dataset, batch_size, workers, shuffle, seed=0):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers,
              pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
              generator=torch.Generator().manual_seed(seed))
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **kw)


def mirror(path, target_dir):
    """Atomically copy a local file into a (Drive) folder."""
    if target_dir is None:
        return
    target_dir = Path(target_dir); target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / (Path(path).name + ".copying")
    shutil.copy2(path, tmp); os.replace(tmp, target_dir / Path(path).name)


def save_json(path, value, target_dir=None):
    Path(path).write_text(json.dumps(value, indent=2, default=float) + "\n")
    mirror(path, target_dir)
