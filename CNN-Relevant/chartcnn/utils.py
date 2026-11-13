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


def stage_bundle(drive_dir, local_dir):
    """Copy + unzip `cnn78u_colab_bundle.zip` to fast local disk; verify checksums.
    Returns the path of the `prepared/` data folder."""
    drive_dir, local_dir = Path(drive_dir), Path(local_dir)
    bundle = drive_dir / "cnn78u_colab_bundle.zip"
    if not bundle.is_file():
        raise FileNotFoundError(f"Put cnn78u_colab_bundle.zip in {drive_dir}")
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
