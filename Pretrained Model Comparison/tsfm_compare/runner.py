"""Run one foundation model over the shared samples, with on-disk caching.

Forecasts are saved as .npy (local run folder, mirrored to Drive), so a Colab disconnect
only costs the model that was running. Re-running a cell reuses finished forecasts.
"""
import time
from pathlib import Path

import numpy as np

from chartcnn.utils import mirror

from . import forecasters
from .samples import close_contexts, ohlcv_contexts, slot_times


def make_inputs(store, prepared, keys, context, horizon, needs):
    """Build only the arrays a model asks for (Kronos also needs candles and timestamps)."""
    batch = {"close": close_contexts(store, keys, context)}
    if "ohlcv" in needs:
        batch["ohlcv"] = ohlcv_contexts(prepared, store, keys, context)
    if "t_past" in needs:
        batch["t_past"], batch["t_future"] = slot_times(store, keys, context, horizon)
    return batch


def cached_forecast(model, inputs, horizon, path, drive_dir=None):
    path = Path(path)
    if path.exists():
        print(f"  reuse {path.name}")
        return np.load(path), None
    start = time.perf_counter()
    q = model.forecast(inputs, horizon)
    seconds = time.perf_counter() - start
    np.save(path, q)
    mirror(path, drive_dir)
    return q, seconds


def run_model(spec, store, prepared, keys, context, horizon, device, out_dir, drive_dir=None,
              smoke=False, model=None):
    """Forecast validation and calibration keys with one model.

    Returns dict(name, n_params, q={'valid','calib'}, seconds). Pass `model` to reuse an
    already-loaded wrapper (e.g. the fine-tuned Chronos-2).
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = model or forecasters.build(spec, device).load()
    q, seconds = {}, {}
    for part in ("valid", "calib"):
        inputs = make_inputs(store, prepared, keys[part], context, horizon, model.needs)
        q[part], seconds[part] = cached_forecast(model, inputs, horizon,
                                                 out_dir / f"{model.name}_{part}.npy", drive_dir)
        print(f"  {model.name} {part}: {len(q[part]):,} forecasts"
              + (f" in {seconds[part]:.0f}s" if seconds[part] else ""))
    return dict(name=model.name, n_params=model.n_params, q=q, seconds=seconds["valid"], model=model)
