"""Bundle v2 and label rules.

1. v2 keeps every window: no window may be dropped because its next-bar return is zero, which would
   select windows on future data. The Colab fallback (chartcnn.utils.rescreen_all_windows, v1 zip
   -> v2 windows) must give exactly the windows of us_pipeline.data.sample_indices with
   drop_zero_returns=False, which built the v2 zip.
2. Labels use unadjusted prices, and a label whose horizon contains an ex-date is dropped
   (chartcnn.data.build_split).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "CNN-Relevant")]

try:
    import torch  # noqa: F401  (chartcnn imports torch)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from us_pipeline.data import sample_indices

STOCKS, DAYS, SLOTS = 3, 6, 78


def fake_bundle(folder, zero_every=7, split_day=None):
    """A tiny v1-style prepared folder: 3 stocks x 6 sessions of 5-minute bars."""
    rng = np.random.default_rng(0)
    t = DAYS * SLOTS
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (STOCKS, t)), axis=1))
    close[:, ::zero_every] = np.roll(close, 1, axis=1)[:, ::zero_every]      # some zero returns
    features = np.repeat(close[:, :, None], 5, axis=2)
    features[:, :, 1] *= 1.001; features[:, :, 2] *= 0.999; features[:, :, 4] = 1000.0
    valid = np.ones((STOCKS, t), bool)
    valid[1, 100:110] = False                                                   # a gap
    mult = np.ones((STOCKS, t, 2))
    if split_day is not None:                       # 2:1 split on stock 0 at the open of split_day
        mult[0, :split_day * SLOTS, 0] = 0.5        # stored = raw * M; history halves
    days = pd.bdate_range("2023-03-27", periods=DAYS)
    clock = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(f"{d.date()} 09:35", tz="America/New_York"), periods=SLOTS, freq="5min")
        for d in days]))
    config = dict(lookback=78, horizon=1, stride=1, bar_minutes=5, allow_cross_session=True,
                  zero_return_epsilon=0.0, train_end=str(days[2].date()), validation_end=str(days[4].date()))
    permnos = [10001, 10002, 10003]
    v1, stats = sample_indices(valid, features[:, :, 3], clock, config, permnos)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "features.npy", features); np.save(folder / "valid.npy", valid)
    np.save(folder / "multipliers.npy", mult)
    for name, frame in v1.items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    manifest = dict(config=dict(config, lookback=78), permnos=permnos, decision_clock_ns=clock.as_unit("ns").asi8.tolist(),
                    splits={k: {"samples": len(v)} for k, v in v1.items()})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return features, valid, clock, config, permnos, stats


class BundleV2(unittest.TestCase):
    def test_pipeline_keeps_zero_return_windows(self):
        with tempfile.TemporaryDirectory() as d:
            features, valid, clock, config, permnos, v1_stats = fake_bundle(Path(d))
            v2, stats = sample_indices(valid, features[:, :, 3], clock, dict(config, drop_zero_returns=False), permnos)
            self.assertGreater(v1_stats["zero_return"], 0)
            self.assertEqual(stats["zero_return"], 0)
            self.assertEqual(stats["kept"], v1_stats["kept"] + v1_stats["zero_return"])

    @unittest.skipUnless(HAVE_TORCH, "chartcnn needs torch")
    def test_colab_fallback_matches_pipeline(self):
        from chartcnn.utils import rescreen_all_windows
        with tempfile.TemporaryDirectory() as d:
            features, valid, clock, config, permnos, _ = fake_bundle(Path(d))
            expected, _ = sample_indices(valid, features[:, :, 3], clock, dict(config, drop_zero_returns=False), permnos)
            rescreen_all_windows(Path(d))
            for name, frame in expected.items():
                got = pd.read_parquet(Path(d) / f"{name}.parquet")
                pd.testing.assert_frame_equal(got.reset_index(drop=True), frame.reset_index(drop=True),
                                              check_dtype=False)

    @unittest.skipUnless(HAVE_TORCH, "chartcnn needs torch")
    def test_labels_unadjusted_and_ex_date_horizons_dropped(self):
        from chartcnn.data import Store, build_split
        with tempfile.TemporaryDirectory() as d:
            features, *_ = fake_bundle(Path(d), split_day=4)
            from chartcnn.utils import rescreen_all_windows
            rescreen_all_windows(Path(d))
            store = Store(Path(d))
            frame, _ = build_split(store, "validation", "vol", 12, 1)
            s, t = frame.security_index.to_numpy(), frame.end_index.to_numpy()
            crosses = (s == 0) & (t < 4 * SLOTS) & (t + 12 >= 4 * SLOTS)
            self.assertFalse(crosses.any(), "a label spanning the ex-date was kept")
            raw = features[:, :, 3] / np.load(Path(d) / "multipliers.npy")[:, :, 0]
            np.testing.assert_allclose(frame.forward_return, raw[s, t + 12] / raw[s, t] - 1, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
