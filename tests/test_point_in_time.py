"""Model inputs must be adjusted to each window's own anchor (its last input bar), never to the
end of the data. The stored bars are on the end-of-data basis with per-bar multipliers; dividing a
window by its anchor bar's multipliers must reproduce the point-in-time formula exactly:

    price  = raw[d] * cfacpr[a] / cfacpr[d] * prod(1 - D[e] / P[e-1] for d < e <= a)
    volume = raw[d] * cfacshr[d] / cfacshr[a]
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adjust_dividend_bars import apply_adjustments, prepare_adjustments

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    """Import one module by path (chartcnn/__init__ needs torch; images.py only needs numpy)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASES = [load("us_images", "us_pipeline/images.py").prediction_basis,
         load("chartcnn_images", "CNN-Relevant/chartcnn/images.py").prediction_basis]
POINT_IN_TIME = load("chartcnn_images", "CNN-Relevant/chartcnn/images.py").point_in_time_window

DAYS = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]
RAW_CLOSE = np.array([100., 99., 50., 49., 48.])   # 2-for-1 split on day 2
CFACPR = np.array([2., 2., 1., 1., 1.])
CFACSHR = np.array([2., 2., 1., 1., 1.])
RAW_VOLUME = np.array([1000., 1100., 2400., 2300., 2200.])


def inputs(dividends):
    """dividends: {ex-date index: cash per share}."""
    f = pd.DataFrame({"permno": 1, "date": DAYS, "cfacpr": CFACPR, "cfacshr": CFACSHR})
    r = f.copy()
    r["prc"] = RAW_CLOSE
    e = pd.DataFrame([dict(permno=1, distcd=1232, divamt=cash, facpr=0., facshr=0., exdt=DAYS[i])
                      for i, cash in dividends.items()],
                     columns=["permno", "distcd", "divamt", "facpr", "facshr", "exdt"])
    return f, e, r


def stored_bars(dividends):
    """One bar per day, adjusted exactly as the pipeline stores it (end-of-data basis)."""
    factors = prepare_adjustments(*inputs(dividends))[0]
    bars = pd.DataFrame({"date": DAYS, "permno": 1, "ticker": "X", "status": "observed",
                         "open": RAW_CLOSE, "high": RAW_CLOSE + 1, "low": RAW_CLOSE - 1,
                         "close": RAW_CLOSE, "vwap": RAW_CLOSE, "volume": RAW_VOLUME, "trade_count": 1})
    out = pd.concat([apply_adjustments(bars.iloc[[i]].reset_index(drop=True), factors) for i in range(len(DAYS))])
    x = out[["open", "high", "low", "close", "volume"]].to_numpy(float)
    m = out[["price_adjustment_multiplier", "volume_adjustment_multiplier"]].to_numpy(float)
    return x, m


def policy_close(anchor, dividends):
    """The CLAUDE.md formula, computed from raw prices and CRSP inputs only."""
    out = []
    for d in range(anchor + 1):
        mult = CFACPR[anchor] / CFACPR[d]
        for e, cash in dividends.items():
            if d < e <= anchor:
                prior = RAW_CLOSE[e - 1] * CFACPR[e] / CFACPR[e - 1]   # prior close in ex-date share units
                mult *= 1 - cash / prior
        out.append(RAW_CLOSE[d] * mult)
    return np.array(out)


class PointInTimeTests(unittest.TestCase):
    DIVIDENDS = {1: 1.0, 3: 0.5}            # one before the split, one after

    def test_window_matches_anchor_formula(self):
        x, m = stored_bars(self.DIVIDENDS)
        for anchor in range(len(DAYS)):
            for basis in BASES:
                w = basis(x[:anchor + 1], m[anchor])
                np.testing.assert_allclose(w[:, 3], policy_close(anchor, self.DIVIDENDS), rtol=1e-12)
                np.testing.assert_allclose(w[:, 4], RAW_VOLUME[:anchor + 1] * CFACSHR[:anchor + 1] / CFACSHR[anchor],
                                           rtol=1e-12)

    def test_anchor_bar_keeps_its_traded_price(self):
        x, m = stored_bars(self.DIVIDENDS)
        for anchor in range(len(DAYS)):
            w = POINT_IN_TIME(x, m, anchor, anchor + 1)
            self.assertAlmostEqual(w[-1, 3], RAW_CLOSE[anchor], places=10)
            self.assertAlmostEqual(w[-1, 4], RAW_VOLUME[anchor], places=8)

    def test_events_after_the_anchor_change_nothing(self):
        x_without, m_without = stored_bars({1: 1.0})
        x_with, m_with = stored_bars({1: 1.0, 3: 0.5, 4: 2.0})   # extra dividends after anchor 2
        for anchor in (0, 1, 2):
            np.testing.assert_allclose(POINT_IN_TIME(x_with, m_with, anchor, anchor + 1),
                                       POINT_IN_TIME(x_without, m_without, anchor, anchor + 1), rtol=1e-12)
        # ...while the stored end-of-data prices do move, which is why they are never used directly.
        self.assertFalse(np.allclose(x_with[:3, 3], x_without[:3, 3]))

    def test_future_split_does_not_leak_into_price_level(self):
        x, m = stored_bars({})
        w = POINT_IN_TIME(x, m, 1, 2)          # anchor before the split on day 2
        np.testing.assert_allclose(w[:, 3], RAW_CLOSE[:2], rtol=1e-12)   # pre-split dollars, not halved
        self.assertAlmostEqual(x[0, 3], RAW_CLOSE[0] / 2)                # the stored series is halved


if __name__ == "__main__":
    unittest.main()
