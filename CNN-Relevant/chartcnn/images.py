"""Chart rendering: one 78-bar OHLCV window -> one 234x234 grayscale image.

Same geometry as `us_pipeline/images.py` (the CNN78u pipeline), reproduced here so the
package is self-contained.

Layout (each bar is 3 pixels wide):
    column 3i     : open tick
    column 3i + 1 : high-low bar (white) and, in the bottom band, volume (gray)
    column 3i + 2 : close tick
The price axis is window-local: the window's lowest low is the bottom of the price
area and its highest high is the top. Volume is scaled by the window's maximum volume.
"""
import numpy as np

PIXELS_PER_BAR = 3


def render_window(window: np.ndarray, include_volume: bool = True) -> np.ndarray:
    """Render an (L, 5) OHLCV array into a uint8 image of shape (3L, 3L)."""
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("Expected a complete, positive OHLCV window")

    length = len(x)
    size = PIXELS_PER_BAR * length
    volume_height = max(1, size // 5) if include_volume else 0
    price_height = size - volume_height - (1 if include_volume else 0)   # 1-pixel gap
    image = np.zeros((size, size), dtype=np.uint8)

    # Map prices to rows (row 0 = top of the image).
    low, high = float(x[:, 2].min()), float(x[:, 1].max())
    if high == low:
        rows = np.full((length, 4), price_height // 2, dtype=int)
    else:
        rows = price_height - 1 - np.rint((x[:, :4] - low) / (high - low) * (price_height - 1)).astype(int)

    max_volume = x[:, 4].max()
    for i, (o, h, l, c) in enumerate(rows):
        image[o, 3 * i] = 255                 # open
        image[h:l + 1, 3 * i + 1] = 255       # high-low
        image[c, 3 * i + 2] = 255             # close
        if include_volume:
            bar = int(np.rint(x[i, 4] / max_volume * volume_height))
            if bar:
                image[size - bar:, 3 * i + 1] = 180
    return image


def prediction_basis(window: np.ndarray, multipliers) -> np.ndarray:
    """Undo end-of-sample split/dividend scaling using the window endpoint's own
    multipliers, so the chart only uses information available at decision time."""
    x = np.array(window, dtype=float, copy=True)
    price, volume = np.asarray(multipliers, dtype=float)
    if not np.isfinite([price, volume]).all() or min(price, volume) <= 0:
        raise ValueError("Invalid prediction-time multipliers")
    x[:, :4] /= price
    x[:, 4] /= volume
    return x
