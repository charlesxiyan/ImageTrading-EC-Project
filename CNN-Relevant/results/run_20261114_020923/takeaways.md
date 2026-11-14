**vol_h1** — best CNN is CNN (plain): AUC 0.687 [0.676, 0.698], above a coin flip.
- Residual − plain AUC: -0.0024 [-0.0050, +0.0008] → no significant difference (residual has 2.6× the parameters).
- CNN (plain) − 1D CNN on raw OHLCV (AUC 0.690): -0.0033 [-0.0078, +0.0009] → no significant difference: the picture adds nothing measurable.
- CNN (plain) − HGB AUC: -0.0171 [-0.0216, -0.0121] → HGB beats CNN.
- Probabilities: log loss +11.4% vs training prior; mean predicted − observed = +0.049 (calibration shift between train and validation).

**xsdir_h12** — best CNN is CNN (plain): AUC 0.508 [0.502, 0.514], above a coin flip.
- Residual − plain AUC: -0.0093 [-0.0167, -0.0013] → plain is better (residual has 2.6× the parameters).
- CNN (plain) − 1D CNN on raw OHLCV (AUC 0.495): +0.0126 [+0.0027, +0.0217] → the image helps.
- CNN (plain) − HGB AUC: +0.0048 [-0.0037, +0.0145] → CNN and HGB are tied.
- Probabilities: log loss -0.2% vs training prior; mean predicted − observed = +0.002.
- Top-minus-bottom decile: -0.34 bps per 12 bars, t = -0.20 (gross; |t| < 2 means no reliable spread).