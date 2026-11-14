# ImageTrading-EC-Project

Chart-image models for five-minute US equity bars, and the data pipeline that feeds them.
Code and documentation only: data, prepared samples, run artifacts and model weights stay
local and are ignored by git.

## Layout

| Path | What it is |
|---|---|
| `CNN-Relevant/` | Chart CNN baseline (plain and residual/ResNet variants) and a 1D CNN on raw OHLCV, executed notebook, diagnostics and run results. See its README. |
| `Pretrained Model Comparison/` | Pretrained time-series foundation models (Chronos-2, TimesFM, Moirai, Kronos) against the chart CNN on the same samples, executed notebook and results. See its README. |
| `build_dynamic_universe.py` | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 universe from WRDS (CRSP/Compustat). |
| `download_intraday_bars.py`, `validate_intraday_demo.py`, `inspect_wrds_demo.py` | Five-minute OHLCV bars aggregated from TAQ trades, and their checks. See `INTRADAY_DEMO.md`. |
| `adjust_bar_prices.py` | Split / corporate-action price adjustment of the bars (CRSP `cfacpr`). |
| `fetch_distribution_inputs.py`, `adjust_dividend_bars.py`, `validate_dividend_bars.py` | Cash-dividend-inclusive price and share-volume adjustment. |
| `us_pipeline/`, `configs/us_5min.json` | CNN78u sample preparation, image rendering and training. See `US_PIPELINE.md`. |
| `tests/` | Offline unit tests for the pipeline. |

Documentation: [DATA_PIPELINE.md](DATA_PIPELINE.md) (universe, bar standard and price
adjustment), [INTRADAY_DEMO.md](INTRADAY_DEMO.md) (TAQ bar download),
[US_PIPELINE.md](US_PIPELINE.md) (CNN78u pipeline).

## Latest results (2023 H1 data, April 2023 validation)

| model | parameters | `vol_h1` AUC | `xsdir_h12` AUC |
|---|---|---|---|
| Chart CNN, plain | 776,450 | 0.687 | 0.508 |
| Chart CNN, residual | 2,047,362 | 0.685 | 0.499 |
| 1D CNN on raw OHLCV (`series1d`) | 720,770 | 0.690 | 0.495 |
| HGB on 20 hand features | 14,762-18,300 tree nodes | 0.705 | 0.503 |

Volatility (`vol_h1`) is learnable, but the chart image adds nothing over the raw numbers and
hand features do best. Cross-sectional direction (`xsdir_h12`) has no signal. Zero-shot
foundation models (11M-331M parameters) do worse than the small trained models on `vol_h1`;
Kronos, at 0.699 on a validation subsample, is the closest. Details, intervals and run IDs are in
[CNN-Relevant/README.md](CNN-Relevant/README.md) and
[Pretrained Model Comparison/README.md](Pretrained%20Model%20Comparison/README.md).

## Setup

```bash
python -m pip install -r requirements-pipeline.txt   # plus wrds / psycopg2 for WRDS access
python -m unittest discover -s tests -v
```

Data defaults live under `~/Market Data/`. WRDS scripts take the username from
`WRDS_USERNAME` (else the local login name); credentials are handled by the WRDS client and
are never stored in this repository.
