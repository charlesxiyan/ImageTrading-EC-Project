# ImageTrading-EC-Project

Chart-image models for five-minute US equity bars, and the data pipeline that feeds them.
Code and documentation only: data, prepared samples, run artifacts and model weights stay
local and are ignored by git.

## Layout

| Path | What it is |
|---|---|
| `CNN-Relevant/` | Chart CNN baseline (plain and residual/ResNet variants), notebook, diagnostics. See its README. |
| `Pretrained Model Comparison/` | Comparison of pretrained time-series foundation models on the same samples. See its README. |
| `build_dynamic_universe.py` | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 universe from WRDS (CRSP/Compustat). |
| `download_intraday_bars.py`, `validate_intraday_demo.py`, `inspect_wrds_demo.py` | Five-minute OHLCV bars aggregated from TAQ trades, and their checks. See `INTRADAY_DEMO.md`. |
| `adjust_bar_prices.py` | Split / corporate-action price adjustment of the bars (CRSP `cfacpr`). |
| `fetch_distribution_inputs.py`, `adjust_dividend_bars.py`, `validate_dividend_bars.py` | Cash-dividend-inclusive price and share-volume adjustment. |
| `us_pipeline/`, `configs/us_5min.json` | CNN78u sample preparation, image rendering and training. See `US_PIPELINE.md`. |
| `tests/` | Offline unit tests for the pipeline. |

Documentation: [DATA_PIPELINE.md](DATA_PIPELINE.md) (universe, bar standard and price
adjustment), [INTRADAY_DEMO.md](INTRADAY_DEMO.md) (TAQ bar download),
[US_PIPELINE.md](US_PIPELINE.md) (CNN78u pipeline).

## Setup

```bash
python -m pip install -r requirements-pipeline.txt   # plus wrds / psycopg2 for WRDS access
python -m unittest discover -s tests -v
```

Data defaults live under `~/Market Data/`. WRDS scripts take the username from
`WRDS_USERNAME` (else the local login name); credentials are handled by the WRDS client and
are never stored in this repository.
