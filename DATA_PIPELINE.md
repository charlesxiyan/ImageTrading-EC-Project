# Data pipeline: universe, intraday bars and price adjustment

Moved from the README of ImageTrading-CV-Project. Commands assume the repository root.

## Current intraday data standard

The 2023 H1 demo contains five-minute OHLCV derived from TAQ trades by server-side SQL aggregation. It is not bid/ask data or a reconstructed order book.

- Use each date's historical S&P 500 / NASDAQ 100 / DJIA 30 union, identified by PERMNO. Preserve historical ticker and TAQ symbol root/suffix separately.
- Use New York regular-session timestamps and half-open intervals in [09:30,16:00). Bars are labeled by start time; complete bar information is available after the interval ends.
- Require positive trade price and size, correction code `00`, and sale conditions containing only spaces, `@`, `E`, or `F`. NULL conditions, odd lots, special auctions, extended-hours, and other conditions outside this whitelist are excluded.
- OHLC, VWAP, volume, and trade count use the same eligible subset. Volume is filtered eligible share volume, not total tape volume.
- Preserve empty intervals as missing prices with zero eligible volume/count; do not forward-fill. Preserve halted index members with explicit identifier/mapping status.
- Raw prices remain unadjusted in the source directory; CRSP daily factors are stored separately. File checksums, membership coverage, interval grids, unique keys, and OHLC ordering have been checked.
- This is basic screening, not comprehensive cleaning: quote-based outlier checks and full cancellation/correction reconstruction remain outstanding.

The downloaded dataset covers 124 trading dates with 5,057,130 interval records, including 224,156 empty intervals (4.43%). See its manifest and quality report for exact provenance.

## Latest-basis historical price adjustment

The section below describes the earlier corporate-action-only version. For the current cash-dividend-inclusive price and share-volume version, see [Cash-dividend-inclusive price and volume](#cash-dividend-inclusive-price-and-volume).

`adjust_bar_prices.py` adjusts historical OHLC and VWAP to each PERMNO's last available factor date at or before the final input date:

```text
adjusted_price[t] = raw_price[t] * CFACPR[base] / CFACPR[t]
```

This is the requested forward adjustment of historical prices to the latest available price basis. For a 2-for-1 split, pre-split prices are multiplied by 0.5 and prices on the anchor date remain unchanged. The base factor need not equal 1; the ratio cancels factor components belonging to events after the chosen basis. Reverse splits can increase historical prices, so the multiplier is not artificially capped at 1.

**Scope:** CRSP split/distribution adjustment, including applicable stock splits, stock dividends, spin-offs, and rights factors. CFACPR does **not** provide ordinary cash-dividend reinvestment; the output is not a cash-dividend-adjusted total-return series. [CRSP methodology](https://www.crsp.org/wp-content/uploads/guides/CRSP_Calculations_and_Index_Methodologies.pdf).

```bash
# Inspect and validate every source file without writing output.
python adjust_bar_prices.py --dry-run

# Create a separate adjusted dataset; never overwrite the raw bars.
python adjust_bar_prices.py
```

Defaults read `~/Market Data/demo_2023_h1` and write a new sibling directory named `demo_2023_h1_price_adjusted`. Use `--input-root` and `--output` to change paths. Existing output directories are rejected to prevent accidental overwrites or double adjustment.

Adjusted files replace only `open/high/low/close/vwap`. Original eligible volume, trade count, identifiers, timestamps, status, and missingness are preserved. Daily CFACPR, the anchor date/factor, and the applied multiplier are added for auditing. `adjustment_anchors.parquet` records each security's actual anchor, including earlier anchors for securities that exited or stopped trading. Missing exact-day factors for any available price cause an error; missing factors on wholly empty rows remain explicit and are never filled from a future date.

The output manifest records the method, source checksums, per-file checksums, row counts, and affected PERMNOs. Latest-basis historical prices are a research representation, not an archive of as-published quotes. Model inputs rebase each window to its own anchor; see [Point-in-time model inputs](#point-in-time-model-inputs-adjust-history-only-never-the-future).

## Cash-dividend-inclusive price and volume

`fetch_distribution_inputs.py` retrieves CRSP distribution events (`crsp_a_stock.dsedist`) and daily reference prices/factors (`crsp_a_stock.dsf`) for the raw demo's PERMNOs. It includes 60 calendar days of preceding daily references, so an early-period ex-date can use a prior close. Connections use SSL and the user's local PostgreSQL authentication; no passwords are stored in source files.

`adjust_dividend_bars.py` starts from the original raw bars, not the previously adjusted output. It combines the CFACPR price ratio with proportional cash-dividend factors. Each same-day group of eligible cash dividends is summed, and the previous CRSP daily close is converted to ex-date share units first:

```text
reference_close = abs(previous_close) * CFACPR[exdate] / CFACPR[previous_date]
cash_event_factor = 1 - sum(cash_dividends_on_exdate) / reference_close
cash_multiplier[t] = product(cash_event_factor for t < exdate <= security_anchor)
adjusted_price[t] = raw_price[t] * CFACPR[anchor] / CFACPR[t] * cash_multiplier[t]
adjusted_volume[t] = raw_volume[t] * CFACSHR[t] / CFACSHR[anchor]
```

The dividend is applied strictly before its ex-date, not its payment date. Prices on each security's anchor date are unchanged. The cash factor never scales volume: share-volume adjustment reflects stock splits and stock dividends only. Adjusted share volume can be fractional and is stored as floating point; `volume_raw` retains the actual eligible integer share count. `trade_count` remains the original eligible print count. Original OHLC and VWAP are retained in `open_raw/high_raw/low_raw/close_raw/vwap_raw`.

The demo adapter explicitly classifies the verified USD cash codes 1212, 1222, 1232, 1233, 1242, 1252, and 1272, including special cash dividends. Corporate event codes 3763 and 5523 are already represented by the cumulative corporate factors and are not counted again as cash dividends. Unsupported relevant codes, a cash event with nonzero corporate factors, invalid dividend/reference values, duplicate events, or inconsistent factor vintages cause an error. Extend classification before using other periods; do not silently treat every positive `DIVAMT` as cash. [CRSP distribution code definitions](https://www.crsp.org/wp-content/uploads/DistributionCode.html).

```bash
# Fetch inputs once into a new directory.
python fetch_distribution_inputs.py

# Generate a new adjusted dataset from raw bars and those inputs.
python adjust_dividend_bars.py

# Verify every saved file and formula against raw bars.
python validate_dividend_bars.py
```

Defaults store inputs in `~/Market Data/demo_2023_h1_distribution_inputs` and adjusted data in `~/Market Data/demo_2023_h1_dividend_adjusted`. Existing directories are rejected. The raw and earlier corporate-action-only datasets remain intact.

The new output includes daily bars, `daily_adjustment_factors.parquet`, `adjustment_anchors.parquet`, `distribution_audit.parquet`, `cash_dividend_factors.parquet`, and a checksum manifest. Every source distribution receives a treatment flag; events outside a security's adjustable observation span are recorded but do not change its history. Negative CRSP daily PRC values are interpreted using their absolute value, with the quote-reference flag retained in the cash-event audit. Missing-price intervals are never filled.

The completed 2023 H1 run processed 813 distribution records: 807 applicable cash records, 3 corporate events handled by cumulative factors, and 3 records outside the security's adjustable history. Cash records form 800 security/ex-date groups across 405 securities. The result contains 124 files and 5,057,130 rows; 2,892,281 price rows receive additional cash adjustment, and 6,474 volume rows receive share adjustment (PCAR and MNST). All 224,156 missing-close intervals are preserved. Nine targeted dividend-adjustment tests passed, and all output files passed the independent read-back validation recorded in `validation_report.json`.

This is proportional dividend-adjusted chart data: it removes the theoretical cash-dividend price step while preserving actual market movements. It is **not** an exact cash-dividend reinvestment total-return index, and adjusted prices multiplied by share-adjusted volume are not historical dollar turnover. Use raw prices and `volume_raw` for historical transaction amounts. All original TAQ filters and incomplete-cleaning limitations remain applicable. A point-in-time forecasting pipeline must use the factor basis available at each prediction time.


## Point-in-time model inputs (adjust history only, never the future)

Model inputs are adjusted to the anchor session, the session of each window's last input bar,
never to the end of the data. Only one copy of the data is stored: the dividend-adjusted bars
keep `*_raw` columns and the per-bar multipliers, and `us_pipeline` writes `features.npy`
(end-of-data basis, `raw * M`) together with `multipliers.npy` (`M` for price, `V` for volume).
When a window is cut, every bar d is divided by the anchor bar's multipliers:

```text
price[d]  = raw[d] * M[d] / M[a] = raw[d] * CFACPR[a] / CFACPR[d] * product(1 - D[e] / P[e-1], d < e <= a)
volume[d] = raw[d] * V[d] / V[a] = raw[d] * CFACSHR[d] / CFACSHR[a]
```

- The anchor bar keeps its traded price and volume.
- An event with an ex-date after the anchor cancels out, so it cannot affect the window.
- Most windows (78 bars, about one session) contain no event, so their prices are the raw prices.
- Stored levels are never used directly. Returns, hand features and the close series given
  to foundation models only use ratios between two bars up to the anchor, which equal the
  point-in-time ratios.
- **Labels** use unadjusted (traded) prices: `raw[t+H] / raw[t] = stored[t+H] / stored[t] * M[t] / M[t+H]`.
  Inside one session no ex-date can fall between the two bars. A horizon that crosses the overnight
  gap into an ex-date session is dropped (`chartcnn.data.build_split`), because its unadjusted return
  would contain the split or dividend jump: about 0.02% of windows at H=1 and 0.2% at H=12.

Code: `prediction_basis` / `point_in_time_window` in `CNN-Relevant/chartcnn/images.py` (chart,
1D CNN and Kronos inputs) and `prediction_basis` in `us_pipeline/images.py`.
`tests/test_point_in_time.py` checks the formula against raw CRSP inputs, that the anchor keeps its
traded price, and that dividends or splits after the anchor change nothing.

### Window screening: bundle v2

The first prepared bundle (`artifacts/cnn78u_5min_h1`, `cnn78u_colab_bundle.zip`) dropped every
window whose next-bar return was exactly zero (163,566 windows, 4.6%). That selects windows on
future data and biases every other target built from them (volatility, cross-sectional direction,
longer horizons). Bundle v2 keeps them: same bars, `drop_zero_returns: false` in
`configs/us_5min_v2.json`. Direction targets drop zero returns themselves, per target and horizon.

```bash
# Rebuild the window lists from an existing prepared folder (bars are unchanged) and audit them.
python -m us_pipeline rescreen --config configs/us_5min_v2.json --source artifacts/cnn78u_5min_h1/prepared
python -m us_pipeline audit --config configs/us_5min_v2.json
python colab/build_colab_bundle.py --version v2          # colab/cnn78u_colab_bundle_v2.zip
```

v2 has 1,796,575 / 537,272 / 1,248,499 train / validation / test windows; every v1 window is in it
with the same return, and the added ones all have a zero next-bar return. On Colab, if only the v1
zip is on Drive, `chartcnn.utils.stage_bundle` derives the v2 windows from it at start-up
(`rescreen_all_windows`, identical output; `tests/test_bundle_v2.py`).

## Dynamic US equity universe

`build_dynamic_universe.py` builds the daily **S&P 500 ∪ NASDAQ 100 ∪ Dow Jones Industrial Average 30** security universe for an inclusive date range. Each date receives the tickers effective on that historical date. Securities appearing in multiple indices are included once; distinct share classes remain separate securities.

### Installation and execution

Use Python 3.9+ on a machine with WRDS access or in WRDS JupyterHub:

```bash
python -m pip install wrds
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --inspect
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --start 2020-01-01 --end 2020-12-31
```

The official WRDS Python client handles authentication. Do not store passwords in source files. `--inspect` reads metadata and prints candidate table columns, NASDAQ/Dow index names and IDs, and visible related tables, including TAQ tables. Visibility alone does not establish full subscription access or data quality; actual queries also check read permissions.

If the index name cannot be resolved uniquely, select its `index_id` from the **exact index name** in the inspection report, preserving leading zeros:

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --ndx100-id VERIFIED_NDX_GVKEYX --djia30-id VERIFIED_DJIA_GVKEYX
```

These IDs are placeholders and must be replaced. Do not select the NASDAQ Composite, a NASDAQ financial index, or the Dow Jones Transportation Average. The script does not guess fixed IDs or substitute QQQ/DIA holdings for index constituents.

The default includes **every calendar day**, including weekends, using the membership effective on each date. For trading dates only:

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --days trading
```

Trading dates come from dates actually present in CRSP `dsi`, rather than a Monday-through-Friday calendar. Requests outside the available calendar range are rejected, including an end date beyond the database's latest date.

### Outputs and Python usage

Outputs default to this project's `data/universe/` directory. Repeated runs overwrite files with the same names; use a separate `--output` directory for each research interval.

- `daily_tickers.json`: a sorted, deduplicated ticker list per date, such as `{"2020-01-02": ["AAPL", "..."], ...}`.
- `daily_universe.csv`: `date,permno,ticker,indices`, with one row per date and PERMNO. Index memberships are separated by `|`.
- `manifest.json`: source tables, index IDs, requested dates, generation time, and daily security counts for each index and their union.

```python
import json
from pathlib import Path

# Run from the project root.
pools = json.loads(Path('data/universe/daily_tickers.json').read_text())
tickers = pools['2020-01-02']
```

All default output paths are within this repository.

### Sources and limitations

1. **S&P 500**: tries `crsp_a_indexes.dsp500list`, then `crsp.dsp500list`; requires `permno/start/ending`.
2. **NASDAQ 100 and DJIA 30**: tries `comp.idxcst_his`, then `comp_na_daily_all.idxcst_his`; uses `gvkey/iid/gvkeyx/from/thru`. Index names are resolved through `idx_index`, requiring `gvkeyx` (or `gvkey`) and `conm` (or `conmx`). **These are candidate adapters; historical coverage for these indices must be verified in your account.**
3. Compustat constituents are mapped from `gvkey+iid` to PERMNO through the historical CCM `ccmxpf_linktable`, using security-level links, effective dates, and link types `LC/LU/LS`. Ambiguous mappings cause an error; the script does not arbitrarily select a link or substitute the company's current primary security.
4. PERMNO is mapped to the **historical ticker** through CRSP `dsenames` / `msenames` and its `namedt/nameendt` intervals. Current `comp.security.tic` values are not used to backfill history. Punctuation and share-class notation are preserved.
5. Date intervals are inclusive. NULL end dates are treated as ongoing; NULL membership start dates are not backfilled. Other interval conventions or schemas require an adapter based on the relevant data dictionary.

Override table names with `--sp500-table`, `--members-table`, `--headers-table`, `--links-table`, `--names-table`, or `--calendar-table`. Names must use the `schema.table` format, and required columns are validated. If your account only exposes newer CRSP CIZ tables, the script reports a table or column mismatch. **Automatic CIZ conversion is not implemented**; add an adapter using your account's actual data dictionary rather than merely renaming a table.

Before writing outputs, the script checks that all three indices are nonempty on every requested date, each constituent maps to a historical ticker, and mappings are unambiguous. **Nonempty results do not prove source completeness or freshness.** Historical records may be revised, vendor updates may lag, and NULL end dates may reflect stale records. Verify WRDS update dates and historical coverage, and inspect daily counts and abrupt changes in `manifest.json`. Counts are not fixed at 500/100/30 because company counts and security counts can differ.

A WRDS account does not grant access to every underlying database. WRDS states that historical S&P 500 constituent access requires a CRSP subscription. Compustat's licensing for S&P index constituents has changed, so `idxcst_his` cannot be assumed to cover every index. Missing indices, dates without membership records, missing links, or missing tickers prevent new outputs from being generated. Resolve subscription or coverage issues first. The script does not backfill with current constituents or silently reduce the universe to S&P 500 alone. Failed runs leave existing output files untouched; check the manifest before using an earlier result.

The result reconstructs membership by **historical effective date**. It is not an archive of information as originally published. Backtests must still respect signal timing, order timing, and announcement availability. Membership alone does not establish tradability; suspensions, price availability, and liquidity require separate filters. Do not exclude historical members based on future delisting information.

### High-frequency data integration

This step does not download intraday prices or depend on an unverified TAQ schema. Before integrating high-frequency data, confirm NYSE TAQ subscription access, available years, trades/quotes versus aggregated bars, timestamps, trade conditions, and cancellation/correction fields. **A CRSP historical ticker is not a universal join key for TAQ tables.** Map it by date and share class to the actual TAQ symbol root/suffix or other relevant fields. PERMNO is retained for security identification and auditing; this does not imply that TAQ tables contain PERMNO.

### Validation

```bash
python -m unittest discover -s tests -v
```

Offline tests cover deduplication, index memberships, historical ticker changes, weekend retention, missing indices/mappings, ambiguous links, ticker collisions, output serialization, and SQL identifier validation. Live WRDS access, source columns, and the three index histories have now been checked for the 2023 H1 intraday demo; see [INTRADAY_DEMO.md](INTRADAY_DEMO.md) and the data manifest for its actual completion status. This does not establish coverage for other periods. The standalone universe builder remains strict about inactive historical names; the demo downloader separately audits last-known identifiers for halted members rather than silently dropping them.

References:

- [WRDS: downloading historical S&P 500 constituents and CRSP subscription requirements](https://wrds-www.wharton.upenn.edu/pages/grid-items/downloading-the-sp-500-constituents/)
- [WRDS: historical S&P 500 constituent changes](https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/)
- [CRSP: official CCM documentation, hosted by WRDS](https://wrds-www.wharton.upenn.edu/documents/415/CRSPAccess_Software_Guide_CCM_part.pdf)
