# Long-range US equity archive

The archive starts on January 1, 2016. Its endpoint is the latest joint coverage of subscribed annual CRSP CIZ security history, daily data, S&P 500 membership, and TAQ. On November 14, 2026, joint coverage ended on December 31, 2025, although TAQ tables extended through November 13, 2026. The later TAQ period is deferred until compatible reference data are available. Coverage is frozen per archive; use a new archive directory for a refreshed snapshot.

Run from `DataDownload/` in the project environment. Policy version 4 archives go to a new directory:

```bash
cd DataDownload
WRDS_USERNAME=<your WRDS user> python -u download_us_equity_archive.py \
  --output ~/"Market Data/US Equity/v4/01_raw"
```

The 2016-2025 `wrds_5min_2016_latest_v1` archive was built with trade-filter policy version 1, which dropped the regular
(blank, NULL in WRDS) sale condition of every non-Nasdaq-listed stock. See `../DATA_PIPELINE.md`. Do not build on it.

Use `--limit-days 1` for a pilot. This still fetches the full membership and corporate-action reference history before downloading one bar partition. Repeat the command without this option to resume. Dependencies include pandas, pyarrow, psycopg2, and pandas_market_calendars. Authentication uses existing PostgreSQL credentials; no credentials are stored in the archive. A process lock prevents concurrent writers. Completed daily partitions are verified against SHA-256 before skipping them. Writes use temporary files followed by atomic replacement. Query failures and unresolved ambiguous mappings stop the run. Missing mappings are retained as explicitly missing security-day bars and audited; they are never matched using ticker alone. A five-letter root without suffix is also queried in its split form (in 2016 the TAQ master writes Nasdaq share classes whole, e.g. `GOOGL`, while the trades table stores `GOOG` + `L`, and the WRDS daily link has no 2016 rows for them), and bars are joined on root and suffix written together (see `DataCheck/SPLIT_SYMBOLS_2026-11-20.md`).

## Directory layout

- `00_metadata`: frozen source coverage, configuration, exchange calendar, source membership intervals, CCM links, security history, and download progress.
- `01_universe/year=YYYY`: daily historical membership by PERMNO, daily ticker lists, and source membership mappings.
- `02_identifier_mapping/year=YYYY`: daily TAQ mappings, original link/master evidence, and unmatched-member audit.
- `03_raw_bars_5min/year=YYYY/month=MM`: one unadjusted Parquet partition per session.
- `04_corporate_actions/year=YYYY`: complete CIZ distribution records for every security ever in the requested universe.
- `05_daily_reference/year=YYYY`: complete CIZ daily records, including cumulative price/share factors, returns with/without dividends, distributions, identifiers, prices, and volume.

Reference histories include periods outside index membership. The first annual reference partition includes 60 calendar days before the requested start to support prior-close calculations. Reference data are not restricted to dates on which a stock is an index member. The output does not yet contain adjusted bars: the old demo distribution-code whitelist must not be applied blindly to CIZ or ten years of corporate actions. Cash-dividend price adjustments must not be used as share-volume adjustments.

## Universe and identifiers

The pool is the daily union of S&P 500, NASDAQ 100, and DJIA 30 constituents, deduplicated by PERMNO. S&P membership comes from CRSP CIZ; NASDAQ and DJIA membership comes from Compustat index IDs 000208 and 000005 with date-effective CCM security links. Membership endpoints are inclusive, following the existing project convention. These historical tables are not an as-published vintage database. Retrospective corrections remain possible.

Daily TAQ symbols use date-effective CIZ CUSIP matches to the daily TAQ master, cross-checked against the WRDS CRSP–TAQ daily link when available. The daily link currently ends in 2024. Symbol changes and share classes are preserved. Daily ticker lists prefer CIZ `tradingsymbol` over the base `ticker` field: the latter can be identical for different share classes. Source fields remain available for auditing. Historical inactive ticker reuse is retained by PERMNO, so the ticker-only list can contain fewer entries than the security table. When several master symbols share a CUSIP, a unique match to the CIZ trading symbol selects the regular security. A link/master disagreement is resolved only when the master mapping is independently confirmed by that trading symbol; all candidates and disagreements are saved. Unresolved ambiguity stops the download. If an expired security and its active successor reuse a TAQ symbol on the same date, the expired security keeps missing bars and an audit record instead of inheriting the successor's prices. Some corporate-action dates have inconsistent effective dates between CRSP and the TAQ master. Unmatched security-days retain `no_symbol_mapping` bars and an audit record, including when the member is active. They must be excluded from model windows. A member with expired security history may retain its explicitly flagged last historical ticker, but no prices are invented. Older TAQ master rows may have a null test-symbol flag; date-effective CUSIP matching still applies and explicit test symbols are excluded.

## Bar policy and limitations

Five-minute intervals use New York exchange time and left/start labels. NYSE session calendars include holidays and early closes (typically 42 intervals on half-days and 78 on full days). Only the official regular session is retained, excluding the exact closing timestamp.

Eligible trades have positive price and size, correction code `00`, and sale conditions that are blank (NULL in WRDS) or contain only spaces, `@`, `E`, `F` or `I` (odd lot). Volume and eligible trade count include odd lots; volume is eligible share volume, not full consolidated tape volume. OHLC and VWAP come from round lots; a bar with no round lot is priced by exchange odd lots, and odd lots reported to a FINRA TRF (`ex = 'D'`) never set a price (policy version 4, `DataDownload/README.md`). A bar with only TRF odd lots keeps its volume, has no price and has status `no_priced_trades`. SIP timestamps, nanoseconds, sequence number, trade ID, and exchange determine ordering. These are server-aggregated trades, not quote bars or reconstructed order books. Special auction conditions are excluded. There is no quote-based outlier filter or reconstruction of later cancellations. This is a conservative research filter, not official exchange OHLCV.

Empty intervals retain missing prices and zero eligible volume/count, with explicit status. No interpolation, resampling imputation, or forward filling is performed. Training-window construction must reject windows containing missing bars. Early closes also require window logic that understands the exchange calendar; do not reinterpret them as missing full-session observations.

`00_metadata/progress.json` is authoritative for completed bar days. Reference files or a directory name alone do not imply that the full download has completed. WRDS query failures, permissions, connectivity, and source mismatches can interrupt the job; inspect the log, resolve the cause, and resume.

## Initial live verification (November 14, 2026)

Historical membership covers 2,514 sessions and 775 distinct ever-member PERMNOs, with 516–531 securities per day. All ten annual mapping tables were checked. There are 113 unmatched security-days out of 1,314,573 security-days; these remain missing and auditable. This count does not include mapped securities with no eligible trades.

The first partition, January 4, 2016, contains 41,028 bars for 526 securities: 40,543 observed and 485 empty intervals. Its SHA-256, unique keys, full-session grid, and close time were checked. Calendar and grid tests cover early closes, extraordinary closures, daylight saving, and missing active-member mappings. Four archive tests and six existing aggregation/resume tests passed.

An unattended job can be run with `run_archive_job.py OUTPUT_DIRECTORY` (in this folder) using the project Python. This wrapper writes `00_metadata/job_exit.json` on completion or failure. The launched local job writes console output to `download.log`; `job_start.json` records its process ID. On macOS the launched job uses `caffeinate -i` to prevent idle sleep while downloading. Power loss, closing the laptop, or loss of connectivity may still interrupt it. No automatic retry or external monitoring is configured. The normal downloader command resumes successfully committed partitions.

For reading a single Parquet file under a Hive-style `year=...` folder without adding inferred partition columns, use `pd.read_parquet(path, partitioning=None)`.
