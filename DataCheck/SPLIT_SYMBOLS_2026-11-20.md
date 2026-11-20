# Policy v4 pilot and the 2016 share-class symbols — November 20, 2026

Reviewer: Tian Guo. The v4 pilot from `DataDownload/README.md` was run on WRDS's own JupyterHub with the code
on `main` (945183b), plus two local stand-ins that change no results: WRDS's internal login instead of
pgpass, and an offline copy of the NYSE calendar (`pandas_market_calendars` cannot be installed there; the
copy gives the identical 2,514-session schedule with its 21 early closes).

## 1. Pilot result: all three pass marks met

Five sessions (2016-01-04, 2017-06-01, 2018-06-01, 2023-06-01, 2025-06-02), 720 securities, 203,736 bars,
8.6 minutes in all.

| Check | Pass mark | Result |
|---|---|---|
| Volume kept, other / Nasdaq-listed | 0.7-1.4 | 0.99; by session 87/89, 88/88, 85/86, 77/78, 74/76 (%) |
| Last-bar close within 0.1% of CRSP | ≥ 95% | 97.9% (within 0.5%: 100%, max 0.40%) |
| Day high and low within 0.5% of CRSP | ≥ 95% | 99.8%; none more than 1% off |

`check_sale_conditions.py --date 2016-01-04` matches the archive: AAPL and MSFT stored most regular trades
with a NULL condition that day (51% and 61% of volume), JPM and KO none.

Two numbers in the output are artefacts of a pilot on non-consecutive days: the daily return correlation
(returns between pilot days years apart) and the securities without a complete window (93 of the 720
securities are members on only one of the five days, so a 78-bar window plus the next bar cannot form).

Download time per session: 39-57 s for 2016-2018, 85-93 s for 2023 and 2025. At the pilot average of 62 s,
the full 2,514 sessions take about 43 hours.

## 2. Defect found: share classes mapped to a symbol that never trades (2016)

On 2016-01-04 six securities had no eligible trade all day: GOOGL, CMCSA, DISCA, DISCK, LBTYA, LVNTA.

Cause:
- The WRDS daily link (`wrdsapps_link_crsp_taqm.taqmclink`) has no 2016 rows for these securities,
  so `mapping()` falls back to the CRSP CUSIP plus TAQ master match.
- Until 2016-10-21 the 2016 TAQ master writes these Nasdaq share classes as one word: `symbol_15`,
  `symbol_root` and `cqs_symbol` are all `GOOGL` and `sym_root`/`sym_suffix` are empty. The trades table
  (`ctm_YYYYMMDD`) stores the same security as `GOOG` + `L`, `CMCS` + `A`, `DISC` + `K` throughout 2016-2017.
- `mapping()` splits `symbol_15` on a space, so it produced `GOOGL` with no suffix, which matches no trade.

Scope, from the mapping of all ten years built with the pilot's code: 1,316 security-days, all in 2016
(about 1% of 2016's 132,565). GOOGL, CMCSA, DISCA, DISCK, LBTYA and LVNTA from 2016-01-04 to 2016-10-21
(204 sessions each); BATRA and BATRK from 2016-04-18 to 2016-06-21 (46 each). From 2016-10-24 the master
writes `GOOG L`, and no other year has a master-only five-character symbol without suffix.

No check caught it: the downloader keeps such days as empty bars, and `check_bar_coverage.py` only showed a
slightly higher share of empty bars for the Nasdaq group.

## 3. Fix

Charles's PR #11 (`symbol_params` and the combined-key join in `archive_grid`, merged as a3e71a3): a
five-letter root without suffix is also queried as four letters plus suffix, and bars are joined on root
and suffix written together, so `GOOGL` + '' and `GOOG` + `L` are the same key. Two spellings of one key
trading in the same session stop the run.

This PR adds the check that would have caught it: `check_bar_coverage.py` section 5 counts security-days
with CRSP volume but no observed bar, with a pass mark of at most 0.1% (`DataCheck/README.md`).

## 4. Verification

A second pilot on WRDS (8 sessions: the five above plus 2016-05-02, 2016-10-21 and 2016-10-24, around the
dates where the master's spelling changes) was run with an equivalent fix that tries the split spelling per
day. It switched the six securities on 2016-01-04 and 2016-10-21, all eight (with BATRA and BATRK) on
2016-05-02, and none on 2016-10-24, as expected from the mapping.

Charles's merged code was then run on WRDS for 2016-01-04 and 2016-05-02 against the same mapping: its bars
are identical to the second pilot's for every security and every column (OHLC, volume, trade count, VWAP,
status). GOOGL, CMCSA, DISCA, DISCK, LBTYA and LVNTA have 78 observed bars on both days; BATRA 72 and
BATRK 74 on 2016-05-02.

| `check_bar_coverage.py --archive --every 1` | First pilot (before the fix) | Second pilot |
|---|---|---|
| Sessions / security-days | 5 / 2,612 | 8 / 4,191 |
| Volume kept, other / Nasdaq-listed | 0.99 | 0.99 |
| Last-bar close within 0.1% of CRSP | 97.9% | 98.1% |
| Day high and low within 0.5% of CRSP | 99.8% | 99.8% (2 more than 1% off, max 1.46%) |
| Section 5: CRSP volume but no bar | 6 (0.23%, FAIL) | 0 (pass) |
| Empty bars, Nasdaq-listed group | 0.78% | 0.01% |

The run time per session in the second pilot was 37-57 s for 2016-2018 and 90-97 s for 2023 and 2025.
