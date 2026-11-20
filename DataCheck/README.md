# DataCheck: acceptance checks for downloaded bars

Step 2 of the data pipeline. No archive is adjusted, prepared or bundled before it passes these checks
(rule in `../DATA_PIPELINE.md`).

| Script | What it checks |
|---|---|
| `check_bar_coverage.py` | Offline, against CRSP daily data. It reports by listing group the last bar's close against the CRSP close, the share of CRSP volume kept, and empty bars and complete windows, and, for an archive, the bar highs and lows against CRSP's daily high and low. Use `--archive` for an archive folder; it reads every 21st session by default (`--every`) and adds a table by year. |
| `check_sale_conditions.py` | Read-only WRDS query for one session: volume kept by each filter version, per symbol and sale condition. |
| `validate_intraday_demo.py` | File, grid and key checks of the 2023 H1 demo download. |

**Pass criteria for an archive:**
- the Nasdaq and other groups keep a similar median share of CRSP volume in every year (ratio between 0.7 and 1.4);
- at least 95% of last-bar closes are within 0.1% of the CRSP close;
- at least 95% of security-days have their highest bar high and lowest bar low within 0.5% of CRSP's daily high and low;
- at most 0.1% of security-days with CRSP volume have no observed bar at all (a whole empty day points to a wrong symbol mapping).

```bash
python DataCheck/check_bar_coverage.py --archive --root ~/"Market Data/US Equity/v4/01_raw"
python DataCheck/check_sale_conditions.py --date 2023-01-09 --symbols JPM KO AAPL MSFT SPY --detail
```

[DATA_REVIEW_2026-11-19.md](DATA_REVIEW_2026-11-19.md) is the review that found the policy version 1 defect.
[ODD_LOT_PRICES_2026-11-20.md](ODD_LOT_PRICES_2026-11-20.md) is the one that found the policy version 3 price defect.
[SPLIT_SYMBOLS_2026-11-20.md](SPLIT_SYMBOLS_2026-11-20.md) is the v4 pilot and the 2016 share-class mapping defect it found.

**v1 archive measured on 2026-11-20 with `--archive`.** One session per year; the table gives the median share of CRSP volume kept.

| | 2016-01 | 2017-06 | 2018-06 | 2020-06 | 2023-06 | 2025-06 |
|---|---|---|---|---|---|---|
| other listings | 77% | 77% | 25% | 17% | 12% | 11% |
| Nasdaq-listed | 27% | 78% | 73% | 65% | 52% | 48% |

What the table shows:
- The bias changes sign: early in 2016 the Nasdaq group lost most of its volume, and from 2018 the other group did.
- The fall in the Nasdaq column tracks the rising share of odd lots, which v1 also excluded.
