#!/usr/bin/env python3
"""Check downloaded five-minute bars against the CRSP daily file saved next to them.

Offline. Two layouts:
- demo (download_intraday_bars.py): `bars_5min/*.parquet` and `crsp_daily_adjustment_factors.parquet`;
- archive (download_us_equity_archive.py, `--archive`): `03_raw_bars_5min/year=/month=/` partitions and the
  CRSP CIZ `05_daily_reference/`. Every `--every`-th session is read (default 21, about one a month), so
  the whole 2016-2025 archive is checked in a few minutes.
It answers five questions:

1. Are the prices right?   last bar close vs CRSP close, and daily returns vs CRSP daily returns.
2. How much volume is kept? eligible bar volume / CRSP daily volume, per listing group.
3. How many bars are empty, and how many complete 78-bar windows (+ next bar) survive?
4. Are the bar highs and lows clean? highest bar high / lowest bar low vs CRSP's daily high / low
   (archive layout only). Policy version 3 let odd lots reported to a FINRA TRF set bar prices, and a few
   print far from the market: in a 20-stock sample, 59 of 220 security-days had a day high or low more
   than 1% away from CRSP's, none with policy version 4 (see ODD_LOT_PRICES_2026-11-20.md).
5. Does every security that traded get bars? Security-days with CRSP volume but no observed bar point to a
   wrong symbol mapping. In 2016 eight Nasdaq share classes (GOOGL, CMCSA, DISCA, ...) were mapped to a
   symbol that never trades and came out empty on most 2016 days; see SPLIT_SYMBOLS_2026-11-20.md.

NASDAQ 100 membership is used as the Tape C (Nasdaq-listed) group, because NDX members must be
Nasdaq-listed; the other group is mostly Tape A (NYSE-listed) with some Nasdaq names mixed in.
In the archive the groups come from CRSP's primary exchange instead (Q = Nasdaq, else other), which is exact.
With a tape-neutral trade filter the two groups keep a similar share of volume. Policy version 1
kept about 55% for the NDX group and 11% for the rest in 2023 (see DATA_REVIEW_2026-11-19.md). In the
2016-2025 v1 archive the gap changes sign over time: in January 2016 Nasdaq names kept 27% against 77%,
in June 2017 both about 77%, and from 2018 the other group kept 10-25% against 50-75%.

    python DataCheck/check_bar_coverage.py --root "~/Market Data/demo_2023_h1"
    python DataCheck/check_bar_coverage.py --archive --root "~/Market Data/US Equity/v4/01_raw"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['date', 'bar_start', 'permno', 'ticker', 'indices', 'high', 'low', 'close', 'volume', 'status']


def load(root):
    files = sorted((root/'bars_5min').glob('*.parquet'))
    if not files:
        raise SystemExit(f'No bar files under {root}/bars_5min')
    bars = pd.concat([pd.read_parquet(f, columns=COLUMNS) for f in files], ignore_index=True)
    bars['slot'] = ((bars.bar_start.dt.hour*60 + bars.bar_start.dt.minute) - 570)//5
    bars['observed'] = bars.status.eq('observed')
    bars['tape_c'] = bars.indices.str.contains('NDX100')
    crsp = pd.read_parquet(root/'crsp_daily_adjustment_factors.parquet')
    crsp['permno'] = crsp.permno.astype('int64')
    crsp['prc'] = crsp.prc.astype(float).abs()
    crsp['vol'] = crsp.vol.astype(float)
    return bars, crsp[['date', 'permno', 'prc', 'vol']]


def load_archive(root, every):
    files = sorted((root/'03_raw_bars_5min').glob('year=*/month=*/*.parquet'))
    if not files:
        raise SystemExit(f'No bar partitions under {root}/03_raw_bars_5min')
    files = files[::every]
    bars = pd.concat([pd.read_parquet(f, columns=COLUMNS) for f in files], ignore_index=True)
    bars['date'] = bars.date.astype(str)
    bars['slot'] = ((bars.bar_start.dt.hour*60 + bars.bar_start.dt.minute) - 570)//5
    bars['observed'] = bars.status.eq('observed')
    years = sorted({d[:4] for d in bars.date.unique()})
    crsp = pd.concat([pd.read_parquet(root/'05_daily_reference'/f'year={y}'/'daily_reference.parquet',
                                      columns=['permno', 'dlycaldt', 'dlyprc', 'dlyvol', 'dlyhigh', 'dlylow',
                                               'primaryexch'])
                      for y in years], ignore_index=True)
    crsp['date'] = pd.to_datetime(crsp.dlycaldt).dt.strftime('%Y-%m-%d')
    crsp = crsp[crsp.date.isin(set(bars.date))].drop_duplicates(['date', 'permno'])
    crsp['permno'] = crsp.permno.astype('int64')
    crsp['prc'] = pd.to_numeric(crsp.dlyprc, errors='coerce').abs()
    crsp['vol'] = pd.to_numeric(crsp.dlyvol, errors='coerce')
    crsp['hi'] = pd.to_numeric(crsp.dlyhigh, errors='coerce').abs()
    crsp['lo'] = pd.to_numeric(crsp.dlylow, errors='coerce').abs()
    bars = bars.merge(crsp[['date', 'permno', 'primaryexch']], on=['date', 'permno'], how='left')
    bars['tape_c'] = bars.primaryexch.eq('Q')
    return bars.drop(columns='primaryexch'), crsp[['date', 'permno', 'prc', 'vol', 'hi', 'lo']]


def complete_windows(bars, lookback=78):
    """Share of member bars that end a complete window: `lookback` observed bars plus the next bar."""
    days = {d: i for i, d in enumerate(sorted(bars.date.unique()))}
    position = bars.date.map(days).to_numpy()*78 + bars.slot.to_numpy()
    total = len(days)*78
    rows = []
    for permno, index in bars.groupby('permno').indices.items():
        valid = np.zeros(total, bool); member = np.zeros(total, bool)
        valid[position[index]] = bars.observed.to_numpy()[index]; member[position[index]] = True
        missing = np.r_[0, np.cumsum(~valid)]
        ends = np.arange(lookback-1, total-1)
        ok = (missing[ends+2] - missing[ends-lookback+1]) == 0
        rows.append((bool(bars.tape_c.to_numpy()[index[-1]]), int(member[ends].sum()), int(ok.sum())))
    frame = pd.DataFrame(rows, columns=['tape_c', 'member_bars', 'windows'])
    frame['kept'] = frame.windows/frame.member_bars
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path.home()/'Market Data/demo_2023_h1')
    parser.add_argument('--archive', action='store_true', help='archive layout (download_us_equity_archive.py)')
    parser.add_argument('--every', type=int, default=21, help='archive: read every n-th session')
    args = parser.parse_args()
    root = args.root.expanduser()
    bars, crsp = load_archive(root, args.every) if args.archive else load(root)
    observed = bars[bars.observed].sort_values(['date', 'permno', 'slot'])
    daily = observed.groupby(['date', 'permno']).agg(
        tape_c=('tape_c', 'last'), last=('close', 'last'), volume=('volume', 'sum'),
        high=('high', 'max'), low=('low', 'min')).reset_index()
    daily = daily.merge(crsp, on=['date', 'permno'])
    daily['close_gap'] = (daily['last']/daily.prc - 1).abs()
    daily['volume_kept'] = daily.volume/daily.vol
    daily = daily.sort_values(['permno', 'date'])
    daily['r_bar'] = daily.groupby('permno')['last'].pct_change()
    daily['r_crsp'] = daily.groupby('permno').prc.pct_change()
    returns = daily.dropna(subset=['r_bar', 'r_crsp'])
    per_security = returns.groupby('permno')[['r_bar', 'r_crsp']].corr().unstack().iloc[:, 1]

    print(f'{root}: {bars.date.nunique()} sessions, {bars.permno.nunique()} securities, {len(bars):,} bars')
    print('\n1. Prices')
    print(f'   last bar close vs CRSP close: within 0.1% for {(daily.close_gap < .001).mean():.1%}, '
          f'within 0.5% for {(daily.close_gap < .005).mean():.2%}, max {daily.close_gap.max():.2%}')
    print(f'   daily return correlation with CRSP: pooled {returns.r_bar.corr(returns.r_crsp):.4f}, '
          f'worst security {per_security.min():.4f}')
    windows = complete_windows(bars)
    print('\n2-3. Coverage by listing group')
    names = (('Nasdaq-listed (CRSP Q)', 'other listings') if args.archive
             else ('NASDAQ 100 members (Tape C)', 'other members (mostly Tape A)'))
    for flag, name in ((True, names[0]), (False, names[1])):
        d = daily[daily.tape_c == flag]; b = bars[bars.tape_c == flag]; w = windows[windows.tape_c == flag]
        print(f'   {name:31s} volume kept: median {d.volume_kept.median():.1%} '
              f'(10%-90%: {d.volume_kept.quantile(.1):.1%}-{d.volume_kept.quantile(.9):.1%}) | '
              f'empty bars {1 - b.observed.mean():.2%} | complete windows {w.windows.sum()/w.member_bars.sum():.1%} | '
              f'securities with no window {int((w.windows == 0).sum())} of {len(w)}')
    kept = daily.groupby('tape_c').volume_kept.median()
    ratio = kept[False]/kept[True]
    print(f'\n   volume kept, other / Nasdaq group: {ratio:.2f} '
          f'({"similar: filter looks tape-neutral" if .7 < ratio < 1/.7 else "FAIL: the trade filter treats the tapes differently"})')
    if {'hi', 'lo'} <= set(daily):
        gap = pd.concat([(daily.high/daily.hi - 1).abs(), (daily.low/daily.lo - 1).abs()], axis=1).max(axis=1)
        within = (gap < .005).mean()
        print('\n4. Bar highs and lows')
        print(f'   highest bar high and lowest bar low vs CRSP daily high/low: within 0.5% for {within:.1%}, '
              f'more than 1% away for {int((gap > .01).sum())} of {gap.notna().sum()} security-days, max {gap.max():.2%} '
              f'({"pass" if within >= .95 else "FAIL: prints far from the market set bar prices"})')
    days = bars.groupby(['date', 'permno']).agg(observed=('observed', 'any'), ticker=('ticker', 'last'),
                                                status=('status', 'first')).reset_index()
    days = days.merge(crsp[['date', 'permno', 'vol']], on=['date', 'permno'])
    dark = days[~days.observed & days.vol.gt(0)]
    share = len(dark)/len(days)
    print('\n5. Whole days without a bar')
    print(f'   security-days where CRSP shows volume but no bar was observed: {len(dark)} of {len(days)} '
          f'({share:.2%}; {"pass" if share <= .001 else "FAIL: check the symbol mapping"})')
    if len(dark):
        print(f'   by status: {({k: int(v) for k, v in dark.status.value_counts().items()})}; most frequent: '
              + ', '.join(f'{t} ({n})' for t, n in dark.ticker.value_counts().head(10).items()))
    if args.archive:
        by_year = daily.assign(year=daily.date.str[:4]).groupby(['year', 'tape_c']).volume_kept.median().unstack()
        by_year.columns = ['other', 'Nasdaq']
        print('\n   volume kept by year (median):')
        print(by_year.map(lambda x: f'{x:.0%}').to_string())


if __name__ == '__main__':
    main()
