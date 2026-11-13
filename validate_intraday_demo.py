"""Validate all completed daily bar files against the download manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def validate(output):
    manifest = json.loads((output/'manifest.json').read_text())
    completed = manifest['completed']
    expected_dates = set(manifest['requested_trading_days'])
    if manifest['status'] != 'complete' or set(completed) != expected_dates:
        raise RuntimeError('The requested date range is not complete')
    universe = pd.read_parquet(output/'daily_universe.parquet')
    rows = observed = empty = volume = trades = size = 0
    inactive = []
    no_trade = []
    max_move = []
    expected_intervals = 390//manifest['config']['minutes']
    for day,quality in sorted(completed.items()):
        file = output/quality['file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != quality['sha256']:
            raise RuntimeError(f'Checksum mismatch: {file}')
        data = pd.read_parquet(file)
        if not data.date.eq(day).all() or data.duplicated(['permno','bar_start']).any():
            raise RuntimeError(f'Invalid date/key: {day}')
        members = universe[universe.date.eq(day)]
        if set(data.permno) != set(members.permno):
            raise RuntimeError(f'Membership mismatch: {day}')
        if not data.groupby('permno').size().eq(expected_intervals).all():
            raise RuntimeError(f'Incomplete interval grid: {day}')
        session_start=pd.Timestamp(day+' 09:30:00',tz='America/New_York')
        offset=(data.bar_start-session_start)/pd.Timedelta(minutes=manifest['config']['minutes'])
        if not (offset.between(0,expected_intervals-1) & offset.eq(offset.round())).all():
            raise RuntimeError(f'Invalid session timestamp grid: {day}')
        if not ((data.bar_end-data.bar_start)==pd.Timedelta(minutes=manifest['config']['minutes'])).all():
            raise RuntimeError(f'Invalid bar duration: {day}')
        good = data.status.eq('observed')
        if data.loc[good,['open','high','low','close','vwap']].isna().any().any():
            raise RuntimeError(f'Missing observed price: {day}')
        if not data.loc[~good,['open','high','low','close','vwap']].isna().all().all():
            raise RuntimeError(f'Unexpected filled empty price: {day}')
        if not (data.loc[~good,['volume','trade_count']]==0).all().all():
            raise RuntimeError(f'Unexpected empty volume: {day}')
        if not (data.loc[good,'high']>=data.loc[good,['open','close','low']].max(axis=1)).all():
            raise RuntimeError(f'Invalid high: {day}')
        if not (data.loc[good,'low']<=data.loc[good,['open','close','high']].min(axis=1)).all():
            raise RuntimeError(f'Invalid low: {day}')
        if len(data) != quality['grid_rows'] or int(good.sum()) != quality['observed_rows']:
            raise RuntimeError(f'Manifest count mismatch: {day}')
        rows += len(data)
        observed += int(good.sum())
        empty += int((~good).sum())
        volume += int(data.volume.sum())
        trades += int(data.trade_count.sum())
        size += file.stat().st_size
        if quality['members_with_no_eligible_trades']:
            no_trade.append({'date':day,'permnos':quality['members_with_no_eligible_trades']})
        inactive.extend(members[members.identifier_status.eq('last_known_inactive')][['date','permno','ticker']].to_dict('records'))
        flagged = data[good & (data.high/data.low>1.2)]
        if not flagged.empty:
            max_move.extend(flagged[['date','permno','ticker','bar_start','low','high']].to_dict('records'))
    counts = universe.groupby('date').size()
    summary = {
        'status':'validated_complete','start':manifest['config']['start'],'end':manifest['config']['end'],
        'trading_days':len(completed),'bar_minutes':manifest['config']['minutes'],
        'daily_security_count_min':int(counts.min()),'daily_security_count_max':int(counts.max()),
        'distinct_permnos':int(universe.permno.nunique()),'security_days':len(universe),
        'grid_rows':rows,'observed_rows':observed,'empty_rows':empty,
        'empty_fraction':empty/rows,'eligible_trade_count':trades,'eligible_volume':volume,
        'bar_file_bytes':size,'inactive_identifier_security_days':inactive,
        'security_days_without_eligible_trades':no_trade,
        'bars_with_high_low_ratio_above_1_2':max_move,
        'limitations':manifest['config']['policy'],
    }
    path=output/'data_quality_summary.json'
    path.write_text(json.dumps(summary,indent=2,default=str)+'\n',encoding='utf-8')
    (output/'README.md').write_text(
        '# US equity intraday demo\n\n'
        f'Period: {summary["start"]} through {summary["end"]}; '
        f'{summary["trading_days"]} trading days; {summary["bar_minutes"]}-minute bars.\n\n'
        'Universe: daily historical union of S&P 500, NASDAQ 100, and DJIA 30, '
        'deduplicated by PERMNO.\n\n'
        f'Rows: {rows:,}; observed: {observed:,}; empty: {empty:,} '
        f'({empty/rows:.2%}). Compressed daily bar files: {size/1e6:.2f} MB.\n\n'
        'Prices are raw and unadjusted. Volume counts only eligible filtered trades, '
        'not total tape volume. Bars cover [09:30,16:00) US Eastern and exclude '
        'special auction conditions, odd lots, and other conditions outside the '
        'documented conservative whitelist. Empty intervals are not forward-filled. '
        'This is a research demo, not an official consolidated OHLCV product.\n\n'
        'Daily files are in `bars_5min/`. Membership, TAQ mappings, and CRSP daily '
        'adjustment factors are stored as separate Parquet files. See '
        '`manifest.json` for the exact policy, sources, file checksums, and daily '
        'quality counts, and `data_quality_summary.json` for validation results.\n\n'
        'Source code and detailed documentation: '
        '`INTRADAY_DEMO.md` in the ImageTrading-EC-Project repository.\n',
        encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,(dict,list))},indent=2))
    print(f'Quality report: {path}')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path.home()/'Market Data/demo_2023_h1')
    validate(parser.parse_args().output)
