#!/usr/bin/env python3
"""Download server-aggregated TAQ bars for a historical three-index universe.

The default demo covers 2023 H1. Raw trades never leave WRDS. Prices and eligible
volume are unadjusted; this conservative demo filter is not official tape OHLCV.
"""
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from build_dynamic_universe import build_universe, fetch_daily, records, table_ident

POLICY = {
    'version': 1,
    'timezone': 'America/New_York',
    'session': '[09:30:00,16:00:00)',
    'bar_label': 'left; timestamps indicate interval start',
    'trade_filter': "price>0, size>0, tr_corr='00', tr_scond contains only spaces/@/E/F",
    'volume': 'Sum of shares for eligible filtered trades only; not total tape volume',
    'auctions': 'Special opening/closing sale conditions excluded; 16:00 prints excluded',
    'adjustment': 'Raw unadjusted OHLCV; CRSP daily split factors saved separately',
    'ordering': 'SIP time_m, time_m_nano, tr_seqnum, tr_id, ex',
    'empty_bars': 'Preserved with missing OHLC and zero eligible volume/trade_count; no fill',
    'quality_limitations': 'No quote-based outlier filter or reconstruction of later cancellations; demo only',
    'mapping': 'WRDS daily taqmclink plus date-effective CRSP NCUSIP to daily TAQ master CUSIP fallback; ambiguity rejected',
    'calendar': 'CRSP trading dates; this adapter supports 2023 H1 regular sessions only',
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def write_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd')
    os.replace(temporary, path)


def aggregate_sql(table, minutes):
    if minutes not in (1, 5, 30):
        raise ValueError('Supported intervals: 1, 5, 30 minutes')
    return f'''WITH eligible AS (
        SELECT sym_root, coalesce(sym_suffix,'') AS sym_suffix,
            floor((extract(epoch FROM time_m)-34200)/{minutes * 60})::int AS bucket,
            price, size, time_m, coalesce(time_m_nano,0) AS nano,
            tr_seqnum, tr_id, ex
        FROM {table_ident(table)}
        WHERE sym_root = ANY(%(roots)s)
          AND time_m >= '09:30:00'::time AND time_m < '16:00:00'::time
          AND price > 0 AND size > 0 AND tr_corr='00'
          AND tr_scond IS NOT NULL
          AND translate(tr_scond, ' @EF', '') = ''
    ) SELECT sym_root, sym_suffix, bucket,
        (array_agg(price ORDER BY time_m,nano,tr_seqnum,tr_id,ex))[1] AS open,
        max(price) AS high, min(price) AS low,
        (array_agg(price ORDER BY time_m DESC,nano DESC,tr_seqnum DESC,tr_id DESC,ex DESC))[1] AS close,
        sum(size)::bigint AS volume, count(*)::bigint AS trade_count,
        sum(price::double precision*size)/sum(size) AS vwap
      FROM eligible
      WHERE (sym_root || '|' || sym_suffix) = ANY(%(symbols)s)
      GROUP BY sym_root,sym_suffix,bucket ORDER BY sym_root,sym_suffix,bucket'''


def historical_universe(db, args):
    import pandas as pd
    request = argparse.Namespace(start=args.start, end=args.end, days='trading',
        ndx100_id=None, djia30_id=None,
        **{role+'_table': None for role in ('sp500','names','links','members','headers','calendar')})
    days, rows, sources = fetch_daily(db, request)
    inactive = []
    for row in rows:
        if row['ticker'] is None:
            if row['permno'] is None:
                raise RuntimeError(f'Missing PERMNO: {row}')
            prior = records(db, '''SELECT ticker,namedt,nameendt FROM crsp_a_stock.dsenames
                WHERE permno=%(permno)s AND namedt<=%(date)s AND ticker IS NOT NULL
                ORDER BY namedt DESC LIMIT 1''',
                {'permno': int(row['permno']), 'date': str(row['date'])[:10]})
            if not prior:
                raise RuntimeError(f'No historical identifier for {row}')
            row['ticker'] = prior[0]['ticker']
            inactive.append({**row, 'identifier_status': 'last_known_inactive',
                             'name_valid_through': prior[0]['nameendt']})
    daily, long_rows = build_universe(days, rows)
    inactive_keys = {(str(r['date'])[:10], int(r['permno'])) for r in inactive}
    frame = pd.DataFrame(long_rows)
    frame['identifier_status'] = ['last_known_inactive' if (r.date,int(r.permno)) in inactive_keys
                                  else 'date_effective' for r in frame.itertuples()]
    sources['inactive_name_audit'] = inactive
    return days, daily, frame, sources


def get_mapping(db, universe, start, end):
    import pandas as pd
    permnos = sorted(int(p) for p in universe.permno.unique())
    frame = db.raw_sql('''SELECT date,permno,sym_root,sym_suffix,ticker,match_lvl
        FROM wrdsapps_link_crsp_taqm.taqmclink
        WHERE date BETWEEN %(start)s AND %(end)s AND permno=ANY(%(permnos)s)''',
        params={'start':start,'end':end,'permnos':permnos})
    frame['date'] = pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
    frame['permno'] = frame.permno.astype('int64')
    frame['sym_suffix'] = frame.sym_suffix.fillna('').str.strip()
    frame['sym_root'] = frame.sym_root.str.strip()
    frame = frame.merge(universe[['date','permno']], on=['date','permno'], how='inner').drop_duplicates()
    frame['mapping_source'] = 'wrds_daily_link'
    master = db.raw_sql('''SELECT m.date,n.permno,m.symbol_15,m.cusip
        FROM taqm_2023.mastm_2023 m
        JOIN crsp_a_stock.dsenames n ON substring(m.cusip,1,8)=n.ncusip
          AND m.date BETWEEN n.namedt AND n.nameendt
        WHERE m.date BETWEEN %(start)s AND %(end)s AND n.permno=ANY(%(permnos)s)
          AND m.test_symbol_flag='N' ''',
        params={'start':start,'end':end,'permnos':permnos})
    master['date'] = pd.to_datetime(master.date).dt.strftime('%Y-%m-%d')
    master['permno'] = master.permno.astype('int64')
    master = master.merge(universe[['date','permno','ticker']],on=['date','permno'],how='inner')
    parts = master.symbol_15.str.strip().str.split()
    if not parts.map(len).isin([1,2]).all():
        raise RuntimeError('Unexpected TAQ master symbol formatting')
    master['sym_root'] = parts.str[0]
    master['sym_suffix'] = parts.map(lambda p: p[1] if len(p)==2 else '')
    master = master.drop_duplicates(['date','permno','sym_root','sym_suffix'])
    if master.duplicated(['date','permno'],keep=False).any():
        raise RuntimeError('Ambiguous historical CUSIP-to-master mapping')
    shared = frame.merge(master,on=['date','permno'],suffixes=('_link','_master'))
    if ((shared.sym_root_link!=shared.sym_root_master)
            | (shared.sym_suffix_link!=shared.sym_suffix_master)).any():
        raise RuntimeError('Daily link disagrees with historical CUSIP/master mapping')
    fallback = master.merge(frame[['date','permno']],on=['date','permno'],how='left',indicator=True)
    fallback = fallback[fallback['_merge'].eq('left_only')].copy()
    fallback['match_lvl'] = float('nan')
    fallback['mapping_source'] = 'historical_cusip_daily_master'
    frame = pd.concat([frame,fallback[frame.columns]],ignore_index=True)
    frame = frame.sort_values(['date','permno']).reset_index(drop=True)
    if frame.duplicated(['date','permno'],keep=False).any():
        raise RuntimeError('Ambiguous daily PERMNO-to-TAQ mapping; inspect mapping source')
    if frame.duplicated(['date','sym_root','sym_suffix'],keep=False).any():
        raise RuntimeError('Multiple PERMNOs map to the same daily TAQ symbol')
    if frame.sym_root.isna().any() or frame.sym_root.eq('').any():
        raise RuntimeError('Empty TAQ symbol root')
    return frame


def complete_grid(day, universe, mapping, bars, minutes):
    import numpy as np
    import pandas as pd
    members = universe[universe.date.eq(day)].copy()
    symbols = mapping[mapping.date.eq(day)].drop(columns=['date','ticker'])
    members = members.merge(symbols,on='permno',how='left',validate='one_to_one')
    if 'mapping_source' not in members:
        members['mapping_source'] = 'wrds_daily_link'
    missing_map = members.sym_root.isna()
    unexpected = members[missing_map & members.identifier_status.eq('date_effective')]
    if not unexpected.empty:
        raise RuntimeError(f'{day}: active members lack TAQ mapping: {unexpected.permno.tolist()}')
    buckets = pd.DataFrame({'bucket':range(390//minutes)})
    grid = members.merge(buckets,how='cross')
    bars['sym_suffix'] = bars.sym_suffix.fillna('').str.strip()
    bars['sym_root'] = bars.sym_root.str.strip()
    bars['bucket'] = bars.bucket.astype('int32')
    result = grid.merge(bars,on=['sym_root','sym_suffix','bucket'],how='left',validate='many_to_one')
    has_trade = result.trade_count.notna()
    result['status'] = np.where(result.sym_root.isna(),'no_symbol_mapping',
                              np.where(has_trade,'observed','no_eligible_trades'))
    for col in ['volume','trade_count']:
        result[col] = result[col].fillna(0).astype('int64')
    for col in ['open','high','low','close','vwap']:
        result[col] = pd.to_numeric(result[col],errors='raise').astype('float64')
    result['bar_start'] = (pd.Timestamp(day+' 09:30:00',tz='America/New_York')
                           + pd.to_timedelta(result.bucket*minutes,unit='min'))
    result['bar_end'] = result.bar_start + pd.Timedelta(minutes=minutes)
    observed = result[result.status.eq('observed')]
    if not ((observed.high >= observed[['open','close','low']].max(axis=1)).all()
            and (observed.low <= observed[['open','close','high']].min(axis=1)).all()
            and (observed[['open','high','low','close']] > 0).all().all()):
        raise RuntimeError(f'{day}: invalid OHLC values')
    if result.duplicated(['permno','bar_start']).any():
        raise RuntimeError('Duplicate security/bar timestamp')
    columns = ['date','bar_start','bar_end','permno','ticker','indices','sym_root','sym_suffix',
               'identifier_status','mapping_source','match_lvl','open','high','low','close','volume','trade_count','vwap','status']
    quality = {'date':day,'members':len(members),'grid_rows':len(result),
               'observed_rows':len(observed),'empty_rows':int((~has_trade).sum()),
               'unmapped_members':int(missing_map.sum()),
               'members_with_no_eligible_trades':sorted(int(x) for x in
                   set(members.permno)-set(observed.permno)),
               'eligible_trade_count':int(result.trade_count.sum()),
               'eligible_volume':int(result.volume.sum()),
               'match_level_counts':{str(k):int(v) for k,v in members.match_lvl.value_counts().items()}}
    return result[columns].sort_values(['permno','bar_start']).reset_index(drop=True), quality


def download(db, args):
    import pandas as pd
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    bar_dir = output/'bars_5min' if args.minutes == 5 else output/f'bars_{args.minutes}min'
    bar_dir.mkdir(exist_ok=True)
    print('Loading and validating historical membership...',flush=True)
    days, daily, universe, sources = historical_universe(db,args)
    mapping = get_mapping(db,universe,args.start,args.end)
    universe_hash = hashlib.sha256(universe.to_csv(index=False).encode()).hexdigest()
    mapping_hash = hashlib.sha256(mapping.to_csv(index=False).encode()).hexdigest()
    config = {'start':args.start,'end':args.end,'minutes':args.minutes,'policy':POLICY,
              'universe_sha256':universe_hash,'mapping_sha256':mapping_hash}
    run_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest = {'config':config,'run_id':run_id,'sources':sources,
                'requested_trading_days':days,'completed':{},'status':'running'}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['run_id'] != run_id and old.get('completed'):
            raise RuntimeError('Output directory belongs to a different configuration/source snapshot; use a new directory')
        manifest['completed'] = old.get('completed',{})
    write_parquet(universe,output/'daily_universe.parquet')
    write_parquet(mapping,output/'daily_taq_mapping.parquet')
    write_json(output/'daily_tickers.json',daily)
    write_json(manifest_path,manifest)
    # Preserve adjustment factors separately. Do not silently alter raw tape prices.
    factors = db.raw_sql('''SELECT date,permno,cfacpr,cfacshr,prc,vol
        FROM crsp_a_stock.dsf WHERE date BETWEEN %(start)s AND %(end)s
        AND permno=ANY(%(permnos)s)''',params={'start':args.start,'end':args.end,
        'permnos':sorted(int(x) for x in universe.permno.unique())})
    factors['date'] = pd.to_datetime(factors.date).dt.strftime('%Y-%m-%d')
    factors = factors.merge(universe[['date','permno']],on=['date','permno'],how='inner')
    write_parquet(factors,output/'crsp_daily_adjustment_factors.parquet')
    selected_days = days[:args.limit_days] if args.limit_days else days
    pending_days = []
    for number, day in enumerate(selected_days,1):
        destination = bar_dir/f'{day}.parquet'
        previous = manifest['completed'].get(day)
        if previous and destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == previous.get('sha256'):
                print(f'[{number}/{len(selected_days)}] {day}: verified existing file, skipped',flush=True)
                continue
        pending_days.append(day)
    local = threading.local()
    connections = []

    def aggregate_day(day):
        if args.workers == 1:
            connection = db
        else:
            if not hasattr(local, 'connection'):
                import wrds
                local.connection = wrds.Connection(wrds_username=args.username,
                    wrds_connect_args={'connect_timeout':30})
                local.connection.raw_sql("SELECT set_config('statement_timeout','900000',false)")
                connections.append(local.connection)
            connection = local.connection
        daily_mapping = mapping[mapping.date.eq(day)]
        roots = sorted(set(daily_mapping.sym_root))
        symbols = sorted(set(daily_mapping.sym_root+'|'+daily_mapping.sym_suffix))
        table = f'taqm_2023.ctm_{day.replace("-", "")}'
        print(f'{day}: aggregating {len(daily_mapping)} mapped securities on WRDS...',flush=True)
        bars = connection.raw_sql(aggregate_sql(table,args.minutes),params={'roots':roots,'symbols':symbols})
        result,quality = complete_grid(day,universe,mapping,bars,args.minutes)
        return day,result,quality

    def commit_day(day,result,quality):
        destination = bar_dir/f'{day}.parquet'
        write_parquet(result,destination)
        quality.update({'file':str(destination.relative_to(output)),
                        'bytes':destination.stat().st_size,
                        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
        manifest['completed'][day] = quality
        manifest['updated_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(manifest_path,manifest)
        print(f'[{len(manifest["completed"])}/{len(days)}] {day}: saved {len(result):,} bars; '
              f'observed={quality["observed_rows"]:,}; bytes={quality["bytes"]:,}',flush=True)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    try:
        futures = [executor.submit(aggregate_day,day) for day in pending_days]
        for future in as_completed(futures):
            commit_day(*future.result())
    except BaseException as exc:
        for future in futures:
            future.cancel()
        manifest['status'] = 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        write_json(manifest_path,manifest)
        raise
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
        for connection in connections:
            connection.close()
    manifest['status'] = 'complete' if set(days) <= set(manifest['completed']) else 'partial'
    manifest['total_bytes'] = sum(x['bytes'] for x in manifest['completed'].values())
    manifest['total_rows'] = sum(x['grid_rows'] for x in manifest['completed'].values())
    write_json(manifest_path,manifest)
    print(f'Status={manifest["status"]}; days={len(manifest["completed"])}/{len(days)}; '
          f'rows={manifest["total_rows"]:,}; bytes={manifest["total_bytes"]:,}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',default='2023-01-01')
    parser.add_argument('--end',default='2023-06-30')
    parser.add_argument('--minutes',type=int,choices=[1,5,30],default=5)
    parser.add_argument('--username',default=os.environ.get('WRDS_USERNAME') or getpass.getuser(),
                        help='WRDS username (default: $WRDS_USERNAME, else the local login name)')
    parser.add_argument('--output',type=Path,default=Path.home()/'Market Data/demo_2023_h1')
    parser.add_argument('--limit-days',type=int,help='Validate a bounded pilot before the full run')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=1,
                        help='Independent WRDS connections; only the parent writes files/manifest')
    args=parser.parse_args()
    start,end=dt.date.fromisoformat(args.start),dt.date.fromisoformat(args.end)
    if not dt.date(2023,1,1)<=start<=end<=dt.date(2023,6,30):
        parser.error('This verified adapter is limited to 2023 H1; extend session/schema handling before other dates')
    if args.limit_days is not None and args.limit_days<=0:
        parser.error('--limit-days must be positive')
    import wrds
    db=wrds.Connection(wrds_username=args.username,wrds_connect_args={'connect_timeout':30})
    try:
        # Bound any single server-side aggregation; completed days are resumable.
        db.raw_sql("SELECT set_config('statement_timeout','900000',false)")
        download(db,args)
    finally:
        db.close()


if __name__=='__main__':
    main()
