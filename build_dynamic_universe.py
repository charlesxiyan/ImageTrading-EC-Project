#!/usr/bin/env python3
"""Build date-effective SP500 ∪ NDX100 ∪ DJIA30 universes using WRDS.

Python >=3.9. Only live WRDS access needs the third-party `wrds` package.
Run `python build_dynamic_universe.py --help` or see README.md.
"""
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

INDICES = ('SP500', 'NDX100', 'DJIA30')
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data' / 'universe'
TABLES = {
    'sp500': ['crsp_a_indexes.dsp500list', 'crsp.dsp500list'],
    'names': ['crsp_a_stock.dsenames', 'crsp.dsenames', 'crsp.msenames'],
    'links': ['crsp.ccmxpf_linktable', 'crsp_a_ccm.ccmxpf_linktable'],
    'members': ['comp.idxcst_his', 'comp_na_daily_all.idxcst_his'],
    'headers': ['comp.idx_index', 'comp_na_daily_all.idx_index'],
    'calendar': ['crsp_a_indexes.dsi', 'crsp.dsi'],
}


def ident(value):
    """Quote identifiers; never interpolate unchecked user input into SQL."""
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', value):
        raise ValueError(f'Invalid SQL identifier: {value!r}')
    return '"' + value + '"'


def table_ident(value):
    parts = value.split('.')
    if len(parts) != 2:
        raise ValueError('Table must be schema.table')
    return '.'.join(map(ident, parts))


def records(db, sql, params=None):
    return db.raw_sql(sql, params=params or {}).to_dict(orient='records')


def columns(db, table):
    schema, name = table.split('.')
    rows = records(db, '''select column_name from information_schema.columns
        where table_schema=%(schema)s and table_name=%(table)s''',
        {'schema': schema, 'table': name})
    return {r['column_name'] for r in rows}


def choose(db, role, override=None):
    candidates = [override] if override else TABLES[role]
    for table in candidates:
        table_ident(table)
        cols = columns(db, table)
        if cols:
            # Also check SELECT permission, without fetching any financial data.
            records(db, f'SELECT * FROM {table_ident(table)} LIMIT 0')
            return table, cols
    raise RuntimeError(f'No accessible {role} table: {candidates}. Run --inspect; '
                       'check subscriptions or supply the matching --*-table option.')


def require(cols, expected, table):
    missing = set(expected) - cols
    if missing:
        raise RuntimeError(f'{table} missing {sorted(missing)}; run --inspect. '
                           'This adapter requires date-effective legacy columns.')


def index_headers(db, table, cols):
    key = 'gvkeyx' if 'gvkeyx' in cols else 'gvkey'
    name = 'conm' if 'conm' in cols else 'conmx'
    require(cols, [key, name], table)
    return records(db, f'''SELECT {ident(key)} AS index_id, {ident(name)} AS name
        FROM {table_ident(table)} WHERE lower({ident(name)}) LIKE '%%nasdaq%%'
        OR lower({ident(name)}) LIKE '%%dow%%'
        OR lower({ident(name)}) LIKE '%%industrial%%' ''')


def resolve_index(rows, label, override):
    if override:
        matches = [r for r in rows if str(r['index_id']) == override]
    else:
        aliases = {
            'NDX100': {'NASDAQ 100', 'NASDAQ 100 INDEX', 'NASDAQ 100 INDEX COMPOSITE'},
            'DJIA30': {'DOW JONES INDUSTRIAL AVERAGE', 'DOW JONES INDUSTRIALS',
                       'DOW JONES INDUSTRIALS 30 STK',
                       'DOW JONES 30 INDUSTRIALS', 'DOW JONES INDUSTRIAL AVERAGE INDEX'},
        }
        matches = [r for r in rows if re.sub(r'[^A-Z0-9]+', ' ',
                    str(r['name']).upper()).strip() in aliases[label]]
    ids = {str(r['index_id']) for r in matches}
    if len(ids) != 1:
        raise RuntimeError(f'Cannot uniquely identify {label}. Run --inspect and pass '
                           f'--{label.lower()}-id using the verified index header. '
                           f'Candidates: {rows}')
    return ids.pop()


def inspect(db):
    report = {'tables': {}, 'index_candidates': [], 'errors': {}}
    for role, candidates in TABLES.items():
        for table in candidates:
            try:
                cols = columns(db, table)
                if cols:
                    report['tables'][table] = sorted(cols)
                    if role == 'headers':
                        report['index_candidates'].extend(index_headers(db, table, cols))
            except Exception as exc:
                report['errors'][table] = str(exc)
    # Discover alternatives, including CIZ and TAQ, without assuming their layout.
    report['related_catalog'] = records(db, '''SELECT table_schema, table_name
        FROM information_schema.tables WHERE table_schema NOT IN
        ('pg_catalog','information_schema') AND
        (table_name ILIKE '%%500list%%' OR table_name ILIKE '%%idxcst%%'
         OR table_name ILIKE '%%idx_index%%' OR table_name ILIKE '%%stocknames%%'
         OR table_schema ILIKE 'taq%%') ORDER BY table_schema, table_name''')
    return report


def fetch_daily(db, args):
    selected = {role: choose(db, role, getattr(args, role + '_table'))
                for role in ('sp500', 'names', 'links', 'members', 'headers')}
    required = {
        'sp500': ['permno', 'start', 'ending'],
        'names': ['permno', 'ticker', 'namedt', 'nameendt'],
        'links': ['gvkey', 'liid', 'lpermno', 'linkdt', 'linkenddt', 'linktype'],
        'members': ['gvkey', 'iid', 'gvkeyx', 'from', 'thru'],
    }
    for role, expected in required.items():
        require(selected[role][1], expected, selected[role][0])
    headers = index_headers(db, *selected['headers'])
    params = {'start': args.start, 'end': args.end,
              'ndx': resolve_index(headers, 'NDX100', args.ndx100_id),
              'djia': resolve_index(headers, 'DJIA30', args.djia30_id)}
    if args.days == 'trading':
        cal, cols = choose(db, 'calendar', args.calendar_table)
        require(cols, ['date'], cal)
        # Reject dates beyond the available calendar instead of silently truncating.
        bounds = records(db, f'SELECT min(date) AS lo, max(date) AS hi FROM {table_ident(cal)}')[0]
        if str(bounds['lo'])[:10] > args.start or str(bounds['hi'])[:10] < args.end:
            raise RuntimeError(f'Request is outside available CRSP calendar: {bounds}. '
                               'Use a covered end date or --days calendar.')
        day_sql = f'SELECT DISTINCT date::date AS date FROM {table_ident(cal)} WHERE date BETWEEN %(start)s AND %(end)s'
    else:
        day_sql = "SELECT generate_series(%(start)s::date, %(end)s::date, interval '1 day')::date AS date"
    days = records(db, day_sql, params)
    if not days:
        raise RuntimeError('No requested dates in the selected calendar.')
    sp, names, links, members = [table_ident(selected[k][0]) for k in ('sp500', 'names', 'links', 'members')]
    # Endpoints are inclusive (FROM/THRU and CRSP START/ENDING).
    # LEFT JOIN keeps failed identifier mappings visible to the validation layer.
    sql = f'''WITH days AS ({day_sql}), membership AS (
        SELECT d.date, 'SP500'::text AS index_name, s.permno,
               'permno:' || s.permno::text AS member_id
        FROM days d JOIN {sp} s ON d.date >= s.start
          AND (s.ending IS NULL OR d.date <= s.ending)
        UNION ALL
        SELECT d.date, CASE WHEN m.gvkeyx=%(ndx)s THEN 'NDX100' ELSE 'DJIA30' END,
               l.lpermno AS permno, m.gvkey || ':' || m.iid AS member_id
        FROM days d JOIN {members} m ON d.date >= m."from"
          AND (m.thru IS NULL OR d.date <= m.thru)
          AND m.gvkeyx IN (%(ndx)s, %(djia)s)
        LEFT JOIN {links} l ON l.gvkey=m.gvkey AND l.liid=m.iid
          AND l.linktype IN ('LC','LU','LS') AND l.lpermno > 0
          AND (l.linkdt IS NULL OR d.date >= l.linkdt)
          AND (l.linkenddt IS NULL OR d.date <= l.linkenddt)
    ) SELECT DISTINCT m.date, m.index_name, m.member_id, m.permno, n.ticker
      FROM membership m LEFT JOIN {names} n ON n.permno=m.permno
        AND m.date >= n.namedt AND (n.nameendt IS NULL OR m.date <= n.nameendt)
      ORDER BY m.date, m.index_name, m.member_id'''
    result = records(db, sql, params)
    return [str(r['date'])[:10] for r in days], result, {
        'tables': {k: v[0] for k, v in selected.items()},
        'index_ids': {'NDX100': params['ndx'], 'DJIA30': params['djia']},
        'calendar_table': cal if args.days == 'trading' else None,
    }


def build_universe(days, rows):
    """Validate complete daily mappings and return (daily dict, long-form rows)."""
    pools = {day: {} for day in sorted(set(days))}
    seen_indices = {day: set() for day in pools}
    member_mappings = {}
    ticker_mappings = {}
    for row in rows:
        day = str(row['date'])[:10]
        index = row['index_name']
        if day not in pools or index not in INDICES:
            raise ValueError(f'Unexpected date/index in source: {row}')
        ticker = row['ticker']
        raw_permno = row['permno']
        if raw_permno is None or str(raw_permno) in ('nan', '<NA>') or not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f'Unmapped member; refusing incomplete universe: {row}')
        permno = int(raw_permno)
        ticker = ticker.strip()  # Preserve CRSP share-class punctuation.
        if permno <= 0 or float(raw_permno) != permno:
            raise ValueError(f'Invalid PERMNO: {row}')
        key = (day, index, row['member_id'])
        mapping = (permno, ticker)
        if key in member_mappings and member_mappings[key] != mapping:
            raise ValueError(f'Ambiguous historical member mapping: {key}')
        member_mappings[key] = mapping
        tkey = (day, ticker)
        if tkey in ticker_mappings and ticker_mappings[tkey] != permno:
            raise ValueError(f'Ticker maps to multiple securities on {day}: {ticker}')
        ticker_mappings[tkey] = permno
        seen_indices[day].add(index)
        item = pools[day].setdefault(permno, {'ticker': ticker, 'indices': set()})
        if item['ticker'] != ticker:
            raise ValueError(f'Conflicting tickers for {permno} on {day}')
        item['indices'].add(index)
    for day, present in seen_indices.items():
        if present != set(INDICES):
            raise ValueError(f'{day}: missing historical constituents for {set(INDICES)-present}. '
                             'Check index ID, permissions and historical coverage.')
    daily, long_rows = {}, []
    for day, pool in pools.items():
        daily[day] = sorted(item['ticker'] for item in pool.values())
        for permno, item in sorted(pool.items()):
            long_rows.append({'date': day, 'permno': permno, 'ticker': item['ticker'],
                              'indices': '|'.join(sorted(item['indices']))})
    return daily, long_rows


def save(output, daily, rows, metadata):
    output.mkdir(parents=True, exist_ok=True)
    # No files are created until every requested date passes validation.
    (output / 'daily_tickers.json').write_text(json.dumps(daily, indent=2) + '\n')
    with (output / 'daily_universe.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['date', 'permno', 'ticker', 'indices'])
        writer.writeheader()
        writer.writerows(rows)
    counts = {day: {'union': len(tickers), **{index: 0 for index in INDICES}}
              for day, tickers in daily.items()}
    for row in rows:
        for index in row['indices'].split('|'):
            counts[row['date']][index] += 1
    metadata.update({'generated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                     'daily_counts': counts,
                     'semantics': 'Date-effective historical membership; inclusive endpoints. '
                     'Not a vintage/as-published database; not proof of intraday tradability. '
                     'NULL interval ends are treated as ongoing; verify source update vintage.'})
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', help='Inclusive YYYY-MM-DD')
    parser.add_argument('--end', help='Inclusive YYYY-MM-DD')
    parser.add_argument('--username', help='WRDS username; never pass a password here')
    parser.add_argument('--inspect', action='store_true', help='Print accessible schema and index candidates; no universe built')
    parser.add_argument('--days', choices=['calendar', 'trading'], default='calendar',
                        help='Default: every calendar day, including weekends')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--ndx100-id', help='Verified Compustat index GVKEYX')
    parser.add_argument('--djia30-id', help='Verified Compustat index GVKEYX')
    for role in TABLES:
        parser.add_argument('--' + role + '-table', help='Override schema.table; see --inspect')
    args = parser.parse_args()
    if not args.inspect:
        try:
            start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        except (TypeError, ValueError):
            parser.error('--start and --end must be YYYY-MM-DD')
        if start > end:
            parser.error('--start must not exceed --end')
        args.start, args.end = start.isoformat(), end.isoformat()
    try:
        import wrds
    except ImportError:
        parser.exit(1, 'Install live-access dependencies: python -m pip install wrds\n')
    db = wrds.Connection(**({'wrds_username': args.username} if args.username else {}))
    try:
        if args.inspect:
            print(json.dumps(inspect(db), indent=2, default=str))
        else:
            days, rows, metadata = fetch_daily(db, args)
            daily, long_rows = build_universe(days, rows)
            metadata.update({'start': args.start, 'end': args.end, 'days': args.days})
            save(args.output, daily, long_rows, metadata)
            print(f'Wrote {len(daily)} dates / {len(long_rows)} security-days to {args.output.resolve()}')
    finally:
        db.close()


if __name__ == '__main__':
    main()
