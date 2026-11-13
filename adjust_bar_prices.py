#!/usr/bin/env python3
"""Rebase raw intraday prices to each security's latest available price basis.

Uses CRSP CFACPR ratios for split/distribution adjustment, excluding ordinary cash
dividends. All code and reports use English. No network connection is required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'vwap']
DEFAULT_ROOT = Path.home()/'Market Data/demo_2023_h1'
METHOD = 'crsp_cfacpr_latest_basis_v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_factors(factors, end_date):
    """Select auditable per-PERMNO anchors; never key adjustments by ticker."""
    required = {'date', 'permno', 'cfacpr'}
    if not required <= set(factors):
        raise ValueError(f'Missing factor columns: {required-set(factors)}')
    f = factors[list(required)].copy()
    f['date'] = pd.to_datetime(f.date, errors='raise').dt.strftime('%Y-%m-%d')
    f = f[f.date.le(end_date)].copy()
    if f.duplicated(['date', 'permno']).any():
        raise ValueError('Duplicate date/PERMNO factors; refusing an ambiguous join')
    f['cfacpr'] = pd.to_numeric(f.cfacpr, errors='raise').astype('float64')
    if not (np.isfinite(f.cfacpr) & f.cfacpr.gt(0)).all():
        raise ValueError('CFACPR must be finite and strictly positive')
    anchors = f.sort_values(['permno','date']).groupby('permno',sort=True).tail(1)
    anchors = anchors.rename(columns={'date':'adjustment_base_date','cfacpr':'base_cfacpr'})
    f = f.merge(anchors,on='permno',validate='many_to_one')
    f['price_adjustment_multiplier'] = f.base_cfacpr/f.cfacpr
    return f, anchors.sort_values('permno').reset_index(drop=True)


def adjust_frame(bars, factors):
    """Apply one positive multiplier to every price in each security/day."""
    required = {'date','permno','status',*PRICE_COLUMNS}
    if not required <= set(bars):
        raise ValueError(f'Missing bar columns: {required-set(bars)}')
    if 'price_adjustment_multiplier' in bars or 'price_adjustment_method' in bars:
        raise ValueError('Input appears adjusted already; refusing double adjustment')
    original = bars.copy()
    original['date'] = pd.to_datetime(original.date,errors='raise').dt.strftime('%Y-%m-%d')
    result = original.merge(factors,on=['date','permno'],how='left',validate='many_to_one',sort=False)
    has_price = original[PRICE_COLUMNS].notna().any(axis=1).to_numpy()
    missing = result.price_adjustment_multiplier.isna().to_numpy()
    if (has_price & missing).any():
        bad = result.loc[has_price & missing,['date','permno']].drop_duplicates().to_dict('records')
        raise ValueError(f'Prices without exact-day CFACPR: {bad[:10]}')
    # Missing-price halted rows may lack factors; retain their missing values.
    for column in PRICE_COLUMNS:
        result[column] = pd.to_numeric(result[column],errors='raise') * result.price_adjustment_multiplier
        if not np.array_equal(result[column].isna(),original[column].isna()):
            raise ValueError(f'Adjustment changed missingness in {column}')
        finite = result[column].dropna().astype(float)
        if not (np.isfinite(finite) & finite.gt(0)).all():
            raise ValueError(f'Invalid adjusted prices in {column}')
    observed = result.status.eq('observed')
    if not (result.loc[observed,'high'] >= result.loc[observed,['open','close','low']].max(axis=1)).all():
        raise ValueError('Invalid adjusted high')
    if not (result.loc[observed,'low'] <= result.loc[observed,['open','close','high']].min(axis=1)).all():
        raise ValueError('Invalid adjusted low')
    for column in original.columns.difference(PRICE_COLUMNS):
        pd.testing.assert_series_equal(original[column].reset_index(drop=True),
            result[column].reset_index(drop=True),check_names=False)
    result['price_adjustment_method'] = METHOD
    result['adjustment_status'] = np.where(missing,'missing_factor_no_price','adjusted')
    return result


def run(input_root, output, dry_run=False):
    source_manifest = input_root/'manifest.json'
    source = json.loads(source_manifest.read_text())
    if source.get('status') != 'complete':
        raise ValueError('Source download must be complete')
    items = sorted(source['completed'].items())
    if not items or set(source['completed']) != set(source['requested_trading_days']):
        raise ValueError('Source manifest does not cover all requested dates')
    if output.resolve() == input_root.resolve() or output.exists():
        raise ValueError('Use a new output directory; source data and existing outputs are protected')
    factor_path = input_root/'crsp_daily_adjustment_factors.parquet'
    factors, anchors = prepare_factors(pd.read_parquet(factor_path),items[-1][0])
    report = {
        'method':METHOD,'formula':'adjusted_price = raw_price * base_cfacpr / daily_cfacpr',
        'scope':'CRSP split/distribution price adjustment; ordinary cash dividends excluded',
        'price_columns':PRICE_COLUMNS,'volume_policy':'Original eligible share volume unchanged',
        'anchor_policy':'Last factor date per PERMNO at or before the final input trading date',
        'requested_base_date':items[-1][0],
        'inactive_security_policy':'Earlier last-available anchor retained explicitly; missing OHLC never filled',
        'source_root':str(input_root.resolve()),'source_manifest_sha256':sha256(source_manifest),
        'source_factors_sha256':sha256(factor_path),'days':len(items),'rows':0,
        'price_changed_rows':0,'missing_factor_no_price_rows':0,'changed_permnos':[],
        'status':'validated_dry_run' if dry_run else 'writing','files':{},
    }
    changed_ids=set()
    # Preflight every file before creating output, including exact-day coverage.
    for day,entry in items:
        path=input_root/entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
        bars=pd.read_parquet(path)
        if not bars.date.astype(str).eq(day).all():
            raise ValueError(f'Source date mismatch: {path}')
        adjusted=adjust_frame(bars,factors[factors.date.eq(day)])
        changed=(~np.isclose(adjusted.price_adjustment_multiplier,1.,rtol=1e-12,atol=1e-12)
                 & adjusted.close.notna())
        changed_ids.update(int(x) for x in adjusted.loc[changed,'permno'].unique())
        report['rows']+=len(adjusted)
        report['price_changed_rows']+=int(changed.sum())
        report['missing_factor_no_price_rows']+=int(adjusted.adjustment_status.eq('missing_factor_no_price').sum())
        report['files'][day]={'source':entry['file'],'source_sha256':entry['sha256']}
    report['changed_permnos']=sorted(changed_ids)
    if not dry_run:
        output.mkdir(parents=True)
        anchors.to_parquet(output/'adjustment_anchors.parquet',index=False,compression='zstd')
        destination_dir=output/'bars_5min'
        destination_dir.mkdir()
        for day,entry in items:
            path=input_root/entry['file']
            if sha256(path)!=entry['sha256']:
                raise ValueError('Source changed after preflight')
            frame=adjust_frame(pd.read_parquet(path),factors[factors.date.eq(day)])
            target=destination_dir/f'{day}.parquet'
            frame.to_parquet(target,index=False,compression='zstd')
            report['files'][day].update({'file':str(target.relative_to(output)),
                                        'sha256':sha256(target),'rows':len(frame)})
        report['status']='complete'
        (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path,help='New directory; defaults to a sibling ending _price_adjusted')
    parser.add_argument('--dry-run',action='store_true',help='Validate the complete dataset without writing files')
    args=parser.parse_args()
    output=args.output or args.input_root.with_name(args.input_root.name+'_price_adjusted')
    run(args.input_root,output,args.dry_run)


if __name__=='__main__':
    main()
