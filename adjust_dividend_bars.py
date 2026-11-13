#!/usr/bin/env python3
"""Create split/distribution plus cash-dividend adjusted bars, with share-adjusted volume.

Chart adjustment, not an exact dividend-reinvestment total-return index. Raw price
and volume fields are retained. No network access is needed after inputs are fetched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adjust_bar_prices import PRICE_COLUMNS, prepare_factors, sha256

METHOD='crsp_cash_dividend_chart_and_share_volume_v1'
# Verified USD cash codes present in this demo, from CRSP's distribution code map.
CASH_CODES={1212,1222,1232,1233,1242,1252,1272}
CORPORATE_CODES={3763,5523}


def normalize_dates(frame,column):
    frame=frame.copy()
    frame[column]=pd.to_datetime(frame[column],errors='raise').dt.strftime('%Y-%m-%d')
    return frame


def prepare_adjustments(raw_factors,events,references):
    raw=normalize_dates(raw_factors,'date')
    f,anchors=prepare_factors(raw,str(raw.date.max()))
    raw['cfacshr']=pd.to_numeric(raw.cfacshr,errors='raise').astype(float)
    if not (np.isfinite(raw.cfacshr)&raw.cfacshr.gt(0)).all():
        raise ValueError('Invalid share factors')
    anchors=anchors.merge(raw[['date','permno','cfacshr']],left_on=['adjustment_base_date','permno'],
        right_on=['date','permno'],validate='one_to_one').drop(columns='date').rename(columns={'cfacshr':'base_cfacshr'})
    f=f.rename(columns={'price_adjustment_multiplier':'split_price_multiplier'})
    f=f.merge(raw[['date','permno','cfacshr']],on=['date','permno'],validate='one_to_one')
    f=f.merge(anchors[['permno','base_cfacshr']],on='permno',validate='many_to_one')
    f['volume_adjustment_multiplier']=f.cfacshr/f.base_cfacshr
    f['cash_dividend_multiplier']=1.
    refs=normalize_dates(references,'date').sort_values(['permno','date'])
    if refs.duplicated(['permno','date']).any():
        raise ValueError('Duplicate daily references')
    comparison=raw.merge(refs,on=['permno','date'],suffixes=('_raw','_ref'),validate='one_to_one')
    if len(comparison)!=len(raw):
        raise ValueError('Reference daily data do not cover original factors')
    for col in ['cfacpr','cfacshr']:
        if not np.allclose(comparison[col+'_raw'].astype(float),comparison[col+'_ref'].astype(float),rtol=1e-10,atol=1e-10):
            raise ValueError('Daily factor vintage differs from raw dataset: '+col)
    events=normalize_dates(events,'exdt')
    if events.duplicated().any():
        raise ValueError('Duplicate distribution records; refusing double counting')
    bounds=raw.groupby('permno').date.agg(['min','max'])
    audit=events.copy()
    audit['treatment']=''
    for i,event in audit.iterrows():
        p=int(event.permno); code=int(event.distcd)
        if p not in bounds.index or not bounds.at[p,'min']<event.exdt<=bounds.at[p,'max']:
            audit.at[i,'treatment']='outside_adjustable_history'
        elif code in CASH_CODES:
            if not np.isfinite(float(event.divamt)) or event.divamt<=0:
                raise ValueError('Invalid cash dividend')
            if not np.isclose(float(event.facpr),0.) or not np.isclose(float(event.facshr),0.):
                raise ValueError('Cash event already has corporate factors; extend policy before applying')
            audit.at[i,'treatment']='cash_chart_factor'
        elif code in CORPORATE_CODES:
            audit.at[i,'treatment']='covered_by_cfacpr_cfacshr'
        else:
            raise ValueError(f'Unsupported distribution code {code}; classify before adjusting')
    details=[]
    for (permno,exdt),group in audit[audit.treatment.eq('cash_chart_factor')].groupby(['permno','exdt']):
        history=refs[refs.permno.eq(permno)]
        previous=history[history.date.lt(exdt)].tail(1)
        current=history[history.date.eq(exdt)]
        if len(previous)!=1 or len(current)!=1:
            raise ValueError(f'Missing prior/ex-date reference for {permno} {exdt}')
        previous,current=previous.iloc[0],current.iloc[0]
        price=abs(float(previous.prc))
        cf_previous,cf_ex=float(previous.cfacpr),float(current.cfacpr)
        if not all(np.isfinite(x) and x>0 for x in [price,cf_previous,cf_ex]):
            raise ValueError('Invalid prior closing reference or corporate factor')
        # CRSP reports the dividend in ex-date share units. Convert prior close to
        # that same price basis before calculating the proportional cash factor.
        reference=price*cf_ex/cf_previous
        cash=float(group.divamt.sum())
        multiplier=1-cash/reference
        if not 0<multiplier<1:
            raise ValueError(f'Invalid cash adjustment factor for {permno} {exdt}')
        mask=f.permno.eq(permno)&f.date.lt(exdt)
        f.loc[mask,'cash_dividend_multiplier']*=multiplier
        details.append(dict(permno=int(permno),exdt=exdt,cash_amount=cash,
            distribution_records=len(group),previous_date=previous.date,
            previous_price=price,previous_price_was_quote=bool(previous.prc<0),
            previous_cfacpr=cf_previous,exdate_cfacpr=cf_ex,
            previous_close_on_exdate_basis=reference,cash_multiplier=multiplier))
    f['price_adjustment_multiplier']=f.split_price_multiplier*f.cash_dividend_multiplier
    return f,anchors,audit,pd.DataFrame(details)


def apply_adjustments(bars,factors):
    if 'price_adjustment_method' in bars or 'price_adjustment_multiplier' in bars:
        raise ValueError('Use raw bars; refusing double adjustment')
    original=normalize_dates(bars,'date')
    result=original.merge(factors,on=['date','permno'],how='left',validate='many_to_one',sort=False)
    missing=result.price_adjustment_multiplier.isna()
    if (missing&original[PRICE_COLUMNS].notna().any(axis=1)).any():
        raise ValueError('Observed prices without exact-day factors')
    if (result.volume_adjustment_multiplier.isna()&original.volume.ne(0)).any():
        raise ValueError('Nonzero volume without a share factor')
    for col in PRICE_COLUMNS:
        result[col+'_raw']=original[col]
        result[col]=original[col]*result.price_adjustment_multiplier
        if not result[col].isna().equals(original[col].isna()):
            raise ValueError('Missing prices were changed')
        values=result[col].dropna()
        if not (np.isfinite(values)&values.gt(0)).all():
            raise ValueError('Nonpositive/nonfinite adjusted price')
    result['volume_raw']=original.volume
    result['volume']=original.volume.astype(float)*result.volume_adjustment_multiplier
    # Wholly empty rows without factors remain genuinely zero-volume intervals.
    result.loc[missing&original.volume.eq(0),'volume']=0.
    for col in original.columns.difference(PRICE_COLUMNS+['volume']):
        pd.testing.assert_series_equal(original[col].reset_index(drop=True),result[col].reset_index(drop=True),check_names=False)
    good=result.status.eq('observed')
    if not ((result.loc[good,'high']>=result.loc[good,['open','close','low']].max(axis=1)).all()
            and (result.loc[good,'low']<=result.loc[good,['open','close','high']].min(axis=1)).all()):
        raise ValueError('Invalid adjusted OHLC')
    result['price_adjustment_method']=METHOD
    result['adjustment_status']=np.where(missing,'missing_factor_no_price','adjusted')
    return result


def run(raw_root,inputs,output):
    if output.exists():
        raise ValueError('Output already exists; use a new directory')
    source=json.loads((raw_root/'manifest.json').read_text())
    if source.get('status')!='complete' or set(source['completed'])!=set(source['requested_trading_days']):
        raise ValueError('Incomplete raw dataset')
    factor_file=raw_root/'crsp_daily_adjustment_factors.parquet'
    event_file=inputs/'distributions.parquet'
    reference_file=inputs/'daily_references.parquet'
    factors,anchors,audit,cash=prepare_adjustments(pd.read_parquet(factor_file),
        pd.read_parquet(event_file),pd.read_parquet(reference_file))
    report={'status':'preflight','method':METHOD,'raw_root':str(raw_root.resolve()),
        'cash_price_formula':'product(1 - sum(cash_amount) / previous_close_on_exdate_basis), for t < exdate <= security_anchor',
        'price_formula':'raw_price * base_cfacpr / cfacpr[t] * cash_multiplier[t]',
        'volume_formula':'raw_volume * cfacshr[t] / base_cfacshr; no cash dividend scaling',
        'return_interpretation':'Proportional dividend-adjusted chart prices, not exact reinvested total returns',
        'event_classification':'Verified demo USD cash codes only; unknown codes fail',
        'source_manifest_sha256':sha256(raw_root/'manifest.json'),
        'factor_sha256':sha256(factor_file),'events_sha256':sha256(event_file),
        'references_sha256':sha256(reference_file),'distribution_records':len(audit),
        'treatment_counts':audit.treatment.value_counts().to_dict(),'cash_exdate_groups':len(cash),
        'cash_affected_permnos':int(cash.permno.nunique()) if len(cash) else 0,
        'days':len(source['completed']),'rows':0,'cash_changed_rows':0,'volume_changed_rows':0,
        'missing_factor_no_price_rows':0,'completed':{}}
    for day,item in sorted(source['completed'].items()):
        path=raw_root/item['file']
        if sha256(path)!=item['sha256']:
            raise ValueError('Raw source checksum mismatch')
        adjusted=apply_adjustments(pd.read_parquet(path),factors[factors.date.eq(day)])
        report['rows']+=len(adjusted)
        report['cash_changed_rows']+=int((adjusted.cash_dividend_multiplier.lt(1-1e-12)&adjusted.close.notna()).sum())
        report['volume_changed_rows']+=int((~np.isclose(adjusted.volume,adjusted.volume_raw)).sum())
        report['missing_factor_no_price_rows']+=int(adjusted.adjustment_status.eq('missing_factor_no_price').sum())
    output.mkdir(parents=True)
    bars_dir=output/'bars_5min'
    bars_dir.mkdir()
    for frame,name in [(anchors,'adjustment_anchors'),(factors,'daily_adjustment_factors'),
                       (audit,'distribution_audit'),(cash,'cash_dividend_factors')]:
        frame.to_parquet(output/(name+'.parquet'),index=False,compression='zstd')
    for day,item in sorted(source['completed'].items()):
        path=raw_root/item['file']
        if sha256(path)!=item['sha256']:
            raise ValueError('Raw source changed after preflight')
        adjusted=apply_adjustments(pd.read_parquet(path),factors[factors.date.eq(day)])
        target=bars_dir/f'{day}.parquet'
        adjusted.to_parquet(target,index=False,compression='zstd')
        report['completed'][day]={'file':str(target.relative_to(output)),'rows':len(adjusted),
                                 'sha256':sha256(target),'raw_sha256':item['sha256']}
    report['status']='complete'
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.md').write_text(
        '# Dividend-adjusted intraday demo\n\n'
        'Prices include CFACPR corporate-action adjustments and proportional cash-dividend chart adjustments. '
        'Volume is adjusted only by CFACSHR, never by the cash dividend factor. '
        'Original prices and eligible volume are retained in `*_raw` columns. '
        'Trade counts and missing-price intervals are unchanged. All original trade filters still apply.\n\n'
        'This chart convention does not exactly reproduce a cash-dividend reinvestment total-return index. '
        'Use ex-dates, not payment dates. Each security uses its last available input date as the basis. '
        'Historical adjusted levels are not as-published executable prices.\n\n'
        'See `manifest.json`, `distribution_audit.parquet`, `cash_dividend_factors.parquet`, '
        '`daily_adjustment_factors.parquet`, and `adjustment_anchors.parquet` for provenance.\n')
    print(json.dumps({k:v for k,v in report.items() if k!='completed'},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path.home()/'Market Data/demo_2023_h1')
    p.add_argument('--inputs',type=Path,default=Path.home()/'Market Data/demo_2023_h1_distribution_inputs')
    p.add_argument('--output',type=Path,default=Path.home()/'Market Data/demo_2023_h1_dividend_adjusted')
    a=p.parse_args()
    run(a.raw_root,a.inputs,a.output)
