"""Align source bars before screening windows; never compress missing time slots."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
FEATURES=['open','high','low','close','volume']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path):
    config=json.loads(Path(path).read_text())
    if config['lookback']!=78 or config['bar_minutes']!=5:
        raise ValueError('The active pipeline uses 78 five-minute input intervals (CNN78u)')
    if config.get('model',{}).get('name')!='CNN78u':
        raise ValueError('The active model must be CNN78u')
    if config['horizon']<1 or config['stride']<1 or config['zero_return_epsilon']<0:
        raise ValueError('Invalid horizon, stride, or zero-return tolerance')
    if not config['train_end']<config['validation_end']:
        raise ValueError('Chronological boundaries must be increasing')
    if not config['allow_cross_session'] and config['lookback']+config['horizon']>390//config['bar_minutes']:
        raise ValueError('Lookback plus horizon cannot fit inside one session; shorten them or allow cross-session windows')
    for key in ('artifact_root','data_root'):
        path=Path(config[key]).expanduser()
        config[key]=str(path if path.is_absolute() else PROJECT/path)
    return config


def sample_indices(valid,close,timestamps,config,permnos):
    """Use the original trading-clock grid for both lookbacks and return horizons."""
    l,h,stride=config['lookback'],config['horizon'],config['stride']
    clock=pd.DatetimeIndex(timestamps)
    dates=np.array(clock.tz_convert('America/New_York').strftime('%Y-%m-%d'))
    cut1=int(np.flatnonzero(dates<=config['train_end'])[-1]) if np.any(dates<=config['train_end']) else -1
    cut2=int(np.flatnonzero(dates<=config['validation_end'])[-1]) if np.any(dates<=config['validation_end']) else -1
    parts={s:[] for s in ('train','validation','test')}
    stats={'candidate_windows':0,'invalid_input':0,'invalid_future':0,
           'cross_session_disallowed':0,'purged_split_boundary':0,'zero_return':0,'kept':0}
    if len(clock)<l+h:
        raise ValueError('Not enough time slots for lookback plus horizon')
    ends=np.arange(l-1,len(clock)-h,stride,dtype=np.int32)
    for stock,permno in enumerate(permnos):
        stats['candidate_windows']+=len(ends)
        bad=np.r_[0,np.cumsum(~np.asarray(valid[stock],dtype=bool))]
        good_input=(bad[ends+1]-bad[ends-l+1])==0
        stats['invalid_input']+=int((~good_input).sum())
        e=ends[good_input]
        good_future=(bad[e+h+1]-bad[e+1])==0
        stats['invalid_future']+=int((~good_future).sum());e=e[good_future]
        if not config['allow_cross_session']:
            same=dates[e-l+1]==dates[e+h]
            stats['cross_session_disallowed']+=int((~same).sum());e=e[same]
        splits=np.where(e<=cut1,0,np.where(e<=cut2,1,2))
        limits=np.where(splits==0,cut1,np.where(splits==1,cut2,len(clock)-1))
        contained=e+h<=limits
        stats['purged_split_boundary']+=int((~contained).sum())
        e,splits=e[contained],splits[contained]
        returns=close[stock,e+h]/close[stock,e]-1.
        nonzero=np.abs(returns)>config['zero_return_epsilon']
        stats['zero_return']+=int((~nonzero).sum())
        e,splits,returns=e[nonzero],splits[nonzero],returns[nonzero]
        stats['kept']+=len(e)
        for number,name in enumerate(parts):
            mask=splits==number
            if mask.any():
                t=e[mask];r=returns[mask]
                parts[name].append(pd.DataFrame({'security_index':np.full(len(t),stock,dtype=np.int32),
                    'permno':np.full(len(t),int(permno),dtype=np.int64),'end_index':t,
                    'label':(r>0).astype('int8'),'forward_return':r,
                    'decision_time':clock[t],'label_end_time':clock[t+h]}))
    output={}
    for name,frames in parts.items():
        output[name]=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=[
            'security_index','permno','end_index','label','forward_return','decision_time','label_end_time'])
    return output,stats


def prepare(config):
    root=Path(config['data_root']);out=Path(config['artifact_root'])/'prepared'
    if out.exists():
        raise ValueError(f'Prepared output exists: {out}; choose a new artifact_root')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('method')!='crsp_cash_dividend_chart_and_share_volume_v1':
        raise ValueError('Expected complete verified dividend-adjusted demo')
    entries=sorted(manifest['completed'].items())
    if not entries or entries[0][0]<'2023-01-01' or entries[-1][0]>'2023-06-30':
        raise ValueError('Extend the session-calendar adapter before using another period')
    if len(entries)!=manifest['days']:
        raise ValueError('Source manifest day count mismatch')
    # Check the raw trading calendar so a missing whole session cannot be bridged.
    raw_manifest=json.loads((Path(manifest['raw_root'])/'manifest.json').read_text())
    if [d for d,_ in entries]!=sorted(raw_manifest['requested_trading_days']):
        raise ValueError('Source is missing an expected trading session')
    all_ids=set()
    for day,item in entries:
        path=root/item['file']
        if digest(path)!=item['sha256']:
            raise ValueError(f'Checksum mismatch: {day}')
        all_ids.update(int(x) for x in pd.read_parquet(path,columns=['permno']).permno.unique())
    ids=sorted(all_ids);lookup={p:i for i,p in enumerate(ids)}
    slots=390//config['bar_minutes'];t=len(entries)*slots;n=len(ids)
    out.mkdir(parents=True)
    features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float64',shape=(n,t,5))
    valid=np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype='bool',shape=(n,t))
    scales=np.lib.format.open_memmap(out/'multipliers.npy',mode='w+',dtype='float64',shape=(n,t,2))
    features[:]=np.nan;valid[:]=False;scales[:]=np.nan
    clock=[]
    for number,(day,item) in enumerate(entries):
        frame=pd.read_parquet(root/item['file'])
        if frame.duplicated(['permno','bar_start']).any() or not frame.date.eq(day).all():
            raise ValueError('Duplicate bar key or inconsistent date')
        starts=pd.DatetimeIndex(frame.bar_start)
        origin=pd.Timestamp(day+' 09:30',tz='America/New_York')
        offsets=np.asarray((starts-origin)/pd.Timedelta(minutes=5))
        if not (np.isfinite(offsets)&(offsets==np.floor(offsets))&(offsets>=0)&(offsets<78)).all():
            raise ValueError('Unexpected source session grid')
        if not (frame.groupby('permno').size()==78).all():
            raise ValueError('Source does not preserve every expected five-minute slot')
        si=frame.permno.map(lookup).to_numpy(dtype=np.int32);ti=offsets.astype(np.int32)
        x=frame[FEATURES].to_numpy(dtype=float)
        good=(frame.status.eq('observed').to_numpy()&np.isfinite(x).all(axis=1)
            &(x>0).all(axis=1)&(x[:,1]>=x[:,:4].max(axis=1))&(x[:,2]<=x[:,:4].min(axis=1)))
        mult=frame[['price_adjustment_multiplier','volume_adjustment_multiplier']].to_numpy(dtype=float)
        if not (np.isfinite(mult[good])&(mult[good]>0)).all():
            raise ValueError('Observed bar has invalid adjustment multipliers')
        daily=np.full((n,78,5),np.nan);ok=np.zeros((n,78),dtype=bool)
        daily_mult=np.full((n,78,2),np.nan)
        daily[si,ti]=x;ok[si,ti]=good;daily_mult[si,ti]=mult
        sl=slice(number*slots,(number+1)*slots)
        features[:,sl]=daily;valid[:,sl]=ok;scales[:,sl]=daily_mult
        clock.extend(pd.date_range(origin,periods=slots,freq=f'{config["bar_minutes"]}min')
                     +pd.Timedelta(minutes=config['bar_minutes']))
    features.flush();valid.flush();scales.flush()
    samples,screening=sample_indices(valid,features[:,:,3],clock,config,ids)
    summary={}
    for split,frame in samples.items():
        if frame.empty:
            raise ValueError(f'No {split} samples; inspect boundaries/window/session settings')
        frame.to_parquet(out/f'{split}.parquet',index=False,compression='zstd')
        counts={str(int(k)):int(v) for k,v in frame.label.value_counts().sort_index().items()}
        majority=max(counts.values())/len(frame)
        summary[split]={'samples':len(frame),'label_counts':counts,
            'positive_fraction':float(frame.label.mean()),'majority_fraction':majority,
            'decision_start':str(frame.decision_time.min()),'decision_end':str(frame.decision_time.max()),
            'label_end_max':str(frame.label_end_time.max())}
    metadata={'status':'complete','config':config,'source_manifest_sha256':digest(root/'manifest.json'),
        'shape':[n,t,5],'permnos':ids,'features':FEATURES,'decision_clock_ns':pd.DatetimeIndex(clock).asi8.tolist(),
        'screening':screening,'splits':summary,'label_definition':'1 if adjusted_close[t+H]/adjusted_close[t]-1 > 0; 0 if < 0; zero returns excluded',
        'label_interpretation':'Dividend-adjusted chart-price return, not exact reinvested total return',
        'price_basis':'features.npy holds end-of-data-basis prices (raw * multipliers.npy). Model inputs divide each window by its last bar\'s multipliers (point in time: anchor = last input bar); never use stored levels directly.',
        'preprocessing':'No imputation, SMOTE, class balancing, or global fitted scaler. Window-local OHLC axis and volume maximum only.',
        'session_policy':'Consecutive scheduled bars; normal overnight/holiday closures allowed by config; missing bars/membership gaps never compressed.'}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'screening':screening,'splits':summary,'prepared':str(out)},indent=2))
    return metadata


def audit_prepared(config):
    """Check saved sample indices against aligned time slots and source features."""
    root=Path(config['artifact_root'])/'prepared'
    m=json.loads((root/'manifest.json').read_text())
    x=np.load(root/'features.npy',mmap_mode='r');valid=np.load(root/'valid.npy',mmap_mode='r')
    clock=pd.to_datetime(m['decision_clock_ns'],unit='ns',utc=True).tz_convert('America/New_York')
    dates=np.array(clock.strftime('%Y-%m-%d'))
    l,h=m['config']['lookback'],m['config']['horizon']
    checked=0;report={}
    for name in ('train','validation','test'):
        frame=pd.read_parquet(root/f'{name}.parquet')
        for stock,g in frame.groupby('security_index',sort=False):
            e=g.end_index.to_numpy(dtype=int)
            if len(np.unique(e))!=len(e) or not g.permno.eq(m['permnos'][stock]).all():
                raise ValueError('Duplicate sample or inconsistent security identity')
            bad=np.r_[0,np.cumsum(~valid[stock])]
            if not (bad[e+h+1]-bad[e-l+1]==0).all():
                raise ValueError('Sample contains a missing input/future slot')
            r=x[stock,e+h,3]/x[stock,e,3]-1
            np.testing.assert_allclose(g.forward_return,r,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(g.label,(r>0).astype('int8'))
            np.testing.assert_array_equal(g.decision_time.to_numpy(),clock[e].to_numpy())
            np.testing.assert_array_equal(g.label_end_time.to_numpy(),clock[e+h].to_numpy())
            if name=='train':
                good=dates[e+h]<=config['train_end']
            elif name=='validation':
                good=(dates[e]>config['train_end'])&(dates[e+h]<=config['validation_end'])
            else:
                good=dates[e]>config['validation_end']
            if not good.all():
                raise ValueError('Sample crosses chronological split boundary')
        if len(frame)!=m['splits'][name]['samples']:
            raise ValueError('Manifest sample count mismatch')
        report[name]=len(frame);checked+=len(frame)
    if checked!=m['screening']['kept']:
        raise ValueError('Total sample count mismatch')
    report={'status':'validated_complete','checked_samples':checked,'splits':report,
        'checks':['complete original-grid inputs and future horizons','no missing-slot bridging',
                  'PERMNO identity','label sign and exact H-bar endpoint','split boundary purging',
                  'saved decision and label timestamps','manifest sample counts']}
    (root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report
