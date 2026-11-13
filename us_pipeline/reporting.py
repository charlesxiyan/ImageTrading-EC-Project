"""Classification, probability, stability, and dependent-sample evaluation reports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,
    confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score)


def binary_metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all():
        raise ValueError('Expected nonempty aligned binary labels/probabilities')
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid probabilities')
    predicted=(p>=threshold).astype(int);clipped=np.clip(p,1e-7,1-1e-7)
    both=len(np.unique(y))==2
    return {'samples':len(y),'positive_fraction':float(y.mean()),'threshold':threshold,
        'accuracy':float(accuracy_score(y,predicted)),
        'balanced_accuracy':float(balanced_accuracy_score(y,predicted)),
        'precision_up':float(precision_score(y,predicted,zero_division=0)),
        'recall_up':float(recall_score(y,predicted,zero_division=0)),
        'f1_up':float(f1_score(y,predicted,zero_division=0)),
        'f1_macro':float(f1_score(y,predicted,labels=[0,1],average='macro',zero_division=0)),
        'mcc':float(matthews_corrcoef(y,predicted)) if both else None,
        'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'average_precision_up':float(average_precision_score(y,p)) if both else None,
        'log_loss':float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
        'brier_score':float(np.mean((p-y)**2)),
        'confusion_matrix_true_rows':confusion_matrix(y,predicted,labels=[0,1]).tolist()}


def calibration_table(y,p,bins=10):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    bucket=np.minimum((p*bins).astype(int),bins-1)
    rows=[]
    for i in range(bins):
        mask=bucket==i
        rows.append({'bin':i,'lower':i/bins,'upper':(i+1)/bins,'samples':int(mask.sum()),
            'mean_probability_up':float(p[mask].mean()) if mask.any() else None,
            'observed_up_fraction':float(y[mask].mean()) if mask.any() else None})
    frame=pd.DataFrame(rows)
    ece=float((frame.samples*(frame.mean_probability_up-frame.observed_up_fraction).abs()).sum()/len(y))
    return frame,ece


def paired_block_interval(frame,majority_class,block_days=5,repetitions=2000,seed=69):
    """Resample consecutive decision-day blocks, keeping all stocks/windows per day."""
    decision=pd.to_datetime(frame.decision_time).dt.tz_convert('America/New_York')
    x=pd.DataFrame({'date':decision.dt.strftime('%Y-%m-%d'),
        'correct':(frame.probability_up.to_numpy()>=.5)==frame.label.to_numpy(),
        'baseline_correct':frame.label.eq(majority_class).to_numpy()})
    daily=x.groupby('date',sort=True).agg(n=('correct','size'),correct=('correct','sum'),baseline=('baseline_correct','sum'))
    n=len(daily);point=float((daily.correct.sum()-daily.baseline.sum())/daily.n.sum())
    report={'estimate_accuracy_difference':point,'decision_days':n,'requested_block_days':block_days,
        'repetitions':repetitions,'seed':seed,
        'interpretation':'Paired moving-day-block bootstrap, conditional on this fixed model; not an IID-window interval or model-selection uncertainty'}
    if n<2:
        return {**report,'effective_block_days':None,'lower_95':None,'upper_95':None}
    block=min(block_days,n);rng=np.random.default_rng(seed);v=daily[['n','correct','baseline']].to_numpy(dtype=float)
    samples=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        indices=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        a=v[indices].sum(axis=0);samples.append((a[1]-a[2])/a[0])
    low,high=np.quantile(samples,[.025,.975])
    return {**report,'effective_block_days':block,'lower_95':float(low),'upper_95':float(high)}


def grouped_scores(frame,key,majority_class):
    x=frame.copy();p=x.probability_up.to_numpy();y=x.label.to_numpy()
    x['correct']=((p>=.5)==y).astype(int)
    x['baseline_correct']=(y==majority_class).astype(int)
    clipped=np.clip(p,1e-7,1-1e-7)
    x['cross_entropy']=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    result=x.groupby(key,dropna=False,sort=True).agg(samples=('label','size'),
        accuracy=('correct','mean'),majority_baseline_accuracy=('baseline_correct','mean'),
        positive_fraction=('label','mean'),log_loss=('cross_entropy','mean')).reset_index()
    result['accuracy_difference']=result.accuracy-result.majority_baseline_accuracy
    return result


def write_evaluation(frame,output,training_prior,split,timing,seed=69,bootstrap_repetitions=2000,smoke_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    majority=int(training_prior>=.5);y=frame.label.to_numpy();p=frame.probability_up.to_numpy()
    score=binary_metrics(y,p)
    calibration,ece=calibration_table(y,p);score['binary_probability_ece_10_bins']=ece
    baselines=[]
    for name,prob in [('CNN78u',p),('training_prior',np.full(len(y),training_prior))]:
        baselines.append({'name':name,**binary_metrics(y,prob)})
    for name,ret in [('last_bar_direction',frame.last_bar_return),('lookback_direction',frame.lookback_return)]:
        direction=np.where(ret>0,1,np.where(ret<0,0,majority))
        row={'name':name,**binary_metrics(y,direction.astype(float))}
        # Hard directional rules do not supply calibrated probability forecasts.
        for key in ['log_loss','brier_score','roc_auc','average_precision_up']:
            row[key]=None
        baselines.append(row)
    baseline_table=pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix_true_rows'} for r in baselines])
    decision=pd.to_datetime(frame.decision_time).dt.tz_convert('America/New_York')
    label_end=pd.to_datetime(frame.label_end_time).dt.tz_convert('America/New_York')
    decorated=frame.copy();decorated['decision_date']=decision.dt.strftime('%Y-%m-%d')
    decorated['decision_month']=decision.dt.strftime('%Y-%m')
    decorated['target_session']=np.where(decision.dt.date==label_end.dt.date,'intraday','overnight_or_holiday')
    groups={name:grouped_scores(decorated,key,majority) for name,key in [
        ('by_day','decision_date'),('by_month','decision_month'),('by_security','permno'),('by_target_session','target_session')]}
    coverage=[]
    confidence=np.maximum(p,1-p)
    for threshold in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
        mask=confidence>=threshold
        coverage.append({'confidence_threshold':threshold,'samples':int(mask.sum()),'coverage':float(mask.mean()),
            'accuracy':float(((p[mask]>=.5)==y[mask]).mean()) if mask.any() else None})
    interval=paired_block_interval(frame,majority,repetitions=bootstrap_repetitions,seed=seed)
    report={'split':split,'smoke_only':smoke_only,'metrics':score,'training_prior_probability':training_prior,
        'majority_class_fixed_from_training':majority,'timing':timing,
        'paired_accuracy_difference_interval':interval,
        'cautions':['Average precision is AP, not trapezoidal PR area.',
                    'Overlapping windows are dependent; the interval uses decision-day blocks.',
                    'Probability buckets and coverage curves are descriptive, not test-selected trading thresholds.',
                    'Accuracy does not establish profitability or executable returns.']}
    frame.to_parquet(output/'predictions.parquet',index=False,compression='zstd')
    calibration.to_csv(output/'calibration.csv',index=False)
    baseline_table.to_csv(output/'baselines.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'confidence_coverage.csv',index=False)
    for name,table in groups.items():
        table.to_csv(output/(name+'.csv'),index=False)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    make_figures(frame,calibration,groups['by_day'],pd.DataFrame(coverage),output,score,split,smoke_only)
    return report


def make_figures(frame,calibration,daily,coverage,output,score,split,smoke_only):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.suptitle('SMOKE CHECK: not a performance estimate' if smoke_only else split.capitalize()+' evaluation',fontsize=14)
    confusion=np.array(score['confusion_matrix_true_rows'])
    axes[0,0].imshow(confusion,cmap='Blues');axes[0,0].set(title='Confusion matrix',xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1])
    for i in range(2):
        for j in range(2):
            axes[0,0].text(j,i,f'{confusion[i,j]:,}',ha='center',va='center',
                color='white' if confusion[i,j]>confusion.max()/2 else 'black')
    y=frame.label.to_numpy();p=frame.probability_up.to_numpy()
    if len(np.unique(y))==2:
        fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axes[0,1].plot(fpr,tpr,label=f'ROC AUC={score["roc_auc"]:.4f}');axes[0,1].plot([0,1],[0,1],'--',color='gray')
        axes[0,1].legend();axes[0,2].plot(recall,precision,label=f'AP={score["average_precision_up"]:.4f}')
        axes[0,2].axhline(y.mean(),linestyle='--',color='gray');axes[0,2].legend()
    axes[0,1].set(title='ROC',xlabel='False positive rate',ylabel='True positive rate')
    axes[0,2].set(title='Precision-recall',xlabel='Recall',ylabel='Precision')
    keep=calibration.samples.gt(0)
    axes[1,0].plot(calibration.loc[keep,'mean_probability_up'],calibration.loc[keep,'observed_up_fraction'],'o-')
    axes[1,0].plot([0,1],[0,1],'--',color='gray');axes[1,0].set(title='Probability calibration',xlabel='Mean predicted P(up)',ylabel='Observed up fraction')
    axes[1,1].plot(pd.to_datetime(daily.decision_date),daily.accuracy,label='CNN78u')
    axes[1,1].plot(pd.to_datetime(daily.decision_date),daily.majority_baseline_accuracy,label='Training-majority baseline')
    axes[1,1].legend();axes[1,1].tick_params(axis='x',rotation=35);axes[1,1].set(title='Daily accuracy',ylabel='Accuracy')
    axes[1,2].plot(coverage.coverage,coverage.accuracy,'o-');axes[1,2].set(title='Confidence / coverage (descriptive)',xlabel='Fraction retained',ylabel='Accuracy')
    fig.tight_layout();fig.savefig(output/'evaluation.png',dpi=150);plt.close(fig)
