"""Portable Colab experiment runner with timing, epoch resume, and held-out reports."""
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from tqdm.auto import tqdm

from .data import digest
from .dataset import ImageDataset
from .models import build_model
from .reporting import binary_metrics,write_evaluation

DEFAULTS={'epochs':3,'batch_size':64,'learning_rate':.001,'weight_decay':.0001,
          'num_workers':2,'seed':69,'amp':False,'patience_epochs':3,'smoke':False}


def sync_cuda(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def runtime_info(device):
    import importlib.metadata
    versions={}
    for name in ['torch','numpy','pandas','pyarrow','scikit-learn','Pillow']:
        versions[name]=importlib.metadata.version(name)
    result={'python':sys.version,'platform':platform.platform(),'device':str(device),'versions':versions,
            'cuda_runtime':torch.version.cuda,'torch_threads':torch.get_num_threads()}
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        result.update(gpu=props.name,gpu_total_bytes=props.total_memory)
    return result


def seed_worker(worker):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def settings_for(values):
    s={**DEFAULTS,**values}
    if s['epochs']<1 or s['batch_size']<1 or s['num_workers']<0 or s['patience_epochs']<1:
        raise ValueError('Invalid epoch/batch/worker/patience settings')
    return s


def load_data(prepared):
    prepared=Path(prepared)
    m=json.loads((prepared/'manifest.json').read_text());c=m['config']
    if m.get('status')!='complete' or c['lookback']!=78 or c['horizon']!=1 or c['bar_minutes']!=5:
        raise ValueError('Expected complete CNN78u five-minute, next-bar prepared data')
    files=['manifest.json','features.npy','multipliers.npy','valid.npy',
           'train.parquet','validation.parquet','test.parquet']
    fingerprint={name:digest(prepared/name) for name in files}
    signature=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    return m,c,signature


def make_loader(dataset,s,training=False):
    options={'batch_size':s['batch_size'],'shuffle':training,'drop_last':False,
             'num_workers':s['num_workers'],'pin_memory':torch.cuda.is_available(),
             'generator':torch.Generator().manual_seed(s['seed']),'worker_init_fn':seed_worker}
    if s['num_workers']:
        # Limit queued image tensors in hosted runtimes with small shared memory.
        options.update(persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    return DataLoader(dataset,**options)


def view(dataset,smoke,seed):
    if not smoke:
        return dataset
    rng=np.random.default_rng(seed)
    return Subset(dataset,rng.choice(len(dataset),min(16,len(dataset)),replace=False).tolist())


def mirror(path,persistent):
    if persistent is None:
        return
    persistent=Path(persistent);persistent.mkdir(parents=True,exist_ok=True)
    target=persistent/path.name;temporary=target.with_name(target.name+'.copying')
    shutil.copy2(path,temporary);os.replace(temporary,target)


def save_json(path,value,persistent=None):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
    mirror(path,persistent)


def save_checkpoint(path,value,persistent=None):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary)
    os.replace(temporary,path);mirror(path,persistent)


def predict(model,dataset,loader,device,amp=False,progress=True):
    model.eval();probabilities=[];indices=[]
    sync_cuda(device);start=time.perf_counter()
    with torch.inference_mode():
        for x,_,index in tqdm(loader,desc='Inference',disable=not progress,leave=False):
            with torch.autocast(device_type=device.type,enabled=amp):
                scores=model(x.to(device,non_blocking=True))
            probabilities.append(scores.float().softmax(1)[:,1].cpu().numpy())
            indices.append(index.numpy())
    sync_cuda(device);elapsed=time.perf_counter()-start
    selected=np.concatenate(indices);p=np.concatenate(probabilities)
    result=dataset.samples.iloc[selected].copy().reset_index(drop=True)
    result['probability_up']=p
    if dataset._features is None:
        dataset.window(int(selected[0]))
    x=dataset._features;s=result.security_index.to_numpy();t=result.end_index.to_numpy();l=dataset.config['lookback']
    result['last_bar_return']=x[s,t,3]/x[s,t-1,3]-1
    result['lookback_return']=x[s,t,3]/x[s,t-l+1,3]-1
    timing={'end_to_end_seconds':elapsed,'samples':len(result),'samples_per_second':len(result)/elapsed,
        'milliseconds_per_sample_amortized':elapsed/len(result)*1000,
        'scope':'DataLoader, rendering, transfer, inference, and probability collection; excludes metrics/file serialization'}
    return result,timing


def _signature(data_signature,config,s):
    # Epoch count may be extended on an epoch-boundary resume; other settings fixed.
    relevant={k:v for k,v in s.items() if k!='epochs'}
    sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    return hashlib.sha256(json.dumps({'data':data_signature,'model':config['model'],
        'settings':relevant,'sources':sources},sort_keys=True).encode()).hexdigest()


def train_experiment(prepared,run_dir,settings,device=None,persistent_dir=None,resume=False,progress=True):
    started=time.perf_counter();s=settings_for(settings)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if s['amp'] and device.type!='cuda':
        raise ValueError('Mixed precision is supported here only on CUDA')
    random.seed(s['seed']);np.random.seed(s['seed']);torch.manual_seed(s['seed'])
    m,config,data_signature=load_data(prepared);signature=_signature(data_signature,config,s)
    run=Path(run_dir)
    if run.exists() and not resume:
        raise ValueError('Run directory exists; choose a new RUN_ID or resume explicitly')
    run.mkdir(parents=True,exist_ok=True)
    if (run/'test/metrics.json').exists():
        raise ValueError('This run has already evaluated test; do not continue training against that holdout')
    training=ImageDataset(prepared,'train');validation=ImageDataset(prepared,'validation')
    prior=float(training.samples.label.mean())
    train_view=view(training,s['smoke'],s['seed']);validation_view=view(validation,s['smoke'],s['seed']+1)
    train_loader=make_loader(train_view,s,True);validation_loader=make_loader(validation_view,s,False)
    model=build_model(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=s['amp'])
    history=[];best=float('inf');stale=0;first_epoch=1;step=0
    if resume:
        checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['signature']!=signature:
            raise ValueError('Resume settings, source code, or data differ from checkpoint')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']);history=checkpoint['history']
        best=checkpoint['best_loss'];stale=checkpoint['stale'];step=checkpoint['step']
        first_epoch=checkpoint['completed_epoch']+1
        torch.set_rng_state(checkpoint['torch_rng']);random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        if device.type=='cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print(f'Resuming at epoch {first_epoch}; completed epochs are not repeated')
    if stale>=s['patience_epochs']:
        raise ValueError('The saved run already stopped for patience; retain the original completed experiment')
    setup_seconds=time.perf_counter()-started
    session_start=time.perf_counter()
    for epoch in range(first_epoch,s['epochs']+1):
        model.train();train_loader.generator.manual_seed(s['seed']+epoch)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync_cuda(device);epoch_start=time.perf_counter();train_start=epoch_start
        total_loss=torch.zeros((),device=device);correct=torch.zeros((),device=device,dtype=torch.int64)
        seen=0;host_wait=0.;iterator=iter(train_loader)
        iterator_progress=tqdm(range(len(train_loader)),desc=f'Epoch {epoch}/{s["epochs"]}',disable=not progress)
        for batch in iterator_progress:
            waiting=time.perf_counter();x,y,_=next(iterator);host_wait+=time.perf_counter()-waiting
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=s['amp']):
                scores=model(x);loss=nn.functional.cross_entropy(scores,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            n=len(y);total_loss+=loss.detach()*n;correct+=(scores.detach().argmax(1)==y).sum()
            seen+=n;step+=1
            if batch%100==0:
                if not torch.isfinite(loss).item():
                    raise ValueError('Nonfinite training loss')
                iterator_progress.set_postfix(loss=float((total_loss/seen).item()))
        sync_cuda(device);train_seconds=time.perf_counter()-train_start
        validation_frame,val_timing=predict(model,validation,validation_loader,device,s['amp'],progress)
        val_metrics=binary_metrics(validation_frame.label,validation_frame.probability_up)
        improved=val_metrics['log_loss']<best
        if improved:
            best=val_metrics['log_loss'];stale=0
        else:
            stale+=1
        record={'epoch':epoch,'optimizer_steps_total':step,'train_samples':seen,
            'train_loss':float((total_loss/seen).item()),'train_accuracy':float((correct/seen).item()),
            'train_seconds':train_seconds,'train_samples_per_second':seen/train_seconds,
            'host_data_wait_seconds_not_additive':host_wait,
            'validation_seconds':val_timing['end_to_end_seconds'],
            **{'val_'+key:val_metrics[key] for key in ['log_loss','accuracy','balanced_accuracy','roc_auc','average_precision_up','brier_score']},
            'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
            'gpu_peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else None}
        history.append(record)
        checkpoint={'signature':signature,'data_signature':data_signature,'config':config,'settings':s,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'completed_epoch':epoch,'step':step,'history':history,'best_loss':best,'stale':stale,
            'training_prior':prior,'smoke_only':s['smoke'],'torch_rng':torch.get_rng_state(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        saving=time.perf_counter()
        if improved:
            save_checkpoint(run/'best.pt',checkpoint,persistent_dir)
        save_checkpoint(run/'last.pt',checkpoint,persistent_dir)
        record['checkpoint_sync_seconds']=time.perf_counter()-saving
        record['epoch_wall_seconds']=time.perf_counter()-epoch_start
        pd.DataFrame(history).to_csv(run/'history.csv',index=False);mirror(run/'history.csv',persistent_dir)
        save_json(run/'epoch_status.json',{'completed_epoch':epoch,'best_validation_log_loss':best,
            'smoke_only':s['smoke'],'test_evaluated':False},persistent_dir)
        print(json.dumps(record),flush=True)
        if stale>=s['patience_epochs']:
            break
    # Best checkpoint determines the final validation report; no test access here.
    best_checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_checkpoint['model'])
    val_frame,val_timing=predict(model,validation,validation_loader,device,s['amp'],progress)
    validation_report=write_evaluation(val_frame,run/'validation',prior,'validation',val_timing,
        seed=s['seed'],bootstrap_repetitions=100 if s['smoke'] else 2000,smoke_only=s['smoke'])
    sync_tree(run/'validation',None if persistent_dir is None else Path(persistent_dir)/'validation')
    peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    report={'status':'smoke_only' if s['smoke'] else 'training_complete','settings':s,
        'signature':signature,'data_signature':data_signature,'runtime':runtime_info(device),
        'completed_epochs':len(history),'best_epoch':best_checkpoint['completed_epoch'],
        'best_validation_log_loss':best_checkpoint['best_loss'],'history':history,
        'setup_seconds_this_session':setup_seconds,'wall_seconds_this_session':time.perf_counter()-session_start,
        'training_seconds_recorded':sum(r['train_seconds'] for r in history),
        'validation_seconds_recorded':sum(r['validation_seconds'] for r in history),
        'main_process_peak_rss_bytes_excludes_workers':int(peak_rss),
        'parameters':sum(p.numel() for p in model.parameters()),'test_evaluated':False,
        'validation_metrics':validation_report['metrics'],
        'timing_notes':['Epoch training time includes loading/rendering/transfers/optimization.',
                        'Host data wait may overlap GPU work; do not add it to GPU time.',
                        'Final validation reporting and serialization are additional overhead.',
                        'Resume timings in a last checkpoint may omit its final sync overhead.']}
    save_json(run/'training_report.json',report,persistent_dir)
    history_figure(history,run/'training_curves.png');mirror(run/'training_curves.png',persistent_dir)
    return report


def sync_tree(source,destination):
    if destination is None:
        return
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for path in Path(source).iterdir():
        if path.is_file():
            mirror(path,destination)


def history_figure(history,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=pd.DataFrame(history);fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(h.epoch,h.train_loss,label='Train');axes[0].plot(h.epoch,h.val_log_loss,label='Validation')
    axes[0].set(title='Cross-entropy',xlabel='Epoch');axes[0].legend()
    axes[1].plot(h.epoch,h.train_accuracy,label='Train');axes[1].plot(h.epoch,h.val_accuracy,label='Validation')
    axes[1].set(title='Accuracy',xlabel='Epoch');axes[1].legend()
    axes[2].bar(h.epoch,h.train_seconds,label='Train');axes[2].bar(h.epoch,h.validation_seconds,bottom=h.train_seconds,label='Validation')
    axes[2].set(title='Timed data/model stages',xlabel='Epoch',ylabel='Seconds');axes[2].legend()
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def benchmark_model(model,x,device,warmup=10,repetitions=30,amp=False):
    model.eval();x=x.to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type,enabled=amp):
                model(x)
        sync_cuda(device);times=[]
        for _ in range(repetitions):
            if device.type=='cuda':
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast(device_type=device.type,enabled=amp):
                    model(x)
                end.record();end.synchronize();milliseconds=start.elapsed_time(end)
            else:
                started=time.perf_counter()
                model(x);milliseconds=(time.perf_counter()-started)*1000
            times.append(milliseconds)
    median=float(np.median(times))
    return {'batch_size':len(x),'warmup':warmup,'repetitions':repetitions,
        'median_batch_ms':median,'p95_batch_ms':float(np.quantile(times,.95)),
        'median_ms_per_sample_amortized':median/len(x),'model_samples_per_second':len(x)*1000/median,
        'scope':'Preloaded device tensor; excludes loading, rendering, host-to-device copy, and report serialization'}


def benchmark_checkpoint(prepared,run_dir,device=None,persistent_dir=None,warmup=10,repetitions=30):
    run=Path(run_dir);device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    data=ImageDataset(prepared,'validation')
    sample=torch.stack([data[i][0] for i in range(min(checkpoint['settings']['batch_size'],len(data)))])
    model=build_model(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model'])
    results=[benchmark_model(model,sample[:1],device,warmup,repetitions,checkpoint['settings']['amp']),
             benchmark_model(model,sample,device,warmup,repetitions,checkpoint['settings']['amp'])]
    report={'checkpoint_sha256':digest(run/'best.pt'),'probe_split':'validation',
            'runtime':runtime_info(device),'benchmarks':results,'smoke_only':checkpoint['smoke_only']}
    save_json(run/'hardware_benchmark.json',report,persistent_dir)
    return report


def final_test(prepared,run_dir,device=None,persistent_dir=None,progress=True):
    run=Path(run_dir);device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint_path=run/'best.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['smoke_only']:
        raise ValueError('A smoke checkpoint cannot provide a formal held-out test result')
    _,config,data_signature=load_data(prepared)
    if data_signature!=checkpoint['data_signature']:
        raise ValueError('Test data fingerprint differs from training data')
    result_path=run/'test/metrics.json'
    if result_path.exists():
        prior=json.loads(result_path.read_text())
        if prior.get('checkpoint_sha256')!=digest(checkpoint_path):
            raise ValueError('Test was already evaluated with a different checkpoint')
        return prior
    s=checkpoint['settings'];dataset=ImageDataset(prepared,'test');loader=make_loader(dataset,s,False)
    model=build_model(config).to(device);model.load_state_dict(checkpoint['model'])
    frame,timing=predict(model,dataset,loader,device,s['amp'],progress)
    report=write_evaluation(frame,run/'test',checkpoint['training_prior'],'test',timing,seed=s['seed'])
    benchmark_path=run/'hardware_benchmark.json'
    benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else benchmark_checkpoint(
        prepared,run,device,persistent_dir)
    if benchmark['checkpoint_sha256']!=digest(checkpoint_path):
        raise ValueError('Hardware benchmark used a different checkpoint')
    report['model_only_benchmarks']=benchmark['benchmarks']
    report['checkpoint_sha256']=digest(checkpoint_path)
    save_json(run/'test/metrics.json',report)
    sync_tree(run/'test',None if persistent_dir is None else Path(persistent_dir)/'test')
    return report
