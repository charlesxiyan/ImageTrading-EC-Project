"""Unweighted training, fixed chronological validation, and explicit test evaluation."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset

from .dataset import ImageDataset
from .models import build_model
from .data import digest


def device_for(name):
    if name!='auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def metrics(labels,probabilities,loss,majority_class):
    y=np.asarray(labels);p=np.asarray(probabilities);pred=p>=.5
    counts=[[int(((y==i)&(pred==j)).sum()) for j in (0,1)] for i in (0,1)]
    recalls=[counts[i][i]/sum(counts[i]) for i in (0,1) if sum(counts[i])]
    return {'samples':len(y),'loss':loss/len(y),'accuracy':float((pred==y).mean()),
        'balanced_accuracy':float(np.mean(recalls)),
        'training_majority_class_baseline_accuracy':float((y==majority_class).mean()),
        'confusion_matrix_true_rows':counts}


def evaluate(model,loader,device,majority_class,max_batches=None,predictions=False):
    model.eval();labels=[];probabilities=[];indices=[];loss=0.
    with torch.no_grad():
        for batch,(x,y,i) in enumerate(loader):
            if max_batches is not None and batch>=max_batches:
                break
            output=model(x.to(device));target=y.to(device)
            loss+=float(nn.functional.cross_entropy(output,target,reduction='sum').item())
            labels.extend(y.numpy().tolist())
            probabilities.extend(output.softmax(dim=1)[:,1].cpu().numpy().tolist())
            indices.extend(i.numpy().tolist())
    if not labels:
        raise ValueError('Empty evaluation')
    result=metrics(labels,probabilities,loss,majority_class)
    return result,(indices,probabilities) if predictions else None


def train(config,smoke=False):
    settings=config['training'];seed=config['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    device=device_for(settings['device'])
    prepared=Path(config['artifact_root'])/'prepared'
    train_data=ImageDataset(prepared,'train');val_data=ImageDataset(prepared,'validation')
    data_keys=set(config)-{'training','seed','model'}
    if any(train_data.config.get(k)!=config[k] for k in data_keys):
        raise ValueError('Config differs from prepared dataset; use its original config')
    output=Path(config['artifact_root'])/('smoke' if smoke else 'training')
    if output.exists():
        raise ValueError(f'Output exists: {output}; move it or use another artifact root')
    output.mkdir(parents=True)
    batch_size=min(settings['batch_size'],4) if smoke else settings['batch_size']
    if smoke:
        rng=np.random.default_rng(seed)
        chosen=rng.choice(len(train_data),size=min(len(train_data),batch_size*2),replace=False)
        train_view=Subset(train_data,chosen.tolist())
        val_view=Subset(val_data,rng.choice(len(val_data),size=min(len(val_data),batch_size*2),replace=False).tolist())
    else:
        train_view,val_view=train_data,val_data
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train_view,batch_size=batch_size,shuffle=True,generator=generator,
        num_workers=settings['num_workers'],drop_last=False)
    validation=DataLoader(val_view,batch_size=batch_size,shuffle=False,
        num_workers=settings['num_workers'],drop_last=False)
    model=build_model(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    counts=train_data.samples.label.value_counts();majority=int(counts.idxmax())
    best=float('inf');stale=0;step=0;history=[];stopped=False

    def check(epoch):
        nonlocal best,stale,stopped
        result,_=evaluate(model,validation,device,majority)
        result.update({'step':step,'epoch':epoch})
        history.append(result)
        print(json.dumps(result),flush=True)
        if result['loss']<best:
            best=result['loss'];stale=0
            torch.save({'state_dict':model.state_dict(),'config':config,
                'prepared_manifest_sha256':digest(prepared/'manifest.json'),'step':step,
                'validation':result,'majority_class':majority,'smoke_only':smoke},output/'best.pt')
        else:
            stale+=1
        stopped=stale>=settings['patience_checks']
        model.train()  # Restore dropout and BatchNorm after every validation call.

    for epoch in range(1,(1 if smoke else settings['epochs'])+1):
        model.train()
        for x,y,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device))
            loss.backward();optimizer.step();step+=1
            if settings['validate_every_batches']>0 and step%settings['validate_every_batches']==0:
                check(epoch)
                if stopped:
                    break
        if not history or history[-1]['step']!=step:
            check(epoch)
        if stopped:
            break
    report={'mode':'smoke_only_not_performance_estimate' if smoke else 'training',
        'device':str(device),'optimizer_steps':step,'validation_checks':len(history),
        'parameters':sum(p.numel() for p in model.parameters()),'history':history,
        'model':config['model'],'convolutional_blocks':len(model.channels),
        'local_receptive_field':list(model.convolutional_receptive_field),
        'torch_version':torch.__version__,'test_evaluated':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))
    return report


def test(config,checkpoint):
    torch.set_num_threads(2)
    device=device_for(config['training']['device'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('smoke_only'):
        raise ValueError('Smoke checkpoint cannot be used for final test evaluation')
    prepared=Path(config['artifact_root'])/'prepared'
    if saved['config']!=config or saved['prepared_manifest_sha256']!=digest(prepared/'manifest.json'):
        raise ValueError('Checkpoint/prepared data mismatch')
    output=Path(config['artifact_root'])/'test_evaluation'
    if output.exists():
        raise ValueError('Test results already exist; keep the original evaluation immutable')
    data=ImageDataset(prepared,'test')
    loader=DataLoader(data,batch_size=config['training']['batch_size'],shuffle=False,num_workers=0)
    model=build_model(config).to(device);model.load_state_dict(saved['state_dict'])
    report,pred=evaluate(model,loader,device,saved['majority_class'],predictions=True)
    output.mkdir(parents=True)
    rows=data.samples.iloc[pred[0]].copy();rows['probability_up']=pred[1]
    rows.to_parquet(output/'predictions.parquet',index=False,compression='zstd')
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
