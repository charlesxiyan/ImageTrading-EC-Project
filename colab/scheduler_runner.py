"""CNN78u validation-driven scheduler experiment; training/validation only."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

from us_pipeline.dataset import ImageDataset
from us_pipeline.models import build_model
from us_pipeline.reporting import binary_metrics, write_evaluation
from us_pipeline.experiment import (make_loader, predict, save_json, save_checkpoint,
                                    mirror, sync_tree, sync_cuda, runtime_info)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_indices(n, fraction=.2, seed=69):
    if not 0 < fraction <= 1 or n < 1:
        raise ValueError('Invalid sampling contract')
    count = max(1, int(np.floor(n * fraction)))
    return np.sort(np.random.default_rng(seed).choice(n, count, replace=False))


def probability_summary(p):
    p = np.asarray(p, dtype=np.float64)
    return dict(probability_std=float(p.std()), probability_min=float(p.min()),
                probability_max=float(p.max()), unique_probabilities=len(np.unique(p)),
                predicted_up_fraction=float((p >= .5).mean()))


def collapse_signal(health):
    # Both constant predictions AND near-zero feature parameters are required.
    # This is a numerical-failure guard, not a significance test for weak alpha.
    return (health['train']['probability_std'] <= 1e-8
            and health['validation']['probability_std'] <= 1e-8
            and health['feature_parameter_max_abs'] <= 1e-8)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe_tensor(ds, indices):
    rows = [ds[int(i)] for i in indices]
    return torch.stack([r[0] for r in rows]), torch.tensor([r[1] for r in rows])


def health_check(model, probes, device, gradients, step):
    was_training = model.training
    model.eval(); result = {'step': step, 'gradient_norms_last_training_batch': gradients}
    with torch.no_grad():
        for name, (x, y) in probes.items():
            p = torch.cat([model(batch.to(device)).softmax(1)[:, 1].cpu()
                           for batch in x.split(16)]).numpy()
            result[name] = {**probability_summary(p),
                            'accuracy': float(((p >= .5) == y.numpy()).mean()),
                            'log_loss': binary_metrics(y.numpy(), p)['log_loss']}
        weights = {name: float(p.detach().abs().max()) for name,p in model.blocks.named_parameters()
                   if name.endswith('weight')}
        result['feature_parameter_max_abs'] = max(weights.values())
        result['feature_weight_max_abs'] = weights
        result['batchnorm_scale_max_abs'] = {str(i):float(m.weight.abs().max())
            for i,m in enumerate(model.blocks) if isinstance(m, nn.BatchNorm2d)}
        z = probes['train'][0][:16].to(device); activation = {}
        for i, layer in enumerate(model.blocks):
            z = layer(z)
            if isinstance(layer, nn.ReLU):
                activation[str(i)] = {'zero_fraction':float((z==0).float().mean()),
                    'between_sample_std':float(z.std(dim=0, unbiased=False).mean())}
        result['relu_activations'] = activation
    model.train(was_training)
    result['collapse_signal'] = collapse_signal(result)
    return result


class ScheduleControl:
    """Shared absolute improvement tolerance for scheduling and early stopping."""
    def __init__(self, optimizer, settings):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=settings['lr_factor'],
            patience=settings['lr_patience'], threshold=settings['min_delta'],
            threshold_mode='abs', cooldown=0, min_lr=settings['min_lr'])
        self.best = float('inf')
        self.stale = 0
        self.patience = settings['early_stop_patience']
        self.min_delta = settings['min_delta']

    def step(self, validation_loss):
        if not np.isfinite(validation_loss):
            raise ValueError('Nonfinite scheduler metric')
        if validation_loss < self.best - self.min_delta:
            self.best = validation_loss; self.stale = 0
        else:
            self.stale += 1
        self.scheduler.step(validation_loss)
        return self.stale >= self.patience

    def state_dict(self):
        return {'scheduler':self.scheduler.state_dict(), 'best':self.best, 'stale':self.stale}

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state['scheduler'])
        self.best = state['best']; self.stale = state['stale']


def run_experiment(prepared, root, persistent, indices, probe_ids, settings, weight_decay, smoke=False):
    """Epoch-boundary resume; terminal collapsed/completed runs are read-only cached."""
    if weight_decay != 0:
        raise ValueError('This protocol fixes weight decay at zero')
    root, persistent = Path(root), Path(persistent)
    root.mkdir(parents=True, exist_ok=True); persistent.mkdir(parents=True, exist_ok=True)
    train = ImageDataset(prepared, 'train'); validation = ImageDataset(prepared, 'validation')
    if train.config['lookback'] != 78 or train.config['horizon'] != 1:
        raise ValueError('Expected 78-bar input and next-bar target')
    config = train.config
    contract = {'settings':settings,'weight_decay':weight_decay,'config':config,'smoke':smoke,
                'indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
                'probes':{k:v.tolist() for k,v in probe_ids.items()},
                'source_sha256':sha(__file__),
                'data':{n:sha(Path(prepared)/n) for n in ['manifest.json','features.npy',
                       'multipliers.npy','valid.npy','train.parquet','validation.parquet']}}
    signature = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if (root/'contract.json').exists():
        if json.loads((root/'contract.json').read_text()) != contract:
            raise ValueError('Run contract changed; use a new experiment ID')
    save_json(root/'contract.json', contract, persistent)
    if (root/'summary.json').exists():
        return json.loads((root/'summary.json').read_text())
    seed_all(settings['seed'])
    model = build_model(config).to(device := torch.device(settings['device']))
    initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                         for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=weight_decay)
    schedule = ScheduleControl(optimizer, settings)
    loader_settings = {**settings, 'amp':False}
    train_loader = make_loader(Subset(train, indices.tolist()), loader_settings, True)
    val_view = Subset(validation, probe_ids['validation'].tolist()) if smoke else validation
    val_loader = make_loader(val_view, loader_settings, False)
    probes = {'train':probe_tensor(train,probe_ids['train']),
              'validation':probe_tensor(validation,probe_ids['validation'])}
    prior = float(train.samples.iloc[indices].label.mean())
    history=[]; health=[]; best=float('inf'); first=1; steps=0; strikes=0
    early_stopped=False
    if (root/'last.pt').exists():
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if ck['signature'] != signature: raise ValueError('Resume signature mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        schedule.load_state_dict(ck['schedule'])
        early_stopped=ck['early_stopped']
        history=ck['history']; health=ck['health']; best=ck['best']; steps=ck['steps']
        strikes=ck['strikes']; first=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:
        health.append(health_check(model,probes,device,{},0))
    save_json(root/'health.json',health,persistent)
    stopped=False; stop_reason=None
    for epoch in ([] if early_stopped else range(first,settings['epochs']+1)):
        lr_used=optimizer.param_groups[0]['lr']
        model.train(); train_loader.generator.manual_seed(settings['seed']+epoch)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        sync_cuda(device); started=time.perf_counter(); total=correct=seen=0; monitor_seconds=0.
        for batch_index,(x,y,_) in enumerate(train_loader):
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):
                stopped=True; stop_reason='nonfinite_loss'; break
            loss.backward()
            monitor=(steps+1)%settings['monitor_every']==0 or batch_index+1==len(train_loader)
            gradients={n:float(p.grad.detach().double().norm()) for n,p in model.named_parameters()
                       if p.grad is not None} if monitor else {}
            if monitor and not all(np.isfinite(list(gradients.values()))):
                stopped=True; stop_reason='nonfinite_gradient'; break
            optimizer.step(); steps+=1
            seen+=len(y); total+=float(loss.detach())*len(y); correct+=int((logits.argmax(1)==y).sum())
            if monitor:
                sync_cuda(device); t=time.perf_counter()
                h=health_check(model,probes,device,gradients,steps); h['epoch']=epoch
                health.append(h); strikes=strikes+1 if h['collapse_signal'] else 0
                save_json(root/'health.json',health,persistent)
                sync_cuda(device); monitor_seconds+=time.perf_counter()-t
                print(json.dumps({'epoch':epoch,'step':steps,'train_loss':total/seen,
                    'probe_probability_std':h['validation']['probability_std'],
                    'feature_max_abs':h['feature_parameter_max_abs'],'collapse_strikes':strikes}),flush=True)
                if strikes>=3:
                    stopped=True; stop_reason='three_consecutive_feature_collapse_checks'; break
        sync_cuda(device); train_wall=time.perf_counter()-started
        frame,timing=predict(model,validation,val_loader,device,False,not smoke)
        if not np.isfinite(frame.probability_up).all():
            stopped=True; stop_reason='nonfinite_validation_probability'
            metrics={}
        else:
            metrics=binary_metrics(frame.label,frame.probability_up)
        if not stopped:
            early_stopped=schedule.step(metrics['log_loss'])
        row={'epoch':epoch,'learning_rate_used':lr_used,
             'learning_rate_next':optimizer.param_groups[0]['lr'],
             'early_stop_bad_epochs':schedule.stale,
             'train_probe_eval_loss':health[-1]['train']['log_loss'],
             'train_probe_eval_accuracy':health[-1]['train']['accuracy'],'steps_total':steps,'samples_seen':seen,'partial_epoch':stopped,
             'train_loss':total/seen if seen else None,'train_accuracy':correct/seen if seen else None,
             'train_wall_seconds_including_monitor':train_wall,'monitor_seconds':monitor_seconds,
             'validation_seconds':timing['end_to_end_seconds'],
             'train_samples_per_second_including_monitor':seen/train_wall,
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
             **{'val_'+k:v for k,v in metrics.items() if not isinstance(v,list)}}
        history.append(row)
        ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
            'schedule':schedule.state_dict(),'early_stopped':early_stopped,
            'signature':signature,'history':history,'health':health,'steps':steps,'epoch':epoch,
            'strikes':strikes,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if not stopped and metrics['log_loss']<best:
            best=metrics['log_loss']; ck['best']=best
            save_checkpoint(root/'best.pt',ck,persistent)
            frame.to_parquet(root/'best_validation_predictions.parquet',index=False)
            mirror(root/'best_validation_predictions.parquet',persistent)
        ck['best']=best
        save_checkpoint(root/('stopped.pt' if stopped else 'last.pt'),ck,persistent)
        save_json(root/'history.json',history,persistent)
        save_json(root/'health.json',health,persistent)
        print(json.dumps(row),flush=True)
        if stopped or early_stopped: break
    # A best pre-collapse checkpoint remains explicitly distinguished from terminal health.
    evaluation=None
    if (root/'best_validation_predictions.parquet').exists():
        frame=pd.read_parquet(root/'best_validation_predictions.parquet')
        evaluation=write_evaluation(frame,root/'validation',prior,'validation',{},
            seed=settings['seed'],bootstrap_repetitions=100 if smoke else 2000,smoke_only=smoke)
        sync_tree(root/'validation',persistent/'validation')
    result={'status':'stopped_unhealthy' if stopped else ('early_stopped' if early_stopped else 'completed_budget'),
            'schedule':schedule.state_dict(),
            'best_epoch':torch.load(root/'best.pt',map_location='cpu',weights_only=False)['epoch'] if (root/'best.pt').exists() else None,
            'stop_reason':stop_reason or ('validation_plateau' if early_stopped else None),'smoke_only':smoke,'test_evaluated':False,
            'weight_decay':weight_decay,'initial_parameters_sha256':initial_sha,
            'train_subset_sha256':contract['indices_sha256'],'training_samples':len(indices),
            'training_prior':prior,'steps':steps,'history':history,
            'final_health':health[-1],'runtime':runtime_info(device),
            'best_validation_metrics':evaluation['metrics'] if evaluation else None,
            'interpretation':'Diagnostic only. Completing the budget is not proof of generalization.'}
    save_json(root/'summary.json',result,persistent)
    return result
