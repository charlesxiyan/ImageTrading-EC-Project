"""Read-only checkpoint diagnosis and train-only memorization control.

Run from the project root with python -m diagnostics.check_baseline.
No test examples are loaded. Diagnostic training never overwrites the baseline.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from us_pipeline.dataset import ImageDataset
from us_pipeline.models import build_model


def stats(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    return dict(samples=len(y), accuracy=float(((p >= .5) == y).mean()),
                positive_fraction=float(y.mean()), predicted_up_fraction=float((p >= .5).mean()),
                probability_min=float(p.min()), probability_max=float(p.max()),
                probability_std=float(p.std()), unique_probabilities=len(np.unique(p)))


def tensors(ds, ids):
    items = [ds[int(i)] for i in ids]
    return torch.stack([v[0] for v in items]), torch.tensor([v[1] for v in items])


def activations(model, x):
    rows = []
    model.eval()
    with torch.no_grad():
        z = x
        for i, layer in enumerate(model.blocks):
            z = layer(z)
            rows.append(dict(layer=f'blocks.{i}:{type(layer).__name__}',
                             mean=float(z.mean()), std=float(z.std()),
                             zero_fraction=float((z == 0).float().mean()),
                             between_sample_std=float(z.std(dim=0).mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', default='artifacts/cnn78u_5min_h1/prepared')
    ap.add_argument('--checkpoint', default='artifacts/cnn78u_5min_h1/colab_best.pt')
    ap.add_argument('--output', default='artifacts/baseline_diagnostics')
    ap.add_argument('--samples', type=int, default=512)
    ap.add_argument('--memorize-samples', type=int, default=64)
    ap.add_argument('--steps', type=int, default=300)
    args = ap.parse_args()
    if args.samples < 2 or args.memorize_samples < 2 or args.steps < 1:
        ap.error('At least two samples and one step are required')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(69)
    rng = np.random.default_rng(69)
    # This is the user's own checkpoint produced by the local project runner.
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ck['config']); model.load_state_dict(ck['model']); model.eval()
    report = {'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'completed_epoch': ck['completed_epoch'], 'test_evaluated': False,
              'runtime': {'torch': torch.__version__, 'device': 'cpu'}, 'splits': {}}
    train = ImageDataset(args.prepared, 'train')
    assert train.config == ck['config'], 'Prepared config differs from checkpoint'
    names = ['manifest.json', 'features.npy', 'multipliers.npy', 'valid.npy',
             'train.parquet', 'validation.parquet', 'test.parquet']
    # Hash test bytes only for provenance; never deserialize or evaluate test samples.
    fingerprints = {n: hashlib.sha256((Path(args.prepared) / n).read_bytes()).hexdigest() for n in names}
    signature = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    assert signature == ck['data_signature'], 'Prepared data differs from checkpoint'
    report['data_fingerprint_matches_checkpoint'] = True
    report['weight_stats'] = {n: dict(max_abs=float(t.double().abs().max()),
        l2_float64=float(t.double().norm()), finite=bool(t.isfinite().all()))
        for n,t in ck['model'].items() if t.is_floating_point()}
    features = np.load(Path(args.prepared) / 'features.npy', mmap_mode='r')
    for split in ('train', 'validation'):
        ds = train if split == 'train' else ImageDataset(args.prepared, split)
        ids = rng.choice(len(ds), min(args.samples, len(ds)), replace=False)
        probabilities = []; hashes = []; mismatches = 0
        for ids_batch in np.array_split(ids, max(1, int(np.ceil(len(ids) / 16)))):
            x, y = tensors(ds, ids_batch)
            with torch.no_grad(): probabilities.extend(model(x).softmax(1)[:, 1].tolist())
            hashes.extend(hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in x)
            for i in ids_batch:
                row = ds.samples.iloc[int(i)]; s, t = int(row.security_index), int(row.end_index)
                expected = int(features[s, t+1, 3] > features[s, t, 3])
                mismatches += expected != int(row.label)
        report['splits'][split] = stats(probabilities, ds.samples.iloc[ids].label)
        report['splits'][split].update(unique_images=len(set(hashes)), label_mismatches=int(mismatches))
        np.savez(out / f'{split}_sample.npz', indices=ids, probabilities=probabilities)
    ids = rng.choice(len(train), args.memorize_samples, replace=False)
    x, y = tensors(train, ids)
    report['checkpoint_activations'] = activations(model, x[:16])
    probe = copy.deepcopy(model); probe.train()
    for layer in probe.modules():
        if isinstance(layer, nn.Dropout): layer.p = 0
    with torch.no_grad(): report['checkpoint_batch_statistics_dropout_off'] = stats(probe(x[:16]).softmax(1)[:, 1], y[:16])
    # One backward pass on a disposable copy; never update the baseline.
    probe.zero_grad(); nn.functional.cross_entropy(probe(x[:16]), y[:16]).backward()
    report['checkpoint_gradient_norms'] = {n: float(p.grad.norm()) if p.grad is not None else None for n,p in probe.named_parameters()}
    (out / 'checkpoint_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    # Same architecture, fresh weights, dropout off, no weight decay: capacity check only.
    torch.manual_seed(69)
    fresh = build_model(train.config)
    for layer in fresh.modules():
        if isinstance(layer, nn.Dropout): layer.p = 0
    optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3, weight_decay=0)
    initial = {n: p.detach().clone() for n,p in fresh.named_parameters()}
    history = []; started = time.perf_counter(); passed = False
    for step in range(1, args.steps + 1):
        fresh.train(); optimizer.zero_grad()
        logits = fresh(x); loss = nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite memorization loss')
        loss.backward()
        if step == 1:
            report['fresh_gradient_norms'] = {n: float(p.grad.norm()) for n,p in fresh.named_parameters()}
        optimizer.step()
        if step == 1 or step % 10 == 0:
            fresh.eval()
            with torch.no_grad():
                scores = fresh(x); evaluation_loss = float(nn.functional.cross_entropy(scores, y))
                accuracy = float((scores.argmax(1) == y).float().mean())
            row = dict(step=step, train_mode_loss=float(loss.detach()), eval_loss=evaluation_loss,
                       eval_accuracy=accuracy, seconds=time.perf_counter()-started)
            history.append(row); print(json.dumps(row), flush=True)
            if accuracy >= .98 and evaluation_loss < .1:
                passed = True; break
    report['memorization'] = dict(samples=len(y), indices=ids.tolist(), steps=step, passed=passed,
        purpose='Capacity check only, not a generalization estimate', dropout=0, weight_decay=0,
        history=history, parameter_update_norms={n:float((p.detach()-initial[n]).norm()) for n,p in fresh.named_parameters()})
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('Saved:', out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
