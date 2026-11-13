"""Training loop, early stopping, resume, and the one-call experiment runner."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import fit_baselines
from .data import Store, build_split, make_dataset
from .evaluation import comparison_table, decile_report, metrics
from .models import build_model, print_parameter_count
from .utils import make_loader, mirror, save_json, seed_all


@torch.no_grad()
def predict(model, loader, device):
    """P(label = 1) for every sample, in dataset order."""
    model.eval()
    out = np.empty(len(loader.dataset))
    for x, _, idx in loader:
        out[idx.numpy()] = model(x.to(device, non_blocking=True)).float().softmax(1)[:, 1].cpu().numpy()
    return out


def train_one_epoch(model, loader, optimizer, device, log_every):
    model.train()
    seen = loss_sum = correct = 0
    start = time.perf_counter()
    for b, (x, y, _) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        seen += len(y); loss_sum += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item()
        if b % log_every == 0:
            print(f"  batch {b}/{len(loader)}  loss {loss_sum / seen:.4f}  acc {correct / seen:.4f}  "
                  f"{seen / (time.perf_counter() - start):.0f} img/s", flush=True)
    return dict(train_loss=loss_sum / seen, train_accuracy=correct / seen,
                train_seconds=time.perf_counter() - start)


def fit(model, train_loader, valid_loader, y_valid, prior, settings, device, out_dir, drive_dir=None):
    """Adam + early stopping on validation log loss. Resumes from out_dir/last.pt if present."""
    out_dir = Path(out_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    history, best, bad, first = [], np.inf, 0, 1
    if (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        history, best, bad, first = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        if ck["finished"]:
            first = settings["epochs"] + 1
        print(f"Resuming after epoch {ck['epoch']}")

    for epoch in range(first, settings["epochs"] + 1):
        train_loader.generator.manual_seed(settings["seed"] + epoch)
        row = dict(epoch=epoch, **train_one_epoch(model, train_loader, optimizer, device,
                                                  max(1, len(train_loader) // 5)))
        p = predict(model, valid_loader, device)
        row.update({"val_" + k: v for k, v in metrics(y_valid, p, prior).items()})
        history.append(row)
        print(f"epoch {epoch}: train_loss {row['train_loss']:.4f} | val AUC {row['val_roc_auc']:.4f} "
              f"| val log loss {row['val_log_loss']:.4f} ({row['val_log_loss_gain_vs_prior_pct']:+.2f}% vs prior) "
              f"| prob std {row['val_prob_std']:.3f}")

        if row["val_log_loss"] < best:
            torch.save(dict(model=model.state_dict(), epoch=epoch, settings=settings), out_dir / "best.pt")
            np.save(out_dir / "best_valid_prob.npy", p)
            mirror(out_dir / "best.pt", drive_dir)
        bad = 0 if row["val_log_loss"] < best - settings["min_delta"] else bad + 1
        best = min(best, row["val_log_loss"])
        finished = bad >= settings["patience"] or epoch == settings["epochs"]
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                        best=best, bad=bad, epoch=epoch, finished=finished), out_dir / "last.pt")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        for name in ("last.pt", "history.csv"):
            mirror(out_dir / name, drive_dir)
        if bad >= settings["patience"]:
            print(f"Early stop: {bad} epochs without improvement")
            break
    return pd.DataFrame(history)


def run_experiment(prepared, exp, settings, device, out_dir, drive_dir=None, smoke=False):
    """Build labels -> fit baselines -> train CNN -> compare on validation."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(settings["seed"])
    store = Store(prepared)
    stride = exp.get("stride", settings["stride"])          # per-experiment override
    train, median = build_split(store, "train", exp["target"], exp["horizon"], stride)
    valid, _ = build_split(store, "validation", exp["target"], exp["horizon"], stride, median)
    if smoke:                                   # tiny code check only
        train = train.sample(256, random_state=0).reset_index(drop=True)
        times = valid.end_index.drop_duplicates().sample(2, random_state=0)
        valid = valid[valid.end_index.isin(times)].reset_index(drop=True)
    print(f"{exp['name']} [{settings.get('arch', 'plain')}, stride {stride}]: {len(train):,} train / {len(valid):,} validation samples; "
          f"positive share {train.label.mean():.3f} / {valid.label.mean():.3f}")
    if median is not None:
        median.to_csv(out_dir / "train_abs_return_median.csv")

    scores, prior, _ = fit_baselines(store, exp["target"], train, valid, settings["seed"],
                                     max_train=2_000 if smoke else 300_000)

    model = build_model(settings).to(device)
    print_parameter_count(model, f"model ({settings.get('arch', 'plain')})")
    w = settings["num_workers"]
    train_loader = make_loader(make_dataset(prepared, train, settings.get("arch", "plain")), settings["batch_size"], w, True, settings["seed"])
    valid_loader = make_loader(make_dataset(prepared, valid, settings.get("arch", "plain")), settings["eval_batch_size"], w, False)
    history = fit(model, train_loader, valid_loader, valid.label.to_numpy(), prior, settings, device,
                  out_dir, drive_dir)

    scores = {"CNN": np.load(out_dir / "best_valid_prob.npy"), **scores}
    table = comparison_table(valid.label.to_numpy(), scores, prior)
    deciles, spread = decile_report(valid, scores["CNN"], exp["target"])
    _, hgb_spread = decile_report(valid, scores["hgb_hand_features"], exp["target"])
    table.to_csv(out_dir / "validation_comparison.csv"); deciles.to_csv(out_dir / "validation_deciles.csv")
    save_json(out_dir / "summary.json", dict(experiment=exp, best_epoch=int(history.loc[history.val_log_loss.idxmin(), "epoch"]),
              long_short_cnn=spread, long_short_hgb=hgb_spread, smoke=smoke), drive_dir)
    for name in ("validation_comparison.csv", "validation_deciles.csv"):
        mirror(out_dir / name, drive_dir)
    return dict(history=history, table=table, deciles=deciles, long_short=dict(cnn=spread, hgb=hgb_spread),
                median=median, prior=prior, valid=valid, scores=scores)


def evaluate_test(prepared, exp, settings, device, out_dir, median, prior, drive_dir=None, smoke=False):
    """One-shot held-out evaluation of best.pt. Do not tune anything after looking at this."""
    store = Store(prepared)
    test, _ = build_split(store, "test", exp["target"], exp["horizon"], exp.get("stride", settings["stride"]), median)
    if smoke:
        test = test[test.end_index.isin(test.end_index.drop_duplicates().sample(2, random_state=0))]
        test = test.reset_index(drop=True)
    model = build_model(settings).to(device)
    model.load_state_dict(torch.load(Path(out_dir) / "best.pt", map_location=device, weights_only=False)["model"])
    p = predict(model, make_loader(make_dataset(prepared, test, settings.get("arch", "plain")), settings["eval_batch_size"],
                                   settings["num_workers"], False), device)
    result = dict(experiment=exp, test=metrics(test.label.to_numpy(), p, prior),
                  long_short=decile_report(test, p, exp["target"])[1])
    save_json(Path(out_dir) / "test_summary.json", result, drive_dir)
    return result
