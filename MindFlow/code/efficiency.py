"""Step 7a: efficient computing for the chart models.

Two questions, both answered on the model that MindFlow settles on (any trained network of a Step 3a
grid run or a Step 4a indicator run can be loaded):

  inference   How much smaller and faster does the trained CNN get when it is run in bfloat16 on the
              GPU or quantized to INT8 on the CPU (post-training static quantization, calibrated on
              training windows only), and how much validation AUC does that cost?
  training    Can a large pretrained ViT be adapted cheaply? DINOv2 ViT-S/14 (22M parameters) is
              fine-tuned with LoRA (low-rank adapters on the attention projections, ~0.15M trainable
              parameters) on the same rows and inputs as the CNN, and compared with the CNN, the
              scratch ViT and full fine-tuning on trainable parameters, GPU memory, time and AUC.

Every model is scored on identical validation rows; differences are paired by resampling whole days.
Nothing here changes inputs, labels or the price adjustment: datasets come from chartcnn / indicator_cnn.
"""
import copy
import io
import json
import math
import operator
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

__version__ = "2026-11-25.1"

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ----------------------------------------------------------------------------- inputs

class Inputs(Dataset):
    """Uniform (image, tab, label, index) items for every source. `rows` index the tab array."""

    def __init__(self, base, tab=None, rows=None):
        self.base, self.tab, self.rows = base, tab, rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if len(item) == 4:                                    # indicator_cnn.Charts already has the tab part
            return item
        x, y, idx = item
        t = torch.zeros(0) if self.tab is None else torch.from_numpy(self.tab[self.rows[i]])
        return x, t, y, idx


def loader(dataset, batch_size, workers, shuffle=False, seed=0):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
              generator=torch.Generator().manual_seed(seed))
    if workers:
        kw.update(persistent_workers=False, prefetch_factor=4)
    return DataLoader(dataset, **kw)


class Source:
    """A trained network with everything needed to rebuild its inputs and its rows.

    model     the trained network (float32, CPU, eval) with forward(x, t)
    valid     validation rows exactly as scored by the source run (with `label`, `decision_date`)
    train     the source run's training rows (calibration and LoRA)
    saved     the source run's own validation scores for `valid` (a reproduction check)"""

    def __init__(self, name, label, model, valid, train, saved, dataset, input_hw, n_tab, settings, cell, kind,
                 fresh, references=None):
        self.name, self.label, self.model, self.valid, self.train, self.saved = name, label, model, valid, train, saved
        self.dataset, self.input_hw, self.n_tab, self.settings, self.cell, self.kind = dataset, input_hw, n_tab, settings, cell, kind
        self.fresh = fresh                                     # () -> untrained copy of the architecture
        self.references = references or {}                    # other models' saved scores on `valid`


class Net(nn.Module):
    """forward(x, t) = image head(features(x)) [+ tab MLP(t)]: one interface for every network, and a
    clean graph for quantization (no device moves inside). `fp32_head` keeps the ViT head out of autocast."""

    def __init__(self, features, head, tab=None, fp32_head=False):
        super().__init__()
        self.features, self.head, self.tab, self.fp32_head = features, head, tab, fp32_head

    def forward(self, x, t):
        z = self.features(x)
        if self.fp32_head:
            with torch.autocast(z.device.type, enabled=False):
                out = self.head(z.float())
        else:
            out = self.head(z)
        if self.tab is not None:
            out = out + self.tab(t)
        return out


def _wrap_grid(model):
    return Net(model.features, model.head, None, fp32_head=model.arch.startswith("vit")).eval()


def _wrap_indicator(model):
    assert model.cnn is not None, "efficiency needs an image model (not tab_only)"
    return Net(model.cnn.features, model.cnn.head, model.tab if model.n_tab else None).eval()


def _same_rows(a, b, what):
    keys = ["security_index", "end_index"]
    assert len(a) == len(b) and (a[keys].to_numpy() == b[keys].to_numpy()).all(), f"{what}: rows differ from the source run"


def load_grid_source(prepared, run_dir, cell_name, arch, smoke=False):
    """A network of a Step 2a/3a grid run (best checkpoint of `cell_name`/`arch`)."""
    import chartcnn
    from chartcnn import grid
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "settings.json").read_text())["settings"]
    info = json.loads((run_dir / cell_name / "done.json").read_text())
    cell = info["cell"]
    pred = pd.read_parquet(run_dir / cell_name / "predictions.parquet")
    _, nn_train, valid, _ = grid._cell_rows(prepared, cell, s, smoke)
    _same_rows(valid, pred, cell_name)
    assert len(nn_train) == info["nn_train_samples"], "training rows differ from the source run (keep its settings)"
    model = chartcnn.build_model(grid.arch_settings(s, arch))
    ck = torch.load(run_dir / cell_name / arch / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    n = 1 + len(chartcnn.PANELS[arch])
    refs = {c[6:]: pred[c].to_numpy(float) for c in pred.columns if c.startswith("score_") and c != f"score_cnn_{arch}"}
    return Source(name=arch, label=grid.ARCH_LABEL.get(arch, arch), model=_wrap_grid(model).float(), valid=valid,
                  train=nn_train, saved=pred[f"score_cnn_{arch}"].to_numpy(float),
                  dataset=lambda frame, split: Inputs(chartcnn.make_dataset(prepared, frame, arch)),
                  input_hw=(234 * n, 234), n_tab=0, settings=dict(s, arch=arch), cell=cell, kind="grid",
                  fresh=lambda: _wrap_grid(chartcnn.build_model(grid.arch_settings(s, arch))), references=refs)


def load_indicator_source(prepared, store, run_dir, cell_name, variant, tab_fn=None):
    """A network of a Step 4a indicator run or the Step 6b final run (best checkpoint of `cell_name`/`variant`).
    `tab_fn(store, windows)` gives the raw numbers (default: the 38 hand + SPY numbers of Step 4a)."""
    import indicator_cnn as ic
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "settings.json").read_text())["settings"]
    pre = json.loads((run_dir / "preprocessing.json").read_text())
    info = json.loads((run_dir / cell_name / "done.json").read_text())
    cell = info["cell"]
    pred = pd.read_parquet(run_dir / cell_name / "predictions.parquet")
    W = {k: pd.read_parquet(run_dir / f"windows_{k}.parquet") for k in ("train", "valid")}
    train, median = ic.cell_rows(store, cell, "train", W["train"])
    valid, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)
    _same_rows(valid, pred, cell_name)
    uses_tab = ic.VARIANTS[variant][2]
    tab = {}
    if uses_tab:
        scaler = ic.TabScaler.__new__(ic.TabScaler)
        st = pre["tab"]
        scaler.columns, scaler.mean, scaler.std = st["columns"], np.asarray(st["mean"]), np.asarray(st["std"])
        tab = {k: scaler((tab_fn or ic.tab_features)(store, W[k])) for k in ("train", "valid")}
    n_tab = next(iter(tab.values())).shape[1] if uses_tab else 0
    scale, halflife = tuple(pre["ewsd_log_scale"]), pre.get("halflife", s["halflife"])

    def build():
        return ic.IndicatorCNN(s, n_tab, image=variant not in ic.IMAGE_FREE)

    model = build()
    model.load_state_dict(torch.load(run_dir / cell_name / variant / "best.pt", map_location="cpu", weights_only=False)["model"])

    def dataset(frame, split):
        t = tab[split][frame.row.to_numpy()] if uses_tab else None
        return Inputs(ic.Charts(prepared, frame, variant, scale, halflife, t))

    refs = {c[6:]: pred[c].to_numpy(float) for c in pred.columns if c.startswith("score_") and c != f"score_ind_{variant}"}
    return Source(name=variant, label=ic.VARIANT_LABEL.get(variant, variant), model=_wrap_indicator(model).float(),
                  valid=valid, train=train, saved=pred[f"score_ind_{variant}"].to_numpy(float), dataset=dataset,
                  input_hw=(ic.SIZE + ic.BAND, ic.SIZE), n_tab=n_tab, settings=dict(s, variant=variant), cell=cell,
                  kind="indicator", fresh=lambda: _wrap_indicator(build()), references=refs)


def every_kth_day(frame, k, max_rows=None, seed=0):
    """Whole validation days (every k-th), optionally fewer days so that at most `max_rows` rows remain.
    Returns positions into `frame`."""
    days = np.sort(frame.decision_date.unique())[::max(1, int(k))]
    if max_rows is not None:
        sizes = frame.decision_date.value_counts()
        keep = max(1, int(max_rows / max(1.0, sizes[days].mean())))
        if keep < len(days):
            days = np.sort(np.random.default_rng(seed).choice(days, keep, replace=False))
    return np.flatnonzero(frame.decision_date.isin(days).to_numpy())


# ----------------------------------------------------------------------------- precision variants

def model_bytes(model):
    """Serialized size of the state dict (what would ship with the model)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_model(model, x, t, device, precision):
    """Logits under one precision. fp32 / bf16 (fp16 where bf16 is missing) / int8 (already converted)."""
    if precision == "bf16" and device.type == "cuda":
        with torch.autocast("cuda", dtype=_amp_dtype(device)):
            return model(x, t)
    return model(x, t)


def quantize_static(model, calib_batches):
    """INT8 post-training static quantization (FX graph mode, x86/fbgemm backend): conv/linear weights
    per channel, activations per tensor, BatchNorm folded into the convolutions. Ranges are observed on
    `calib_batches` (training windows only). A tab MLP, if any, stays float32 (it is tiny)."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = engine
    qmap = get_default_qconfig_mapping(engine).set_module_name("tab", None).set_object_type(operator.add, None)
    m = copy.deepcopy(model).cpu().float().eval()
    x0, t0 = calib_batches[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(m, qmap, example_inputs=(x0, t0))
        with torch.inference_mode():
            for x, t in calib_batches:
                prepared(x, t)
        return convert_fx(prepared).eval()


def quantize_dynamic(model):
    """INT8 dynamic quantization of every Linear layer (weights int8, activations quantized on the fly);
    the usual choice for transformers, whose compute is almost all Linear layers."""
    from torch.ao.quantization import quantize_dynamic as qd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qd(copy.deepcopy(model).cpu().float().eval(), {nn.Linear}, dtype=torch.qint8)


def has_conv(model):
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


def calibration_batches(source, n, batch_size, seed=0):
    rows = source.train.sample(min(n, len(source.train)), random_state=seed).reset_index(drop=True)
    ds = source.dataset(rows, "train")
    out = []
    for x, t, _, _ in loader(ds, batch_size, 0):
        out.append((x.float(), t.float()))
    return out


@torch.inference_mode()
def score(model, dataset, device, precision, batch_size, workers):
    """P(y = 1) for every item of `dataset`, in order."""
    model.eval()
    out = np.empty(len(dataset))
    for x, t, _, idx in loader(dataset, batch_size, workers):
        x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
        out[idx.numpy()] = run_model(model, x, t, device, precision).float().softmax(1)[:, 1].cpu().numpy()
    return out


def variants(source_model, calib, device, kind="auto"):
    """name -> (model, device, precision, description) for one trained network."""
    gpu = device.type == "cuda"
    out = {}
    if gpu:
        out["fp32 GPU"] = (copy.deepcopy(source_model).to(device).float(), device, "fp32", "float32 weights and math")
        out[f"{'bf16' if _amp_dtype(device) == torch.bfloat16 else 'fp16'} GPU"] = (
            out["fp32 GPU"][0], device, "bf16", "autocast: matmuls/convs in 16-bit, weights float32")
    cpu = torch.device("cpu")
    out["fp32 CPU"] = (copy.deepcopy(source_model).cpu().float(), cpu, "fp32", "float32 on the CPU")
    if kind == "static" or (kind == "auto" and has_conv(source_model)):
        out["INT8 CPU (static PTQ)"] = (quantize_static(source_model, calib), cpu, "int8",
                                        "weights and activations int8, calibrated on training windows")
    else:
        out["INT8 CPU (dynamic)"] = (quantize_dynamic(source_model), cpu, "int8", "Linear weights int8, activations quantized per batch")
    return out


# ----------------------------------------------------------------------------- latency and memory

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def latency(model, x, t, device, precision, reps=20, warmup=3):
    """Median milliseconds per forward pass of one batch (inputs already on the device)."""
    model.eval()
    for _ in range(warmup):
        run_model(model, x, t, device, precision)
    _sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_model(model, x, t, device, precision)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


@torch.inference_mode()
def peak_inference_memory(model, x, t, device, precision):
    """Peak GPU memory (MB) of one forward pass, weights included."""
    if device.type != "cuda":
        return np.nan
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    run_model(model, x, t, device, precision); _sync(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def benchmark(name, model, device, precision, x, t, batches, reps=20):
    """Latency per batch size for one variant; inputs are a fixed batch of real charts."""
    rows = []
    for b in batches:
        xb, tb = x[:b].to(device), t[:b].to(device)
        if len(xb) < b:                                    # repeat to fill a large batch
            k = math.ceil(b / len(xb))
            xb, tb = xb.repeat(k, 1, 1, 1)[:b], tb.repeat(k, 1)[:b]
        ms = latency(model, xb, tb, device, precision, reps=max(3, reps if b <= 64 else reps // 2))
        rows.append(dict(variant=name, batch=b, ms_per_batch=ms, ms_per_image=ms / b, images_per_s=1000 * b / ms,
                         peak_memory_mb=peak_inference_memory(model, xb, tb, device, precision)))
    return rows


# ----------------------------------------------------------------------------- LoRA ViT

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, W frozen, A (r x in) and B (out x r) trained; B starts at zero, so the
    adapted network starts exactly as the pretrained one (Hu et al., 2021)."""

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale

    def merged(self):
        """A plain Linear with W + scale * B A: no extra cost at inference."""
        m = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            m.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
            if self.base.bias is not None:
                m.bias.copy_(self.base.bias)
        return m


def _replace(module, names, make):
    hits = 0
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in names:
                setattr(parent, child_name, make(child)); hits += 1
    return hits


class ChartBackbone(nn.Module):
    """A pretrained DINOv2 ViT on the chart: [B, 1, H, 234] in [0, 1] -> resized so that 234 px -> 224 px
    (height rounded to the 14-px patch grid), copied to 3 channels, ImageNet-normalised. Position
    embeddings are interpolated to the taller grid by the model itself. Output: [CLS, mean patch token]."""

    def __init__(self, hub, input_hw, tiny=False, seed=0):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        torch.manual_seed(seed)
        if tiny:                                              # offline code check only
            self.net = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                                intermediate_size=64, image_size=224, patch_size=14))
        else:
            try:
                self.net = Dinov2Model.from_pretrained(hub, attn_implementation="sdpa")
            except (TypeError, ValueError):
                self.net = Dinov2Model.from_pretrained(hub)
        h, w = input_hw
        self.size = (max(14, round(h * 224 / w / 14) * 14), 224)
        self.register_buffer("mean", torch.tensor(IMAGENET[0]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET[1]).view(1, 3, 1, 1))
        self.dim = 2 * self.net.config.hidden_size

    def forward(self, x):
        x = F.interpolate(x.float(), size=self.size, mode="bilinear", antialias=True, align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        h = self.net(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)


def vit_model(mode, input_hw, n_tab=0, hub="facebook/dinov2-small", r=8, alpha=16, targets=("query", "value", "q_proj", "v_proj"),
              lora_dropout=0.0, tab_hidden=64, tiny=False, seed=0):
    """mode 'lora': backbone frozen, LoRA on `targets` + head trained; 'full': everything trained;
    'frozen': only the head (a linear probe trained end to end). `targets` are Linear layer names: the
    attention query/value projections are `query`/`value` in transformers 4 and `q_proj`/`v_proj` in 5."""
    bb = ChartBackbone(hub, input_hw, tiny=tiny, seed=seed)
    if mode in ("lora", "frozen"):
        bb.net.requires_grad_(False)
    if mode == "lora":
        hits = _replace(bb.net, set(targets), lambda lin: LoRALinear(lin, r, alpha, lora_dropout))
        assert hits, f"no Linear layers named {targets} in the backbone"
    head = nn.Sequential(nn.LayerNorm(bb.dim), nn.Linear(bb.dim, 2))
    nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
    tab = None
    if n_tab:
        tab = nn.Sequential(nn.Linear(n_tab, tab_hidden), nn.GELU(), nn.Linear(tab_hidden, tab_hidden), nn.GELU(),
                            nn.Linear(tab_hidden, 2))
        nn.init.zeros_(tab[-1].weight); nn.init.zeros_(tab[-1].bias)
    return Net(bb, head, tab, fp32_head=True)


def merge_lora(model):
    """Copy of a LoRA model with every adapter folded into its Linear layer (plain ViT for inference)."""
    m = copy.deepcopy(model).cpu()
    for parent in list(m.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return m.eval()


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def _lr(base, step, warmup):
    return base * min(1.0, (step + 1) / warmup) if warmup else base


def fit_vit(model, train_ds, es_ds, y_es, settings, device, out_dir, drive_dir=None, log=print):
    """AdamW on the trainable parameters, linear warmup, gradient clipping, bfloat16 autocast; early stop on
    the log loss of the early-stopping days. Saves the trainable weights only (best.pt, last.pt)
    and resumes from last.pt (local, else Drive) after a disconnect."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=settings["weight_decay"])
    history, best, bad, first = [], np.inf, 0, 1
    for src in (out_dir, drive_dir):
        if src is not None and (Path(src) / "last.pt").exists():
            ck = torch.load(Path(src) / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ck["trainable"], strict=False); opt.load_state_dict(ck["optimizer"])
            history, best, bad, first = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
            if drive_dir is not None and (Path(drive_dir) / "best.pt").exists() and not (out_dir / "best.pt").exists():
                shutil.copy2(Path(drive_dir) / "best.pt", out_dir / "best.pt")
            log(f"  resuming after epoch {ck['epoch']}")
            if bad >= settings["patience"]:
                first = settings["epochs"] + 1
            break
    train_dl = loader(train_ds, settings["batch_size"], settings["num_workers"], True, settings["seed"])
    es_dl = loader(es_ds, settings["eval_batch_size"], settings["num_workers"])
    dtype = _amp_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first, settings["epochs"] + 1):
        train_dl.generator.manual_seed(settings["seed"] + epoch)
        model.train()
        seen = loss_sum = 0
        start, n = time.perf_counter(), len(train_dl)
        for b, (x, t, y, _) in enumerate(train_dl, 1):
            step = (epoch - 1) * n + b - 1
            for g in opt.param_groups:
                g["lr"] = _lr(settings["lr"], step, settings["warmup_steps"])
            x, t, y = x.to(device, non_blocking=True), t.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x, t)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, settings["grad_clip"])
            opt.step()
            seen += len(y); loss_sum += loss.item() * len(y)
            if b % max(1, n // 5) == 0:
                log(f"    batch {b}/{n} loss {loss_sum / seen:.4f} {seen / (time.perf_counter() - start):.0f} img/s")
        p = score(model, es_ds, device, "bf16", settings["eval_batch_size"], settings["num_workers"])
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_seconds=time.perf_counter() - start,
                   images_per_s=seen / (time.perf_counter() - start),
                   es_log_loss=log_loss(y_es, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                   es_auc=roc_auc_score(y_es, p) if 0 < y_es.mean() < 1 else np.nan, es_prob_std=float(p.std()),
                   peak_train_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else np.nan)
        history.append(row)
        log(f"  epoch {epoch}: train loss {row['train_loss']:.4f} | early-stop days AUC {row['es_auc']:.4f}, log loss "
            f"{row['es_log_loss']:.5f}, prob std {row['es_prob_std']:.3f} | {row['train_seconds'] / 60:.1f} min")
        if row["es_log_loss"] < best:
            torch.save(dict(trainable=trainable_state(model), epoch=epoch), out_dir / "best.pt")
            _mirror(out_dir / "best.pt", drive_dir)
        bad = 0 if row["es_log_loss"] < best - settings["min_delta"] else bad + 1
        best = min(best, row["es_log_loss"])
        torch.save(dict(trainable=trainable_state(model), optimizer=opt.state_dict(), history=history, best=best, bad=bad,
                        epoch=epoch), out_dir / "last.pt")
        _mirror(out_dir / "last.pt", drive_dir)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        _mirror(out_dir / "history.csv", drive_dir)
        if bad >= settings["patience"]:
            log(f"  early stop: {bad} epochs without improvement")
            break
    best_ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ck["trainable"], strict=False)
    return pd.DataFrame(history), int(best_ck["epoch"])


def _mirror(path, target_dir):
    if target_dir is None:
        return
    target_dir = Path(target_dir); target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / (Path(path).name + ".copying")
    shutil.copy2(path, tmp); os.replace(tmp, target_dir / Path(path).name)


# ----------------------------------------------------------------------------- training cost

def train_step_cost(name, model, batch, device, lr=1e-4, steps=20, warmup=3):
    """ms per optimizer step and peak GPU memory at one batch size (bf16 autocast, AdamW on trainable params).
    The batch is fixed and already rendered, so chart drawing is excluded."""
    model = model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    x, t, y = (v.to(device) for v in batch)
    dtype = _amp_dtype(device)
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(x, t)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    total, trainable = count(model)
    ms = 1000 * float(np.median(times))
    out = dict(model=name, parameters=total, trainable=trainable, trainable_share=trainable / total,
               batch=len(y), ms_per_step=ms, images_per_s=1000 * len(y) / ms,
               peak_train_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else np.nan)
    del opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- tables

def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def day_boot(frame, scores, n_boot=300, seed=0):
    """AUC of every score on `n_boot` resamples of whole days (the same days for every score)."""
    rng = np.random.default_rng(seed)
    y = frame.label.to_numpy()
    codes, _ = pd.factorize(frame.decision_date)
    days = [np.flatnonzero(codes == d) for d in range(codes.max() + 1)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
        draws.append({n: _auc(y[idx], s[idx]) for n, s in scores.items()})
    return pd.DataFrame(draws)


def compare(frame, scores, reference, n_boot=300, seed=0):
    """AUC with day-bootstrap 95% interval, and the paired difference to `reference` on the same rows."""
    y = frame.label.to_numpy()
    boot = day_boot(frame, scores, n_boot, seed)
    diff = boot.sub(boot[reference], axis=0)
    rows = []
    for n, s in scores.items():
        r = dict(model=n, rows=len(y), days=frame.decision_date.nunique(), auc=_auc(y, s),
                 auc_low=boot[n].quantile(0.025), auc_high=boot[n].quantile(0.975),
                 diff_vs_ref=_auc(y, s) - _auc(y, scores[reference]),
                 diff_low=diff[n].quantile(0.025), diff_high=diff[n].quantile(0.975),
                 score_corr_vs_ref=float(np.corrcoef(s, scores[reference])[0, 1]),
                 max_abs_prob_diff=float(np.abs(s - scores[reference]).max()))
        r["verdict"] = "reference" if n == reference else (
            "better" if r["diff_low"] > 0 else "worse" if r["diff_high"] < 0 else "tied")
        rows.append(r)
    return pd.DataFrame(rows)
