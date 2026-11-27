"""Step 8a: what did the pure-image CNN of Step 6d learn? (data-based; the weights stay fixed, nothing is trained)

The weights alone say little: training moved them only ~5% from their random start, and a weight's size ignores how
large the activation it multiplies is. Every analysis here runs the trained model on real 2023 validation pictures:

  occlusion        replace one part of the picture (a strip, the chart's price area, the session-open line, the open
                   ticks, ...) by the average picture there and measure the AUC drop (paired day bootstrap)
  time occlusion   the same for blocks of 6 bars across the time-aligned rows
  number strip     set one of the 43 drawn numbers to zero (its training mean) at a time (pure_all only)
  activation x weight   the head's input activations times its weights: which of the 4 x 78 positions and which of
                   the 256 channels move the prediction from picture to picture; channel ablation of the top ones
  saliency         Grad-CAM on the last convolution and gradient x input, averaged over pictures, and per region
  probing          how well the known numbers (the 43 numbers, time of day) can be read linearly from the CNN's
                   pooled features (cross-validated by day)

Pictures are drawn once (pure_all); pure_key's picture is its first 423 rows (identical by construction).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import indicator_cnn as ic
import pure_image as pi
from indicator_cnn import autocast

__version__ = "2026-11-27.2"

PRICE_ROWS = ic.SIZE - ic.SIZE // 5 - 1          # 187: price area of the chart (then a blank row, then volume)
MARKER = pi.MARKER / 255.0
N_NUMBERS = 43                                   # Step 6b's 43 numbers, in the order the strip draws them


def regions(variant):
    """name -> (first row, end row, label) of each part of the picture."""
    g, out, r = pi.GAP, {}, 0
    out["chart_price"] = (0, PRICE_ROWS, "chart: price bars")
    out["chart_volume"] = (PRICE_ROWS + 1, ic.SIZE, "chart: volume bars")
    r = ic.SIZE
    out["ewsd"] = (r, r + ic.BAND, "stock EWSD band"); r += ic.BAND + g
    out["spy"] = (r, r + pi.SPY_BAND, "SPY EWSD band"); r += pi.SPY_BAND + g
    out["volume"] = (r, r + pi.VOL_BAND, "volume band"); r += pi.VOL_BAND + g
    out["rv"] = (r, r + pi.RV_ROWS, "multi-day RV bars"); r += pi.RV_ROWS
    if pi.VARIANTS[variant]:
        r += g
        out["numbers"] = (r, r + pi.NUM_STRIP, "number strip (43 numbers)")
    assert out[list(out)[-1]][1] == pi.height(variant)
    return out


TIME_ROWS = (0, regions("pure_key")["volume"][1])     # rows whose columns are bars (chart .. volume band)


# ----------------------------------------------------------------------------- pictures

class _Raw(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return torch.from_numpy(self.ds.image(i)), i


def render(ds, workers=2, batch=256):
    """All pictures of a pi.PureCharts dataset as one uint8 tensor (N, H, 234)."""
    out = None
    for x, idx in DataLoader(_Raw(ds), batch_size=batch, num_workers=workers):
        if out is None:
            out = torch.empty((len(ds),) + tuple(x.shape[1:]), dtype=torch.uint8)
        out[idx] = x
    return out


def load(variant, settings, ckpt, device):
    m = pi.PureCNN(settings, variant)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return m.to(device).eval()


def batches(X, variant, batch, device):
    h = pi.height(variant)
    for a in range(0, len(X), batch):
        yield a, X[a:a + batch, :h].to(device, non_blocking=True).float().div_(255.0)[:, None]


# ----------------------------------------------------------------------------- occlusion

def ops_for(variant, fills=("mean", "zero"), block=6):
    """(name, group, label, function(x, mean picture) -> x) for one model."""
    R, ops = regions(variant), []

    def fill(r0, r1, kind, c0=0, c1=ic.SIZE):
        def f(x, m):
            x = x.clone()
            x[:, :, r0:r1, c0:c1] = m[:, :, r0:r1, c0:c1] if kind == "mean" else 0.0
            return x
        return f

    for kind in fills:
        for k, (r0, r1, lab) in R.items():
            ops.append((f"{kind}:{k}", f"region ({kind} fill)", lab, fill(r0, r1, kind)))
        ops.append((f"{kind}:chart", f"region ({kind} fill)", "whole chart (price + volume)", fill(0, ic.SIZE, kind)))

    def erase(col0, what):
        def f(x, m):
            x = x.clone()
            part = x[:, :, :PRICE_ROWS, col0::3] if col0 is not None else x[:, :, :ic.SIZE]
            hit = (part - (1.0 if what == "tick" else MARKER)).abs() < 1e-3
            part[hit] = 0.0
            return x
        return f

    ops += [("erase:marker", "erase", "session-open line", erase(None, "marker")),
            ("erase:open", "erase", "open ticks (left column)", erase(0, "tick")),
            ("erase:highlow", "erase", "high-low bars (middle column)", erase(1, "tick")),
            ("erase:close", "erase", "close ticks (right column)", erase(2, "tick"))]
    for b in range(0, ic.LOOKBACK, block):
        e = min(b + block, ic.LOOKBACK)
        lab = f"bars t-{ic.LOOKBACK - 1 - b}..t-{ic.LOOKBACK - e}"
        ops.append((f"time:{b:02d}", "time block (mean fill)", lab, fill(*TIME_ROWS, "mean", 3 * b, 3 * e)))
    if pi.VARIANTS[variant]:
        r0 = R["numbers"][0]
        mid = r0 + pi.NUM_STRIP // 2

        def zero_number(j):
            def f(x, m):
                x = x.clone()
                c = pi.NUM_WIDTH * j
                x[:, :, r0:r0 + pi.NUM_STRIP, c:c + 3] = 0.0
                x[:, :, mid, c:c + 3] = 120 / 255.0
                return x
            return f
        for j in range(N_NUMBERS):
            ops.append((f"number:{j:02d}", "number set to 0", f"number {j}", zero_number(j)))
    return ops


@torch.no_grad()
def run_ops(model, X, variant, ops, mean_picture, device, batch, amp):
    """Probability of the high-volatility class for the unchanged picture ('base') and under every op."""
    out = {"base": np.empty(len(X))} | {n: np.empty(len(X)) for n, *_ in ops}
    m = mean_picture[:, :, :pi.height(variant)].to(device)
    for a, x in batches(X, variant, batch, device):
        for name, f in [("base", lambda x, m: x)] + [(n, f) for n, _, _, f in ops]:
            with autocast(device, amp):
                z = model(f(x, m))
            out[name][a:a + len(x)] = z.float().softmax(1)[:, 1].cpu().numpy()
    return out


def mean_picture(X, variant, batch=2048):
    h = pi.height(variant)
    s = torch.zeros(h, ic.SIZE, dtype=torch.float64)
    for a in range(0, len(X), batch):
        s += X[a:a + batch, :h].double().sum(0)
    return (s / len(X) / 255.0).float()[None, None]


def _log_loss(y, p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def drop_table(frame, scores, ops, n_boot, cell="2023", seed=0):
    """Per op: AUC and its drop, log loss and its rise, Brier score, each against the unchanged picture,
    with 95% paired day-bootstrap intervals (the same day draws for AUC and log loss)."""
    pred = frame[["decision_date", "label"]].copy()
    names = ["base"] + [n for n, *_ in ops]
    for n in names:
        pred[f"score_{n.replace(':', '__')}"] = scores[n]
    boot = ic.day_bootstrap(pred, [n.replace(":", "__") for n in names], n_boot, seed)
    y = pred.label.to_numpy().astype(np.float64)
    codes, _ = pd.factorize(pred.decision_date)
    days = [np.flatnonzero(codes == d) for d in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))]) for _ in range(n_boot)]
    base = ic._auc(y, scores["base"])
    base_ll = _log_loss(y, scores["base"])
    base_draws = np.array([_log_loss(y[i], scores["base"][i]) for i in draws])
    rows = []
    for n, group, label, _ in ops:
        d = (boot["base"] - boot[n.replace(":", "__")]).to_numpy()
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        ll = _log_loss(y, scores[n])
        r = np.array([_log_loss(y[i], scores[n][i]) for i in draws]) - base_draws
        llo, lhi = np.nanpercentile(r, [2.5, 97.5])
        rows.append(dict(op=n, group=group, part=label, auc=ic._auc(y, scores[n]), drop=base - ic._auc(y, scores[n]),
                         low=lo, high=hi, verdict="matters" if lo > 0 else "hurts" if hi < 0 else "no effect",
                         log_loss=ll, log_loss_rise=ll - base_ll, ll_low=llo, ll_high=lhi,
                         ll_verdict="matters" if llo > 0 else "hurts" if lhi < 0 else "no effect",
                         brier=float(np.mean((np.asarray(scores[n]) - y) ** 2))))
    return dict(auc=base, log_loss=base_ll, brier=float(np.mean((np.asarray(scores["base"]) - y) ** 2))), pd.DataFrame(rows)


# ----------------------------------------------------------------------------- activation x weight

def head_delta(model):
    """Head weights for (high - low) class, as (256, 4, 78): logit difference = sum(features * this) + bias."""
    w = model.cnn.head[-1].weight.detach().float()
    return (w[1] - w[0]).reshape(model.cnn.feature_shape)


@torch.no_grad()
def contributions(model, X, variant, device, batch, amp):
    """Per position and per channel: mean and std over pictures of their contribution to the logit difference;
    and pooled features for probing (per channel: mean over the picture, and the last bar's column)."""
    wd = head_delta(model).to(device)
    feats = {}
    hook = model.cnn.features.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    C, Hh, T = wd.shape
    s_pos = torch.zeros(Hh, T, device=device, dtype=torch.float64); q_pos = s_pos.clone()
    s_ch = torch.zeros(C, device=device, dtype=torch.float64); q_ch = s_ch.clone()
    s_bin = torch.zeros(Hh, device=device, dtype=torch.float64); q_bin = s_bin.clone()
    probe = np.empty((len(X), 2 * C), np.float32)
    for a, x in batches(X, variant, batch, device):
        with autocast(device, amp):
            model(x)
        f = feats["f"].float()                                   # (B, 256, 4, 78)
        c = f * wd
        pos = c.sum(1).double(); ch = c.sum((2, 3)).double(); bn = c.sum((1, 3)).double()
        s_pos += pos.sum(0); q_pos += (pos ** 2).sum(0)
        s_ch += ch.sum(0); q_ch += (ch ** 2).sum(0)
        s_bin += bn.sum(0); q_bin += (bn ** 2).sum(0)
        probe[a:a + len(x)] = torch.cat([f.mean((2, 3)), f[:, :, :, -1].mean(2)], 1).cpu().numpy()
    hook.remove()
    n = len(X)
    sd = lambda s, q: ((q / n - (s / n) ** 2).clamp_min(0) ** 0.5).cpu().numpy()
    return dict(pos_mean=(s_pos / n).cpu().numpy(), pos_std=sd(s_pos, q_pos), ch_mean=(s_ch / n).cpu().numpy(),
                ch_std=sd(s_ch, q_ch), bin_mean=(s_bin / n).cpu().numpy(), bin_std=sd(s_bin, q_bin), probe=probe)


@torch.no_grad()
def channel_ablation(model, X, variant, y, frame, order, ks, device, batch, amp, n_random=3, seed=0):
    """AUC with the top-k channels (by `order`) of the head's input set to zero, vs k random channels."""
    rng = np.random.default_rng(seed)
    C = model.cnn.feature_shape[0]
    sets = {f"top{k}": list(order[:k]) for k in ks}
    for k in ks:
        for r in range(n_random):
            sets[f"rand{k}_{r}"] = list(rng.choice(C, k, replace=False))
    state = {"drop": None}

    def hook(m, i, o):
        if state["drop"] is not None:
            o = o.clone(); o[:, state["drop"]] = 0.0
        return o
    h = model.cnn.features.register_forward_hook(hook)
    out = {}
    for name, chans in [("base", None)] + list(sets.items()):
        state["drop"] = chans
        p = np.empty(len(X))
        for a, x in batches(X, variant, batch, device):
            with autocast(device, amp):
                p[a:a + len(x)] = model(x).float().softmax(1)[:, 1].cpu().numpy()
        out[name] = ic._auc(y, p)
    h.remove()
    rows = []
    for k in ks:
        rand = [out[f"rand{k}_{r}"] for r in range(n_random)]
        rows.append(dict(k=k, auc_top=out[f"top{k}"], drop_top=out["base"] - out[f"top{k}"],
                         auc_random_mean=float(np.mean(rand)), drop_random_mean=out["base"] - float(np.mean(rand))))
    return out["base"], pd.DataFrame(rows)


# ----------------------------------------------------------------------------- saliency

def saliency(model, X, variant, device, batch, layer=10):
    """Grad-CAM at the last convolution block (per picture, H' x 78) and |gradient x input| (H x 234), both for the
    logit difference. Returns (cams [N, H', 78] float16, mean |g*x| map, mean Grad-CAM map)."""
    acts = {}
    hk = model.cnn.features[layer].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    cams, gx_sum = [], None
    for a, x in batches(X, variant, batch, device):
        x = x.clone().requires_grad_(True)
        z = model(x)
        target = (z[:, 1] - z[:, 0]).sum()
        A = acts["a"]
        gA, gx = torch.autograd.grad(target, [A, x])
        w = gA.mean((2, 3), keepdim=True)
        cam = torch.relu((w * A).sum(1)).detach()
        cams.append(cam.half().cpu())
        g = (gx * x).abs().sum(0)[0].detach().double()
        gx_sum = g if gx_sum is None else gx_sum + g
    hk.remove()
    cams = torch.cat(cams)
    return cams.numpy(), (gx_sum / len(X)).cpu().numpy(), cams.float().mean(0).numpy()


def region_shares(attr_map, variant):
    """Share of a (H x 234) attribution map in each region, and its share of the picture's rows."""
    R = regions(variant)
    total = attr_map.sum()
    return pd.DataFrame([dict(region=k, part=lab, share=attr_map[r0:r1].sum() / total,
                              share_of_rows=(r1 - r0) / attr_map.shape[0]) for k, (r0, r1, lab) in R.items()])


# ----------------------------------------------------------------------------- probing

def probe(features, targets, groups, alphas=(0.1, 1, 10, 100, 1000), folds=5):
    """Cross-validated R^2 (folds by day) of a ridge regression from the CNN's pooled features to each target."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Xf = np.asarray(features, np.float64)
    rows = []
    for name in targets.columns:
        y = targets[name].to_numpy(np.float64)
        ok = np.isfinite(y)
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(folds).split(Xf[ok], y[ok], groups[ok]):
            idx = np.flatnonzero(ok)
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(Xf[idx[tr]], y[idx[tr]])
            pred[idx[te]] = m.predict(Xf[idx[te]])
        r2 = 1 - np.nanmean((y[ok] - pred[ok]) ** 2) / np.nanvar(y[ok])
        rows.append(dict(target=name, r2=float(r2), n=int(ok.sum())))
    return pd.DataFrame(rows)
