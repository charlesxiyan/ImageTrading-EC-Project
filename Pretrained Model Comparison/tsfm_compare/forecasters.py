"""Thin, uniform wrappers around pretrained time-series foundation models.

Every wrapper exposes
    .load()                      download weights, move to device, print parameter count
    .n_params                    total parameters (advisor rule: always reported)
    .forecast(batch, horizon)    -> quantiles, shape (n, horizon, 9) at levels 0.1 ... 0.9
    .unload()                    free GPU memory before the next model

`batch` is a dict prepared by the runner:
    close   (n, L)     close / last close   (all models)
    ohlcv   (n, L, 5)  raw candles           (Kronos only)
    t_past, t_future   bar timestamps        (Kronos only)

Forecasts are of the *normalised* close, so q - 1 is a forecast return.
"""
import gc
import sys
import time

import numpy as np
import torch

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def count_parameters(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


class Forecaster:
    name = "base"
    needs = ("close",)                       # which inputs the runner must build

    def __init__(self, model_id, device, batch_size=512, **kw):
        self.model_id, self.device, self.batch_size, self.kw = model_id, str(device), batch_size, kw
        self.n_params = None

    # --- to implement -----------------------------------------------------------
    def _load(self):                          # returns the torch modules to count
        raise NotImplementedError

    def _forecast(self, batch, horizon):      # one mini-batch -> (n, horizon, 9)
        raise NotImplementedError

    # --- shared -----------------------------------------------------------------
    def load(self):
        start = time.perf_counter()
        modules = self._load()
        self.n_params = count_parameters(*modules)
        print(f"{self.name} [{self.model_id}]: {self.n_params:,} parameters "
              f"(loaded in {time.perf_counter() - start:.0f}s)")
        return self

    @torch.no_grad()
    def forecast(self, batch, horizon, log_every=20):
        n = len(batch["close"])
        out = np.empty((n, horizon, len(QUANTILES)), np.float32)
        steps = range(0, n, self.batch_size)
        start = time.perf_counter()
        for k, i in enumerate(steps, 1):
            sl = slice(i, i + self.batch_size)
            out[sl] = self._forecast({key: v[sl] for key, v in batch.items()}, horizon)
            if k % log_every == 0 or k == len(steps):
                done = min(i + self.batch_size, n)
                print(f"  {self.name}: {done:,}/{n:,} series, {done / (time.perf_counter() - start):.0f}/s",
                      flush=True)
        return np.sort(out, axis=-1)               # guard against quantile crossing

    def unload(self):
        for attr in list(vars(self)):
            if attr not in ("name", "model_id", "device", "batch_size", "kw", "n_params"):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- Chronos
class Chronos(Forecaster):
    """Amazon Chronos-2 (default) or Chronos-Bolt. `pip install chronos-forecasting`."""
    name = "chronos"

    def _load(self):
        from chronos import BaseChronosPipeline
        use_bf16 = self.device.startswith("cuda") and torch.cuda.is_bf16_supported()   # T4: no native bf16 -> fp32
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device,
                                                        torch_dtype=dtype)
        return [self.pipe.model]

    def _forecast(self, batch, horizon):
        x = torch.from_numpy(batch["close"])
        if "chronos-2" in self.model_id or type(self.pipe).__name__ == "Chronos2Pipeline":
            x = x[:, None, :]                       # (n, variates=1, L)
            q, _ = self.pipe.predict_quantiles(x, prediction_length=horizon, quantile_levels=QUANTILES,
                                               batch_size=self.batch_size)
            q = torch.stack(q)[:, 0]               # list of (1, h, 9) -> (n, h, 9)
        else:
            q, _ = self.pipe.predict_quantiles(x, prediction_length=horizon, quantile_levels=QUANTILES)
        return q.float().cpu().numpy()

    def finetune(self, series, horizon, context_length, mode="lora", learning_rate=1e-5,
                 num_steps=1000, batch_size=256, output_dir="chronos2_finetuned"):
        """Light fine-tune of Chronos-2 on the training-period price series (forecast loss).

        This adapts the forecaster to 5-minute equity prices; it never sees our labels.
        Returns a new Chronos wrapper holding the fine-tuned pipeline.
        """
        if not hasattr(self.pipe, "fit"):
            raise RuntimeError("fine-tuning is implemented for Chronos-2 only")
        if mode == "lora":
            try:
                import peft  # noqa: F401  (fails e.g. on an incompatible torchao)
            except ImportError as err:
                print(f"LoRA unavailable ({err}); falling back to full fine-tune with lr 1e-6")
                mode, learning_rate = "full", 1e-6
        start = time.perf_counter()
        pipe = self.pipe.fit(inputs=series, prediction_length=horizon, context_length=context_length,
                             finetune_mode=mode, learning_rate=learning_rate, num_steps=num_steps,
                             batch_size=batch_size, output_dir=output_dir, min_past=context_length,
                             report_to="none")
        tuned = Chronos(self.model_id, self.device, self.batch_size)
        tuned.name = f"{self.name}_ft"
        tuned.pipe = pipe
        tuned.n_params = count_parameters(pipe.model)
        trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
        print(f"{tuned.name}: {tuned.n_params:,} parameters ({trainable:,} trainable during {mode} fine-tune), "
              f"{num_steps} steps in {time.perf_counter() - start:.0f}s")
        return tuned


# ----------------------------------------------------------------------------- TimesFM
class TimesFM(Forecaster):
    """Google TimesFM 3.0 (default) or 2.5. `pip install timesfm[torch]`.

    TimesFM 3.0 weights carry a non-commercial licence (fine for coursework/research);
    2.5 weights are Apache-2.0. If 3.0 fails to load, 2.5 is used instead.
    """
    name = "timesfm"
    FALLBACK = "google/timesfm-2.5-200m-pytorch"

    def _load(self):
        if "2.5" not in self.model_id:
            try:
                from timesfm3 import TimesFM3Forecaster
                self.v3 = TimesFM3Forecaster.from_pretrained(self.model_id, device=self.device,
                                                             per_core_batch_size=self.batch_size)
                return [self.v3.model]
            except Exception as err:                            # gated repo, old package, ...
                print(f"TimesFM 3.0 unavailable ({type(err).__name__}: {err}); using {self.FALLBACK}")
                self.model_id = self.FALLBACK
        import timesfm
        self.v25 = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        self._compiled_for = None
        return [self.v25.model]

    def _forecast(self, batch, horizon):
        series = list(batch["close"])
        if hasattr(self, "v3"):
            outs = self.v3.predict_batch(series, horizon=horizon, return_quantiles=True)
            return np.stack([o.quantiles for o in outs])           # (n, h, 9)
        import timesfm
        if self._compiled_for != (len(series[0]), horizon):
            ctx = int(np.ceil(len(series[0]) / 32) * 32)            # multiple of the input patch
            self.v25.compile(timesfm.ForecastConfig(
                max_context=ctx, max_horizon=max(128, horizon), normalize_inputs=True,
                per_core_batch_size=self.batch_size, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
            self._compiled_for = (len(series[0]), horizon)
        _, q = self.v25.forecast(horizon=horizon, inputs=series)
        return np.asarray(q)[:, :, 1:10]                            # column 0 is the mean


# ----------------------------------------------------------------------------- Moirai
class Moirai(Forecaster):
    """Salesforce Moirai 2.0. `pip install --no-deps uni2ts` (see notebook install cell)."""
    name = "moirai"

    def _load(self):
        from uni2ts.model.moirai2 import Moirai2Module
        self.module = Moirai2Module.from_pretrained(self.model_id)
        self._models = {}
        return [self.module]

    def _model(self, context_length, horizon):
        from uni2ts.model.moirai2 import Moirai2Forecast
        key = (context_length, horizon)
        if key not in self._models:
            self._models[key] = Moirai2Forecast(
                module=self.module, prediction_length=horizon, context_length=context_length,
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(self.device).eval()
        return self._models[key]

    def _forecast(self, batch, horizon):
        x = batch["close"]
        model = self._model(x.shape[1], horizon)
        q = model.predict(past_target=[row[:, None] for row in x])  # (n, 9, h[, 1])
        q = np.asarray(q).reshape(len(x), len(QUANTILES), horizon)
        return q.transpose(0, 2, 1)


# ----------------------------------------------------------------------------- Kronos
class Kronos(Forecaster):
    """Kronos: a foundation model for financial candlesticks (OHLCV), arXiv 2508.02739.

    Not on PyPI: the notebook clones github.com/shiyu-coder/Kronos and passes its path
    as `repo`. Kronos samples future candles autoregressively; we draw `samples` paths
    per series and convert them to empirical quantiles of the close.
    """
    name = "kronos"
    needs = ("close", "ohlcv", "t_past", "t_future")

    def _load(self):
        sys.path.insert(0, str(self.kw["repo"]))
        from model import Kronos as KronosModel, KronosTokenizer
        self.tokenizer = KronosTokenizer.from_pretrained(self.kw.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base"))
        self.model = KronosModel.from_pretrained(self.model_id)
        self.tokenizer.to(self.device).eval(); self.model.to(self.device).eval()
        return [self.model, self.tokenizer]

    @staticmethod
    def _stamps(t):
        t = np.asarray(t, "datetime64[m]")
        minute = (t.astype(np.int64) % 60)
        hour = (t.astype("datetime64[h]").astype(np.int64) % 24)
        days = t.astype("datetime64[D]")
        weekday = (days.astype(np.int64) + 3) % 7                   # 1970-01-01 was a Thursday
        month_start = days.astype("datetime64[M]")
        day = (days - month_start.astype("datetime64[D]")).astype(np.int64) + 1
        month = month_start.astype(np.int64) % 12 + 1
        return np.stack([minute, hour, weekday, day, month], -1).astype(np.float32)

    def _forecast(self, batch, horizon):
        from model.kronos import auto_regressive_inference
        samples, clip = self.kw.get("samples", 4), 5
        ohlcv = batch["ohlcv"].astype(np.float64)
        amount = ohlcv[:, :, 4:5] * ohlcv[:, :, :4].mean(-1, keepdims=True)
        x = np.concatenate([ohlcv, amount], -1)                       # open high low close volume amount
        mean, std = x.mean(1, keepdims=True), x.std(1, keepdims=True)
        x = np.clip((x - mean) / (std + 1e-5), -clip, clip)
        rep = lambda a: torch.from_numpy(np.repeat(a, samples, axis=0)).float().to(self.device)
        pred = auto_regressive_inference(
            self.tokenizer, self.model, rep(x), rep(self._stamps(batch["t_past"])),
            rep(self._stamps(batch["t_future"])), max_context=512, pred_len=horizon, clip=clip,
            T=self.kw.get("temperature", 1.0), top_k=0, top_p=self.kw.get("top_p", 0.9), sample_count=1)
        pred = pred[:, -horizon:, :].reshape(len(x), samples, horizon, 6)
        close = pred[..., 3] * (std[:, 0, 3, None, None] + 1e-5) + mean[:, 0, 3, None, None]  # (n, samples, h)
        close = close / ohlcv[:, -1, 3, None, None]                                       # normalise by last close
        return np.quantile(close, QUANTILES, axis=1).transpose(1, 2, 0)


REGISTRY = {"chronos": Chronos, "timesfm": TimesFM, "moirai": Moirai, "kronos": Kronos}


def build(spec, device):
    """spec = dict(kind='chronos', model_id='amazon/chronos-2', batch_size=..., **extra)."""
    spec = dict(spec)
    cls = REGISTRY[spec.pop("kind")]
    name = spec.pop("name", None)
    f = cls(spec.pop("model_id"), device, spec.pop("batch_size", 512), **spec)
    if name:
        f.name = name
    return f
