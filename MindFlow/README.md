# MindFlow: the comparison experiments, in the order of the argument

Each step answers one question and motivates the next. All steps use the v4 bundle
(`alldata_bundle_v4.zip`), the same split (train 2016–2022, validation 2023, test 2024–2025) and, within a step,
identical rows for every model. Every model reports its parameter count (HGB: tree nodes).

| step | question | notebook | models |
|---|---|---|---|
| 1a | Is the Jiang–Kelly–Xiu daily chart signal there on our large caps? | `Step1a_Daily_JKX_CNN` | JKX CNN I5/I20 (5 seeds), HGB, reversal/momentum rules |
| 1b | Do pretrained models find it? | `Step1b_Daily_Pretrained` | frozen DINOv2-S / CLIP-B/16 / random ViT-S + probes on the JKX images; Chronos-2, TimesFM, Moirai 2, Kronos zero-shot; random-walk yardstick |
| 2a | On 5-minute charts, does any architecture predict direction? | `Step2a_Intraday_Direction_Models` | image CNN (± SPY panel), ResNet CNN, 1D CNN (± SPY), T2T-style ViT (± SPY), HGB (± SPY), rules |
| 2b | Do pretrained models predict 5-minute direction? | `Step2b_Intraday_Direction_Pretrained` | frozen encoders + probes, time-series foundation models, joined with 2a |
| 3a | Same models, volatility target: signal, and who wins? | `Step3a_Intraday_Volatility_Models` | as 2a |
| 3b | Pretrained models on volatility | `Step3b_Intraday_Volatility_Pretrained` | as 2b, joined with 3a |
| 4a | Add information to the picture until the image model reaches HGB; is the image still useful? | `Step4a_Add_Image_Information` | ladder: chart → + SPY panel → + fixed-scale EW volatility band → + MA → + 38 numbers; control: the 38 numbers alone (MLP) |
| 4b | Which of the 38 hand features matter? | `Step4b_Feature_Selection` | HGB: permutation importance, group drop / keep-only, forward selection (chosen on 2022, confirmed on 2023) |
| 5 | One figure, one parameter table | `Step5_Summary` | reads every step's newest run |
| 6a | Does market-regime information (VIX, multi-day realized vol, VIXY) help HGB? | `Step6a_VIX_Regime` | HGB with and without regime features, same rows |
| 6d | Can a model that sees only pictures reach the final model if the numbers are drawn into the picture? | `Step6d_Pure_Image` | CNN on chart + EWSD + fixed-scale strips (time of day, SPY volatility, volume, multi-day RV), with and without all 43 numbers drawn as bars; no numbers input; paired with 6b and 4a |
| 7a | Efficient computing for the final model (run after it is chosen) | `Step7a_Efficient_Computing` | float32 / bfloat16 GPU and INT8 CPU inference (static PTQ for CNNs, dynamic for ViTs): AUC, size, latency, memory; DINOv2 ViT-S/14 + LoRA vs full fine-tune, frozen and scratch ViT: trainable parameters, memory, time, AUC |

## Running on Colab

1. Drive folder `Colab Notebooks/EC/Project/Demo/MindFlow/` holds `alldata_bundle_v4.zip`, the `code/` folder and the
   notebooks. Runs are written to `MindFlow/runs/<step>/<RUN_ID>/`.
2. Run the notebooks in order (1a → 5). A later step finds the newest finished run of the step it compares with;
   set its `REFERENCE_RUN` / `STEP1A_RUN` to pin one.
3. Each notebook has `QUICK = True` for a short first pass and resumes after a disconnect when `RUN_ID` is set to the
   printed id. Notebooks in one runtime share the unpacked bundle.
4. The held-out test section of each notebook is run once, at the end (`RUN_TEST = True`).

## Code (`code/`)

- `chartcnn/` 5-minute data, labels, image/1D CNNs, T2T-style ViT, HGB baselines, the horizon × target grid
- `daily_jkx.py` daily bars, JKX images and CNN
- `daily_pretrained.py` frozen encoders + probes and foundation models on the daily rows of Step 1a
- `frozen_probe.py` frozen DINOv2 / CLIP encoders and probes on 5-minute charts
- `pretrained_ts.py`, `tsfm_compare/` time-series foundation models (zero-shot), contexts and scores
- `indicator_cnn.py` chart variants (MA, EW volatility band), hybrid CNN + feature MLP, the features-only control
- `feature_select.py` HGB feature selection
- `pure_image.py` pure-image pictures (the numbers drawn on fixed scales) and their CNN (Step 6d)
- `mindflow.py` shared setup, run discovery, parameter tables, summary figure

## Repository QA

```
python MindFlow/tools/build_notebooks.py               # writes the notebooks into MindFlow/
python MindFlow/tools/smoke.py MindFlow <work dir>      # runs all of them on a tiny synthetic bundle
```
The smoke run uses untrained tiny encoders and the random-walk forecaster only, so it needs no downloads.
