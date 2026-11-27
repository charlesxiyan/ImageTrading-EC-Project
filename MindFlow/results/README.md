# MindFlow results

Summary files of each finished step: tables, takeaways, figures and run settings. Validation year 2023, v4 bundle.
Per-row predictions, windows and checkpoints stay with the runs (they are large and hold per-stock rows), and so do
example chart images drawn from real price data. Each folder is one run, named by its `RUN_ID`; it is the run that
Step 5 summarised.

| step | run | headline (validation 2023) |
|---|---|---|
| 1a daily JKX CNN | `step1a_daily_jkx/daily_20261123_082101` | no daily direction signal (CNN, HGB and rules near 0.5) |
| 1b daily pretrained | `step1b_daily_pretrained/dailypre_20261123_200813` | frozen DINOv2/CLIP probes and time-series foundation models also near 0.5 |
| 2a 5-min direction | `step2a_direction_models/grid_20261123_083042` | every network near 0.5; HGB + SPY at most 0.526 (xsdir, 5 min) |
| 2b direction pretrained | `step2b_direction_pretrained/probe_20261124_034956` | probes and foundation models near 0.5 |
| 3a 5-min volatility | `step3a_volatility_models/grid_20261123_083117` | HGB + SPY 0.689 > CNN 0.660 ≈ ViT 0.661; VIXY panel no help |
| 3b volatility pretrained | `step3b_volatility_pretrained/probe_20261124_010911` | probes ≈ CNN; Kronos best foundation model; random walk 0.580 |
| 4a image information | `step4a_image_information/indicator_20261123_200955` | EW volatility band +0.018; CNN + 38 numbers 0.685 ties HGB 0.687 |
| 4b feature selection | `step4b_feature_selection/features_20261123_083258` | 3-feature subset worse than all 38 on 2023, so all 38 kept |
| 5 summary | `step5_summary` | one figure and parameter table over steps 1-4 |
| 6a regime | `step6a_vix_regime/regime_20261124_182927` | daily VIX adds nothing; multi-day realized vol +0.0025 (significant) |
| 6d pure image | `step6d_pure_image/pure_20261127_023749` | CNN with no numbers input, numbers drawn on fixed scales: key strips 0.6866 (ties final 0.6881); all 43 drawn 0.6894 = HGB 43 |
| 8a inside the pure-image CNN | `step8a_inside_pure_cnn/inside_20261127_063450` | 19,762 val rows, fixed 6d models: session-open line is the most important single element (erase -0.017 / -0.024 AUC); whole chart -0.052 / -0.087; newest and oldest 6 bars matter, middle bars ~0; features encode volatility, volume and time of day, not direction |

Steps 6b (final model) and 7a (efficient computing) are added when their runs finish.
