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
| 6b final model | `step6b_final_model/final_20261125_222625` | final CNN `ewsd_tab_rv` (783,556 params) 0.6881 / 0.6760 (5 / 60 min) vs HGB 43 0.6894 / 0.6780; test 2024-25: 0.6974 / 0.6827 vs 0.6983 / 0.6850 (tied at 5 min, -0.002 at 60 min); image adds +0.001 over the same numbers on test |
| 6c volatility use | `step6c_volatility_use/voluse_20261126_045556` | IC vs next abs. return 0.13 (ICIR 1.5, test, 5 min); low-minus-high volatility long-short loses (no low-vol anomaly intraday); volatility targeting halves the variability of daily risk |
| 6d pure image | `step6d_pure_image/pure_20261127_023749` | CNN with no numbers input, numbers drawn on fixed scales: key strips 0.6866 (ties final 0.6881); all 43 drawn 0.6894 = HGB 43 |
| 7b distillation | `step7b_distillation/distill_20261126_232557` | teacher final CNN 0.6881; distilled students beat label-trained ones by +0.0008-0.0010 (signif.) and tie the teacher: small CNN 86,020 params 0.6885 (4.5x faster), half-height 0.6886 (8.3x), 43-number MLP 7,106 params 0.6887 (192x) |
| 7b+ pure-image distillation | `step7b_plus_pure_distillation/puredistill_20261127_063830` | teacher 6d pure_all 0.6894; small pure-image CNN (78,914 params) distilled 0.6892 vs labels 0.6865 (+0.0026 signif.), ties teacher and HGB 43; half-height 0.6889 (7.8x faster); tiny 29,858 params 0.6880 (-0.0013 signif.) |
| 8a inside the pure-image CNN | `step8a_inside_pure_cnn/inside_20261127_063450` | 19,762 val rows, fixed 6d models: session-open line is the most important single element (erase -0.017 / -0.024 AUC); whole chart -0.052 / -0.087; newest and oldest 6 bars matter, middle bars ~0; features encode volatility, volume and time of day, not direction |
| 6f compact pictures | `step6f_compact_images/compact_20261127_230744` | same information as 6d pure_all in denser layouts: 6 channels 0.6894, filled 0.6889, 6 channels at half height 0.6889 all tie pure_all 0.6894; quarter height 0.6883 (-0.0010 signif.); half-height channels 2.7x faster on the GPU at no significant loss |

Step 6b and 6c include the held-out test years 2024-2025 (scored once, after the final model was chosen on 2023). Steps 6d, 6f, 7b, 7b+ and 8a are validation 2023 only; their test runs are not done.
