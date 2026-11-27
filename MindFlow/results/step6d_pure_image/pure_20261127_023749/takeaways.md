Step 6b run final_20261125_222625, final model `ewsd_tab_rv`; strip scales fitted on training windows.

- vol_h1 final model (6b): image + 43 numbers: AUC 0.6881 [0.6796, 0.6961]
- vol_h1 HGB 43 (6b): AUC 0.6894 [0.6806, 0.6975]
- vol_h1 43-number MLP (6b): AUC 0.6878 [0.6792, 0.6962]
- vol_h1 chart + EWSD, image only (4a): AUC 0.6776 [0.6695, 0.6853]
- vol_h1 original chart, image only (4a): AUC 0.6599 [0.6525, 0.6674]
- vol_h1 pure image: chart + EWSD + time, SPY vol, volume, multi-day RV (no numbers): AUC 0.6866 [0.6782, 0.6944], 776,450 parameters
- vol_h1 pure image: pure_key + all 43 numbers drawn as bars (no numbers input): AUC 0.6894 [0.6811, 0.6971], 776,450 parameters
- vol_h1 key strips added to the chart + EWSD picture (both image only): +0.0089 [+0.0066, +0.0112] -> better
- vol_h1 all 43 numbers drawn vs the key strips: +0.0027 [+0.0016, +0.0038] -> better
- vol_h1 pure image (key) vs the final model (image + numbers input): -0.0015 [-0.0034, +0.0005] -> tied
- vol_h1 pure image (all) vs the final model (image + numbers input): +0.0012 [-0.0006, +0.0029] -> tied
- vol_h1 pure image (key) vs HGB 43: -0.0028 [-0.0047, -0.0008] -> worse
- vol_h1 pure image (all) vs HGB 43: -0.0000 [-0.0017, +0.0015] -> tied
- vol_h1 pure image (all) vs the 43-number MLP: +0.0015 [-0.0005, +0.0033] -> tied
