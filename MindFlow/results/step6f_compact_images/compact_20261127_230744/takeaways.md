Step 6d run pure_20261127_023749 (reference `pure_all`), Step 6b run final_20261125_222625.

- pure_all, 466 x 234 (6d): input 1x466x234, lit 9.4%, 776,450 parameters, AUC 0.6894, log loss 0.6278, GPU 1.00x vs 6d, 2.9 min per epoch
- pure_all, a quarter of the height (4x1 max pool): input 1x116x234, lit 12.0%, 776,450 parameters, AUC 0.6883, log loss 0.6286, GPU 2.44x vs 6d, 2.7 min per epoch
- 6 channels, half height (6 x 117 x 234): input 6x117x234, lit 15.1%, 781,250 parameters, AUC 0.6889, log loss 0.6288, GPU 2.75x vs 6d, 5.9 min per epoch
- strips as 6 channels of 234 x 234: input 6x234x234, lit 14.8%, 781,250 parameters, AUC 0.6894, log loss 0.6281, GPU 1.71x vs 6d, 5.9 min per epoch
- pure_all with the empty space filled: input 1x466x234, lit 37.6%, 776,450 parameters, AUC 0.6889, log loss 0.6281, GPU 1.04x vs 6d, 3.9 min per epoch
- vol_h1 quarter height vs pure_all (6d): AUC -0.0010 [-0.0018, -0.0001] -> worse; log loss +0.0008 [+0.0003, +0.0013] -> worse
- vol_h1 strips as channels vs pure_all (6d): AUC +0.0001 [-0.0009, +0.0011] -> tied; log loss +0.0004 [-0.0003, +0.0010] -> tied
- vol_h1 filled picture vs pure_all (6d): AUC -0.0004 [-0.0012, +0.0004] -> tied; log loss +0.0004 [-0.0001, +0.0008] -> tied
- vol_h1 6 channels at half height vs pure_all (6d): AUC -0.0005 [-0.0015, +0.0005] -> tied; log loss +0.0010 [+0.0000, +0.0019] -> worse
- vol_h1 half height vs full height (both channels): AUC -0.0005 [-0.0013, +0.0002] -> tied; log loss +0.0006 [-0.0001, +0.0012] -> tied
- vol_h1 quarter height vs HGB 43: AUC -0.0010 [-0.0024, +0.0004] -> tied; log loss -0.0001 [-0.0013, +0.0010] -> tied
- vol_h1 6 channels at half height vs HGB 43: AUC -0.0005 [-0.0019, +0.0007] -> tied; log loss +0.0000 [-0.0007, +0.0008] -> tied
