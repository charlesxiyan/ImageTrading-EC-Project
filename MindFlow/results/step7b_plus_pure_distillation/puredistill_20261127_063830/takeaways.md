Teacher `pure_all` (Step 6d run pure_20261127_023749); alpha 0.7, T 1.0. Validation 2023, vol_h1 (5 min), same rows for every model; paired day bootstrap.

- teacher: pure image, all numbers drawn (6d): AUC 0.6894 [0.6811, 0.6971], 776,450 parameters
- pure image, key strips (6d): AUC 0.6866 [0.6782, 0.6944]
- final model, image + numbers (6b): AUC 0.6881 [0.6796, 0.6961]
- HGB 43 (6b): AUC 0.6894 [0.6806, 0.6975]
- 43-number MLP (6b): AUC 0.6878 [0.6792, 0.6962]
- small pure-image CNN 16-32-64, distilled: AUC 0.6892 [0.6809, 0.6970], 78,914 parameters
- small pure-image CNN 16-32-64, labels: AUC 0.6865 [0.6783, 0.6941], 78,914 parameters
- small pure-image CNN, half-height picture, distilled: AUC 0.6889 [0.6807, 0.6970], 78,914 parameters
- small pure-image CNN, half-height picture, labels: AUC 0.6885 [0.6803, 0.6965], 78,914 parameters
- tiny pure-image CNN 8-16-32, distilled: AUC 0.6880 [0.6799, 0.6959], 29,858 parameters
- distillation, small CNN: +0.0026 [+0.0019, +0.0034] -> better
- distillation, small CNN on the half-height picture: +0.0004 [-0.0002, +0.0010] -> tied
- distilled small CNN vs the teacher: -0.0002 [-0.0006, +0.0002] -> tied
- distilled half-height CNN vs the teacher: -0.0005 [-0.0011, +0.0002] -> tied
- distilled tiny CNN vs the teacher: -0.0013 [-0.0019, -0.0008] -> worse
- half-height vs full picture (both distilled): -0.0003 [-0.0010, +0.0005] -> tied
- tiny vs small (both distilled): -0.0012 [-0.0016, -0.0006] -> worse
- distilled small CNN vs HGB 43: -0.0002 [-0.0020, +0.0015] -> tied
- distilled half-height CNN vs HGB 43: -0.0005 [-0.0022, +0.0012] -> tied
- distilled small pure-image CNN vs the 6b final model: +0.0011 [-0.0009, +0.0028] -> tied
- speed (L4 GPU, bf16, batch 256): teacher 82.4 ms; small 18.5 ms (4.5x); half-height 10.5 ms (7.8x); tiny 9.7 ms (8.5x). Files: teacher 3.0 MB, small 0.31 MB, tiny 0.12 MB.

Run note: small_kd stalled twice after training (Colab disconnect); resumed with the scoring-only path in pure_distill.py 2026-11-27.2. Test 2024-25 not run.
