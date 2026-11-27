Step 6d run pure_20261127_023749, cell vol_h1, 19,762 validation rows on 241 days of 2023. Drops are AUC(unchanged) - AUC(occluded), 95% day-bootstrap.

**pure_all** (AUC on these rows 0.6844)
- region (mean fill): whole chart (price + volume): drop +0.0518 [+0.0416, +0.0639] matters
- region (mean fill): chart: price bars: drop +0.0113 [+0.0081, +0.0148] matters
- region (mean fill): number strip (43 numbers): drop +0.0082 [+0.0054, +0.0109] matters
- region (mean fill): stock EWSD band: drop +0.0034 [+0.0019, +0.0051] matters
- region (mean fill): multi-day RV bars: drop +0.0021 [+0.0006, +0.0038] matters
- region (mean fill): volume band: drop +0.0011 [+0.0003, +0.0019] matters
- region (mean fill): chart: volume bars: drop -0.0000 [-0.0006, +0.0005] no effect
- region (mean fill): SPY EWSD band: drop -0.0001 [-0.0010, +0.0009] no effect
- erase: session-open line: drop +0.0171 [+0.0129, +0.0230] matters
- erase: high-low bars (middle column): drop +0.0019 [-0.0001, +0.0037] no effect
- erase: close ticks (right column): drop +0.0001 [-0.0002, +0.0004] no effect
- erase: open ticks (left column): drop -0.0001 [-0.0004, +0.0002] no effect
- time blocks with the largest drop: bars t-77..t-72 +0.0182; bars t-5..t-0 +0.0048; bars t-35..t-30 +0.0030
- gradient x input shares: rv 27% (rows 6%), chart_price 22% (rows 40%), ewsd 16% (rows 17%), volume 11% (rows 8%), spy 10% (rows 8%), numbers 9% (rows 9%), chart_volume 4% (rows 10%)
- channels zeroed: top 1 -0.0001 vs random +0.0000; top 5 -0.0001 vs random +0.0001; top 20 +0.0000 vs random -0.0001; top 64 +0.0021 vs random +0.0004

**pure_key** (AUC on these rows 0.6810)
- region (mean fill): whole chart (price + volume): drop +0.0866 [+0.0739, +0.1012] matters
- region (mean fill): chart: price bars: drop +0.0308 [+0.0257, +0.0362] matters
- region (mean fill): stock EWSD band: drop +0.0126 [+0.0094, +0.0164] matters
- region (mean fill): multi-day RV bars: drop +0.0026 [+0.0005, +0.0044] matters
- region (mean fill): volume band: drop +0.0025 [+0.0012, +0.0039] matters
- region (mean fill): SPY EWSD band: drop +0.0007 [-0.0009, +0.0022] no effect
- region (mean fill): chart: volume bars: drop +0.0004 [-0.0002, +0.0010] no effect
- erase: session-open line: drop +0.0236 [+0.0188, +0.0302] matters
- erase: high-low bars (middle column): drop +0.0092 [+0.0058, +0.0127] matters
- erase: close ticks (right column): drop -0.0001 [-0.0006, +0.0003] no effect
- erase: open ticks (left column): drop -0.0003 [-0.0008, +0.0001] no effect
- time blocks with the largest drop: bars t-77..t-72 +0.0211; bars t-5..t-0 +0.0175; bars t-35..t-30 +0.0029
- gradient x input shares: chart_price 27% (rows 44%), ewsd 22% (rows 18%), rv 22% (rows 7%), spy 13% (rows 9%), volume 11% (rows 9%), chart_volume 4% (rows 11%)
- channels zeroed: top 1 -0.0000 vs random -0.0000; top 5 +0.0003 vs random +0.0000; top 20 +0.0006 vs random +0.0000; top 64 +0.0007 vs random +0.0008

Numbers the CNN reads most (pure_all, set to 0): rv_ratio +0.0006, r1 +0.0003, spy_r1 +0.0002, cum12 +0.0002, rel_cum77 +0.0002, r8 +0.0002, mean_abs_r12 +0.0002, spy_rel_volume +0.0002
Best linear read-outs from pure_key (numbers not drawn): spy_rv20d 0.98, spy_rv5d 0.97, rv5d 0.96, rv20d 0.95, rv_ratio 0.93, rel_volume 0.90, spy_realized_vol 0.83, realized_vol 0.82
