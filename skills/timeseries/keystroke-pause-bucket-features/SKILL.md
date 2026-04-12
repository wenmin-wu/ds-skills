---
name: timeseries-keystroke-pause-bucket-features
description: Bucket inter-keystroke latencies into pause-duration ranges (0.5-1s, 1-1.5s, 1.5-2s, 2-3s, >3s) and count per session as hesitation features
---

## Overview

In keystroke / writing-process data, raw event-latency distributions are noisy but their *binned* shapes are highly predictive of quality. Pauses under 0.5s are within-word; 0.5-2s are within-sentence hesitations; >3s are plan/read pauses. Counting events in each bucket per session gives five features that cleanly separate "fast drafter" from "slow reviser" from "struggling" writers, and they are more stable than min/mean/max because outliers fall into the tail bucket instead of distorting the moments.

## Quick Start

```python
# pandas version
logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
logs['time_diff']      = (logs['down_time'] - logs['up_time_lagged']).abs() / 1000  # seconds

g = logs.groupby('id')['time_diff']
feats = pd.DataFrame({
    'initial_pause':   logs.groupby('id')['down_time'].first() / 1000,
    'pauses_half_sec': g.apply(lambda x: ((x > 0.5) & (x < 1  )).sum()),
    'pauses_1_sec':    g.apply(lambda x: ((x > 1  ) & (x < 1.5)).sum()),
    'pauses_1_5_sec':  g.apply(lambda x: ((x > 1.5) & (x < 2  )).sum()),
    'pauses_2_sec':    g.apply(lambda x: ((x > 2  ) & (x < 3  )).sum()),
    'pauses_3_sec':    g.apply(lambda x: (x > 3).sum()),
})
```

## Workflow

1. Compute `time_diff = down_time - lag(up_time)` per session — this is the "idle gap" between keystrokes
2. Drop negative / garbage values (focus switches can create these); take abs() and divide to seconds
3. Count events in each non-overlapping bucket per session id
4. Add `initial_pause` (time before first key) as a separate feature — it captures planning time
5. Feed the 6-ish features into a GBDT alongside aggregated event counts

## Key Decisions

- **Buckets, not quantiles**: quantile-based bins shift across sessions, breaking feature comparability. Hard thresholds are interpretable and stable.
- **Open-ended tail**: `>3s` must be unbounded — capping it loses the "stuck writer" signal.
- **vs. mean/std of latency**: bucket counts outperform summary stats because the distribution is strongly multi-modal.
- **Scale to seconds**: keystroke timestamps are usually milliseconds — divide early to keep thresholds intuitive.

## References

- [LGBM (X2) + NN](https://www.kaggle.com/code/cody11null/lgbm-x2-nn)
- [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/mcpenguin/silver-bullet-single-model-165-features)
