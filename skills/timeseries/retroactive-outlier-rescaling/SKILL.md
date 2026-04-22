---
name: timeseries-retroactive-outlier-rescaling
description: Walk backward through a time series and multiplicatively rescale segments when jumps exceed a fraction of the running mean to correct data collection anomalies
---

# Retroactive Outlier Rescaling

## Overview

Administrative time series (business counts, registrations, economic indicators) sometimes contain abrupt level shifts caused by data collection changes rather than real trends. Walking backward from the most recent value and rescaling earlier segments when the jump exceeds a threshold (e.g., 20% of running mean) corrects these artifacts while preserving genuine trends.

## Quick Start

```python
import numpy as np

def fix_outliers(series, threshold_frac=0.20):
    values = series.values.copy().astype(float)
    n = len(values)
    for i in range(n - 2, -1, -1):
        running_mean = np.mean(values[i + 1 : min(i + 4, n)])
        if running_mean == 0:
            continue
        jump = abs(values[i] - values[i + 1])
        if jump > threshold_frac * running_mean:
            scale = values[i + 1] / values[i] if values[i] != 0 else 1.0
            values[: i + 1] *= scale
    return values

df["value_clean"] = df.groupby("entity_id")["value"].transform(fix_outliers)
```

## Workflow

1. Group by entity, sort by time (most recent last)
2. Walk backward from the second-to-last observation
3. At each step, compute the running mean of the next 2-3 values
4. If `|value[i] - value[i+1]|` exceeds `threshold_frac * running_mean`, compute scale factor
5. Multiply all values before index i by `value[i+1] / value[i]`
6. Use cleaned series for both training and inference

## Key Decisions

- **Threshold**: 0.20 (20%) catches major level shifts without overcorrecting normal volatility
- **Direction**: backward walk anchors to the most recent (most reliable) data
- **Running window**: 2-3 points balances stability vs. sensitivity to recent changes
- **Multiplicative vs. additive**: multiplicative preserves proportional relationships across entities with different scales

## References

- [Better XGB Baseline](https://www.kaggle.com/code/titericz/better-xgb-baseline)
- [GoDaddy: Tune Stacking](https://www.kaggle.com/code/batprem/godaddy-tune-stacking)
