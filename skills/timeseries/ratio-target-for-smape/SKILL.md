---
name: timeseries-ratio-target-for-smape
description: Transform forecasting target to next/current ratio minus one so that optimizing MAE or squared error implicitly minimizes SMAPE
---

# Ratio Target for SMAPE

## Overview

When the evaluation metric is SMAPE, predicting absolute values penalizes errors asymmetrically. Transforming the target to `y_{t+1}/y_t - 1` (the relative change) aligns standard regression losses with SMAPE's symmetric percentage nature. The model learns relative changes, then predictions are converted back via `pred * last_known_value + last_known_value`.

## Quick Start

```python
import pandas as pd

df = df.sort_values(["entity_id", "date"])
df["target_ratio"] = df.groupby("entity_id")["value"].transform(
    lambda s: s.shift(-1) / s - 1
)

# Train model on target_ratio with standard MAE/MSE loss
# At inference:
last_known = df.groupby("entity_id")["value"].last()
df["prediction"] = last_known * (1 + predicted_ratio)
```

## Workflow

1. Sort by entity and time, compute `value_{t+1} / value_t - 1` as the training target
2. Clip extreme ratios (e.g., to [-0.005, 0.005]) to limit outlier influence
3. Train any regression model (XGBoost, LightGBM, linear) on the ratio target
4. At inference, multiply predicted ratio by the last known value to recover the forecast
5. For multi-step horizons, chain predictions or use the last known value for all steps

## Key Decisions

- **Clip bounds**: [-0.005, 0.005] works for slowly-changing density metrics; widen for volatile series
- **Division by zero**: replace zero denominators with a small epsilon or carry forward the last nonzero value
- **Multi-step**: chaining introduces compounding error — for short horizons (1-3 steps), direct last-value scaling is more stable
- **vs log-differencing**: ratio target is simpler and avoids log(0) issues; log-diff is better when changes span orders of magnitude

## References

- [Better XGB Baseline](https://www.kaggle.com/code/titericz/better-xgb-baseline)
- [GoDaddy: Tune Stacking](https://www.kaggle.com/code/batprem/godaddy-tune-stacking)
