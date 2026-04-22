---
name: timeseries-expanding-window-stacking
description: Walk-forward stacking ensemble that trains base models on expanding windows and a meta-learner on their out-of-fold predictions across time
---

# Expanding-Window Stacking

## Overview

Standard K-fold stacking leaks future information in time series. Expanding-window stacking respects temporal order: base models are trained on data up to time t, predict t+1, and the meta-learner trains on these walk-forward out-of-fold predictions. This produces a properly validated stacking ensemble for forecasting tasks.

## Quick Start

```python
import numpy as np
from sklearn.linear_model import Ridge

def expanding_stack(df, base_models, time_col="month", target="y"):
    periods = sorted(df[time_col].unique())
    oof_preds = np.zeros((len(df), len(base_models)))

    for i, cutoff in enumerate(periods[12:], start=12):  # min 12 months training
        train_mask = df[time_col] < cutoff
        val_mask = df[time_col] == cutoff

        for j, model in enumerate(base_models):
            model.fit(df.loc[train_mask, features], df.loc[train_mask, target])
            oof_preds[val_mask, j] = model.predict(df.loc[val_mask, features])

    # Train meta-learner on OOF predictions
    valid_mask = df[time_col] >= periods[12]
    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds[valid_mask], df.loc[valid_mask, target])
    return meta, oof_preds
```

## Workflow

1. Sort data by time, define minimum training window (e.g., 12 months)
2. For each subsequent time period, train all base models on data up to that period
3. Generate out-of-fold predictions for the held-out period
4. Collect all OOF predictions into a matrix
5. Train a meta-learner (Ridge, linear regression) on the OOF prediction matrix
6. For final predictions, train base models on all data, stack with the fitted meta-learner

## Key Decisions

- **Minimum window**: 12 months gives base models enough data; shorter windows increase variance
- **Meta-learner**: Ridge regression with small alpha prevents overfitting to correlated base predictions
- **Base diversity**: combine XGBoost, LightGBM, CatBoost, and linear models for maximum gain
- **Retraining frequency**: retrain base models every period for freshness vs. every N periods for speed
- **vs purged K-fold**: expanding window is stricter (no future leakage at all) but produces fewer OOF samples

## References

- [GoDaddy: Tune Stacking](https://www.kaggle.com/code/batprem/godaddy-tune-stacking)
