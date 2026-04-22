---
name: timeseries-multi-lag-target-features
description: Generate lag features for multiple targets over N days by shifting evaluation dates and self-joining per entity, creating a wide feature matrix of past target values
---

# Multi-Lag Target Features

## Overview

For entity-level time-series forecasting with multiple targets, lagging each target by 1..N days and pivoting into separate columns creates a dense feature matrix capturing recent history. The self-join approach shifts the date column by the lag amount and merges back on (entity_id, date), producing columns like `target1_1`, `target1_2`, ..., `target4_20` — 80 features for 4 targets x 20 lags.

## Quick Start

```python
from datetime import timedelta

TARGETS = ["target1", "target2", "target3", "target4"]
LAGS = list(range(1, 21))

def add_lag(df, lag):
    lagged = df[["entity_id", "date"] + TARGETS].copy()
    lagged["date"] = lagged["date"] + timedelta(days=lag)
    df = df.merge(lagged, on=["entity_id", "date"], suffixes=("", f"_{lag}"), how="left")
    return df

for lag in LAGS:
    df = add_lag(df, lag)

lag_cols = [f"{t}_{l}" for l in reversed(LAGS) for t in TARGETS]
```

## Workflow

1. Start with a long-format DataFrame: one row per (entity_id, date)
2. For each lag value, copy the target columns, shift dates forward by lag days
3. Merge back on (entity_id, date) with a lag suffix
4. Missing lags (start of series or gaps) become NaN — fill with 0 or entity median
5. Use the lag columns as features for any model (GBDT, ANN, linear)

## Key Decisions

- **Lag range**: 1-20 for daily data captures ~3 weeks of history; extend to 30+ for weekly patterns
- **Offset**: add an offset (e.g., 45 days) if targets aren't available until a delay after the evaluation date
- **Fill strategy**: fillna(0) is simple; entity-level median preserves scale differences
- **Memory**: for large datasets, compute lags in chunks or use `shift()` within groups instead of merge
- **At inference**: maintain a rolling history buffer and pivot the last N days into lag columns

## References

- [[Fork of] LightGBM + CatBoost + ANN 2505f2](https://www.kaggle.com/code/somayyehgholami/fork-of-lightgbm-catboost-ann-2505f2)
- [MLB: ANN with Lags TF Keras](https://www.kaggle.com/code/ulrich07/mlb-ann-with-lags-tf-keras)
