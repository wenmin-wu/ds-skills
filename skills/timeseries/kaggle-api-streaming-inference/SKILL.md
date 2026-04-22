---
name: timeseries-kaggle-api-streaming-inference
description: Predict day-by-day via Kaggle's iter_test API while maintaining a rolling history buffer for computing lag features online
---

# Kaggle API Streaming Inference

## Overview

Kaggle time-series competitions often use a streaming API where test data arrives one day at a time. The model must predict using only data available up to that point. This requires maintaining a rolling history buffer, computing lag features on the fly, and appending each day's predictions back to the buffer for use as future lag inputs.

## Quick Start

```python
import pandas as pd
from datetime import timedelta

history = train_df[["entity_id", "date"] + TARGETS].copy()

env = competition.make_env()
for test_df, sample_sub in env.iter_test():
    eval_date = pd.to_datetime(test_df["date"].iloc[0])

    # Compute lag features from history
    lag_features = []
    for lag in range(1, 21):
        lag_date = eval_date - timedelta(days=lag)
        lag_vals = history[history["date"] == lag_date][["entity_id"] + TARGETS]
        lag_vals = lag_vals.rename(columns={t: f"{t}_{lag}" for t in TARGETS})
        lag_features.append(lag_vals)

    features = test_df[["entity_id"]].copy()
    for lf in lag_features:
        features = features.merge(lf, on="entity_id", how="left")
    features = features.fillna(0)

    preds = model.predict(features[lag_cols])
    sample_sub[TARGETS] = preds

    # Update history with today's predictions
    new_row = test_df[["entity_id", "date"]].copy()
    new_row[TARGETS] = preds
    history = pd.concat([history, new_row], ignore_index=True)

    env.predict(sample_sub)
```

## Workflow

1. Initialize history buffer with all available training data
2. For each test day received from `iter_test()`:
   a. Extract the evaluation date
   b. Look up lag values from the history buffer
   c. Construct the feature vector matching the training schema
   d. Predict and submit
   e. Append predictions to history for future lag computation
3. Keep only the last N days in the buffer to limit memory usage

## Key Decisions

- **History pruning**: keep last 30-60 days only; older data is never queried as a lag
- **Missing lags**: fill with 0 or entity-level median from training data
- **Prediction feedback**: using model predictions as future lags compounds errors — consider blending with entity median
- **Deduplication**: if resubmitting, deduplicate history by (entity_id, date) keeping the latest entry
- **Performance**: precompute entity medians and yearly stats before the loop to avoid repeated groupby operations

## References

- [Getting Started with MLB Player Digital Engagement](https://www.kaggle.com/code/ryanholbrook/getting-started-with-mlb-player-digital-engagement)
- [[Fork of] LightGBM + CatBoost + ANN 2505f2](https://www.kaggle.com/code/somayyehgholami/fork-of-lightgbm-catboost-ann-2505f2)
