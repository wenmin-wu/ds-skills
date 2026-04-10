---
name: tabular-per-type-model-training
description: >
  Trains separate models for each discrete category (e.g., molecule type, product class) to capture type-specific patterns.
---
# Per-Type Model Training

## Overview

When a dataset contains distinct subgroups with different distributions (e.g., molecule coupling types, product categories), train a separate model per type. Each model sees only its type's data, learning type-specific feature importance and hyperparameters. This often outperforms a single global model with type as a feature.

## Quick Start

```python
import lightgbm as lgb
from sklearn.model_selection import KFold

def train_per_type(df_train, df_test, target_col, type_col, features, params):
    """Train separate models per category type."""
    predictions = df_test[[type_col]].copy()
    predictions['pred'] = 0.0
    scores = {}

    for t in df_train[type_col].unique():
        mask_train = df_train[type_col] == t
        mask_test = df_test[type_col] == t

        X_t = df_train.loc[mask_train, features]
        y_t = df_train.loc[mask_train, target_col]
        X_test_t = df_test.loc[mask_test, features]

        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(X_t))
        pred = np.zeros(len(X_test_t))

        for train_idx, val_idx in folds.split(X_t):
            model = lgb.LGBMRegressor(**params)
            model.fit(X_t.iloc[train_idx], y_t.iloc[train_idx],
                      eval_set=[(X_t.iloc[val_idx], y_t.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200)])
            oof[val_idx] = model.predict(X_t.iloc[val_idx])
            pred += model.predict(X_test_t) / folds.n_splits

        predictions.loc[mask_test, 'pred'] = pred
        scores[t] = np.mean(np.abs(y_t - oof))

    return predictions, scores
```

## Workflow

1. Identify categorical column that defines distinct subgroups
2. Split train/test by each unique type value
3. Train independent model per type with its own CV
4. Optionally tune hyperparameters per type (some types need more trees)
5. Concatenate per-type predictions for final submission

## Key Decisions

- **When to use**: When types have fundamentally different target distributions or feature importance
- **Min samples**: Each type needs enough data for CV — merge rare types if < 500 samples
- **Shared features**: Use same feature set or customize per type based on importance
- **Evaluation**: Compute metric per type, then average (group-mean) for fair comparison

## References

- Predicting Molecular Properties / CHAMPS Scalar Coupling (Kaggle)
- Source: [brute-force-feature-engineering](https://www.kaggle.com/code/artgor/brute-force-feature-engineering)
