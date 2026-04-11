---
name: tabular-per-target-nan-mask-training
description: >
  Trains independent models per target by masking NaN labels, enabling multi-output regression on datasets where each target has different coverage.
---
# Per-Target NaN-Mask Training

## Overview

Multi-output datasets often have sparse labels — not every sample has values for every target. Naively dropping rows with any NaN discards most of the data. Per-target NaN-mask training fits a separate model for each target using only the rows where that target is non-null. Each model sees the maximum available training data for its specific target. At inference, all models predict on the full test set. This is simpler and often more effective than joint multi-output models that must handle missing labels internally.

## Quick Start

```python
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

def train_per_target(X, y_multi, target_names, params, n_folds=5):
    """Train independent KFold models per sparse target."""
    models = {}
    for i, name in enumerate(target_names):
        y = y_multi[:, i]
        mask = ~np.isnan(y)
        X_valid, y_valid = X[mask], y[mask]

        fold_models = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_valid):
            dtrain = lgb.Dataset(X_valid[train_idx], label=y_valid[train_idx])
            dval = lgb.Dataset(X_valid[val_idx], label=y_valid[val_idx])
            model = lgb.train(params, dtrain, valid_sets=[dval],
                              num_boost_round=2000,
                              callbacks=[lgb.early_stopping(50)])
            fold_models.append(model)
        models[name] = fold_models
    return models

def predict_per_target(models, X_test, target_names):
    """Average fold predictions per target."""
    preds = np.zeros((len(X_test), len(target_names)))
    for i, name in enumerate(target_names):
        fold_preds = [m.predict(X_test) for m in models[name]]
        preds[:, i] = np.mean(fold_preds, axis=0)
    return preds
```

## Workflow

1. For each target column, create a boolean mask of non-null rows
2. Train a KFold CV model using only masked (valid) rows
3. At inference, predict on the full test set with each per-target model
4. Average fold predictions per target

## Key Decisions

- **vs joint model**: Per-target is simpler and avoids NaN-handling complexity; joint models can exploit target correlations
- **Shared features**: All targets use the same feature matrix — only the label mask differs
- **Per-target scaling**: Optionally fit a separate StandardScaler per target's valid subset
- **Evaluation**: Compute metrics per target on its own valid subset, then average

## References

- [NeurIPS 2025 Open Polymer Challenge Tutorial](https://www.kaggle.com/code/alexliu99/neurips-2025-open-polymer-challenge-tutorial)
