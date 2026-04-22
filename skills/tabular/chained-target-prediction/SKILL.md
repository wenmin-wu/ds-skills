---
name: tabular-chained-target-prediction
description: Predict correlated targets sequentially, using earlier target predictions as input features for subsequent targets to exploit inter-target dependencies
---

# Chained Target Prediction

## Overview

When multiple targets are correlated (e.g., different engagement metrics for the same entity), predicting them independently wastes inter-target signal. Chained prediction trains target1 first, then feeds its prediction as an additional feature when training target2, and so on. This captures causal or correlational structure between targets without requiring a multi-output model.

## Quick Start

```python
import lightgbm as lgb
import numpy as np

targets = ["target1", "target2", "target3", "target4"]
models = {}
feature_cols_base = [c for c in X.columns if c not in targets]

for i, target in enumerate(targets):
    feat_cols = feature_cols_base + targets[:i]  # add earlier targets as features
    model = lgb.LGBMRegressor(**params[target])
    model.fit(X_train[feat_cols], y_train[target])
    models[target] = model
    # Add predictions to both train and test for downstream targets
    X_train[target] = np.clip(model.predict(X_train[feat_cols]), 0, 100)
    X_test[target] = np.clip(model.predict(X_test[feat_cols]), 0, 100)
```

## Workflow

1. Order targets by independence (most independent first) or by prediction difficulty (easiest first)
2. Train model for target1 using base features only
3. Append target1 predictions to the feature set
4. Train model for target2 using base features + target1 predictions
5. Continue chaining for all targets
6. At inference, predict sequentially in the same order

## Key Decisions

- **Target order**: predict the most accurately modeled target first — its predictions are most reliable as features
- **Train vs. OOF**: use out-of-fold predictions during training to avoid target leakage; raw predictions are fine at test time
- **Clipping**: clip predictions to valid range before using as features to prevent error propagation
- **vs. multi-output**: chaining is simpler and allows per-target hyperparameter tuning; multi-output shares parameters

## References

- [LightGBM + CatBoost + ANN 2505f2](https://www.kaggle.com/code/lhagiimn/lightgbm-catboost-ann-2505f2)
