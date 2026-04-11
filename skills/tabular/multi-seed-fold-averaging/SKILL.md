---
name: tabular-multi-seed-fold-averaging
description: >
  Trains multiple models per CV fold with different random seeds for augmentation, then averages their predictions to reduce variance from stochastic data generation.
---
# Multi-Seed Fold Averaging

## Overview

When training involves stochastic data augmentation (column shuffling, random oversampling), a single run per fold is noisy — different seeds yield different OOF scores. Train N models per fold, each with a different augmentation seed, and average their predictions before computing the fold's validation score. This smooths out augmentation variance without increasing the number of folds.

## Quick Start

```python
import numpy as np
import lightgbm as lgb

N_SEEDS = 5
oof = np.zeros(len(X_train))

for fold, (trn_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    p_valid = np.zeros(len(val_idx))

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        X_aug, y_aug = augment(X.iloc[trn_idx].values, y.iloc[trn_idx].values)
        trn_data = lgb.Dataset(X_aug, label=y_aug)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params, trn_data, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(500)])
        p_valid += model.predict(X_val) / N_SEEDS

    oof[val_idx] = p_valid
```

## Workflow

1. Set up K-fold cross-validation as usual
2. For each fold, loop N times with different random seeds
3. Apply stochastic augmentation with the current seed
4. Train and predict on the validation set
5. Average the N predictions for the fold's OOF score
6. Average test predictions across all N * K models

## Key Decisions

- **N seeds**: 3-5 is typical; diminishing returns beyond 10
- **When to use**: Only when augmentation is stochastic; deterministic pipelines don't benefit
- **Cost**: N times slower per fold — offset by using fewer folds if needed
- **Alternative**: Fix a single seed and accept the variance; or use a larger validation set

## References

- [LGB 2 leaves + augment](https://www.kaggle.com/code/jiweiliu/lgb-2-leaves-augment)
