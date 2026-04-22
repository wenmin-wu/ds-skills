---
name: tabular-transductive-train-test-transform
description: Fit unsupervised transforms (scaler, PCA, variance filter) on combined train+test data for more stable statistics, especially on small datasets
---

# Transductive Train-Test Transform

## Overview

When the training set is small, fitting a scaler or dimensionality reduction on train alone produces noisy statistics. For unsupervised transforms (StandardScaler, PCA, VarianceThreshold, NMF) that don't use the target, fitting on combined train+test is safe and produces more stable estimates. This is especially valuable when data is partitioned into many small subgroups.

## Quick Start

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

cols = [c for c in train.columns if c not in ['id', 'target', 'group']]

for g in train['group'].unique():
    tr = train[train['group'] == g][cols]
    te = test[test['group'] == g][cols]

    combined = pd.concat([tr, te])
    pipe = Pipeline([
        ('vt', VarianceThreshold(threshold=1.5)),
        ('scaler', StandardScaler())
    ])
    combined_t = pipe.fit_transform(combined)
    X_train = combined_t[:len(tr)]
    X_test = combined_t[len(tr):]
    # ... train model ...
```

## Workflow

1. Concatenate train and test feature matrices (exclude target and ID columns)
2. Fit unsupervised transforms on the combined data
3. Transform both portions
4. Split back into train and test by original lengths
5. Train supervised model on the transformed train set only

## Key Decisions

- **Safe transforms**: StandardScaler, MinMaxScaler, PCA, VarianceThreshold, NMF — all unsupervised, no leakage
- **Unsafe transforms**: target encoding, frequency encoding on target — DO NOT fit on test
- **When beneficial**: small datasets (< 1000 rows), partitioned data with < 100 rows per group
- **Large datasets**: minimal benefit — train-only fitting is already stable

## References

- [Pseudo Labeling - QDA - [0.969]](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969)
- [Quadratic Discriminant Analysis](https://www.kaggle.com/code/speedwagon/quadratic-discriminant-analysis)
