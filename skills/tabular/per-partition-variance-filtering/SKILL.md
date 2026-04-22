---
name: tabular-per-partition-variance-filtering
description: Apply VarianceThreshold within each data partition on combined train+test to select informative features per subgroup
---

# Per-Partition Variance Filtering

## Overview

When a dataset contains a categorical variable that defines distinct subpopulations, features that are informative in one partition may be constant (zero-variance) in another. Applying VarianceThreshold per partition instead of globally selects the right feature subset for each subgroup. Fitting on combined train+test (transductive) gives a more stable variance estimate, especially for small partitions.

## Quick Start

```python
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

cols = [c for c in train.columns if c not in ['id', 'target', 'group']]
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for g in train['group'].unique():
    tr = train[train['group'] == g]
    te = test[test['group'] == g]

    combined = pd.concat([tr[cols], te[cols]])
    sel = VarianceThreshold(threshold=1.5)
    combined_t = sel.fit_transform(combined)

    X_train = combined_t[:len(tr)]
    X_test = combined_t[len(tr):]
    # ... train model on X_train, predict X_test ...
```

## Workflow

1. Identify the partitioning column (categorical/group variable)
2. For each partition value: subset train and test
3. Concatenate train+test features, fit VarianceThreshold
4. Transform both train and test with the partition-specific selector
5. Train a model on the reduced feature set for this partition

## Key Decisions

- **Threshold value**: 1.5-2.0 typically reduces 255 features to ~40; tune per dataset
- **Combined train+test**: safe because VarianceThreshold is unsupervised (no target leakage)
- **Global vs per-partition**: per-partition selects 2-3x fewer features with better signal per subgroup
- **Pipeline**: combine with StandardScaler in a Pipeline for cleaner code

## References

- [Pseudo Labeling - QDA - [0.969]](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969)
- [Quadratic Discriminant Analysis](https://www.kaggle.com/code/speedwagon/quadratic-discriminant-analysis)
