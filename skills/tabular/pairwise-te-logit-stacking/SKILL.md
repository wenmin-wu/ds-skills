---
name: tabular-pairwise-te-logit-stacking
description: >
  Generates all C(n,2) pairwise feature combinations, target-encodes each pair with cuML TargetEncoder, then applies logit polynomial expansion (z, z^2, z^3) for stacking with cuML LogisticRegression.
---

# Pairwise TE Logit Stacking

## Overview

For n categorical/discretized features, generate all n*(n-1)/2 pairwise combinations. Target-encode each pair to get a probability estimate, then transform through `logit(p), logit(p)^2, logit(p)^3` to create polynomial features in logit space. These features feed into a cuML LogisticRegression, producing predictions highly diverse from tree-based models -- ideal for ensembling.

## Quick Start

```python
from cuml.preprocessing import TargetEncoder
from cuml.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np
import cudf

features = list(combinations(feature_cols, 2))  # C(n, 2) pairs
te_train = cudf.DataFrame()
te_test = cudf.DataFrame()

for f1, f2 in features:
    col_name = f'{f1}__{f2}'
    train[col_name] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[col_name] = test[f1].astype(str) + '_' + test[f2].astype(str)

    te = TargetEncoder(n_folds=5, smooth=10)
    te_train[col_name] = te.fit_transform(train[col_name], train[target])
    te_test[col_name] = te.transform(test[col_name])

# Logit polynomial expansion
def logit_expand(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p))
    return np.column_stack([z, z**2, z**3])

X_logit_train = logit_expand(te_train.to_numpy())
X_logit_test = logit_expand(te_test.to_numpy())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_logit_train)
X_test_scaled = scaler.transform(X_logit_test)

model = LogisticRegression(C=0.5, max_iter=1000)
model.fit(X_scaled, y_train)
```

## Workflow

1. Select feature columns (categorical or binned numerical)
2. Generate all `C(n, 2)` pairwise string combinations
3. Target-encode each pair using `cuml.preprocessing.TargetEncoder(n_folds=5)`
4. Clip TE probabilities to `[eps, 1-eps]`, compute `z = logit(p)`
5. Expand to `[z, z^2, z^3]` for each pair -- total features = 3 * C(n, 2)
6. StandardScale, then fit `cuml.linear_model.LogisticRegression(C=0.5)`
7. Use OOF predictions as a stacking feature alongside tree-model predictions

## Key Decisions

- **Why pairwise?** Captures 2-way interactions that single-feature TE misses
- **Why logit transform?** Converts bounded probabilities to unbounded space where linear models work naturally
- **Why polynomial (z^2, z^3)?** Allows the linear model to fit non-linear decision boundaries without tree splits
- **C=0.5:** light regularization; with many logit features, prevents overfitting
- **cuML:** GPU-accelerated TE and LR handle the O(n^2) feature explosion efficiently
- **Diversity:** logistic regression on logit-polynomials produces predictions uncorrelated with tree models, improving ensemble gain

## References

- Source: "ChatGPT Vibe Coding 3xGPU Models" (Kaggle Playground Series S6E3)
