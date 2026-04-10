---
name: tabular-inner-kfold-target-encoding
description: >
  Computes leak-free target encoding statistics (mean, std, min, max) using nested inner KFold within each outer CV fold, preventing target leakage that occurs with naive groupby-based encoding.
---

# Inner KFold Target Encoding

## Overview

Naive target encoding leaks information because the target statistics for a row's category include that row's own target value. This skill uses a nested (inner) KFold loop: within each outer training fold, an inner 5-fold CV computes category-level aggregates only on held-in data, then applies them to the held-out inner fold. The result is a leak-free set of TE features (mean via sklearn, std/min/max via groupby).

## Quick Start

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import pandas as pd

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in outer_cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = y.iloc[train_idx]

    # Mean TE via sklearn (handles inner CV internally)
    te = TargetEncoder(cv=5, smooth="auto")
    X_tr_te = te.fit_transform(X_tr[cat_cols], y_tr)
    X_val_te = te.transform(X_val[cat_cols])

    # Std/Min/Max TE via inner KFold
    for col in cat_cols:
        oof_stats = pd.DataFrame(index=X_tr.index)
        for in_tr, in_val in inner_cv.split(X_tr, y_tr):
            stats = X_tr.iloc[in_tr].groupby(col)[target].agg(['std','min','max'])
            mapped = X_tr.iloc[in_val][[col]].join(stats, on=col)
            oof_stats.loc[mapped.index] = mapped[['std','min','max']].values
        # Val/test: use full outer-fold training stats
        full_stats = X_tr.groupby(col)[target].agg(['std','min','max'])
        X_val[[f'{col}_std', f'{col}_min', f'{col}_max']] = \
            X_val[[col]].join(full_stats, on=col)[['std','min','max']].values
```

## Workflow

1. Define outer `StratifiedKFold(n_splits=5)` for model evaluation
2. For each outer fold, split into train/val
3. Run inner `StratifiedKFold(n_splits=5)` on the outer training set
4. Inner folds: `groupby(cat_col)[target].agg(['std','min','max'])` on held-in, apply to held-out
5. For val/test: compute stats on the full outer training set and map
6. Use `sklearn.TargetEncoder(cv=5)` for mean TE (it does its own internal CV)
7. Concatenate all TE features with original features for model training

## Key Decisions

- **Why nested CV?** Single-fold groupby leaks target info into training rows, inflating CV scores
- **Why std/min/max separately from mean?** sklearn's TargetEncoder only provides mean; additional statistics add signal for tree models
- **Fill NaN stats** with global target mean/std for unseen categories
- **Inner folds = 5** balances computation cost with stable estimates

## References

- Source: "S6E3 Detail EDA + Baseline XGB" (Kaggle Playground Series S6E3)
- sklearn TargetEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
