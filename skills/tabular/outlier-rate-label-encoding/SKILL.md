---
name: tabular-outlier-rate-label-encoding
description: Encode a categorical column by replacing each category with the per-category outlier rate (mean of a binary outlier flag), out-of-fold to avoid leakage — a target-aware encoding tuned to long-tail / sentinel-target problems where a binary classifier signal is more useful than the raw regression mean
---

## Overview

Standard target encoding replaces a category with `mean(y | category)`. When the target has a sentinel-tail problem (Elo's -33.22, churned/non-churned), the regression mean is dominated by the sentinel and washes out subtle category effects. Replace it with `mean(is_outlier | category)` — a per-category outlier *rate*. This is a much more discriminating encoding for the classifier half of an outlier-aware blending pipeline, and remains useful as a feature in the regressor too. As with any target encoding, do it out-of-fold to avoid leakage, and add Bayesian smoothing for low-frequency categories.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

OUT_VAL = -33.21928
train['is_out'] = (train.target == OUT_VAL).astype(int)

def oof_rate_encode(df_tr, df_te, col, target='is_out', k=20):
    prior = df_tr[target].mean()
    enc = np.zeros(len(df_tr))
    kf = KFold(5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(df_tr):
        agg = df_tr.iloc[tr_idx].groupby(col)[target].agg(['mean', 'size'])
        smooth = (agg['mean'] * agg['size'] + prior * k) / (agg['size'] + k)
        enc[va_idx] = df_tr.iloc[va_idx][col].map(smooth).fillna(prior).values
    full_agg = df_tr.groupby(col)[target].agg(['mean', 'size'])
    smooth_full = (full_agg['mean'] * full_agg['size'] + prior * k) / (full_agg['size'] + k)
    te_enc = df_te[col].map(smooth_full).fillna(prior).values
    return enc, te_enc

train['cat3_outrate'], test['cat3_outrate'] = oof_rate_encode(train, test, 'category_3')
```

## Workflow

1. Define the binary outlier flag (`y == sentinel`, `y > threshold`, or any indicator)
2. Pick categorical columns with cardinality between 5 and a few thousand — too few and the encoding is uninformative, too many and the prior dominates
3. KFold over training; for each fold compute per-category outlier rate on the train portion and assign to the validation portion
4. Apply Bayesian smoothing: `(mean·n + prior·k) / (n + k)` with `k` between 10 and 100
5. For the test set use the full-train aggregation
6. Use these encodings as features for both the outlier classifier and (optionally) the bulk regressor

## Key Decisions

- **Outlier rate beats target mean for sentinel problems**: the classifier signal is denser and less skewed.
- **Out-of-fold is non-negotiable**: in-fold encoding leaks the target and inflates validation scores by 0.005-0.02.
- **Smoothing constant `k` is target-dependent**: start at 20, raise it if the encoding overfits low-frequency categories.
- **Fall back to the prior for unseen categories**: never NaN — the model splits on it weirdly.
- **Combine with raw frequency encoding**: `outrate` and `freq` capture orthogonal axes (signal vs. support).
- **Generalizes**: any binary indicator (refund, fraud, churn, conversion) can drive this — not only sentinel detection.

## References

- [My first kernel — Elo Merchant](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)
