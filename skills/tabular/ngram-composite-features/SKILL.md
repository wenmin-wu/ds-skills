---
name: tabular-ngram-composite-features
description: >
  Creates bi-gram and tri-gram composite categorical features by concatenating top categorical columns, then target-encodes the composites. Captures interaction effects that tree models may miss.
---

# N-gram Composite Features

## Overview

Tree-based models can learn feature interactions, but explicit composite categoricals often improve performance by giving the model pre-built interaction signals. This technique selects the top categorical columns (by feature importance), creates all 2-way and 3-way string concatenations, then target-encodes the resulting high-cardinality composites.

## Quick Start

```python
from itertools import combinations
import pandas as pd

# 1. Define top categoricals (from feature importance)
TOP_CATS = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

# 2. Create bi-gram composites
for c1, c2 in combinations(TOP_CATS, 2):
    df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

# 3. Create tri-gram composites (optional, use fewer columns)
for c1, c2, c3 in combinations(TOP_CATS[:4], 3):
    df[f"TG_{c1}_{c2}_{c3}"] = (
        df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
    )

# 4. Target-encode composites (use KFold to avoid leakage)
from sklearn.model_selection import KFold

def target_encode_kfold(df, col, target, n_splits=5):
    encoded = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(df):
        means = df.iloc[tr_idx].groupby(col)[target].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means)
    return encoded.fillna(df[target].mean())

composite_cols = [c for c in df.columns if c.startswith(("BG_", "TG_"))]
for col in composite_cols:
    df[f"TE_{col}"] = target_encode_kfold(df, col, 'target')
```

## Workflow

1. **Identify top categoricals** from a baseline model's feature importance.
2. **Generate bi-grams** (all 2-combinations of top N columns). Use `N <= 8` to keep count manageable.
3. **Optionally generate tri-grams** from top 4-5 columns only (combinatorics explode quickly).
4. **Target-encode** each composite using KFold to prevent leakage.
5. **Train model** with original features + target-encoded composites.
6. **Prune** composites with near-zero importance after first training round.

## Key Decisions

| Decision | Guidance |
|---|---|
| How many base columns | 5-8 for bi-grams, 3-5 for tri-grams. C(8,2)=28 bi-grams is fine; C(8,3)=56 tri-grams gets heavy. |
| Encoding method | KFold target encoding is standard. Alternatives: CatBoost native encoding, leave-one-out. |
| Leakage prevention | Never encode using the full dataset. Always use KFold or fit on train / transform on test. |
| Rare categories | Composites create many rare groups. Apply smoothing or min-count thresholds during encoding. |
| When to use | Most effective when categoricals have strong interactions and the dataset has >10k rows. |

## References

- Kaggle: "S6E3 Ridge XGB N-gram 0.91927 CV" (playground-series-s6e3)
- Micci-Barreca, D. "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems" (SIGKDD Explorations, 2001)
