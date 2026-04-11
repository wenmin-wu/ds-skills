---
name: tabular-collinear-feature-removal
description: >
  Removes redundant features by iterating pairwise Pearson correlations and dropping one member of each pair exceeding a threshold.
---
# Collinear Feature Removal

## Overview

Highly correlated features add redundancy without improving model performance and can destabilize linear models and increase overfitting in tree models on wide datasets. This technique computes the full pairwise correlation matrix, identifies pairs above a threshold (typically 0.8–0.95), and drops one member of each pair. Unlike VIF or PCA, it's interpretable — you know exactly which features were removed and why.

## Quick Start

```python
import pandas as pd
import numpy as np

def remove_collinear(df, threshold=0.8):
    """Remove one feature from each pair with |correlation| > threshold."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop

train_reduced, dropped = remove_collinear(train[numeric_cols], threshold=0.8)
test_reduced = test.drop(columns=dropped)
print(f"Removed {len(dropped)} collinear features")
```

## Workflow

1. Compute absolute Pearson correlation matrix on numeric features
2. Extract upper triangle (avoid counting each pair twice)
3. For each column, check if any correlation exceeds the threshold
4. Drop the later-appearing column in each correlated pair
5. Apply same column drops to test set

## Key Decisions

- **Threshold**: 0.8 is aggressive; 0.95 is conservative. Sweep with CV to find optimal
- **Which to drop**: Default keeps the first column — alternatively keep the one with higher target correlation
- **Scope**: Apply only to numeric features; categorical correlations need Cramér's V
- **Ordering**: Run after feature engineering, before model training

## References

- [Introduction to Manual Feature Engineering](https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering)
