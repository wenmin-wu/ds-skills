---
name: tabular-column-shuffle-augmentation
description: >
  Augments imbalanced tabular data by independently shuffling each feature column within a class, creating synthetic samples that preserve per-column marginal distributions.
---
# Column Shuffle Augmentation

## Overview

For imbalanced binary classification, you need more minority samples but SMOTE-style interpolation doesn't work well with high-dimensional anonymous features. Instead, independently shuffle each column within the minority class — this creates new rows that preserve each feature's marginal distribution while breaking inter-feature correlations. Apply asymmetrically: more copies for the minority class.

## Quick Start

```python
import numpy as np

def augment(X, y, t_pos=2, t_neg=1):
    augmented_X, augmented_y = [X], [y]
    for _ in range(t_pos):
        x1 = X[y == 1].copy()
        for c in range(x1.shape[1]):
            np.random.shuffle(x1[:, c])
        augmented_X.append(x1)
        augmented_y.append(np.ones(len(x1)))
    for _ in range(t_neg):
        x0 = X[y == 0].copy()
        for c in range(x0.shape[1]):
            np.random.shuffle(x0[:, c])
        augmented_X.append(x0)
        augmented_y.append(np.zeros(len(x0)))
    return np.vstack(augmented_X), np.concatenate(augmented_y)

X_aug, y_aug = augment(X_train.values, y_train.values, t_pos=2, t_neg=1)
```

## Workflow

1. Separate samples by class label
2. For each augmentation round, copy the class subset
3. Shuffle each column independently (breaks correlations)
4. Append synthetic rows with correct labels
5. Train on the augmented dataset

## Key Decisions

- **Multiplier ratio**: More copies for minority (2-3x) than majority (0-1x)
- **Per-fold augmentation**: Augment inside CV loop, not before, to avoid leakage
- **Multi-seed averaging**: Train N models with different shuffle seeds, average predictions
- **Trade-off**: Breaks real feature correlations — works best when individual features are informative independently

## References

- [LGB 2 leaves + augment](https://www.kaggle.com/code/jiweiliu/lgb-2-leaves-augment)
