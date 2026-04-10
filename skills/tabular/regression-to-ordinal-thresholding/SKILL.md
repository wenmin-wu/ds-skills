---
name: tabular-regression-to-ordinal-thresholding
description: >
  Converts regression predictions to ordinal classes by optimizing bin thresholds to maximize Quadratic Weighted Kappa.
---
# Regression-to-Ordinal Thresholding

## Overview

When the target is ordinal (e.g., 0-3 rating) but you train as regression, you need optimal thresholds to bin continuous predictions back into classes. Naive rounding loses performance. Instead, optimize thresholds directly against the evaluation metric (e.g., QWK) using Nelder-Mead or percentile-based binning.

## Quick Start

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def optimize_thresholds(y_true, y_pred, n_classes=4):
    """Find optimal thresholds to maximize QWK."""
    def loss(coef):
        bins = [-np.inf] + sorted(coef.tolist()) + [np.inf]
        y_binned = np.digitize(y_pred, bins[1:-1])
        return -qwk(y_true, y_binned)

    initial = np.arange(1, n_classes) - 0.5  # e.g., [0.5, 1.5, 2.5]
    result = minimize(loss, initial, method='nelder-mead')
    return sorted(result.x.tolist())

# Usage
thresholds = optimize_thresholds(y_val, val_preds)
bins = [-np.inf] + thresholds + [np.inf]
final_preds = np.digitize(test_preds, bins[1:-1])
```

## Workflow

1. Train model as regression (continuous target)
2. Generate OOF predictions on validation set
3. Optimize thresholds on OOF using Nelder-Mead to maximize QWK
4. Apply learned thresholds to test predictions

## Key Decisions

- **vs percentile binning**: Percentile bins match training distribution; Nelder-Mead optimizes the metric directly — Nelder-Mead usually wins
- **Initial values**: Start at midpoints (0.5, 1.5, 2.5) for classes 0-3
- **Per-fold thresholds**: Optimize on each fold's OOF, then average thresholds across folds
- **When to use**: Any ordinal classification with a weighted kappa metric

## References

- 2019 Data Science Bowl (Kaggle)
- Source: [quick-and-dirty-regression](https://www.kaggle.com/code/artgor/quick-and-dirty-regression)
