---
name: tabular-group-mean-log-mae-metric
description: >
  Custom evaluation metric that computes log of per-group MAE then averages, penalizing uniformly bad groups.
---
# Group Mean Log MAE Metric

## Overview

When predictions span multiple types or groups with different scales, a global MAE favors high-volume groups. Group Mean Log MAE computes MAE per group, takes the log of each, then averages. The log transform penalizes groups where MAE is large relative to its own scale, ensuring the model performs well across all groups — not just the largest.

## Quick Start

```python
import numpy as np

def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):
    """Compute mean of log(MAE) across groups.

    Args:
        y_true: array of true values
        y_pred: array of predictions
        groups: array of group labels (same length)
        floor: minimum MAE to avoid log(0)
    """
    errors = np.abs(y_true - y_pred)
    unique_groups = np.unique(groups)
    log_maes = []
    for g in unique_groups:
        mask = groups == g
        mae = errors[mask].mean()
        log_maes.append(np.log(max(mae, floor)))
    return np.mean(log_maes)

# LightGBM custom metric
def lgb_group_metric(y_pred, data, groups):
    y_true = data.get_label()
    score = group_mean_log_mae(y_true, y_pred, groups)
    return 'group_log_mae', score, False  # lower is better
```

## Workflow

1. Define group column (e.g., coupling type, product category)
2. Implement metric as standalone function
3. Wrap as custom eval for LightGBM/XGBoost callbacks
4. Monitor per-group MAE during training to catch lagging groups
5. Consider training per-type models if one group dominates the metric

## Key Decisions

- **Floor value**: 1e-9 prevents log(0); adjust if your target scale is very small
- **Log vs raw**: Log penalizes proportionally — a group with MAE 0.01→0.02 matters as much as 1.0→2.0
- **Weighted variant**: Weight groups by sample count if group sizes are extremely imbalanced
- **Early stopping**: Use this metric for early stopping to optimize directly for the competition metric

## References

- Predicting Molecular Properties / CHAMPS Scalar Coupling (Kaggle)
- Source: [molecular-properties-eda-and-models](https://www.kaggle.com/code/artgor/molecular-properties-eda-and-models)
