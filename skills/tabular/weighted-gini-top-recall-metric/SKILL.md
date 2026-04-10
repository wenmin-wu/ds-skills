---
name: tabular-weighted-gini-top-recall-metric
description: Custom ranking metric combining normalized weighted Gini coefficient with top-K% capture rate for imbalanced classification with class-weighted evaluation
domain: tabular
---

# Weighted Gini + Top Recall Metric

## Overview

For imbalanced binary classification (e.g. credit default), combine two complementary ranking signals: (1) normalized weighted Gini coefficient measuring overall ranking quality, and (2) top-K% capture rate measuring recall among the highest-risk predictions. Apply class weights (e.g. 20x for negatives) to account for known subsampling. Final score = 0.5 * (Gini + capture_rate).

## Quick Start

```python
import numpy as np

def weighted_gini_top_recall(y_true, y_pred, neg_weight=20, top_pct=0.04):
    """Combined weighted Gini + top-K% capture rate.
    
    Args:
        y_true: binary labels (0/1)
        y_pred: predicted scores (higher = more positive)
        neg_weight: weight for negative class (compensates subsampling)
        top_pct: fraction of weighted population for capture rate
    """
    idx = np.argsort(y_pred)[::-1]
    y_true, y_pred = y_true[idx], y_pred[idx]
    weight = np.where(y_true == 0, neg_weight, 1)
    
    # Top-K% capture rate
    cum_weight = np.cumsum(weight)
    cutoff = int(top_pct * weight.sum())
    top_mask = cum_weight <= cutoff
    capture = y_true[top_mask].sum() / y_true.sum()
    
    # Normalized weighted Gini
    cum_norm_w = cum_weight / weight.sum()
    total_pos = (y_true * weight).sum()
    lorentz = np.cumsum(y_true * weight) / total_pos
    gini = ((lorentz - cum_norm_w) * weight).sum()
    
    # Perfect Gini (sort by true labels)
    idx_perfect = np.argsort(y_true)[::-1]
    w_p = np.where(y_true[idx_perfect] == 0, neg_weight, 1)
    cum_p = np.cumsum(w_p) / w_p.sum()
    lor_p = np.cumsum(y_true[idx_perfect] * w_p) / (y_true[idx_perfect] * w_p).sum()
    gini_max = ((lor_p - cum_p) * w_p).sum()
    
    return 0.5 * (gini / gini_max + capture)
```

## Key Decisions

- **neg_weight=20**: matches known subsampling ratio; adjust per your dataset
- **top_pct=0.04**: focuses on the riskiest 4% of the weighted population
- **0.5 weighting**: equal blend of Gini and capture; tune if one matters more
- **LightGBM feval**: wrap as `def feval(preds, data): ...` returning (name, value, higher_is_better)

## References

- Source: [xgboost-starter-0-793](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793)
- Competition: American Express - Default Prediction
