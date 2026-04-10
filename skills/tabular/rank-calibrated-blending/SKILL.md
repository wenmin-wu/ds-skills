---
name: tabular-rank-calibrated-blending
description: >
  Blends predictions from multiple models by converting to ranks, weighting, and calibrating back to probabilities via rank-group means from a reference model. Ensures monotonic calibrated output.
---

# Rank-Calibrated Blending

## Overview

Standard prediction averaging can produce poorly calibrated blends when models have different score distributions. Rank-calibrated blending sidesteps this by operating in rank space, then mapping back to calibrated probabilities using a reference model's predictions grouped by rank bucket.

## Quick Start

```python
import numpy as np
from scipy.stats import rankdata

# 1. Rank each model's predictions
rank_a = rankdata(pred_xgb)
rank_b = rankdata(pred_cb)

# 2. Weighted blend in rank space
rank_blend = rank_a * 0.99 + rank_b * 0.01

# 3. Calibrate: map rank groups to reference model's mean prediction
n_bins = 1000
rank_bins = np.digitize(rank_blend, np.linspace(rank_blend.min(), rank_blend.max(), n_bins))
calibrated = np.zeros_like(rank_blend)
for b in np.unique(rank_bins):
    mask = rank_bins == b
    calibrated[mask] = pred_xgb[mask].mean()  # reference model = xgb

# 4. Enforce monotonicity via isotonic regression (optional)
from sklearn.isotonic import IsotonicRegression
calibrated = IsotonicRegression(out_of_bounds='clip').fit_transform(rank_blend, calibrated)
```

## Workflow

1. **Train N models** independently with cross-validation (e.g., XGBoost + CatBoost).
2. **Generate predictions** on the same test set from each model.
3. **Rank predictions** per model using `scipy.stats.rankdata`.
4. **Blend ranks** with chosen weights (tune via CV AUC).
5. **Calibrate** by grouping blended ranks into bins and assigning each bin the mean prediction from a reference model.
6. **Enforce monotonicity** with isotonic regression if needed.

## Key Decisions

| Decision | Guidance |
|---|---|
| Weight selection | Start with the better model at ~0.99, minor model at ~0.01. Tune on CV. |
| Number of bins | 500-1000 bins works well for ~100k samples. More data = more bins. |
| Reference model | Use the single best model (highest CV AUC) as the calibration reference. |
| Monotonicity | Always enforce via `IsotonicRegression` -- rank-to-probability must be non-decreasing. |
| When to use | When models have similar AUC but different score distributions or ranking behavior. |

## References

- Kaggle: "CV AUC 0.91930 XGB&CB + Blend" (playground-series-s6e3)
- `scipy.stats.rankdata` documentation
- `sklearn.isotonic.IsotonicRegression` documentation
