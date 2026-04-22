---
name: timeseries-scaled-pinball-loss
description: Scaled Pinball Loss (SPL) metric for evaluating quantile forecasts, normalized by mean absolute successive differences of training data
---

# Scaled Pinball Loss

## Overview

The Scaled Pinball Loss (SPL) evaluates quantile forecast accuracy while being scale-independent across time series. The numerator is the standard pinball (quantile) loss; the denominator normalizes by the mean absolute successive difference of the training data, similar to how MASE normalizes by naive forecast error. This makes SPL comparable across series with different scales.

## Quick Start

```python
import numpy as np

def scaled_pinball_loss(y_true, y_pred, quantile, y_train):
    """Compute SPL for a single quantile forecast."""
    h = len(y_true)

    # Pinball numerator
    errors = y_true - y_pred
    pinball = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    numerator = np.sum(pinball)

    # Scale denominator: mean absolute successive differences
    N = len(y_train)
    denom = np.sum(np.abs(np.diff(y_train))) / (N - 1)

    return numerator / (h * denom)

def mean_spl(y_true, quantile_preds, quantiles, y_train):
    """Average SPL across all quantiles."""
    scores = []
    for q, pred in zip(quantiles, quantile_preds):
        scores.append(scaled_pinball_loss(y_true, pred, q, y_train))
    return np.mean(scores)

quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
score = mean_spl(y_val, preds_per_quantile, quantiles, y_train)
```

## Workflow

1. Compute pinball loss: `q * e` if `e >= 0`, else `(q-1) * e`
2. Sum over the forecast horizon
3. Compute denominator: mean absolute first-difference of training data
4. Divide numerator by `(h * denominator)` for scale-free metric
5. Average across all quantiles for the final SPL score

## Key Decisions

- **Denominator**: successive differences capture the series' natural variability
- **Per-quantile vs average**: report both for diagnostics; average for leaderboard
- **Flat series**: if `denom ≈ 0`, the series is constant — clip to a small epsilon
- **vs CRPS**: SPL evaluates specific quantiles; CRPS integrates over all quantiles

## References

- [M5 - Sales Uncertainty Prediction](https://www.kaggle.com/code/allunia/m5-sales-uncertainty-prediction)
