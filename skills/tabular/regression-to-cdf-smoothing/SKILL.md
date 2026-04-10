---
name: tabular-regression-to-cdf-smoothing
description: Convert a scalar regression prediction into a smoothed CDF over discrete bins using a linear ramp instead of a hard step
domain: tabular
---

# Regression to CDF Smoothing

## Overview

When the evaluation metric is CRPS on a discrete CDF but your model outputs a scalar prediction, convert it to a smooth CDF using a linear ramp (or sigmoid) centered on the predicted value. A hard step function is overconfident; a ramp with width W spreads probability over ±W bins, hedging against prediction error.

## Quick Start

```python
import numpy as np

def scalar_to_cdf(predictions, n_bins=199, offset=99, ramp_width=10):
    """Convert scalar predictions to smoothed CDFs.
    
    Args:
        predictions: array of scalar predictions
        n_bins: number of CDF bins
        offset: bin index for target=0
        ramp_width: half-width of linear ramp (bins)
    Returns:
        (N, n_bins) CDF array
    """
    cdf = np.zeros((len(predictions), n_bins))
    for i, pred in enumerate(predictions):
        center = int(round(pred)) + offset
        for j in range(n_bins):
            if j >= center + ramp_width:
                cdf[i, j] = 1.0
            elif j >= center - ramp_width:
                cdf[i, j] = (j - center + ramp_width) / (2 * ramp_width)
    return np.clip(cdf, 0, 1)

# Usage with LightGBM
y_pred_scalar = np.mean([m.predict(X_test) for m in models], axis=0)
y_pred_cdf = scalar_to_cdf(y_pred_scalar, ramp_width=10)
```

## Key Decisions

- **Ramp width**: 10 bins is conservative; optimize on validation CRPS — wider = safer, narrower = sharper
- **Linear vs sigmoid**: linear ramp is simple and effective; sigmoid is smoother but adds a hyperparameter
- **Physical clipping**: enforce domain bounds (e.g., can't gain more yards than field remaining)
- **Ensemble first**: average scalar predictions before converting to CDF for best results

## References

- Source: [nfl-simple-model-using-lightgbm](https://www.kaggle.com/code/hukuda222/nfl-simple-model-using-lightgbm)
- Competition: NFL Big Data Bowl
