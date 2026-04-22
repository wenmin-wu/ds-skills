---
name: timeseries-quantile-ratio-scaling
description: Convert point forecasts to prediction intervals by scaling with logit-transformed quantile ratios passed through a Normal CDF
---

# Quantile Ratio Scaling

## Overview

Instead of training separate models for each quantile, convert a single point (median) forecast into a full set of prediction intervals. Apply the logit transform `log(q/(1-q))` to each target quantile, scale by a bandwidth coefficient, pass through the Normal CDF, and normalize so the median ratio equals 1.0. Multiplying point forecasts by these ratios produces calibrated quantile predictions at zero additional training cost.

## Quick Start

```python
import numpy as np
from scipy import stats
import pandas as pd

def get_quantile_ratios(quantiles, coef=0.065):
    """Compute multiplicative ratios for converting point forecasts to quantiles."""
    qs = np.array(quantiles)
    logit_qs = np.log(qs / (1 - qs)) * coef
    ratios = stats.norm.cdf(logit_qs)
    ratios /= ratios[len(ratios) // 2]  # normalize so median = 1.0
    return pd.Series(ratios, index=qs)

quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
ratios = get_quantile_ratios(quantiles, coef=0.065)

# Apply to point forecasts
for q in quantiles:
    forecast_q = point_forecast * ratios[q]
```

## Workflow

1. Define target quantiles (e.g., 9 symmetric quantiles around 0.5)
2. Compute logit transform: `log(q / (1-q))` for each quantile
3. Scale by bandwidth coefficient and pass through Normal CDF
4. Normalize so the 0.5 quantile ratio equals 1.0
5. Multiply point forecast columns by the corresponding ratio

## Key Decisions

- **Bandwidth coef**: controls interval width; 0.05-0.15 typical; tune on validation SPL
- **Symmetry**: logit+Normal CDF produces symmetric intervals around the median
- **Per-level coef**: use different coefficients for different aggregation levels (wider for item-level, narrower for total)
- **vs quantile regression**: this is zero-cost but assumes symmetric uncertainty; quantile regression captures asymmetry

## References

- [From point to uncertainty prediction](https://www.kaggle.com/code/kneroma/from-point-to-uncertainty-prediction)
- [Point to uncertainty - different ranges per level](https://www.kaggle.com/code/szmnkrisz97/point-to-uncertainty-different-ranges-per-level)
