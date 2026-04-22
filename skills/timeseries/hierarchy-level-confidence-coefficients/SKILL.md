---
name: timeseries-hierarchy-level-confidence-coefficients
description: Assign different uncertainty spread coefficients per aggregation level in hierarchical forecasts, reflecting that higher aggregation yields narrower intervals
---

# Hierarchy-Level Confidence Coefficients

## Overview

In hierarchical time series (item → department → category → store → state → total), higher aggregation levels are smoother and have narrower prediction intervals. When converting point forecasts to quantiles via ratio scaling, assign a different bandwidth coefficient per hierarchy level — wider for individual items (high variance), narrower for totals (low variance). This produces better-calibrated intervals across all levels without training separate quantile models.

## Quick Start

```python
from scipy import stats
import numpy as np
import pandas as pd

def get_ratios(quantiles, coef=0.15):
    qs = np.array(quantiles)
    logit_qs = np.log(qs / (1 - qs)) * coef
    ratios = stats.norm.cdf(logit_qs)
    ratios /= ratios[len(ratios) // 2]
    return pd.Series(ratios, index=qs)

qs = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

level_coefs = {
    'id':       0.30,   # individual item-store: widest
    'item_id':  0.15,
    'dept_id':  0.08,
    'cat_id':   0.07,
    'store_id': 0.08,
    'state_id': 0.07,
    'total':    0.05,   # grand total: narrowest
}

level_ratios = {level: get_ratios(qs, coef) for level, coef in level_coefs.items()}
```

## Workflow

1. Define hierarchy levels from finest (item-store) to coarsest (total)
2. Assign a bandwidth coefficient per level (larger = wider intervals)
3. Compute quantile ratios using the level-specific coefficient
4. For each level, sum point forecasts by group, then multiply by level ratios
5. Tune coefficients on validation Scaled Pinball Loss per level

## Key Decisions

- **Coefficient range**: 0.05 (total) to 0.30 (item-level) is typical for retail
- **Cross-level pairs**: state×item, store×dept get interpolated coefficients
- **Tuning**: grid search coefs to minimize SPL on validation set per level
- **vs per-level models**: coefficients are simpler and avoid training 12 separate models

## References

- [Point to uncertainty - different ranges per level](https://www.kaggle.com/code/szmnkrisz97/point-to-uncertainty-different-ranges-per-level)
