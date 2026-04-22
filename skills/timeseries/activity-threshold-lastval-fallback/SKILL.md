---
name: timeseries-activity-threshold-lastval-fallback
description: Override model predictions with last known value for low-activity or low-density entities where learned trends are unreliable
---

# Activity-Threshold Last-Value Fallback

## Overview

In entity-level forecasting, some entities have very low counts or near-zero activity. Models trained on the full population tend to hallucinate trends for these entities, producing predictions worse than simply carrying forward the last observed value. Setting an activity threshold and falling back to last-value carry-forward for entities below it improves overall accuracy.

## Quick Start

```python
def apply_fallback(df, pred_col, value_col, entity_col, threshold=3.0):
    last_values = df.groupby(entity_col)[value_col].last()
    entity_mean = df.groupby(entity_col)[value_col].mean()
    low_activity = entity_mean[entity_mean < threshold].index

    df.loc[df[entity_col].isin(low_activity), pred_col] = (
        df.loc[df[entity_col].isin(low_activity), entity_col].map(last_values)
    )
    return df

df = apply_fallback(df, "prediction", "microbusiness_density", "cfips", threshold=3.0)
```

## Workflow

1. Compute per-entity summary statistics (mean, std, count of nonzero observations)
2. Define an activity threshold based on the target's distribution (e.g., bottom 10th percentile, or absolute cutoff)
3. For entities below the threshold, replace model predictions with the last known value
4. Optionally, apply a blended transition: `alpha * model_pred + (1-alpha) * last_value` near the threshold boundary
5. Validate on holdout — the fallback should improve SMAPE/MAE for the low-activity segment without hurting the rest

## Key Decisions

- **Threshold selection**: use CV to find the optimal cutoff; too high wastes model predictions on viable entities
- **Last value vs. recent average**: last value works best for stable low-activity entities; use trailing mean (3-6 periods) if there's noise
- **Entity-level vs. global**: threshold should be entity-specific, not a global value cutoff, since population varies
- **Blending zone**: a hard cutoff can create discontinuities — blending in a ±20% band around the threshold smooths the transition

## References

- [Better XGB Baseline](https://www.kaggle.com/code/titericz/better-xgb-baseline)
