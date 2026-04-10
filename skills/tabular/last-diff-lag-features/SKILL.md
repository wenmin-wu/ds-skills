---
name: tabular-last-diff-lag-features
description: Compute first-order difference between last and second-to-last rows per entity in panel data to capture recent trend direction and magnitude
domain: tabular
---

# Last-Diff Lag Features

## Overview

For panel/sequential data where each entity has multiple time-stamped rows, compute the first-order difference between the last two observations per entity. This captures the recent trend — whether a feature is increasing or decreasing — which is often more predictive than the raw value for tasks like credit default or churn prediction.

## Quick Start

```python
import pandas as pd
import numpy as np

def last_diff_features(df, group_col, num_features):
    """Compute last-minus-previous difference per entity.
    
    Args:
        df: DataFrame sorted by (group_col, time), multiple rows per entity
        group_col: entity identifier column
        num_features: list of numeric columns to diff
    """
    diffs = []
    ids = []
    for entity_id, group in df.groupby(group_col):
        diff = group[num_features].diff(1).iloc[[-1]].values
        diffs.append(diff)
        ids.append(entity_id)
    
    result = pd.DataFrame(
        np.concatenate(diffs, axis=0),
        columns=[f'{c}_diff1' for c in num_features],
        index=ids
    )
    result.index.name = group_col
    return result
```

## Key Decisions

- **diff(1) on last row**: captures most recent change; diff(2) would span two periods
- **Combine with aggregates**: use alongside mean/std/last for a complete feature set
- **NaN handling**: entities with only one row produce NaN diffs — fill with 0 or drop
- **Vectorized alternative**: `groupby().transform('diff').groupby().tail(1)` avoids the Python loop

## References

- Source: [amex-lgbm-dart-cv-0-7977](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977)
- Competition: American Express - Default Prediction
