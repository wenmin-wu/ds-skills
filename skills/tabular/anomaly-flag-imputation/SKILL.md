---
name: tabular-anomaly-flag-imputation
description: >
  Detects sentinel anomaly values in numeric columns, creates a boolean flag feature, then replaces the sentinel with NaN for proper imputation.
---
# Anomaly Flag Imputation

## Overview

Real-world datasets often encode missing or special-status values as sentinel numbers (e.g., 365243 for "not employed", 999 for "unknown", -1 for "missing"). These break statistical features (mean, std) and mislead models. This pattern: (1) creates a binary flag column indicating the anomaly, (2) replaces the sentinel with NaN so imputers and models handle it correctly. The flag preserves the information that the value was special while the NaN lets downstream code treat it as missing.

## Quick Start

```python
import numpy as np
import pandas as pd

def flag_and_replace(df, col, sentinel):
    """Flag sentinel value as boolean feature, replace with NaN."""
    flag_col = f'{col}_ANOMALY'
    df[flag_col] = (df[col] == sentinel).astype(int)
    df[col] = df[col].replace({sentinel: np.nan})
    return df

# Apply to both train and test identically
for df in [train, test]:
    df = flag_and_replace(df, 'DAYS_EMPLOYED', 365243)
    df = flag_and_replace(df, 'DAYS_LAST_PHONE_CHANGE', 0)
```

## Workflow

1. Inspect numeric columns for suspicious spikes (histogram or `value_counts`)
2. Identify sentinel values — often round numbers, max/min outliers, or documented codes
3. Create a boolean flag column (`col_ANOMALY`) before replacing
4. Replace sentinel with `np.nan`
5. Apply identical transform to train and test

## Key Decisions

- **Detection**: Plot distributions — sentinels appear as isolated spikes far from the main mass
- **Flag vs drop**: Always keep the flag — it often correlates with the target (e.g., "not employed" predicts default)
- **Multiple sentinels**: Some columns have several (e.g., -1 and 999) — flag each separately
- **Order**: Flag first, then replace — reversing loses the information

## References

- [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction)
