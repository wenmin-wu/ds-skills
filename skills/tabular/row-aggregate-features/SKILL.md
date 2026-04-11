---
name: tabular-row-aggregate-features
description: >
  Engineers row-wise statistical features (sum, mean, std, skew, kurtosis, median, min, max) across all numeric columns per sample.
---
# Row Aggregate Features

## Overview

When a dataset has many anonymous or homogeneous numeric columns (e.g., 200 `var_0` to `var_199`), row-level aggregates summarize each sample's overall distribution. Tree models can use these to split on "samples with high variance" or "samples with extreme values" — patterns invisible to individual features.

## Quick Start

```python
import pandas as pd

feature_cols = [c for c in df.columns if c.startswith("var_")]

df["row_sum"]  = df[feature_cols].sum(axis=1)
df["row_mean"] = df[feature_cols].mean(axis=1)
df["row_std"]  = df[feature_cols].std(axis=1)
df["row_min"]  = df[feature_cols].min(axis=1)
df["row_max"]  = df[feature_cols].max(axis=1)
df["row_skew"] = df[feature_cols].skew(axis=1)
df["row_kurt"] = df[feature_cols].kurtosis(axis=1)
df["row_med"]  = df[feature_cols].median(axis=1)
```

## Workflow

1. Identify the block of homogeneous numeric columns
2. Compute row-wise statistics across those columns
3. Add as new features alongside the originals
4. Apply to both train and test identically
5. Let the model decide which aggregates are informative

## Key Decisions

- **Which stats**: Start with mean, std, min, max; add skew/kurtosis if the model benefits
- **Subsets**: Compute over all features or meaningful subgroups (top-importance, positive-only)
- **Percentiles**: `np.percentile(row, [25, 75])` can capture spread better than min/max
- **Nulls**: Use `skipna=True` (default) to handle missing values gracefully

## References

- [Santander EDA and Prediction](https://www.kaggle.com/code/gpreda/santander-eda-and-prediction)
