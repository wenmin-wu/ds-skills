---
name: tabular-relative-deviation-features
description: >
  Computes differences and ratios between group-level aggregates and raw values to capture how each sample deviates from its group.
---
# Relative Deviation Features

## Overview

After computing group aggregates (mean, max, min, std), create features that measure how each row deviates from its group. The difference (value - group_mean) captures absolute deviation; the ratio (value / group_mean) captures relative deviation. These features help tree models split on "is this value unusual for its group?" rather than just raw magnitude.

## Quick Start

```python
import pandas as pd

def add_relative_features(df, group_cols, value_col):
    """Add deviation features: diff and ratio vs group stats."""
    prefix = '_'.join(group_cols) + '_' + value_col

    grp_mean = df.groupby(group_cols)[value_col].transform('mean')
    grp_std = df.groupby(group_cols)[value_col].transform('std')
    grp_max = df.groupby(group_cols)[value_col].transform('max')
    grp_min = df.groupby(group_cols)[value_col].transform('min')

    df[f'{prefix}_mean_diff'] = df[value_col] - grp_mean
    df[f'{prefix}_mean_ratio'] = df[value_col] / grp_mean.clip(lower=1e-9)
    df[f'{prefix}_max_diff'] = grp_max - df[value_col]
    df[f'{prefix}_zscore'] = (df[value_col] - grp_mean) / grp_std.clip(lower=1e-9)
    df[f'{prefix}_range_pos'] = (df[value_col] - grp_min) / (grp_max - grp_min).clip(lower=1e-9)

    return df

# Usage
df = add_relative_features(df, ['molecule_name', 'atom_index_0'], 'dist')
df = add_relative_features(df, ['molecule_name'], 'electronegativity')
```

## Workflow

1. Compute group aggregates (mean, std, max, min) via groupby + transform
2. Create diff features: value minus group stat
3. Create ratio features: value divided by group stat (clip denominator to avoid div-by-zero)
4. Create z-score: (value - mean) / std for normalized deviation
5. Create range position: (value - min) / (max - min) for where in the group range

## Key Decisions

- **Which groups**: Try multiple grouping levels (single column, multi-column) — different granularities capture different patterns
- **Clip denominators**: Always clip to 1e-9 to prevent division by zero
- **Feature explosion**: N groups × M value columns × 5 deviation types adds up — use feature importance to prune
- **Works best with**: Tree models (LightGBM, XGBoost) that can exploit interaction splits

## References

- Predicting Molecular Properties / CHAMPS Scalar Coupling (Kaggle)
- Source: [brute-force-feature-engineering](https://www.kaggle.com/code/artgor/brute-force-feature-engineering)
