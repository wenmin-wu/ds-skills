---
name: tabular-typed-panel-aggregation
description: Aggregate panel/sequential data with type-appropriate statistics — numeric (mean/std/min/max/last) and categorical (count/last/nunique) — then concat into flat features
domain: tabular
---

# Typed Panel Aggregation

## Overview

When customer or entity data arrives as multiple time-stamped rows (panel data), flatten to one row per entity by applying different aggregation functions per column type. Numeric columns get mean, std, min, max, last; categorical columns get count, last, nunique. This type-aware approach extracts richer signals than uniform aggregation.

## Quick Start

```python
import pandas as pd

def typed_panel_agg(df, group_col, cat_features, num_features=None):
    """Aggregate panel data with type-appropriate statistics.
    
    Args:
        df: DataFrame with multiple rows per entity
        group_col: entity identifier column
        cat_features: list of categorical column names
        num_features: list of numeric columns (default: all non-cat)
    """
    if num_features is None:
        num_features = [c for c in df.columns
                        if c not in cat_features + [group_col]]
    
    num_agg = df.groupby(group_col)[num_features].agg(
        ['mean', 'std', 'min', 'max', 'last'])
    num_agg.columns = ['_'.join(x) for x in num_agg.columns]
    
    cat_agg = df.groupby(group_col)[cat_features].agg(
        ['count', 'last', 'nunique'])
    cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]
    
    return pd.concat([num_agg, cat_agg], axis=1)
```

## Key Decisions

- **Numeric stats**: std captures volatility; last captures recency; min/max capture extremes
- **Categorical stats**: nunique captures diversity; count captures engagement frequency
- **GPU acceleration**: use cuDF `groupby().agg()` for datasets with millions of rows
- **Curated stat lists**: optionally select which stats per feature to reduce dimensionality

## References

- Source: [xgboost-starter-0-793](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793)
- Competition: American Express - Default Prediction
