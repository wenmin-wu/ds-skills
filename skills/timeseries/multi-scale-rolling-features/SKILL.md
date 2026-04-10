---
name: timeseries-multi-scale-rolling-features
description: >
  Computes rolling mean/max/std at multiple window sizes plus total variation (abs first differences) for multi-resolution temporal context.
---
# Multi-Scale Rolling Features

## Overview

Create rolling aggregates at multiple window sizes (minutes to hours) to capture patterns at different temporal scales. Add total variation features (rolling stats of absolute first differences) to distinguish gradual drift from sudden jumps. This gives tree models and NNs rich temporal context without explicit sequence modeling.

## Quick Start

```python
import polars as pl

def make_rolling_features(df, value_cols, windows_min, sample_rate=12):
    """Generate multi-scale rolling features.

    Args:
        df: DataFrame with sensor columns
        value_cols: columns to aggregate (e.g., ['enmo', 'anglez'])
        windows_min: list of window sizes in minutes (e.g., [5, 30, 120, 480])
        sample_rate: samples per minute
    """
    features = []
    for mins in windows_min:
        win = sample_rate * mins
        for var in value_cols:
            # Raw signal aggregates
            features.extend([
                pl.col(var).rolling_mean(win, center=True, min_periods=1).alias(f'{var}_{mins}m_mean'),
                pl.col(var).rolling_max(win, center=True, min_periods=1).alias(f'{var}_{mins}m_max'),
                pl.col(var).rolling_std(win, center=True, min_periods=1).alias(f'{var}_{mins}m_std'),
            ])
            # Total variation (abs first differences)
            features.extend([
                pl.col(var).diff().abs().rolling_mean(win, center=True, min_periods=1).alias(f'{var}_tv_{mins}m_mean'),
                pl.col(var).diff().abs().rolling_std(win, center=True, min_periods=1).alias(f'{var}_tv_{mins}m_std'),
            ])
    return df.with_columns(features)
```

## Workflow

1. Choose window sizes spanning short-term (5 min) to long-term (8 hours)
2. Compute rolling mean, max, std for each signal at each scale
3. Add total variation features (rolling stats of `.diff().abs()`)
4. Use `center=True` so features look both forward and backward
5. Cast to smaller dtypes (UInt16/Float32) to reduce memory

## Key Decisions

- **Window sizes**: Powers of 2 or domain-meaningful durations (5m, 30m, 2h, 8h for sleep)
- **center=True**: Avoids lookahead bias only if labels are point-in-time; for segment labels it's fine
- **Total variation**: Distinguishes "quiet but drifting" from "noisy but stable" — critical for activity detection
- **Polars vs Pandas**: Polars is 5-10x faster for rolling operations on large series

## References

- Child Mind Institute - Detect Sleep States (Kaggle)
- Source: [cmi-submit](https://www.kaggle.com/code/tubotubo/cmi-submit)
