---
name: tabular-cyclical-feature-encoding
description: >
  Encodes cyclical features (hour, month, day-of-week) using sine/cosine transforms to preserve circular distance.
---
# Cyclical Feature Encoding

## Overview

Features like hour-of-day or month wrap around (23→0, Dec→Jan). Standard ordinal encoding creates a false distance between the endpoints. Sine/cosine transforms map cyclical features onto a unit circle, preserving the true circular distance between values.

## Quick Start

```python
import numpy as np

def cyclical_encode(values, period):
    """Encode cyclical feature using sin/cos pair.

    Args:
        values: array of numeric values (e.g., hours 0-23)
        period: full cycle length (24 for hours, 12 for months, 7 for weekdays)
    """
    sin = np.sin(2 * np.pi * values / period)
    cos = np.cos(2 * np.pi * values / period)
    return sin, cos

# Usage
df['hour_sin'], df['hour_cos'] = cyclical_encode(df['hour'], 24)
df['month_sin'], df['month_cos'] = cyclical_encode(df['month'], 12)
df['dow_sin'], df['dow_cos'] = cyclical_encode(df['dayofweek'], 7)
```

## Workflow

1. Identify cyclical features (hour, month, day-of-week, minute, week-of-year)
2. Apply sin/cos transform with the correct period for each
3. Drop the original integer column — keep only the sin/cos pair
4. Use both sin AND cos features (one alone is ambiguous)

## Key Decisions

- **Always use both sin and cos**: sin(0) = sin(π) — without cos, midnight and noon look identical
- **Period must be exact**: hours → 24, months → 12, weekdays → 7, minutes → 60
- **Tree models**: May not benefit much — trees can learn splits on integer features directly
- **Linear/NN models**: Significant improvement — they cannot learn circular boundaries otherwise

## References

- 2019 Data Science Bowl (Kaggle)
- Source: [quick-and-dirty-regression](https://www.kaggle.com/code/artgor/quick-and-dirty-regression)
