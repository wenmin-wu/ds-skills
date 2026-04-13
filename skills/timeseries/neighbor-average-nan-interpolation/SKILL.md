---
name: timeseries-neighbor-average-nan-interpolation
description: Fill gaps in a daily exogenous series (oil prices, sensor feeds) by merging against a full calendar to expose NaNs, then replacing each NaN with the midpoint of its nearest valid left and right neighbors, walking outward past consecutive NaN runs
---

## Overview

When you merge a sparse daily series (oil price, weather) against a full calendar for forecasting features, weekends and holidays become NaN rows that break downstream models. `ffill` propagates Monday's value across Sunday, biasing weekend-sensitive signals; `interpolate()` handles short gaps but silently skips leading/trailing NaNs. The midpoint-neighbor recipe is the deterministic workhorse: build the full calendar, expose NaN positions, walk outward past any consecutive NaN run to find the next valid value on each side, and assign the midpoint. Edge NaNs fall back to the single available neighbor.

## Quick Start

```python
import numpy as np
import pandas as pd

calendar = pd.DataFrame({'date': pd.date_range(min_d, max_d)})
oil = calendar.merge(raw_oil, on='date', how='left').reset_index(drop=True)

na_idx = oil.index[oil['dcoilwtico'].isnull()].values
for i in na_idx:
    left = i - 1
    while left >= 0 and pd.isna(oil.loc[left, 'dcoilwtico']):
        left -= 1
    right = i + 1
    while right < len(oil) and pd.isna(oil.loc[right, 'dcoilwtico']):
        right += 1
    if left < 0:
        oil.loc[i, 'dcoilwtico'] = oil.loc[right, 'dcoilwtico']
    elif right >= len(oil):
        oil.loc[i, 'dcoilwtico'] = oil.loc[left, 'dcoilwtico']
    else:
        oil.loc[i, 'dcoilwtico'] = (oil.loc[left, 'dcoilwtico'] +
                                    oil.loc[right, 'dcoilwtico']) / 2
```

## Workflow

1. Build a complete `date_range` calendar spanning the forecast horizon
2. Left-merge the sparse series onto the calendar so all gaps become explicit NaN rows
3. Collect integer indices of every NaN position
4. For each NaN, walk left and right past consecutive NaN runs to find the nearest valid neighbors
5. Assign the midpoint of left + right values; fall back to the single available neighbor for boundary NaNs

## Key Decisions

- **Midpoint vs distance-weighted linear**: midpoint is cheaper and nearly indistinguishable for short gaps in near-stationary signals. Use `pandas.interpolate(method='time')` for long gaps or strong trends.
- **Full calendar merge is mandatory**: without it, weekend/holiday gaps are invisible and `fillna` won't touch them.
- **Boundary fallback**: leading/trailing NaN must copy the one available neighbor — otherwise you get out-of-bounds or NaN leakage into the model.
- **Integer index, not date index**: walking outward with `.loc[i]` requires a sequential integer index, so `reset_index(drop=True)` after the merge.
- **Don't use `ffill` for weekends**: it propagates Friday's value through Sunday, which is exactly wrong for weekend-sensitive signals.

## References

- [Grocery prediction with Neural Network](https://www.kaggle.com/code/bilalyuksel/grocery-prediction-with-neural-network)
