---
name: timeseries-snap-event-interaction-features
description: Build per-state SNAP / event-flag interaction features by multiplying the binary flag with the sales and revenue columns segmented by state, capturing the demand uplift on government-benefit days that affects only specific geographies and product categories
---

## Overview

Retail forecasts in welfare-supported markets (SNAP days in the US, holiday allowances in EU) have to know that demand spikes 30-50% on benefit-credit days, but only in the right state and only for the right product mix. A flat `is_snap_day` flag isn't enough — you need to localize it: SNAP-CA only matters in California stores, SNAP-grocery only matters for food categories. The cleanest encoding is to create one interaction column per (state, snap_status) combination by multiplying the per-row sales by `(state_id == X) * snap_X`. The resulting columns are sparse (most rows are zero) but capture an effect that no untyped event flag can. Same pattern works for any geography-conditional event: holidays, weather alerts, regional promotions.

## Quick Start

```python
import pandas as pd

for state in ['CA', 'TX', 'WI']:
    snap_col = f'snap_{state}'
    df[f'sold_{state}_snap']    = df['sold'] * (df['state_id'] == state) * df[snap_col]
    df[f'sold_{state}_nonsnap'] = df['sold'] * (df['state_id'] == state) * (1 - df[snap_col])
    df[f'rev_{state}_snap']     = df[f'sold_{state}_snap'] * df['sell_price']
    df[f'rev_{state}_nonsnap']  = df[f'sold_{state}_nonsnap'] * df['sell_price']

# Aggregate to (date, state) for cross-state comparison
snap_agg = (
    df.groupby(['date', 'state_id'])
      [[f'sold_{s}_snap'  for s in ['CA','TX','WI']] +
       [f'sold_{s}_nonsnap' for s in ['CA','TX','WI']]]
      .sum()
)
```

## Workflow

1. Identify the geographic and event columns — `state_id`, `snap_CA`, `snap_TX`, `snap_WI` for M5; analogous for any retail panel
2. For every (state, event) pair, create two columns: `event_active` and `event_inactive`
3. Multiply the per-row sales/revenue by both indicators to produce per-row contribution features
4. Use as raw features for LightGBM (interaction discovery is automatic from the splits)
5. Or aggregate to (date, state, category) for hierarchical post-hoc reconciliation
6. Train a lift-estimation model on the aggregated table to quantify the SNAP uplift per category

## Key Decisions

- **Multiplicative encoding, not concatenated one-hot**: tree models can split on the multiplied column directly; one-hot needs the model to discover the interaction.
- **Both `_snap` and `_nonsnap` columns**: the difference `_snap - _nonsnap` is what the model learns the lift from; one column alone hides the baseline.
- **Per-state, not pooled**: SNAP credits hit different stores in different states on different days; pooling washes out the signal.
- **Revenue alongside units**: high-margin items respond differently to SNAP than low-margin; the revenue version captures dollar uplift, the units version captures volume uplift.
- **Apply at the row level, not the panel level**: row-level multiplication is leak-free because each row already knows its `state_id`; panel-level merges can leak between rows.
- **Same pattern for any locale event**: replace `snap_*` with `holiday_*`, `promo_*`, `weather_alert_*`.

## References

- [Time Series Forecasting - EDA + FE + Modelling](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
