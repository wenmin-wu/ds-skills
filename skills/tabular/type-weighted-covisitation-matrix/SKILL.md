---
name: tabular-type-weighted-covisitation-matrix
description: Build item co-visitation matrix from session pairs within a time window, weighting by interaction type (click/cart/order) via GPU self-join
domain: tabular
---

# Type-Weighted Co-visitation Matrix

## Overview

For session-based recommendation, build a co-visitation matrix by self-joining session events within a time window. Weight each co-occurrence by interaction type (e.g. clicks=1, carts=6, orders=3) so high-intent signals dominate. Use RAPIDS cuDF for GPU-accelerated computation on large datasets. Extract top-K co-visited items per anchor for candidate generation.

## Quick Start

```python
import cudf

def build_covisitation_matrix(df, type_weight={0:1, 1:6, 2:3},
                               window_hours=24, top_k=20):
    """Build type-weighted co-visitation from session events.
    
    Args:
        df: DataFrame with columns [session, aid, ts, type]
        type_weight: {event_type: weight} mapping
        window_hours: time window for co-occurrence (hours)
        top_k: top co-visited items to keep per anchor
    """
    window_sec = window_hours * 3600
    df = df.merge(df, on='session')
    df = df.loc[
        ((df.ts_x - df.ts_y).abs() < window_sec) &
        (df.aid_x != df.aid_y)
    ]
    df = df[['session','aid_x','aid_y','type_y']].drop_duplicates(
        ['session','aid_x','aid_y'])
    df['wgt'] = df.type_y.map(type_weight).astype('float32')
    
    # Aggregate and keep top-K per anchor
    pairs = df.groupby(['aid_x','aid_y']).wgt.sum().reset_index()
    pairs = pairs.sort_values(['aid_x','wgt'], ascending=[True, False])
    pairs['n'] = pairs.groupby('aid_x').aid_y.cumcount()
    return pairs.loc[pairs.n < top_k].drop('n', axis=1)
```

## Key Decisions

- **Type weights**: carts >> orders > clicks reflects purchase intent; tune on validation
- **Buy2buy variant**: filter to carts/orders only with a 14-day window for longer-horizon purchase affinity
- **Time-weighted variant**: scale weights linearly by recency (1-4x) to emphasize recent co-visits
- **GPU self-join**: cuDF handles billion-row self-joins; chunk by time period if OOM
- **Top-K per anchor**: 20-40 is typical; higher for cold-start items

## References

- Source: [candidate-rerank-model-lb-0-575](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575)
- Competition: OTTO - Multi-Objective Recommender System
