---
name: tabular-spatial-distance-aggregation
description: Compute min/max/mean/std of Euclidean distances from all entities to a key point, then aggregate per group for spatial feature engineering
domain: tabular
---

# Spatial Distance Aggregation

## Overview

When data has multiple entities per event (players per play, sensors per reading, objects per frame), compute Euclidean distance from each entity to a key reference point, then aggregate with min/max/mean/std per group. Captures spatial density, isolation, and spread in a few dense features.

## Quick Start

```python
import numpy as np
import pandas as pd

def spatial_distance_features(df, ref_x, ref_y, x_col='X', y_col='Y',
                               group_cols=None, prefix='dist'):
    """Compute distance aggregates from entities to a reference point.
    
    Args:
        df: DataFrame with entity positions
        ref_x, ref_y: columns with reference point coordinates
        group_cols: columns defining each group (e.g., [GameId, PlayId])
        prefix: column name prefix
    """
    df = df.copy()
    df[f'{prefix}_to_ref'] = np.sqrt(
        (df[x_col] - df[ref_x]) ** 2 + (df[y_col] - df[ref_y]) ** 2
    )
    aggs = df.groupby(group_cols)[f'{prefix}_to_ref'].agg(
        ['min', 'max', 'mean', 'std']
    ).reset_index()
    aggs.columns = group_cols + [
        f'{prefix}_min', f'{prefix}_max', f'{prefix}_mean', f'{prefix}_std'
    ]
    return aggs

# Usage: distance of all defenders to ball carrier
features = spatial_distance_features(
    defenders, ref_x='carrier_X', ref_y='carrier_Y',
    group_cols=['GameId', 'PlayId'], prefix='def_dist'
)
```

## Key Decisions

- **Min distance**: closest threat/nearest neighbor — strongest single predictor in many spatial tasks
- **Std distance**: measures clustering vs spread — low std means entities are bunched together
- **Subset by role**: compute separately for defense/offense or by entity type for richer signal
- **Add velocity**: extend with relative speed features for dynamic spatial tasks

## References

- Source: [location-eda-8eb410](https://www.kaggle.com/code/bestpredict/location-eda-8eb410)
- Competition: NFL Big Data Bowl
