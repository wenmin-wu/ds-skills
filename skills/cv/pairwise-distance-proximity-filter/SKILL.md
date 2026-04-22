---
name: cv-pairwise-distance-proximity-filter
description: Compute Euclidean distance between entity pairs from tracking data and filter out pairs beyond a threshold to reduce inference candidates
---

# Pairwise Distance Proximity Filter

## Overview

In pairwise interaction detection (player contact, vehicle collision, person re-id), most entity pairs at any given timestep are too far apart to interact. Computing Euclidean distance from tracking/GPS coordinates and filtering pairs beyond a threshold (e.g., 2 yards) removes 80-95% of negative candidates before expensive model inference, dramatically reducing compute and false positive rate.

## Quick Start

```python
import numpy as np
import pandas as pd

def filter_by_distance(df, pos_cols_1, pos_cols_2, threshold=2.0):
    """Filter entity pairs by Euclidean distance.
    df: DataFrame with position columns for both entities
    """
    valid = df[pos_cols_2[0]].notnull()
    dist = np.full(len(df), np.nan)
    dist[valid] = np.sqrt(
        np.square(df.loc[valid, pos_cols_1[0]] - df.loc[valid, pos_cols_2[0]])
        + np.square(df.loc[valid, pos_cols_1[1]] - df.loc[valid, pos_cols_2[1]])
    )
    df['distance'] = dist
    return df.query('not distance > @threshold').reset_index(drop=True)

filtered = filter_by_distance(
    df_pairs,
    ['x_position_1', 'y_position_1'],
    ['x_position_2', 'y_position_2'],
    threshold=2.0
)
```

## Workflow

1. Merge tracking data to get positions for both entities in each pair
2. Compute Euclidean distance (handle NaN for missing positions)
3. Filter pairs beyond the distance threshold
4. Pass only nearby pairs to the model for classification

## Key Decisions

- **Threshold**: domain-specific (2 yards for football contact, 5m for vehicle interaction)
- **NaN handling**: keep NaN-distance pairs (missing tracking) rather than dropping — model can still classify from video
- **query syntax**: `not distance > T` keeps NaN rows; `distance <= T` drops them
- **Distance as feature**: also feed the distance value to the model as a tabular feature

## References

- [NFL 2.5D CNN](https://www.kaggle.com/code/royalacecat/nfl-2-5d-cnn)
- [NFL Player Contact Detection - Getting Started](https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started)
