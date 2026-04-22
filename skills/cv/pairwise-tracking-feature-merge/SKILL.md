---
name: cv-pairwise-tracking-feature-merge
description: Double left-join on tracking data to create pairwise features (positions, velocities, distance) for both entities in an interaction pair
---

# Pairwise Tracking Feature Merge

## Overview

Interaction detection tasks (contact, collision, handoff) require features for both entities in each candidate pair. When tracking data has one row per entity per timestep, a double left-join merges entity 1's features first, then entity 2's features, producing a single row with both sets of kinematics. The resulting pairwise features (positions, speeds, accelerations, relative distance) serve as tabular input alongside visual features.

## Quick Start

```python
import numpy as np
import pandas as pd

def merge_pairwise_features(pairs_df, tracking_df, merge_col='step',
                            feature_cols=None):
    """Merge tracking features for both entities in each pair."""
    if feature_cols is None:
        feature_cols = ['x_position', 'y_position', 'speed', 'direction']

    track = tracking_df[['game_play', merge_col, 'nfl_player_id'] + feature_cols]

    df = (pairs_df
        .merge(track, left_on=['game_play', merge_col, 'player_id_1'],
               right_on=['game_play', merge_col, 'nfl_player_id'], how='left')
        .rename(columns={c: f'{c}_1' for c in feature_cols})
        .drop('nfl_player_id', axis=1)
        .merge(track, left_on=['game_play', merge_col, 'player_id_2'],
               right_on=['game_play', merge_col, 'nfl_player_id'], how='left')
        .rename(columns={c: f'{c}_2' for c in feature_cols})
        .drop('nfl_player_id', axis=1))

    df['distance'] = np.sqrt(
        (df['x_position_1'] - df['x_position_2'])**2
        + (df['y_position_1'] - df['y_position_2'])**2)
    return df

df_pairs = merge_pairwise_features(contact_pairs, tracking, merge_col='step')
```

## Workflow

1. Start with a pairs DataFrame containing (game_play, timestep, entity_1, entity_2)
2. Left-join tracking data on entity_1 → suffix columns with `_1`
3. Left-join tracking data on entity_2 → suffix columns with `_2`
4. Compute derived features: Euclidean distance, relative speed, heading difference
5. Feed as tabular features to the model alongside visual features

## Key Decisions

- **Left join**: preserves all pairs even when tracking is missing for one entity
- **Feature columns**: position, speed, direction, acceleration — all get `_1`/`_2` suffixes
- **Derived features**: distance is most predictive; relative velocity and heading angle add marginal gain
- **String casting**: ensure player IDs match type between pairs and tracking DataFrames

## References

- [NFL 2.5D CNN](https://www.kaggle.com/code/royalacecat/nfl-2-5d-cnn)
- [NFL Player Contact Detection - Getting Started](https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started)
