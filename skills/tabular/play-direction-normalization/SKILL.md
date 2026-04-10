---
name: tabular-play-direction-normalization
description: Mirror spatial coordinates and angles so all plays face the same direction — removes left/right asymmetry from sports and spatial data
domain: tabular
---

# Play Direction Normalization

## Overview

In spatial datasets (sports tracking, autonomous driving, robotics), events can occur in either direction. Normalize by flipping X-coordinates, orientation, and direction angles so all events face the same way. This halves the effective feature space and prevents models from learning spurious left/right patterns.

## Quick Start

```python
import numpy as np

def normalize_direction(df, direction_col='PlayDirection',
                        x_col='X', orient_col='Orientation',
                        dir_col='Dir', field_length=120.0):
    """Mirror coordinates so all plays go left-to-right."""
    df = df.copy()
    is_left = df[direction_col] == 'left'
    
    # Flip X coordinate
    df.loc[is_left, x_col] = field_length - df.loc[is_left, x_col]
    
    # Flip Y coordinate (optional, for symmetry)
    # df.loc[is_left, 'Y'] = field_width - df.loc[is_left, 'Y']
    
    # Flip angles (0-360 degrees)
    for col in [orient_col, dir_col]:
        flipped = 360.0 - df.loc[is_left, col]
        flipped[flipped == 360.0] = 0.0
        df.loc[is_left, col] = flipped
    
    return df

df = normalize_direction(tracking_data)
```

## Key Decisions

- **X-flip mandatory**: spatial position must be mirrored; Y-flip depends on symmetry of the field
- **Angle flip**: subtract from 360° to mirror orientation; handle 360→0 wraparound
- **Apply before feature engineering**: all derived features (distances, velocities) inherit correct orientation
- **Generalizable**: same pattern for any bidirectional spatial data (court sports, road networks)

## References

- Source: [location-eda-8eb410](https://www.kaggle.com/code/bestpredict/location-eda-8eb410)
- Competition: NFL Big Data Bowl
