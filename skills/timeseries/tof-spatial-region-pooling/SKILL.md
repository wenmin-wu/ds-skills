---
name: timeseries-tof-spatial-region-pooling
description: Aggregate high-dimensional spatial sensor grids into hierarchical region statistics at multiple granularities
domain: timeseries
---

# ToF Spatial Region Pooling

## Overview

Spatial sensors (Time-of-Flight, depth cameras, pressure mats) produce pixel grids per timestep. Reduce dimensionality by pooling pixels into spatial regions at multiple scales (2, 4, 8, 16, 32 regions), computing mean/std/min/max per region. Captures both coarse and fine spatial patterns.

## Quick Start

```python
import numpy as np
import pandas as pd

def spatial_region_pooling(pixel_df, n_pixels=64, scales=[2, 4, 8, 16, 32],
                           sentinel=-1):
    """Pool pixel grid into multi-scale region statistics."""
    data = pixel_df.replace(sentinel, np.nan)
    features = {}
    # Global stats
    features['mean'] = data.mean(axis=1)
    features['std'] = data.std(axis=1)
    # Multi-scale regions
    for n_regions in scales:
        region_size = n_pixels // n_regions
        for r in range(n_regions):
            region = data.iloc[:, r * region_size:(r + 1) * region_size]
            features[f'r{n_regions}_{r}_mean'] = region.mean(axis=1)
            features[f'r{n_regions}_{r}_std'] = region.std(axis=1)
            features[f'r{n_regions}_{r}_min'] = region.min(axis=1)
            features[f'r{n_regions}_{r}_max'] = region.max(axis=1)
    return pd.DataFrame(features, index=pixel_df.index)
```

## Key Decisions

- **Sentinel handling**: replace -1 (or other invalid markers) with NaN before aggregation
- **Multi-scale**: coarse regions capture global patterns, fine regions capture local detail
- **Generalizable**: works for any grid sensor — ToF, pressure mats, thermal arrays

## References

- Source: [cmi-detect-behavior-with-sensor-data](https://www.kaggle.com/code/nina2025/cmi-detect-behavior-with-sensor-data)
- Competition: CMI - Detect Behavior with Sensor Data
