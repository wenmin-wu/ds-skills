---
name: timeseries-sigma-clip-outlier-masking
description: Detect and mask outlier data points using iterative sigma-clipping on reference frames or calibration data
domain: timeseries
---

# Sigma-Clip Outlier Masking

## Overview

Iterative sigma-clipping flags values beyond N standard deviations from the mean, recomputing statistics after each round. Use it to detect hot pixels in sensor data, outlier time steps, or anomalous channels. Returns a boolean mask compatible with numpy masked arrays.

## Quick Start

```python
from astropy.stats import sigma_clip
import numpy as np

def mask_outliers(reference_data, signal, sigma=5, maxiters=5):
    """Sigma-clip on reference data, apply mask to signal.
    
    Args:
        reference_data: (H, W) calibration frame (e.g. dark frame)
        signal: (T, H, W) time-series to mask
        sigma: clipping threshold in std deviations
    """
    clipped = sigma_clip(reference_data, sigma=sigma, maxiters=maxiters)
    outlier_mask = clipped.mask  # True where outlier
    # Broadcast to time dimension
    mask_3d = np.broadcast_to(outlier_mask, signal.shape)
    return np.ma.masked_array(signal, mask=mask_3d)

# Then use np.nanmean or .mean() on masked array
clean_mean = masked_signal.mean(axis=(1, 2))
```

## Key Decisions

- **sigma=5**: conservative — only flags extreme outliers (>5σ)
- **maxiters=5**: converges quickly, prevents over-clipping on small samples
- **Reference-based**: clip on calibration data, apply mask to science data — avoids masking real signal
- **astropy.stats.sigma_clip**: handles masked arrays natively; scipy alternative exists

## References

- Source: [update-calibrating-and-binning-astronomical-data](https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data)
- Competition: NeurIPS - Ariel Data Challenge 2024
