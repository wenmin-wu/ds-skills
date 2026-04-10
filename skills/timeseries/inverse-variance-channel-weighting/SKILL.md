---
name: timeseries-inverse-variance-channel-weighting
description: Weight multi-channel signals by inverse per-channel variance with percentile clipping, emphasizing low-noise channels in aggregation
domain: timeseries
---

# Inverse-Variance Channel Weighting

## Overview

When aggregating multi-channel (multi-band, multi-sensor) time series, weight each channel by the inverse of its temporal variance. Noisy channels get downweighted; stable channels dominate. Clip extreme weights at the 5th/95th percentile to prevent any single channel from dominating, then normalize so weights sum to the number of channels (preserving scale).

## Quick Start

```python
import numpy as np

def inverse_variance_weight(data, clip_pct=(5.0, 95.0)):
    """Weight channels by inverse variance.
    
    Args:
        data: (n_timesteps, n_channels) array
        clip_pct: percentile bounds for weight clipping
    Returns:
        weighted data (n_timesteps, n_channels)
    """
    var = np.nanvar(data, axis=0, ddof=1)
    median_var = np.nanmedian(var)
    
    # Replace invalid variances with median
    safe_var = np.where(~np.isfinite(var) | (var <= 0), median_var, var)
    weights = 1.0 / safe_var
    
    # Clip extremes
    lo, hi = np.nanpercentile(weights, clip_pct)
    if lo < hi:
        weights = np.clip(weights, lo, hi)
    
    # Normalize to preserve scale
    n_channels = data.shape[1]
    weights *= n_channels / np.nansum(weights)
    
    return data * weights[None, :]
```

## Key Decisions

- **Percentile clipping**: prevents one ultra-stable channel from getting all the weight
- **Median fallback**: channels with zero/nan variance get median weight, not infinity
- **Scale-preserving normalization**: weights sum to n_channels so overall magnitude is unchanged
- **General purpose**: applies to any multi-channel signal — spectral, multi-sensor, multi-band

## References

- Source: [0-374-lb-score-bronze-medal](https://www.kaggle.com/code/antonoof/0-374-lb-score-bronze-medal)
- Competition: NeurIPS - Ariel Data Challenge 2025
