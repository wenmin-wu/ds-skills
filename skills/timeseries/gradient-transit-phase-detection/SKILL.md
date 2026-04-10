---
name: timeseries-gradient-transit-phase-detection
description: Detect event ingress/egress boundaries by finding steepest gradient on each side of the signal minimum in a smoothed time series
domain: timeseries
---

# Gradient Transit Phase Detection

## Overview

For time series with a dip pattern (e.g. exoplanet transits, signal dropouts), detect the ingress and egress boundaries by: (1) optionally bin and smooth the signal, (2) find the minimum (deepest point), (3) split at the minimum, (4) compute the gradient on each half, (5) argmin of left gradient = ingress, argmax of right gradient = egress. Works for any symmetric or asymmetric dip in a time series.

## Quick Start

```python
import numpy as np
from scipy.signal import savgol_filter

def detect_dip_boundaries(signal, binning=15, smooth_window=11):
    """Detect ingress/egress of a dip in a time series.
    
    Args:
        signal: 1D array of measurements
        binning: temporal binning factor before detection
        smooth_window: Savitzky-Golay filter window (odd integer)
    Returns:
        (ingress_idx, egress_idx) in original time indices
    """
    # Bin for noise reduction
    n = len(signal) // binning
    binned = signal[:n * binning].reshape(n, binning).mean(axis=1)
    
    # Smooth
    smoothed = savgol_filter(binned, smooth_window, polyorder=2)
    
    # Find minimum
    min_idx = np.argmin(smoothed)
    
    # Gradient on each side
    grad_left = np.gradient(smoothed[:min_idx])
    grad_right = np.gradient(smoothed[min_idx:])
    
    # Normalize
    if grad_left.max() > 0: grad_left /= grad_left.max()
    if grad_right.max() > 0: grad_right /= grad_right.max()
    
    ingress = np.argmin(grad_left) * binning
    egress = (np.argmax(grad_right) + min_idx) * binning
    return ingress, egress
```

## Key Decisions

- **Bin first**: reduces noise so gradient isn't dominated by high-frequency jitter
- **Savitzky-Golay**: preserves dip shape better than moving average
- **Normalize gradients**: makes detection robust to varying signal amplitudes
- **Generalizes beyond astronomy**: any dip/trough detection in time series

## References

- Source: [neurips-non-ml-transit-curve-fitting](https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting)
- Competition: NeurIPS - Ariel Data Challenge 2025
