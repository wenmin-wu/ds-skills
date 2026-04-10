---
name: timeseries-transit-depth-polynomial-optimization
description: Estimate event depth by optimizing a scalar scaling factor on the in-event segment that minimizes polynomial baseline residual across the full signal
domain: timeseries
---

# Transit Depth Polynomial Optimization

## Overview

Given a time series with a known dip region (ingress/egress boundaries detected), estimate the dip depth by finding a scalar `s` such that scaling the in-dip segment by `(1+s)` produces the smoothest polynomial fit across the full signal. Minimize the mean absolute residual of the polynomial. This is a physics-free, general-purpose depth estimation technique.

## Quick Start

```python
import numpy as np
from scipy.optimize import minimize

def estimate_depth(signal, ingress, egress, margin=10, poly_deg=3):
    """Estimate dip depth via polynomial baseline optimization.
    
    Args:
        signal: 1D time series
        ingress, egress: dip boundary indices
        margin: buffer around boundaries to exclude
        poly_deg: polynomial degree for baseline
    Returns:
        depth: estimated fractional depth of the dip
    """
    def objective(s):
        corrected = np.concatenate([
            signal[:ingress - margin],
            signal[ingress + margin:egress - margin] * (1 + s[0]),
            signal[egress + margin:]
        ])
        x = np.arange(len(corrected))
        poly = np.poly1d(np.polyfit(x, corrected, poly_deg))
        return np.mean(np.abs(poly(x) - corrected))
    
    result = minimize(objective, x0=[0.0001], method='Nelder-Mead')
    return result.x[0]
```

## Key Decisions

- **Margin buffer**: excludes noisy ingress/egress transition zones
- **poly_deg=3**: captures slow trends; increase for longer baselines
- **Nelder-Mead**: derivative-free, robust for this 1D optimization
- **Mean absolute residual**: more robust to outliers than MSE
- **Generalization**: works for any dip/absorption feature, not just transits

## References

- Source: [neurips-non-ml-transit-curve-fitting](https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting)
- Competition: NeurIPS - Ariel Data Challenge 2025
