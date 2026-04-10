---
name: timeseries-detector-calibration-pipeline
description: Multi-step detector calibration pipeline — ADC inversion, hot/dead pixel masking, nonlinearity correction, dark subtraction, flat-field normalization
domain: timeseries
---

# Detector Calibration Pipeline

## Overview

Raw sensor/detector data requires systematic calibration before analysis. Apply these steps in order: (1) ADC gain/offset inversion to physical units, (2) hot/dead pixel masking via sigma-clip on dark frames, (3) per-pixel polynomial nonlinearity correction, (4) dark current subtraction scaled by integration time, (5) flat-field normalization for pixel-to-pixel sensitivity variation. Applicable to any imaging sensor with calibration frames.

## Quick Start

```python
import numpy as np
from astropy.stats import sigma_clip

def calibrate(signal, dark, flat, dead_mask, linear_corr, dt, gain, offset):
    """Full detector calibration pipeline.
    
    Args:
        signal: (n_frames, height, width) raw detector data
        dark: (height, width) dark current frame
        flat: (height, width) flat field frame
        dead_mask: (height, width) boolean dead pixel map
        linear_corr: (n_coeffs, height, width) polynomial coefficients
        dt: (n_frames,) integration times
        gain, offset: ADC conversion parameters
    """
    # 1. ADC inversion
    signal = signal.astype(np.float64) / gain + offset
    
    # 2. Hot pixel detection + dead pixel masking
    hot_mask = sigma_clip(dark, sigma=5, maxiters=5).mask
    bad = hot_mask | dead_mask
    signal[:, bad] = np.nan
    
    # 3. Per-pixel nonlinearity correction (Horner's method)
    signal = np.clip(signal, 0, None)
    for i in range(signal.shape[1]):
        for j in range(signal.shape[2]):
            c = linear_corr[:, i, j]
            signal[:, i, j] = np.polyval(c, signal[:, i, j])
    
    # 4. Dark current subtraction
    signal -= dark[None, :, :] * dt[:, None, None]
    
    # 5. Flat-field normalization
    flat_masked = flat.copy()
    flat_masked[bad] = np.nan
    signal /= flat_masked[None, :, :]
    return signal
```

## Key Decisions

- **Order matters**: ADC first, then mask, then nonlinearity, then dark, then flat
- **Sigma=5 for hot pixels**: conservative; lower sigma catches warm pixels too
- **Horner's method**: faster polynomial evaluation for per-pixel correction
- **NaN for bad pixels**: downstream code uses nanmean/nanmedian to ignore them

## References

- Source: [neurips-non-ml-transit-curve-fitting](https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting)
- Competition: NeurIPS - Ariel Data Challenge 2025
