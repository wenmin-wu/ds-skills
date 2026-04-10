---
name: timeseries-multiband-lomb-scargle-period
description: Estimate periodicity from irregularly sampled multi-band time series using the multiband Lomb-Scargle periodogram, then phase-fold observations
domain: timeseries
---

# Multiband Lomb-Scargle Period Estimation

## Overview

For irregularly sampled multi-band time series (e.g. photometric light curves), the multiband Lomb-Scargle periodogram estimates the dominant period by jointly fitting all bands. After finding the best period, phase-fold observations (time mod period) to expose the underlying shape. The period and phase-folded features are powerful for classification of periodic signals.

## Quick Start

```python
import numpy as np
from gatspy.periodic import LombScargleMultiband

def estimate_period(times, values, errors, bands, t_min=0.1, t_max=10.0):
    """Estimate period from multi-band irregular time series.
    
    Args:
        times: observation timestamps
        values: measured values (flux)
        errors: measurement uncertainties
        bands: passband identifiers per observation
        t_min, t_max: period search range
    """
    model = LombScargleMultiband(fit_period=True)
    model.optimizer.set(
        period_range=(t_min, t_max),
        first_pass_coverage=5)
    model.fit(times, values, dy=errors, filts=bands)
    return model.best_period

def phase_fold(times, period):
    """Fold time series by estimated period."""
    return (times / period) % 1.0

# Usage
period = estimate_period(df.mjd, df.flux, df.flux_err, df.passband)
df['phase'] = phase_fold(df.mjd, period)
```

## Key Decisions

- **Multiband jointly**: fitting all bands simultaneously is more robust than per-band
- **Period range**: set t_min > median cadence, t_max < half the observation span
- **first_pass_coverage=5**: coarse grid density; increase for more precision
- **Phase features**: after folding, compute per-phase-bin statistics as features

## References

- Source: [the-plasticc-astronomy-starter-kit](https://www.kaggle.com/code/michaelapers/the-plasticc-astronomy-starter-kit)
- Competition: PLAsTiCC Astronomical Classification
