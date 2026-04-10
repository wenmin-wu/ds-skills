---
name: timeseries-flux-snr-weighted-features
description: Engineer SNR-derived features from irregular time series — flux ratio squared, error-weighted mean flux, and normalized amplitude/range features
domain: timeseries
---

# Flux SNR-Weighted Features

## Overview

For irregular time series with measurement errors (e.g. astronomical light curves, sensor data with noise estimates), engineer signal-to-noise features that weight observations by their reliability. Compute (flux/flux_err)^2 as SNR weight, then derive error-weighted mean flux, normalized amplitude, and range-over-mean ratios. These features are far more predictive than raw flux statistics for noisy data.

## Quick Start

```python
import numpy as np

def flux_snr_features(df, id_col='object_id'):
    """Engineer SNR-weighted features from flux + flux_err columns."""
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    
    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'std'],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
    }
    result = df.groupby(id_col).agg(aggs)
    result.columns = ['_'.join(x) for x in result.columns]
    
    # Error-weighted mean flux
    result['flux_w_mean'] = (
        result['flux_by_flux_ratio_sq_sum'] / result['flux_ratio_sq_sum'])
    # Normalized dynamic range
    result['flux_dif2'] = (
        (result['flux_max'] - result['flux_min']) / result['flux_mean'])
    result['flux_dif3'] = (
        (result['flux_max'] - result['flux_min']) / result['flux_w_mean'])
    return result
```

## Key Decisions

- **SNR weighting**: (flux/flux_err)^2 gives inverse-variance weight — standard in astronomy
- **Per-passband**: apply separately per frequency band for multi-band data
- **Skewness of SNR**: captures asymmetry in observation quality distribution
- **Normalized amplitude**: range/mean is scale-invariant across objects

## References

- Source: [simple-neural-net-for-time-series-classification](https://www.kaggle.com/code/meaninglesslives/simple-neural-net-for-time-series-classification)
- Competition: PLAsTiCC Astronomical Classification
