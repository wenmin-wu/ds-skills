---
name: timeseries-multiband-tsfresh-fft
description: Apply tsfresh per-passband feature extraction with FFT coefficients to capture multi-band periodicity from irregular time series
domain: timeseries
---

# Multiband tsfresh FFT Features

## Overview

Use tsfresh's `extract_features` with the `column_kind` parameter to automatically compute features per frequency band in parallel. Include FFT coefficient magnitudes alongside standard statistics (kurtosis, skewness) to capture periodic structure. The per-band FFT features are especially useful for irregularly sampled multi-band data where time-domain periodicity detection is noisy.

## Quick Start

```python
from tsfresh import extract_features

def multiband_tsfresh(df, id_col='object_id', time_col='mjd',
                      band_col='passband', value_col='flux'):
    """Extract per-band features including FFT coefficients.
    
    Args:
        df: DataFrame with multi-band time series
        id_col: entity identifier
        time_col: timestamp column
        band_col: passband/channel column
        value_col: measurement value column
    """
    fcp = {
        'fft_coefficient': [
            {'coeff': 0, 'attr': 'abs'},
            {'coeff': 1, 'attr': 'abs'},
        ],
        'kurtosis': None,
        'skewness': None,
    }
    
    features = extract_features(
        df,
        column_id=id_col,
        column_sort=time_col,
        column_kind=band_col,
        column_value=value_col,
        default_fc_parameters=fcp,
        n_jobs=4,
    )
    return features
```

## Key Decisions

- **column_kind**: tells tsfresh to compute features per band independently
- **FFT coeff 0**: DC component (mean level); coeff 1: dominant frequency amplitude
- **Minimal feature set**: kurtosis + skewness + 2 FFT coefficients keeps dimensionality manageable
- **n_jobs=4**: parallelize across entities; increase for large datasets
- **Extend fcp**: add `'maximum'`, `'minimum'`, `'standard_deviation'` for richer features

## References

- Source: [ideas-from-kernels-and-discussion-lb-1-135](https://www.kaggle.com/code/iprapas/ideas-from-kernels-and-discussion-lb-1-135)
- Competition: PLAsTiCC Astronomical Classification
