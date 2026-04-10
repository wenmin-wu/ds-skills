---
name: timeseries-multiband-color-index-features
description: Compute log-ratio features between adjacent frequency bands as color indices to characterize spectral shape from multi-band time series
domain: timeseries
---

# Multiband Color Index Features

## Overview

For multi-band time series (multiple sensor channels or frequency bands), compute "color indices" — the log-ratio of mean values between adjacent bands. In astronomy, color = -2.5*log10(flux_A/flux_B) captures the spectral energy distribution. In general, cross-band ratios expose relative signal strength across channels, which is more discriminative than per-band statistics alone.

## Quick Start

```python
import numpy as np

def color_index_features(df, id_col, band_col, value_col, band_order):
    """Compute color indices between adjacent bands.
    
    Args:
        df: DataFrame with multi-band time series
        id_col: entity identifier column
        band_col: band/channel identifier column
        value_col: measurement value column
        band_order: list of band names in spectral order
    """
    # Mean value per band per entity
    band_means = df.groupby([id_col, band_col])[value_col].mean().unstack()
    
    features = {}
    for i in range(len(band_order) - 1):
        b1, b2 = band_order[i], band_order[i + 1]
        col_name = f'color_{b1}_minus_{b2}'
        ratio = band_means[b1] / band_means[b2]
        # Log-ratio (color index); handle non-positive values
        features[col_name] = -2.5 * np.log10(ratio.clip(lower=1e-10))
    
    return pd.DataFrame(features)
```

## Key Decisions

- **Adjacent bands**: compute between neighboring bands in spectral order for interpretability
- **Log-ratio**: captures multiplicative differences; more stable than raw ratios
- **Handle negatives**: clip ratios to avoid log of zero/negative; set to NaN if both bands missing
- **Generalization**: any multi-channel sensor data benefits — not just astronomy

## References

- Source: [the-plasticc-astronomy-starter-kit](https://www.kaggle.com/code/michaelapers/the-plasticc-astronomy-starter-kit)
- Competition: PLAsTiCC Astronomical Classification
