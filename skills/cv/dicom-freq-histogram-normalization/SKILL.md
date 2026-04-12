---
name: cv-dicom-freq-histogram-normalization
description: Normalize DICOM pixel values using frequency-equalized histogram bins for globally consistent non-linear intensity mapping
---

## Overview

Standard HU windowing clips to a linear range, losing detail in dense regions of the intensity distribution. Frequency-equalized histogram normalization computes bin edges where each bin contains roughly equal numbers of pixels across a representative sample. This non-linear mapping spreads contrast evenly across the full [0, 1] range, making subtle density differences visible to the CNN.

## Quick Start

```python
import numpy as np
import torch

def freqhist_bins(px, n_bins=20):
    """Compute bin edges where each bin has equal pixel count."""
    imsd = np.sort(px.flatten())
    t = np.concatenate([[0.001],
        np.arange(n_bins) / n_bins + (1 / (2 * n_bins)),
        [0.999]])
    return np.unique(np.quantile(imsd, t))

def hist_scaled(px, bins):
    """Map pixel values through frequency-equalized bins to [0, 1]."""
    return np.interp(px.flatten(), bins,
        np.linspace(0, 1, len(bins))).reshape(px.shape)

# Build global bins from representative sample
sample_pixels = np.concatenate([read_dcm(f).flatten() for f in sample_files])
bins = freqhist_bins(sample_pixels, n_bins=20)

# Apply to any image
normalized = hist_scaled(dcm.pixel_array * slope + intercept, bins)
```

## Workflow

1. Select representative sample stratified by scanner type and label class
2. Concatenate all sample pixel values and compute frequency-equal quantile bins
3. Store bins as a global array (one per dataset)
4. For each image: convert to HU, then interpolate through bins to [0, 1]
5. Compute dataset mean/std from normalized samples for further standardization

## Key Decisions

- **n_bins=20**: Enough resolution to preserve detail, few enough to be stable. Increase for wider HU ranges.
- **Sample selection**: Stratify by BitsStored, PixelRepresentation, and label class to capture the full pixel distribution.
- **vs. linear windowing**: Linear clips extremes. Histogram equalization preserves all density information with uniform contrast.

## References

- [DON'T see like a radiologist! (fastai)](https://www.kaggle.com/code/jhoward/don-t-see-like-a-radiologist-fastai)
