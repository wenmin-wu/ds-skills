---
name: timeseries-correlated-double-sampling
description: Subtract paired reference frames from signal frames to cancel readout noise and common-mode bias
domain: timeseries
---

# Correlated Double Sampling (CDS)

## Overview

For sensors with paired read/reset cycles, subtract even frames from odd frames (or vice versa) to cancel common-mode noise (kTC noise, bias drift, readout offsets). Halves the frame count but dramatically improves SNR. Common in astronomical detectors, IR sensors, and scientific imaging.

## Quick Start

```python
import numpy as np

def correlated_double_sampling(signal):
    """Subtract even (reset) frames from odd (read) frames.
    
    Args:
        signal: (..., T, ...) with T frames in read/reset pairs
    Returns:
        (..., T//2, ...) CDS-processed signal
    """
    return signal[..., 1::2, :] - signal[..., ::2, :]

# Often followed by temporal binning
def cds_and_bin(signal, bin_size=10):
    cds = correlated_double_sampling(signal)
    n_bins = cds.shape[-2] // bin_size
    binned = np.zeros((*cds.shape[:-2], n_bins, *cds.shape[-1:]))
    for i in range(n_bins):
        binned[..., i, :] = cds[..., i*bin_size:(i+1)*bin_size, :].mean(axis=-2)
    return binned
```

## Key Decisions

- **Odd - even**: convention depends on sensor — check if read or reset comes first
- **Pair binning after CDS**: sum/mean consecutive CDS frames to further reduce noise by √N
- **Spatial mean before CDS**: for 2D detectors, average spatial pixels first if only temporal signal matters

## References

- Source: [ariel-only-correlation](https://www.kaggle.com/code/sergeifironov/ariel-only-correlation)
- Competition: NeurIPS - Ariel Data Challenge 2024
