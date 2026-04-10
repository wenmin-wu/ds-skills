---
name: timeseries-temporal-frame-binning
description: Reduce temporal resolution by averaging consecutive frame blocks to improve SNR and compress high-cadence data
domain: timeseries
---

# Temporal Frame Binning

## Overview

High-cadence sensors produce more frames than needed for the signal of interest. Bin consecutive frames by averaging (or summing) to reduce noise by √N and compress data. Apply after noise-removal steps (CDS, dark subtraction) but before feature extraction.

## Quick Start

```python
import numpy as np

def temporal_bin(signal, bin_size, method='mean'):
    """Bin consecutive frames along time axis.
    
    Args:
        signal: (..., T, ...) array with T timesteps
        bin_size: number of frames per bin
        method: 'mean' (improves SNR) or 'sum' (preserves counts)
    """
    T = signal.shape[-2]
    n_bins = T // bin_size
    # Reshape to group frames
    truncated = signal[..., :n_bins * bin_size, :]
    shape = (*truncated.shape[:-2], n_bins, bin_size, *truncated.shape[-1:])
    grouped = truncated.reshape(shape)
    if method == 'mean':
        return grouped.mean(axis=-2)
    return grouped.sum(axis=-2)
```

## Key Decisions

- **Mean vs sum**: mean for averaged signals (flux), sum for count-based (photon counts)
- **SNR improvement**: √N for white noise — binning 16 frames gives 4x SNR boost
- **Truncate remainder**: drop trailing frames that don't fill a complete bin
- **Order matters**: bin after CDS/calibration, before model input

## References

- Source: [update-calibrating-and-binning-astronomical-data](https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data)
- Competition: NeurIPS - Ariel Data Challenge 2024
