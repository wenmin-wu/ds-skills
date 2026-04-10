---
name: timeseries-prediction-smoothing-lpf
description: >
  Applies rolling mean or Butterworth low-pass filter to model predictions for temporal consistency and noise reduction.
---
# Prediction Smoothing / Low-Pass Filter

## Overview

Raw per-timestep predictions from classifiers are noisy — consecutive timesteps may flip between classes. Smooth predictions with a rolling mean or Butterworth low-pass filter before event detection. After filtering, rescale by the RMSE ratio (before/after) to preserve signal energy and calibration.

## Quick Start

```python
import numpy as np
from scipy import signal

def rolling_smooth(predictions, window):
    """Simple rolling mean smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(predictions, kernel, mode='same')

def butterworth_smooth(predictions, sample_rate, cutoff_freq, n_passes=3):
    """Butterworth low-pass filter with energy preservation.

    Args:
        predictions: 1D array of per-timestep scores
        sample_rate: samples per day (e.g., 12*60*24 for 5-sec intervals)
        cutoff_freq: cutoff in cycles per day (e.g., 60 for ~24-min period)
        n_passes: number of filtfilt passes for sharper rolloff
    """
    nyquist = sample_rate / 2.0
    b, a = signal.butter(1, cutoff_freq / nyquist, btype='low')

    before_energy = np.sqrt(np.mean(predictions**2))
    smoothed = predictions.copy()
    for _ in range(n_passes):
        smoothed = signal.filtfilt(b, a, smoothed)
    after_energy = np.sqrt(np.mean(smoothed**2))

    # Rescale to preserve energy
    if after_energy > 0:
        smoothed *= before_energy / after_energy
    return smoothed
```

## Workflow

1. Generate raw per-timestep predictions from model
2. Choose smoothing method: rolling mean (simple) or Butterworth (sharper cutoff)
3. Apply smoothing per prediction column independently
4. Rescale smoothed output to preserve RMSE (energy) of original
5. Feed smoothed predictions into event detection / thresholding

## Key Decisions

- **Rolling vs Butterworth**: Rolling is simpler; Butterworth has sharper frequency cutoff
- **Window / cutoff**: Match to expected event duration — too wide smooths out real events
- **Energy preservation**: Always rescale after filtering — LPF reduces amplitude
- **Apply per-class**: Smooth each event type column independently

## References

- Child Mind Institute - Detect Sleep States (Kaggle)
- Source: [detect-sleep-states-inference](https://www.kaggle.com/code/itsuki9180/detect-sleep-states-inference)
