---
name: timeseries-periodogram-seasonality-detection
description: Use scipy periodogram to identify dominant seasonal frequencies in a time series before selecting Fourier feature orders or ARIMA seasonal parameters
---

# Periodogram Seasonality Detection

## Overview

Before adding seasonal features, determine which frequencies actually exist in the data. The periodogram decomposes a time series into its frequency components, revealing peaks at dominant cycles (weekly, monthly, annual). This avoids overfitting to phantom seasonality and guides the choice of Fourier order or SARIMA seasonal period.

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

fs = 365.25  # samples per year (daily data)
freqs, spectrum = periodogram(
    series.dropna().values,
    fs=fs,
    detrend="linear",
    window="boxcar",
    scaling="spectrum",
)

plt.figure(figsize=(10, 4))
plt.step(freqs, spectrum)
plt.xscale("log")
plt.xlabel("Cycles per year")
plt.ylabel("Power")
plt.axvline(1, color="r", alpha=0.3, label="Annual")
plt.axvline(52, color="g", alpha=0.3, label="Weekly")
plt.legend()
plt.show()

# Find top 3 peaks
top_idx = np.argsort(spectrum)[-3:]
dominant_periods = 1.0 / freqs[top_idx]  # in days
```

## Workflow

1. Detrend the series (linear or differencing) to remove drift
2. Compute the periodogram with `scipy.signal.periodogram`
3. Plot on log-x scale to see peaks across time scales
4. Identify dominant peaks — these are the frequencies worth modeling
5. Use the detected periods to set Fourier orders or seasonal ARIMA parameters

## Key Decisions

- **Sampling frequency**: daily data → `fs=365.25` (cycles/year), hourly → `fs=8766`
- **Detrending**: `"linear"` removes simple trends; for nonstationary data, difference first
- **Window**: `"boxcar"` (no taper) for maximum frequency resolution; `"hann"` reduces spectral leakage
- **Interpretation**: peaks at 1.0 cycles/year = annual, 52 = weekly, 365 = daily patterns
- **Action**: set Fourier order to 2-4x the number of significant peaks

## References

- [Getting Started with MLB Player Digital Engagement](https://www.kaggle.com/code/ryanholbrook/getting-started-with-mlb-player-digital-engagement)
