---
name: timeseries-wavelet-denoising
description: Denoise an erratic 1D series with discrete wavelet decomposition + universal soft thresholding (sigma estimated from MAD of the detail coefficients) to extract the underlying trend/seasonality without lagging the signal — a far better trend extractor than rolling means for spiky retail or sensor data
---

## Overview

Rolling means are the default trend extractor but they have two failures: they lag the signal by `window/2` and they smear sharp legitimate jumps. Wavelet denoising solves both. Decompose the series with `pywt.wavedec`, estimate the noise scale `sigma` from the median-absolute-deviation of the highest-frequency detail coefficients (the MAD-based universal threshold from Donoho-Johnstone), apply hard or soft thresholding to all detail levels, then `waverec` back. The result preserves discontinuities and has *zero phase shift*. Use it as a feature (denoised series alongside raw), as a smoothed target for a regression model that hates noise, or just as a visualization.

## Quick Start

```python
import numpy as np
import pywt

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode='per')
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode='per')

# Usage as a feature
df['sales_denoised'] = denoise_signal(df['sales'].values)
```

## Workflow

1. Choose a wavelet — `db4` (Daubechies-4) is the standard default; `sym8` for smoother trends; `haar` for piecewise-constant signals
2. `pywt.wavedec` to get `[approximation, detail_L, detail_L-1, ..., detail_1]`
3. Estimate `sigma = MAD(detail_1) / 0.6745` — the 0.6745 factor converts MAD to a Gaussian sigma
4. Compute universal threshold `uthresh = sigma * sqrt(2 * log(N))`
5. Apply `pywt.threshold(c, uthresh, mode='hard')` to all detail coefficients (keep the approximation untouched)
6. `pywt.waverec` to reconstruct the cleaned signal — same length as input
7. Use as a feature alongside the raw series, or as a target for a smoother regression

## Key Decisions

- **MAD over std for sigma**: std is dominated by outliers (the very things you're trying to remove); MAD is robust.
- **Hard thresholding for trend extraction, soft for denoising**: hard preserves discontinuity edges; soft produces smoother output with slight shrinkage.
- **`mode='per'` (periodic)**: avoids edge artifacts at series boundaries; `'symmetric'` is the alternative.
- **Don't denoise the target if you'll predict the raw target**: a model trained on the denoised target will be biased low at every spike.
- **Pad to power-of-2 length if needed**: `db4` works on arbitrary lengths but some wavelets need power-of-2 input.
- **vs. EMA / rolling mean**: wavelet has no phase lag and preserves jumps; EMA lags by `1/alpha` and smears jumps.

## References

- [Time Series Forecasting - EDA + FE + Modelling](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
