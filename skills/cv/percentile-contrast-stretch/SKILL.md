---
name: cv-percentile-contrast-stretch
description: Normalize high-dynamic-range satellite or medical imagery to [0,1] using per-channel percentile clipping to suppress outliers while preserving relative contrast
---

# Percentile Contrast Stretch

## Overview

Satellite and medical images often have extreme pixel outliers that wash out simple min-max normalization. Percentile contrast stretching clips each channel at the 2nd and 98th percentiles, then linearly maps to [0,1]. This preserves meaningful contrast while being robust to dead pixels, sensor noise, and atmospheric artifacts.

## Quick Start

```python
import numpy as np

def percentile_stretch(image, lower=2, upper=98):
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        lo = np.percentile(image[:, :, c], lower)
        hi = np.percentile(image[:, :, c], upper)
        out[:, :, c] = (image[:, :, c] - lo) / (hi - lo + 1e-10)
    return np.clip(out, 0, 1)

rgb_stretched = percentile_stretch(rgb_image)
```

## Workflow

1. Compute lower and upper percentiles per channel
2. Linearly map each channel: `(pixel - lo) / (hi - lo)`
3. Clip result to [0, 1]
4. Apply before visualization or as model input normalization

## Key Decisions

- **Percentile range**: 2/98 is standard; use 1/99 for less aggressive clipping
- **Per-channel vs global**: per-channel preserves color balance across bands
- **vs histogram equalization**: percentile stretch is linear and invertible; CLAHE introduces nonlinearity
- **Multi-spectral**: works on any number of channels, not just RGB

## References

- [Full pipeline demo: poly -> pixels -> ML -> poly](https://www.kaggle.com/code/lopuhin/full-pipeline-demo-poly-pixels-ml-poly)
