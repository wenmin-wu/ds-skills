---
name: cv-spectral-index-segmentation
description: Compute normalized spectral band ratios (NDWI, CCCI, NDVI) from multispectral imagery and threshold for binary segmentation of water, vegetation, or other targets
---

# Spectral Index Segmentation

## Overview

Multispectral satellites capture bands beyond visible RGB (NIR, red-edge, SWIR). Normalized difference indices combine these bands into single-channel images where target features (water, vegetation, chlorophyll) have distinct value ranges. A simple threshold on the index produces a binary segmentation mask without any ML model.

## Quick Start

```python
import numpy as np
from skimage.transform import resize

def ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-10)

def ccci(red, red_edge, mir):
    return ((mir - red_edge) / (mir + red_edge + 1e-10) *
            (mir - red) / (mir + red + 1e-10))

def ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-10)

ms = load_16band_image(image_id)
re = resize(ms[5], rgb.shape[:2])
mir = resize(ms[7], rgb.shape[:2])
r = rgb[:, :, 0].astype(float)

index = ccci(r, re, mir)
water_mask = (index > 0.11).astype(np.uint8)
```

## Workflow

1. Load multispectral bands and resize to a common spatial resolution
2. Select bands for the target index (e.g., green+NIR for NDWI)
3. Compute the normalized ratio — values typically range [-1, 1]
4. Threshold to produce a binary mask
5. Optionally apply morphological cleaning (opening/closing)

## Key Decisions

- **Index choice**: NDWI for water, NDVI for vegetation, CCCI for chlorophyll/waterways
- **Threshold tuning**: varies by sensor, scene, and season — tune on labeled samples
- **Band alignment**: always resize lower-res bands to match before computing indices
- **vs ML models**: spectral indices are interpretable baselines; often competitive for simple targets

## References

- [Waterway 0.095 LB](https://www.kaggle.com/code/resolut/waterway-0-095-lb)
