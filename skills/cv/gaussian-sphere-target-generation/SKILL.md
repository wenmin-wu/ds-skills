---
name: cv-gaussian-sphere-target-generation
description: >
  Generates 3D segmentation training targets by placing Gaussian spheres at annotated point coordinates.
---
# Gaussian Sphere Target Generation

## Overview

When ground truth is point annotations (x, y, z) rather than voxel masks, generate training targets by placing 3D Gaussian blobs at each annotated location. Each particle type gets its own channel with a configurable radius. The model learns to predict these soft targets, and centroids are recovered at inference via peak detection or connected components.

## Quick Start

```python
import numpy as np

def generate_gaussian_volume(shape, coords, sigma=3.0):
    volume = np.zeros(shape, dtype=np.float32)
    for z, y, x in coords:
        zz, yy, xx = np.ogrid[
            max(0,int(z)-3*int(sigma)):min(shape[0],int(z)+3*int(sigma)+1),
            max(0,int(y)-3*int(sigma)):min(shape[1],int(y)+3*int(sigma)+1),
            max(0,int(x)-3*int(sigma)):min(shape[2],int(x)+3*int(sigma)+1),
        ]
        d2 = (zz-z)**2 + (yy-y)**2 + (xx-x)**2
        volume[zz, yy, xx] = np.maximum(volume[zz, yy, xx], np.exp(-d2/(2*sigma**2)))
    return volume
```

## Workflow

1. Parse point annotations (z, y, x) per particle class
2. For each class, create a zero volume matching the input shape
3. Place Gaussian blob at each coordinate (clip to volume bounds)
4. Use `np.maximum` to handle overlapping particles (keep brightest value)
5. Stack per-class volumes into multi-channel target tensor

## Key Decisions

- **Sigma**: Match to expected particle radius; too large → merged blobs, too small → hard to learn
- **Per-class channels**: Separate channel per particle type enables multi-class detection
- **Max vs sum**: `max` prevents double-counting overlapping particles; `sum` better for density estimation
- **Truncation**: 3-sigma cutoff balances accuracy vs speed

## References

- [DeepFindET_Train](https://www.kaggle.com/code/kharrington/deepfindet-train)
