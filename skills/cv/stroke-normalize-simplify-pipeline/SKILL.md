---
name: cv-stroke-normalize-simplify-pipeline
description: Normalize raw stroke coordinates to 0-255 range, resample at uniform arc-length spacing, then apply Ramer-Douglas-Peucker simplification
---

# Stroke Normalize-Simplify Pipeline

## Overview

Raw hand-drawn strokes have arbitrary coordinate ranges and irregular point spacing. A three-step pipeline — normalize to [0, 255], resample at uniform arc-length intervals, then simplify with Ramer-Douglas-Peucker — produces clean, compact stroke representations. This reduces point count by 3-5x while preserving shape fidelity, improving both storage efficiency and model performance.

## Quick Start

```python
import numpy as np
import math

def resample(x, y, spacing=1.0):
    output = []
    px, py = x[0], y[0]
    cumlen, offset = 0, 0
    for i in range(1, len(x)):
        dx, dy = x[i] - px, y[i] - py
        seg_len = math.sqrt(dx*dx + dy*dy)
        cumlen += seg_len
        while offset < cumlen:
            t = (offset - (cumlen - seg_len)) / seg_len
            output.append((px + t*dx, py + t*dy))
            offset += spacing
        px, py = x[i], y[i]
    output.append((x[-1], y[-1]))
    return np.array(output)

def normalize_strokes(strokes):
    all_pts = np.concatenate([np.array(s).T for s in strokes])
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    rng = max(mx - mn)
    result = []
    for s in strokes:
        pts = np.array(s, dtype=float).T
        pts = (pts - mn) / rng * 255
        resampled = resample(pts[:, 0], pts[:, 1], spacing=1.0)
        result.append(np.round(resampled).astype(np.uint8).T.tolist())
    return result
```

## Workflow

1. Compute global bounding box across all strokes
2. Normalize coordinates: `(pt - min) / max_range * 255`
3. Resample each stroke at uniform arc-length intervals via linear interpolation
4. Optionally apply RDP simplification to reduce point count further
5. Round to uint8 for compact storage

## Key Decisions

- **Normalization range**: [0, 255] matches standard image coordinate space for rendering
- **Resampling spacing**: 1.0 pixel gives high fidelity; 2.0-3.0 for faster processing
- **RDP epsilon**: 1.0-2.0 removes redundant points; higher values lose fine detail
- **Order**: normalize first (consistent scale), then resample (uniform spacing), then simplify (reduce count)
- **Aspect ratio**: use max(width, height) as the range denominator to preserve aspect ratio

## References

- [Getting Started: Viewing Quick Draw Doodles](https://www.kaggle.com/code/inversion/getting-started-viewing-quick-draw-doodles-etc)
