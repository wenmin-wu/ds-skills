---
name: cv-multi-scale-patch-training-pyramid
description: Generate count-regression training patches at a geometric pyramid of image scales (0.9^k) so one CNN handles within- and between-image object-size variation without explicit anchors
---

## Overview

Aerial-imagery count datasets suffer from inter-image scale drift — different flights, altitudes, and zoom levels produce the same objects at different pixel sizes. Random rescale augmentation helps but is unprincipled; the deterministic fix is a geometric scale pyramid: for each training image, resize at `[1.0, 0.9, 0.81, 0.729, 0.656]`, rebuild the patch/label grid at each scale, tile, and pool everything into one training set. The CNN then sees the same object instances at every reasonable size and learns a size-invariant mapping. Used alongside patch-grid count regression to win on NOAA Steller Sea Lion.

## Quick Start

```python
import cv2
import numpy as np

PATCH = 300
SCALES = [0.9 ** k for k in range(5)]          # 1.0, 0.9, 0.81, 0.729, 0.656

patches, labels = [], []
for r in SCALES:
    img_r = cv2.resize(img, None, fx=r, fy=r)
    grid = build_count_grid(points, scale=r, patch=PATCH, n_classes=n_classes)
    h, w = img_r.shape[:2]
    for i in range(w // PATCH):
        for j in range(h // PATCH):
            y = grid[i, j]
            x = img_r[j*PATCH:(j+1)*PATCH, i*PATCH:(i+1)*PATCH]
            if y.sum() > 0 or np.random.rand() < 0.25:   # 1:3 pos:neg
                patches.append(x); labels.append(y)
```

## Workflow

1. Define a geometric scale pyramid (e.g. `0.9^k` for `k ∈ [0..4]`) covering the expected size range
2. For each scale, resize the image **and rebuild the point-to-grid mapping** at the new resolution
3. Tile the scaled image into `PATCH × PATCH` patches; keep every positive and ~25% of negatives
4. Pool all scales into a single `(patches, labels)` training set and shuffle
5. Train one count-regression CNN on the pooled set — no separate per-scale model

## Key Decisions

- **Geometric, not linear scaling**: `0.9^k` gives perceptually uniform steps; linear steps crowd the small end.
- **Fix patch *pixel* size across scales**: the CNN sees a constant 300×300 input geometry — scale is absorbed into how many objects fit in a patch, not how big they look to the conv filters.
- **Rebuild the grid per scale**: points get remapped, so the label tensor is different at every scale. Don't reuse the base-scale grid.
- **Enforce pos:neg ratio per scale**: rare classes are rarer at the small end; without ratio control they get drowned.
- **Fewer scales, more data**: 5 scales × full dataset usually beats 10 scales × half dataset in practice.

## References

- [Use keras to count sea lions](https://www.kaggle.com/code/outrunner/use-keras-to-count-sea-lions)
