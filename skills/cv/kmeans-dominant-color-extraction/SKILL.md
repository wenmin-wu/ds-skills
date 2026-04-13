---
name: cv-kmeans-dominant-color-extraction
description: Extract an image's dominant RGB color via k-means over pixel-color space and emit three dense features capturing the modal color of the subject
---

## Overview

Mean-channel color is a poor feature because a red product on a gray backdrop averages out to brown. K-means in pixel color space finds the *modal* color cluster — the one a human would name when asked "what color is this?" — and it works without any segmentation. Run k-means (k=5) on the flattened Nx3 pixel matrix, pick the centroid of the most populous cluster, and emit three normalized features `dominant_r/g/b`. Used in Avito Demand Prediction top kernels to capture product-color signal for listing-quality modeling.

## Quick Start

```python
import cv2
import numpy as np

def dominant_color(path, n_colors=5):
    img = cv2.imread(path)                       # BGR
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    b, g, r = centroids[np.argmax(counts)].astype(np.uint8)
    return {'dominant_r': r / 255., 'dominant_g': g / 255., 'dominant_b': b / 255.}
```

## Workflow

1. Load the image with cv2 (BGR) and reshape to `Nx3 float32`
2. Run `cv2.kmeans` with `k=5`, 10 attempts, `EPS + MAX_ITER` criteria
3. Count cluster labels with `np.bincount` and pick the `argmax` cluster — the modal color
4. Convert BGR → RGB, normalize to [0, 1], and emit `dominant_r/g/b` as three features
5. Feed the three columns into the GBDT alongside blur score, dullness, edge density

## Key Decisions

- **Modal cluster, not cluster center of mass**: `argmax(counts)` picks the color a human would name; the mean of centroids just reproduces average color.
- **k=5**: balances speed and color-diversity capture; k=3 merges similar shades, k=10 wastes compute.
- **10 attempts**: reduces local-minima sensitivity — single-init k-means gives unstable features across runs.
- **Normalize by 255**: keeps the feature in [0, 1] so linear models and neural nets both see it cleanly; GBDTs are invariant.
- **Keep BGR ↔ RGB explicit**: silent channel swaps are the #1 bug in this recipe.

## References

- [Ideas for Image Features and Image Quality](https://www.kaggle.com/code/shivamb/ideas-for-image-features-and-image-quality)
