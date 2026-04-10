---
name: cv-connected-components-centroid-extraction
description: >
  Extracts object centers from 3D segmentation masks using connected-component labeling and centroid computation.
---
# Connected-Components Centroid Extraction

## Overview

After segmenting a 3D volume into a binary or multi-class mask, convert pixel-level predictions into discrete object detections. Connected-component labeling groups contiguous voxels, then each component's centroid gives the object's 3D coordinate. Essential for particle picking, cell detection, and nodule localization.

## Quick Start

```python
import cc3d
import numpy as np

def extract_centroids(segmentation, threshold=0.5):
    binary = (segmentation > threshold).astype(np.uint8)
    labels, n = cc3d.connected_components(binary, return_N=True)
    centroids = []
    for label_id in range(1, n + 1):
        coords = np.argwhere(labels == label_id)
        centroids.append(coords.mean(axis=0))
    return np.array(centroids)  # shape: (N, 3) — z, y, x
```

## Workflow

1. Threshold the probability map to binary mask
2. Run 3D connected-component labeling (`cc3d` or `scipy.ndimage.label`)
3. Filter components by volume (discard tiny noise or oversized blobs)
4. Compute centroid of each component via `np.argwhere().mean()`
5. Output list of (z, y, x) coordinates

## Key Decisions

- **Connectivity**: 26-connected (face+edge+corner) catches thin structures; 6-connected (face-only) is stricter
- **Min volume filter**: Remove components below N voxels to suppress false positives
- **cc3d vs scipy**: `cc3d` is 2-10x faster on large volumes
- **Multi-class**: Run per-class or use `cc3d.connected_components` with `connectivity=26`

## References

- [Baseline UNet train + submit](https://www.kaggle.com/code/fnands/baseline-unet-train-submit)
- [3d-unet using 2d image encoder](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder)
