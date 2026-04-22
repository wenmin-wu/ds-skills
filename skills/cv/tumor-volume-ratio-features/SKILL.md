---
name: cv-tumor-volume-ratio-features
description: Extract volumetric features from 3D segmentation masks including scan/tumor pixel ratios, tumor percentage, and tumor centroid coordinates
---

# Tumor Volume Ratio Features

## Overview

3D segmentation masks contain rich structural information beyond the mask itself. Extracting volumetric features — total scan pixels, tumor pixels, tumor-to-scan ratio, and tumor centroid coordinates — creates tabular features that complement image-based models. These features capture tumor size, location, and relative volume that CNNs may struggle to learn directly.

## Quick Start

```python
import numpy as np
from skimage.morphology import binary_closing

def extract_volume_features(scan, segmentation, mask_idx=1):
    """Extract volumetric features from a 3D scan and its segmentation."""
    scan_filled = np.stack([binary_closing(scan[i] > 0) for i in range(scan.shape[0])])
    scan_px = scan_filled.sum()
    tumor_mask = (segmentation == mask_idx) | (segmentation == 4)
    tumor_px = tumor_mask.sum()
    total_px = np.prod(scan.shape)

    z, x, y = tumor_mask.nonzero()
    if len(z) > 0:
        centroid = [np.median(x) / scan.shape[1],
                    np.median(y) / scan.shape[2],
                    np.median(z) / scan.shape[0]]
    else:
        centroid = [0.5, 0.5, 0.5]

    return {
        'scan_ratio': scan_px / total_px,
        'tumor_ratio': tumor_px / total_px,
        'tumor_to_scan': tumor_px / max(scan_px, 1),
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'centroid_z': centroid[2],
    }

features = extract_volume_features(volume, seg_mask, mask_idx=1)
```

## Workflow

1. Binarize the scan with morphological closing to fill holes
2. Count total scan pixels and tumor pixels from the segmentation mask
3. Compute ratios: tumor/total, tumor/scan, scan/total
4. Find tumor centroid via median of nonzero coordinates (normalized to [0,1])
5. Use as tabular features alongside CNN predictions

## Key Decisions

- **binary_closing**: fills small holes in the brain mask for accurate volume estimation
- **Median centroid**: more robust to outlier voxels than mean
- **Normalization**: divide coordinates by volume dimensions for scale invariance
- **Multiple tumor classes**: combine relevant segmentation labels (e.g., necrotic + enhancing)

## References

- [Brain Tumor - Understanding the Data](https://www.kaggle.com/code/susant4learning/brain-tumor-understanding-the-data)
