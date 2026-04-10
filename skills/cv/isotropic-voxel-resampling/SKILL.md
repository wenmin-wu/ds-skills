---
name: cv-isotropic-voxel-resampling
description: Resample 3D CT volumes to uniform voxel spacing using scipy zoom, normalizing physical dimensions across scanners
domain: cv
---

# Isotropic Voxel Resampling

## Overview

CT scanners produce volumes with non-uniform voxel spacing (e.g., 0.7×0.7×2.5 mm). Models expecting consistent spatial resolution need isotropic resampling. Compute the resize factor from original spacing to target (typically 1×1×1 mm), then apply scipy's zoom interpolation. Essential when mixing scans from different scanners or protocols.

## Quick Start

```python
import numpy as np
import scipy.ndimage

def resample(image, scan, new_spacing=[1, 1, 1]):
    """Resample a 3D volume to isotropic voxel spacing.
    
    Args:
        image: (D, H, W) numpy array
        scan: list of pydicom Dataset objects (for metadata)
        new_spacing: target voxel size in mm [z, y, x]
    Returns:
        resampled image, actual new spacing
    """
    spacing = np.array(
        [scan[0].SliceThickness] + list(scan[0].PixelSpacing),
        dtype=np.float32
    )
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize = new_shape / image.shape
    real_spacing = spacing / real_resize
    resampled = scipy.ndimage.zoom(image, real_resize, mode='nearest')
    return resampled, real_spacing

resampled_vol, spacing = resample(hu_volume, slices, [1, 1, 1])
```

## Key Decisions

- **1mm isotropic**: standard target for lung CT; adjust for other modalities
- **Nearest-mode interpolation**: preserves HU values without blending; use spline for smoother results
- **Round then recompute**: avoids fractional voxel drift by rounding new shape first
- **Memory impact**: 2.5mm→1mm triples volume size along z-axis — consider chunked processing

## References

- Source: [pulmonary-dicom-preprocessing](https://www.kaggle.com/code/allunia/pulmonary-dicom-preprocessing)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
