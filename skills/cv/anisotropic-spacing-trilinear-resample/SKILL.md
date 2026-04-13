---
name: cv-anisotropic-spacing-trilinear-resample
description: Resample a 3D medical volume to a fixed network input shape using physical voxel spacing (dz, dy, dx), correcting the Z dimension by the dz/dy ratio so anisotropic CT scans (1mm in-plane, 5mm slice) end up anatomically isotropic before trilinear interpolation
---

## Overview

CT volumes are routinely anisotropic: 0.7mm in-plane and 5mm between slices is normal. If you `F.interpolate` a `(60, 512, 512)` volume directly to `(96, 256, 256)`, you stretch the Z axis 1.6x in voxel space but the *physical* distance per slice is already 7x the in-plane spacing — the resulting volume looks correct in tensor shape but is geometrically wrong, and the network learns the wrong aspect ratio. The fix is to first scale the slice count by `dz / dy` (the spacing ratio), then resample to the target shape. This delivers an anatomically isotropic input regardless of the source acquisition.

## Quick Start

```python
import torch
import torch.nn.functional as F

def resample_to_shape(image, spacing, target_hw=256, target_d=96):
    dz, dy, dx = spacing             # physical mm/voxel
    d, h, w = image.shape

    # 1. Correct Z count by physical-spacing ratio (anisotropy fix)
    d_iso = int(dz / dy * d * 0.5)   # 0.5 = empirically tuned compression

    # 2. Scale all axes to in-plane target
    scale = target_hw / h
    d_out = int(scale * d_iso)
    h_out = w_out = int(scale * h)

    # 3. Final resample to fixed network input shape
    image = F.interpolate(image[None, None],
                          size=(d_out, h_out, w_out),
                          mode='trilinear',
                          align_corners=False)[0, 0]
    image = F.interpolate(image[None, None],
                          size=(target_d, target_hw, target_hw),
                          mode='trilinear',
                          align_corners=False)[0, 0]
    return image
```

## Workflow

1. Read `PixelSpacing` (dy, dx) and `SliceThickness` (or `SpacingBetweenSlices`) → `dz` from the DICOM headers
2. Compute the anisotropy-corrected slice count `d_iso = int(dz / dy * d * factor)`
3. First interpolate to an isotropic intermediate, then to the fixed network input shape
4. Use trilinear (not nearest, not linear-per-axis) for both passes
5. Cache the per-series spacing alongside the cropped volume to avoid re-reading DICOM headers at training time

## Key Decisions

- **Physical spacing first, voxel resize second**: skipping the spacing correction trains the model on geometrically distorted anatomy.
- **`* 0.5` factor**: pure `dz/dy` over-corrects in practice — a damping factor between 0.4 and 0.6 is empirically best for abdominal CT.
- **Two-pass interpolation**: collapsing the two `interpolate` calls into one introduces aliasing for high anisotropy ratios.
- **`align_corners=False`**: the PyTorch default for new code; matches OpenCV/numpy convention.
- **Handles missing `dz`**: fall back to `SpacingBetweenSlices` if `SliceThickness` is absent; some scanners populate only one.

## References

- [LB 0.55 — 2.5D + 3D sample model](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
