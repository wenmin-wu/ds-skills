---
name: cv-ct-z-stack-orientation-flip
description: Detect inverted CT slice ordering by comparing ImagePositionPatient[2] (the Z coordinate) of the first and last DICOM slice in a series, flipping the volume along axis 0 when needed so every patient ends up in canonical head→feet order
---

## Overview

DICOM file names are not a reliable proxy for slice order. The same scanner can write `1.dcm` as the topmost slice on one acquisition and the bottommost on the next, depending on protocol and reconstruction direction. If you train on a mix and don't normalize, augmentations like `flip(axis=0)` become silently inconsistent and the model learns a much fuzzier through-plane signal than it could. The single-line fix: read `ImagePositionPatient[2]` from the first and last DICOM in the series; if the last has a *larger* Z value than the first, the stack is inverted relative to the canonical patient frame and you flip it.

## Quick Start

```python
import dicomsdl
import numpy as np

def canonicalize_z_order(image, dcm_dir, slice_min, slice_max):
    dcm0 = dicomsdl.open(f'{dcm_dir}/{slice_min}.dcm')
    dcmN = dicomsdl.open(f'{dcm_dir}/{slice_max - 1}.dcm')
    z0 = dcm0.ImagePositionPatient[2]
    zN = dcmN.ImagePositionPatient[2]

    if zN > z0:                       # inverted: flip into head→feet order
        image = image[::-1]

    dz = abs((zN - z0) / max(slice_max - slice_min - 1, 1))
    return np.ascontiguousarray(image), dz
```

## Workflow

1. After loading every slice into a `(D, H, W)` array (in filename order), open just the first and last DICOM headers
2. Compare `ImagePositionPatient[2]` — the third element is the Z coordinate in patient space
3. If `zN > z0`, the stack is in feet→head order; reverse it with `image[::-1]`
4. Compute `dz` as the absolute difference divided by `(num_slices - 1)` and store it for the resampling step
5. `np.ascontiguousarray` after the slice-reverse to avoid downstream `RuntimeError: non-contiguous` from torch

## Key Decisions

- **Compare Z, not InstanceNumber**: InstanceNumber can be reset, missing, or inconsistent across vendors; ImagePositionPatient is the geometric ground truth.
- **Absolute `dz`**: after flipping, the sign is meaningless — what the resampler needs is the magnitude.
- **Only open two DICOMs, not all of them**: the orientation check costs ~1ms and avoids re-parsing every header.
- **`np.ascontiguousarray` after `[::-1]`**: numpy reverse-views are non-contiguous and break torch tensor zero-copy paths.
- **Don't sort by Z to "fix" ordering**: just flipping is faster and equivalent for evenly-spaced acquisitions, which is what 99% of CT series are.

## References

- [LB 0.55 — 2.5D + 3D sample model](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
