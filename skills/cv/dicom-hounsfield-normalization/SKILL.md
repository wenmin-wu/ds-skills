---
name: cv-dicom-hounsfield-normalization
description: Convert raw DICOM pixel arrays to Hounsfield Units using per-slice RescaleSlope/RescaleIntercept, with outside-scanner clamping
domain: cv
---

# DICOM Hounsfield Normalization

## Overview

Medical CT scanners store raw pixel values that must be converted to Hounsfield Units (HU) for meaningful analysis. Apply per-slice RescaleSlope and RescaleIntercept from DICOM metadata, then clamp outside-scanner regions to air (−1000 HU). This standardizes pixel values across different scanners and protocols.

## Quick Start

```python
import numpy as np
import pydicom

def transform_to_hu(slices):
    """Convert DICOM slices to Hounsfield Units.
    
    Args:
        slices: list of pydicom Dataset objects (one per CT slice)
    Returns:
        3D numpy array in HU scale
    """
    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    images[images <= -1000] = 0  # outside-scanner → air
    for n in range(len(slices)):
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        if slope != 1:
            images[n] = (slope * images[n].astype(np.float64)).astype(np.int16)
        images[n] += np.int16(intercept)
    return np.array(images, dtype=np.int16)

# Usage
slices = [pydicom.dcmread(f) for f in sorted(dicom_files)]
slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
hu_volume = transform_to_hu(slices)
```

## Key Decisions

- **Per-slice rescale**: slope/intercept can vary between slices in the same series
- **Clamp outside-scanner**: pixels ≤ −1000 are set to 0 (air) to remove scanner artifacts
- **Sort by z-position**: ensures correct spatial ordering for 3D analysis
- **int16 output**: HU range (−1024 to +3071) fits in int16, saves memory vs float

## References

- Source: [pulmonary-dicom-preprocessing](https://www.kaggle.com/code/allunia/pulmonary-dicom-preprocessing)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
