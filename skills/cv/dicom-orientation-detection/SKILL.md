---
name: cv-dicom-orientation-detection
description: >
  Decodes MRI scan plane (axial, coronal, sagittal) from DICOM ImageOrientationPatient direction cosine vectors.
---
# DICOM Orientation Detection

## Overview

MRI studies contain multiple series acquired in different planes. The scan plane (axial, coronal, sagittal) determines what anatomical structures are visible and how they should be processed. DICOM stores this as `ImageOrientationPatient` — six direction cosines defining the row and column directions in patient coordinates. Rounding these cosines and pattern-matching identifies the plane, enabling automated routing of series to the correct preprocessing pipeline or model.

## Quick Start

```python
import pydicom
import numpy as np

def detect_orientation(dicom):
    """Detect scan plane from DICOM ImageOrientationPatient."""
    iop = [round(float(v)) for v in dicom.ImageOrientationPatient]
    x1, y1, z1, x2, y2, z2 = iop

    # Row and column direction cosines determine the plane
    row_dir = np.array([x1, y1, z1])
    col_dir = np.array([x2, y2, z2])
    normal = np.cross(row_dir, col_dir)

    # The dominant axis of the normal vector indicates the plane
    dominant = np.argmax(np.abs(normal))
    planes = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    return planes[dominant]

# Usage
dcm = pydicom.dcmread('path/to/slice.dcm')
plane = detect_orientation(dcm)
print(f"Scan plane: {plane}")

# Route series to correct model/preprocessing
series_planes = {}
for path in series_paths:
    dcm = pydicom.dcmread(path, stop_before_pixels=True)
    series_planes[dcm.SeriesInstanceUID] = detect_orientation(dcm)
```

## Workflow

1. Read `ImageOrientationPatient` from any DICOM slice in the series
2. Round direction cosines to nearest integer (handles slight off-axis tilts)
3. Compute cross product of row and column vectors → normal direction
4. The dominant axis of the normal identifies the scan plane
5. Use plane to route to the correct preprocessing or model

## Key Decisions

- **Rounding**: Necessary for oblique acquisitions that are nearly but not exactly on-axis
- **One slice per series**: All slices in a series share the same orientation — check only one
- **stop_before_pixels**: Use this flag when only reading metadata — avoids loading pixel data
- **Fallback**: If ImageOrientationPatient is missing, fall back to SeriesDescription text parsing

## References

- [RSNA Lumbar Spine Analysis](https://www.kaggle.com/code/satyaprakashshukl/rsna-lumbar-spine-analysis)
