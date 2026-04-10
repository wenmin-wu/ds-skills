---
name: cv-dicom-voi-lut-preprocessing
description: Read DICOM X-ray files with VOI LUT transformation and MONOCHROME1 inversion for correct pixel intensity rendering
domain: cv
---

# DICOM VOI LUT Preprocessing

## Overview

X-ray DICOM files use Value of Interest (VOI) Lookup Tables to map stored pixel values to display-ready intensities. Some scanners store images as MONOCHROME1 (inverted — bright=air, dark=tissue), which must be flipped. Apply VOI LUT first, then fix photometric interpretation, then normalize to 0-255.

## Quick Start

```python
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut=True, fix_monochrome=True):
    """Read and preprocess a DICOM X-ray image.
    
    Args:
        path: path to .dcm file
        voi_lut: apply VOI LUT transform (recommended)
        fix_monochrome: invert MONOCHROME1 images
    Returns:
        uint8 image array (0-255)
    """
    dicom = pydicom.dcmread(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

img = read_xray("patient001.dcm")
```

## Key Decisions

- **VOI LUT vs raw**: VOI LUT applies manufacturer-calibrated display mapping; skip only for custom windowing
- **MONOCHROME1 inversion**: without this fix, lung appears white and bone black — confuses models
- **Min-max normalization**: handles variable bit depths (12-bit, 16-bit) across scanners
- **uint8 output**: compatible with standard CV pipelines and augmentation libraries

## References

- Source: [siim-cov19-efnb7-yolov5-infer](https://www.kaggle.com/code/h053473666/siim-cov19-efnb7-yolov5-infer)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
