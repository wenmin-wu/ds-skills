---
name: cv-dicom-12bit-pixel-correction
description: Fix pixel overflow artifacts in 12-bit unsigned DICOM files where values wrap around at 4096
---

## Overview

Some DICOM files store pixel data as 12-bit unsigned integers (BitsStored=12, PixelRepresentation=0) but report a RescaleIntercept near 0 instead of -1024. This causes pixel values to wrap around at 4096, producing artifacts. Detecting and correcting this before HU conversion prevents corrupted images from entering the training pipeline.

## Quick Start

```python
import pydicom
import numpy as np

def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def safe_pixel_array(dcm):
    if (dcm.BitsStored == 12
        and dcm.PixelRepresentation == 0
        and int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    return dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
```

## Workflow

1. Check DICOM metadata: BitsStored, PixelRepresentation, RescaleIntercept
2. If 12-bit unsigned with suspiciously high intercept (> -100), apply correction
3. Shift pixels by +1000, wrap values >= 4096 back by subtracting 4096
4. Update RescaleIntercept to -1000 and rewrite PixelData
5. Proceed with normal HU conversion using corrected values

## Key Decisions

- **Detection condition**: BitsStored==12 AND PixelRepresentation==0 AND RescaleIntercept > -100. This catches the specific encoder bug without affecting normal files.
- **Shift value**: +1000 followed by modulo 4096 unwraps the overflow. The new intercept (-1000) produces correct HU values.
- **In-place vs copy**: Modifying dcm.PixelData in-place is faster but mutates the object. Clone first if you need the original.

## References

- [RSNA InceptionV3 Keras](https://www.kaggle.com/code/akensert/rsna-inceptionv3-keras-tf1-14-0)
