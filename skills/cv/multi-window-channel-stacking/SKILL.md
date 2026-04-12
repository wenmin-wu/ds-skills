---
name: cv-multi-window-channel-stacking
description: Stack multiple CT window settings (brain, subdural, bone) as separate RGB channels for CNN input
---

## Overview

CT scans contain a wide range of Hounsfield Unit (HU) values. A single window clips most diagnostic detail. Stacking three clinically-relevant windows — brain (W:80, L:40), subdural (W:200, L:80), and soft tissue (W:380, L:40) — as RGB channels preserves all three ranges in one image that standard ImageNet-pretrained CNNs can consume directly.

## Quick Start

```python
import pydicom
import numpy as np

def window_image(img, center, width, intercept, slope):
    img = img * slope + intercept
    img_min = center - width // 2
    img_max = center + width // 2
    return np.clip(img, img_min, img_max)

def multi_window_rgb(dcm):
    px = dcm.pixel_array.astype(np.float32)
    intercept, slope = float(dcm.RescaleIntercept), float(dcm.RescaleSlope)
    brain = window_image(px, 40, 80, intercept, slope)
    subdural = window_image(px, 80, 200, intercept, slope)
    soft = window_image(px, 40, 380, intercept, slope)
    # Normalize each channel to [0, 1]
    brain = (brain - (40 - 40)) / 80
    subdural = (subdural - (80 - 100)) / 200
    soft = (soft - (40 - 190)) / 380
    return np.stack([brain, subdural, soft], axis=-1)
```

## Workflow

1. Read DICOM and extract RescaleIntercept / RescaleSlope
2. Apply three different window center/width pairs to raw pixel array
3. Normalize each windowed image to [0, 1]
4. Stack as 3-channel (H, W, 3) array — feed directly to pretrained CNN

## Key Decisions

- **Window choice**: Brain (W:80 L:40) for parenchyma, Subdural (W:200 L:80) for extra-axial blood, Bone (W:380 L:40) for skull fractures. Adjust for task.
- **Normalization**: Divide by window width after shifting to zero-base. Keeps channels in [0, 1] range.
- **vs. single window**: Single window loses information outside its range. Multi-window retains three diagnostic views simultaneously.

## References

- [RSNA InceptionV3 Keras](https://www.kaggle.com/code/akensert/rsna-inceptionv3-keras-tf1-14-0)
