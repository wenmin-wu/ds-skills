---
name: cv-isotropic-resize-with-padding
description: Resize images preserving aspect ratio then zero-pad to a square to avoid distortion artifacts in face crops or object detection inputs
---

# Isotropic Resize with Padding

## Overview

Naively resizing a rectangular image to a square distorts aspect ratio, creating artifacts that confuse classifiers (especially for faces). Isotropic resize scales the image so the longer side matches the target size, then zero-pads the shorter side. This preserves proportions while producing a fixed-size square input for CNNs.

## Quick Start

```python
import cv2
import numpy as np

def isotropic_resize(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    # Zero-pad to square
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas

face_crop = isotropic_resize(face_crop, 224)
```

## Workflow

1. Compute the scaling factor from the longer side to the target size
2. Resize both dimensions by this factor (shorter side will be < target)
3. Create a zero-filled canvas of target size
4. Place the resized image in the top-left corner
5. Feed the padded square to the CNN

## Key Decisions

- **Padding position**: top-left is simplest; center-padding is slightly better for some models
- **Fill value**: zero (black) is standard; mean pixel value (ImageNet mean) reduces distribution shift
- **Interpolation**: `INTER_AREA` for downsampling (anti-aliased), `INTER_LINEAR` for upsampling
- **vs. letterboxing**: same concept — isotropic resize is letterboxing for square targets
- **vs. center crop**: cropping loses content; padding preserves everything at the cost of wasted pixels

## References

- [Inference Demo](https://www.kaggle.com/code/humananalog/inference-demo)
