---
name: cv-microscope-circular-mask-aug
description: Mask the corners of a dermoscopy image with a random-radius black circle to mimic the dark vignette of a dermatoscope field of view
---

## Overview

About half the dermoscopy images in ISIC datasets have the characteristic dark circular vignette of a dermatoscope; the other half are cropped rectangles. Models pick up the vignette as a shortcut feature correlated with source site, which breaks cross-site generalization. The fix is a symmetric augmentation: apply a random-radius circular mask to train images that don't have one, so the model learns to ignore the vignette. Cheap, interpretable, and directly closes a train/test distribution gap.

## Quick Start

```python
import cv2
import numpy as np
import random

class MicroscopeMask:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        h, w = img.shape[:2]
        circle = (np.ones(img.shape) * 255).astype(np.uint8)
        radius = random.randint(h // 2 - 3, h // 2 + 15)
        circle = cv2.circle(circle, (w // 2, h // 2), radius, (0, 0, 0), -1)
        mask = circle - 255                      # 0 inside circle, -255 outside
        return np.multiply(img, mask)            # zeros outside the circle
```

## Workflow

1. Place the aug in the train transform pipeline before normalization
2. Randomize the radius within a narrow band around `img_size / 2` — keeps the lesion visible while varying the vignette
3. Apply with `p=0.5` since roughly half the images already have a vignette
4. Validate by plotting a grid of augmented samples before kicking off training
5. Track val AUC across sites — this aug specifically lifts underrepresented sites

## Key Decisions

- **Hard black mask, not soft**: dermatoscope vignettes are hard-edged; matching that edge is important for the model to learn the boundary.
- **Random radius**: a fixed radius teaches the model to rely on that exact edge; randomizing forces invariance.
- **Apply before normalization**: normalization expects full-range pixel values; the black mask would shift distribution means if applied after.
- **vs. removing vignettes**: cropping to an inscribed square loses lesion context and shrinks effective resolution.

## References

- [Melanoma. Pytorch starter. EfficientNet](https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet)
