---
name: cv-class-density-threshold-patch-sampling
description: Sample training patches from large images using per-class area-fraction thresholds to ensure each patch contains meaningful object coverage
---

# Class-Density Threshold Patch Sampling

## Overview

Large satellite or histopathology images are too big to feed directly into a CNN. Random cropping produces mostly empty patches for rare classes. Instead, set a minimum area-fraction threshold per class — a patch is accepted only if at least one class exceeds its threshold. Rare classes get lower thresholds (0.1%) to capture sparse features; common classes get higher thresholds (40%) to filter noise.

## Quick Start

```python
import numpy as np
import random

def sample_patches(image, mask, patch_size, n_patches, thresholds):
    h, w = image.shape[:2]
    patches_img, patches_msk = [], []
    area = patch_size * patch_size
    attempts = 0
    while len(patches_img) < n_patches and attempts < n_patches * 20:
        x = random.randint(0, h - patch_size)
        y = random.randint(0, w - patch_size)
        msk_patch = mask[x:x+patch_size, y:y+patch_size]
        for cls_idx, thresh in enumerate(thresholds):
            if msk_patch[:, :, cls_idx].sum() / area > thresh:
                patches_img.append(image[x:x+patch_size, y:y+patch_size])
                patches_msk.append(msk_patch)
                break
        attempts += 1
    return patches_img, patches_msk

thresholds = [0.4, 0.1, 0.1, 0.15, 0.3, 0.05, 0.1, 0.05, 0.001, 0.005]
imgs, msks = sample_patches(image, mask, 256, 5000, thresholds)
```

## Workflow

1. Define per-class area-fraction thresholds based on class frequency
2. Randomly crop a candidate patch from the full image
3. Check if any class exceeds its threshold in the patch
4. Accept the patch if yes, reject and retry if no
5. Apply augmentation to accepted patches before saving

## Key Decisions

- **Threshold per class**: rare classes need low thresholds (0.001); dominant classes need higher (0.3-0.5)
- **Max attempts**: cap retries to avoid infinite loops on empty regions
- **vs weighted sampling**: density thresholds are simpler and guarantee minimum object coverage
- **Patch size**: must be large enough to capture spatial context (128-512 typical for satellite)

## References

- [SegNet_dstl](https://www.kaggle.com/code/ksishawon/segnet-dstl)
