---
name: cv-slice-padding-augmented-duplicates
description: Pad 3D volumes with fewer slices than required by duplicating existing slices with slight brightness variation via convertScaleAbs
---

# Slice Padding with Augmented Duplicates

## Overview

When a 3D volume has fewer slices than the target depth, zero-padding wastes model capacity on empty data. Instead, pad by duplicating randomly chosen existing slices with slight brightness augmentation (`cv2.convertScaleAbs(alpha=1.2)`). This fills the depth dimension with plausible content and acts as a mild data augmentation, better than both zero-padding and plain duplication.

## Quick Start

```python
import cv2
import random
import numpy as np

def pad_volume_with_augmented_slices(slices, target_depth, alpha=1.2, beta=0):
    """Pad a list of slices to target_depth by duplicating with brightness jitter."""
    while len(slices) < target_depth and slices:
        donor = random.choice(slices)
        augmented = cv2.convertScaleAbs(donor, alpha=alpha, beta=beta)
        slices.append(augmented)
    return slices[:target_depth]

flair_slices = [cv2.imread(p, 0) for p in flair_paths]
flair_slices = pad_volume_with_augmented_slices(flair_slices, target_depth=64)
volume = np.stack(flair_slices)
```

## Workflow

1. Load all available slices for the volume
2. If count < target depth, randomly select an existing slice
3. Apply `cv2.convertScaleAbs(alpha, beta)` for slight brightness change
4. Append the augmented copy
5. Repeat until target depth is reached

## Key Decisions

- **alpha**: 1.1-1.3 provides subtle variation; higher values distort tissue contrast
- **Random vs sequential**: random selection provides more diversity than repeating the last slice
- **vs zero-pad**: augmented duplicates provide real texture; zeros create sharp boundary artifacts
- **vs interpolation**: duplication is simpler and preserves original slice quality

## References

- [[TF]: 3D & 2D Model for Brain Tumor Classification](https://www.kaggle.com/code/ipythonx/tf-3d-2d-model-for-brain-tumor-classification)
