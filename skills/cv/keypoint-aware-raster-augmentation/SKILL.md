---
name: cv-keypoint-aware-raster-augmentation
description: Use albumentations keypoint_params to jointly augment BEV rasters and trajectory target points so the spatial transform stays consistent
---

## Overview

BEV rasters paired with trajectory targets (lists of `(x, y)` points) cannot be augmented with image-only transforms — a horizontal flip of the raster without flipping the trajectory produces a garbage training pair. Albumentations supports this natively via `KeypointParams`: pass the trajectory points as keypoints, and the library applies the exact same geometric transform to both the image and the points. This unlocks shift/scale/rotate/flip/cutout on motion-prediction data with zero custom code.

## Quick Start

```python
import albumentations as A
import numpy as np

tfms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                       rotate_limit=15, p=0.7),
    A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.3),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def augment(sample):
    image = sample['image'].transpose(1, 2, 0)            # (C,H,W) -> (H,W,C)
    kps = sample['target_positions'].tolist()             # [(x,y), ...]
    out = tfms(image=image, keypoints=kps)
    sample['image'] = out['image'].transpose(2, 0, 1)
    sample['target_positions'] = np.array(out['keypoints'], dtype=np.float32)
    return sample
```

## Workflow

1. Build an `A.Compose` pipeline with any mix of geometric and pixel transforms
2. Add `keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)` — `remove_invisible=False` is critical so flipped points outside the image aren't silently dropped
3. In the dataset's `__getitem__`, convert the raster from `(C,H,W)` to `(H,W,C)` and pass it with `keypoints=target_points`
4. Albumentations returns a dict with both transformed tensors
5. Transpose back and overwrite the sample — done

## Key Decisions

- **`remove_invisible=False`**: without this, horizontal flip drops negative-x points and desyncs lengths.
- **Match raster pixel space**: the keypoints must be in image pixel coordinates (post world-to-image transform), not world meters.
- **Rotation limit stays small**: BEV data encodes a canonical heading; rotating too far breaks the agent-forward assumption.
- **vs. manual flip/rotate**: hand-rolled augmentation drifts out of sync with the image transform whenever you add a new op. Keypoint-aware pipelines stay consistent by construction.

## References

- [Understanding the data + Catalyst/Kekas baseline](https://www.kaggle.com/code/pestipeti/understanding-the-data-catalyst-kekas-baseline)
