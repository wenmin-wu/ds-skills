---
name: cv-point-centered-fixed-patch-cropping
description: Convert (x, y, class) point annotations into a CNN classification training set by cropping fixed-size square patches centered on each point, using a numpy shape check to silently reject border-clipped crops
---

## Overview

Once you have point annotations (from `dot-annotation-blob-diff-extraction` or similar), the cleanest way to build a classification training set is to crop a fixed-size square patch centered on each point. The one gotcha: numpy slicing silently returns smaller arrays when the slice runs off the image edge, so `img[y-h:y+h, x-h:x+h]` yields a `(PATCH-k, PATCH-k, 3)` patch near borders without throwing. Check `thumb.shape == (PATCH, PATCH, 3)` before appending — it doubles as a border filter that avoids fabricating padded context.

## Quick Start

```python
import cv2
import numpy as np

PATCH = 32
half = PATCH // 2

X, y = [], []
for fname in file_names:
    img = cv2.imread(train_dir + fname)
    for cls_idx, cls in enumerate(classes):
        for (cx, cy) in coords[cls][fname]:
            thumb = img[cy - half:cy + half, cx - half:cx + half, :]
            if thumb.shape == (PATCH, PATCH, 3):    # reject border clips
                X.append(thumb)
                y.append(cls_idx)

X = np.stack(X); y = np.array(y)
```

## Workflow

1. Pick `PATCH` to match the typical object diameter (32 for small sea lions, 64-128 for mid-size targets)
2. For every annotated `(x, y, class)` point, slice `img[y-half:y+half, x-half:x+half, :]`
3. Validate `thumb.shape == (PATCH, PATCH, 3)` — the check filters out any point within `half` pixels of the border
4. Append valid thumbnails and labels into parallel lists
5. `np.stack` at the end into a contiguous training tensor

## Key Decisions

- **Shape equality check over padding**: padding fabricates context that wasn't in the data; shape-check just drops the handful of border points, which is usually harmless.
- **Patch size ≈ object diameter**: larger patches mostly add background and slow training; smaller patches crop the object.
- **Stack once at the end**: appending numpy arrays inside the loop is O(N²); lists + single `np.stack` is O(N).
- **Keep labels as integer class indices**, not one-hot — Keras `sparse_categorical_crossentropy` eats them directly and saves memory on large N.

## References

- [Use keras to classify Sea Lions: 0.91 accuracy](https://www.kaggle.com/code/outrunner/use-keras-to-classify-sea-lions-0-91-accuracy)
