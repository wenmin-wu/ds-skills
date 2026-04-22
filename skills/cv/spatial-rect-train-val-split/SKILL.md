---
name: cv-spatial-rect-train-val-split
description: Split large-image segmentation data into train/val by spatial rectangle regions with border buffer exclusion to prevent patch leakage
---

# Spatial Rectangle Train-Val Split

## Overview

For large-image segmentation (satellite, medical, scroll fragments), random pixel splits leak spatial context between train and val. Instead, designate a rectangular region as validation and use everything else for training. Exclude a buffer zone around the rectangle boundary to prevent overlap between train and val patches.

## Quick Start

```python
import numpy as np

def spatial_split(mask, val_rect, buffer=32):
    """Split pixels into train/val by spatial region.
    val_rect: (x, y, width, height)
    """
    x, y, w, h = val_rect
    valid = np.zeros_like(mask, dtype=bool)
    valid[buffer:mask.shape[0]-buffer, buffer:mask.shape[1]-buffer] = True
    valid &= mask.astype(bool)

    val_region = np.zeros_like(mask, dtype=bool)
    val_region[y:y+h, x:x+w] = True

    val_pixels = np.argwhere(valid & val_region)
    train_pixels = np.argwhere(valid & ~val_region)
    return train_pixels, val_pixels

train_px, val_px = spatial_split(label_mask, val_rect=(1100, 3500, 700, 950))
```

## Workflow

1. Define a rectangular validation region based on visual inspection or coverage analysis
2. Create a border exclusion mask (buffer pixels from image edges)
3. Intersect with the label mask to get valid pixels
4. Split: pixels inside rectangle → val, outside → train
5. Use pixel coordinates to index into the volume for patch extraction

## Key Decisions

- **Buffer size**: should be ≥ half the patch size to prevent train/val patch overlap
- **Rectangle selection**: choose a region with representative label distribution
- **Multiple rectangles**: for k-fold, define k non-overlapping rectangles
- **vs random split**: spatial split prevents the model from memorizing local texture patterns

## References

- [Vesuvius Challenge: Ink Detection tutorial](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial)
