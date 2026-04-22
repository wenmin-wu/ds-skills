---
name: cv-per-class-score-threshold
description: Apply class-specific confidence thresholds by inferring the dominant class per image and indexing into a per-class threshold array
---

# Per-Class Score Threshold

## Overview

Different object classes have different score distributions — small dense cells score lower than large isolated ones. A single global threshold under-filters easy classes and over-filters hard ones. Infer the dominant class per image (mode of predicted classes), then apply that class's optimized threshold. Common in cell segmentation where cell types have very different morphologies.

## Quick Start

```python
import torch
import numpy as np

THRESHOLDS = [0.15, 0.35, 0.55]  # per class, tuned on validation
MIN_PIXELS = [75, 150, 75]        # per class minimum area

def filter_predictions(predictions):
    scores = predictions['scores']
    classes = predictions['pred_classes']
    masks = predictions['pred_masks']

    # Infer dominant class for this image
    dominant_class = torch.mode(classes)[0].item()

    # Apply class-specific threshold
    keep = scores >= THRESHOLDS[dominant_class]
    return masks[keep], scores[keep], dominant_class

masks, scores, cls = filter_predictions(output['instances'])
```

## Workflow

1. Run inference to get per-instance scores, classes, and masks
2. Compute the mode of predicted classes to determine dominant image class
3. Index into per-class threshold and min-area arrays
4. Filter predictions by the class-specific threshold
5. Optionally apply class-specific min-area filtering post-overlap-resolution

## Key Decisions

- **Mode vs majority**: `torch.mode` is simple; for mixed-class images, consider per-instance thresholds instead
- **Threshold tuning**: sweep thresholds per class on validation set, optimize for mAP@IoU
- **Homogeneous assumption**: works best when images contain mostly one class; for mixed scenes, apply per-instance class thresholds
- **Min-area coupling**: pair with per-class min_pixels to also filter by class-specific size

## References

- [Positive score with Detectron 3/3 - Inference](https://www.kaggle.com/code/slawekbiel/positive-score-with-detectron-3-3-inference)
