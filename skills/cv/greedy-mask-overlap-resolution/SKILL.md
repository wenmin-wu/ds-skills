---
name: cv-greedy-mask-overlap-resolution
description: Resolve overlapping instance masks by greedily assigning contested pixels to higher-confidence predictions using a running occupancy map
---

# Greedy Mask Overlap Resolution

## Overview

Instance segmentation models often produce overlapping masks, but many evaluation metrics (and biological reality) require non-overlapping instances. This technique processes masks in descending confidence order and subtracts already-claimed pixels from each new mask. Simple, fast, and effective — commonly used in cell segmentation where dense packing makes overlap inevitable.

## Quick Start

```python
import numpy as np

def resolve_overlaps(masks, scores, min_pixels=75):
    order = np.argsort(-scores)
    used = np.zeros(masks[0].shape, dtype=np.uint8)
    result = []
    for idx in order:
        mask = (masks[idx] > 0).astype(np.uint8)
        mask = mask * (1 - used)  # remove already-claimed pixels
        if mask.sum() >= min_pixels:
            used = np.clip(used + mask, 0, 1)
            result.append(mask)
    return result

# Usage with Mask R-CNN output
masks = output['masks'].cpu().numpy()  # (N, H, W)
scores = output['scores'].cpu().numpy()
clean_masks = resolve_overlaps(masks, scores, min_pixels=75)
```

## Workflow

1. Sort all predicted masks by confidence score descending
2. Initialize an empty occupancy map (zeros, same H×W as image)
3. For each mask: subtract occupied pixels, check remaining area ≥ min_pixels
4. If large enough, add to result and update occupancy map
5. Encode surviving masks as RLE for submission

## Key Decisions

- **Sort by confidence**: highest-confidence masks get priority for contested pixels
- **min_pixels threshold**: discard masks that become too small after overlap removal; tune per dataset (75-150 for cells)
- **Per-class min_pixels**: use different thresholds per class if instance sizes vary significantly
- **Binary occupancy**: simple and fast; for soft overlap, use IoU-based merging instead

## References

- [Sartorius - Starter Torch Mask R-CNN](https://www.kaggle.com/code/julian3833/sartorius-starter-torch-mask-r-cnn-lb-0-273)
- [Positive score with Detectron 3/3 - Inference](https://www.kaggle.com/code/slawekbiel/positive-score-with-detectron-3-3-inference)
