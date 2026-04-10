---
name: cv-hungarian-matching-detection-eval
description: >
  Evaluates 3D object detection by matching predicted and ground-truth coordinates via the Hungarian algorithm, then computing F-beta score.
---
# Hungarian Matching Detection Evaluation

## Overview

Standard IoU-based metrics don't work for point-based 3D detections (particle picking, cell centers). Instead, compute a distance matrix between predicted and ground-truth coordinates, solve the optimal 1-to-1 assignment with the Hungarian algorithm, then count matches within a distance threshold to compute precision, recall, and F-beta.

## Quick Start

```python
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def fbeta_score(pred_coords, gt_coords, threshold=10.0, beta=4.0):
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 1.0
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0
    dist = cdist(pred_coords, gt_coords)
    row_ind, col_ind = linear_sum_assignment(dist)
    tp = sum(dist[r, c] <= threshold for r, c in zip(row_ind, col_ind))
    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    denom = (beta**2 * precision + recall)
    return (1 + beta**2) * precision * recall / denom if denom > 0 else 0
```

## Workflow

1. Collect predicted and ground-truth 3D coordinate arrays
2. Build pairwise Euclidean distance matrix via `cdist`
3. Solve assignment with `linear_sum_assignment` (Hungarian)
4. Apply distance threshold to classify matches as TP or FP
5. Compute F-beta (beta > 1 weights recall more than precision)

## Key Decisions

- **Distance threshold**: Domain-specific (e.g., particle radius in angstroms)
- **Beta value**: Beta=4 heavily penalizes missed detections; beta=1 balances equally
- **Scaling**: Multiply voxel coordinates by voxel spacing for physical-unit distances
- **Large N**: Hungarian is O(n³) — for >10k detections, consider greedy matching

## References

- [Baseline UNet train + submit](https://www.kaggle.com/code/fnands/baseline-unet-train-submit)
- [3d-unet using 2d image encoder](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder)
