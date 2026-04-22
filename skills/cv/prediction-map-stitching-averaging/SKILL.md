---
name: cv-prediction-map-stitching-averaging
description: Stitch overlapping tile predictions into a full-resolution output by accumulating probabilities and dividing by per-pixel overlap counts
---

# Prediction Map Stitching with Averaging

## Overview

When running patch-based inference on large images, overlapping tiles produce multiple predictions for each pixel. Instead of taking the last prediction, accumulate all predictions into a sum array and maintain a parallel count array. Dividing sum by count produces a smooth, averaged prediction map that reduces tile-boundary artifacts.

## Quick Start

```python
import numpy as np
import torch

def stitch_predictions(predictions, xyxys, output_shape):
    pred_map = np.zeros(output_shape, dtype=np.float32)
    count_map = np.zeros(output_shape, dtype=np.float32)

    for pred, (x1, y1, x2, y2) in zip(predictions, xyxys):
        pred_map[y1:y2, x1:x2] += pred.squeeze()
        count_map[y1:y2, x1:x2] += 1.0

    count_map = np.maximum(count_map, 1.0)
    return pred_map / count_map

# During inference loop:
all_preds, all_xyxys = [], []
for images, coords in test_loader:
    with torch.no_grad():
        preds = torch.sigmoid(model(images.cuda())).cpu().numpy()
    all_preds.extend(preds)
    all_xyxys.extend(coords)

result = stitch_predictions(all_preds, all_xyxys, (H, W))
```

## Workflow

1. Extract overlapping tiles with stride < tile_size (e.g., stride=tile_size//2)
2. Run model inference on each tile batch
3. Accumulate sigmoid outputs into a prediction sum array
4. Track overlap counts per pixel
5. Divide sum by count for final averaged prediction

## Key Decisions

- **Stride**: smaller stride = more overlap = smoother boundaries but slower inference
- **Float accumulation**: use float32 for sum array to avoid precision loss
- **Count floor**: clip count_map minimum to 1.0 to avoid division by zero at edges
- **vs max stitching**: averaging is smoother; max preserves high-confidence detections

## References

- [2.5d segmentaion baseline [inference]](https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-inference)
- [2.5d segmentaion baseline [training]](https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-training)
