---
name: cv-test-time-augmentation
description: >
  Applies geometric and color augmentations at inference time and averages predictions to reduce variance.
---
# Test-Time Augmentation (TTA)

## Overview

Generate multiple augmented copies of each input image at inference (flips, 90-degree rotations, brightness/contrast jitter), run the model on all copies, then average the predictions. Reduces prediction variance and improves robustness without retraining.

## Quick Start

```python
import numpy as np

def tta_predict(model, image, n_augments=8):
    preds = []
    for _ in range(n_augments):
        aug = apply_random_augment(image)  # flip, rotate, color jitter
        preds.append(model.predict(aug[np.newaxis]))
    return np.mean(preds, axis=0)

def apply_random_augment(img):
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.flipud(img)
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    return img
```

## Workflow

1. Define augmentation set (flips, rotations, color transforms)
2. For each test image, generate N augmented copies
3. Run model inference on all copies
4. Average (classification) or geometrically reverse + average (segmentation) predictions
5. Use averaged prediction as final output

## Key Decisions

- **N augments**: 4-8 is typical; diminishing returns beyond 16
- **Augment types**: Match training augmentations for best effect
- **Segmentation TTA**: Must reverse spatial transforms before averaging masks
- **Cost**: Inference time scales linearly with N; batch augmented copies for GPU efficiency

## References

- HPA Single Cell Classification competition (Kaggle)
- Source: [hpa-cellwise-classification-inference](https://www.kaggle.com/code/dschettler8845/hpa-cellwise-classification-inference)
