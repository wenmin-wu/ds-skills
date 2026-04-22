---
name: cv-coverage-stratified-split
description: Stratify train/validation split by binned mask coverage percentage to ensure balanced foreground representation in segmentation tasks
---

# Coverage-Stratified Split

## Overview

In segmentation tasks, naive random splits can produce folds with unbalanced foreground/background ratios — some folds get mostly empty masks, others get mostly full masks. Compute per-image mask coverage (foreground pixel ratio), bin into discrete classes, and use stratified splitting on these bins. This ensures each fold sees the full range of mask densities.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split

coverage = masks.sum(axis=(1, 2)) / (masks.shape[1] * masks.shape[2])

def coverage_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i
    return 10

coverage_classes = np.array([coverage_to_class(c) for c in coverage])

X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2,
    stratify=coverage_classes, random_state=42
)
```

## Workflow

1. Compute mask coverage ratio for each training image (sum of foreground pixels / total pixels)
2. Bin coverage into discrete classes (e.g., 0-10% → class 0, 10-20% → class 1, ...)
3. Use binned classes as `stratify` parameter in `train_test_split` or `StratifiedKFold`
4. Validate that each fold has similar coverage distribution

## Key Decisions

- **10 bins**: covers 0-100% in 10% increments — fine enough for most tasks
- **Empty mask handling**: images with 0% coverage form their own bin, preventing empty-mask imbalance
- **vs random split**: critical when dataset has skewed coverage distribution (many empty masks)
- **With KFold**: use `StratifiedKFold(n_splits=5).split(X, coverage_classes)` for cross-validation

## References

- [U-net, dropout, augmentation, stratification](https://www.kaggle.com/code/phoenigs/u-net-dropout-augmentation-stratification)
- [U-net with simple ResNet Blocks v2 (New loss)](https://www.kaggle.com/code/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss)
