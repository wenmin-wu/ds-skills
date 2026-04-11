---
name: tabular-per-fold-threshold-voting-ensemble
description: >
  Binarizes each CV fold's predictions using its own optimized threshold, then majority-votes across folds instead of averaging raw probabilities.
---
# Per-Fold Threshold Voting Ensemble

## Overview

Standard ensembling averages probabilities across folds, then applies a single global threshold. This assumes all folds are calibrated equally — which they often aren't, especially with class imbalance or varying fold distributions. Per-fold threshold voting first optimizes a binary threshold per fold (e.g., via F1 or IoU on validation), binarizes each fold's predictions independently, then takes a majority vote. This is more robust to miscalibrated folds.

## Quick Start

```python
import numpy as np

def per_fold_vote(fold_preds, fold_thresholds):
    """Majority vote after per-fold binarization.

    Args:
        fold_preds: list of arrays, each (N,) with probabilities
        fold_thresholds: list of floats, one threshold per fold
    Returns:
        (N,) binary predictions
    """
    binary_preds = [
        (pred >= thresh).astype(int)
        for pred, thresh in zip(fold_preds, fold_thresholds)
    ]
    # Majority vote: >= 0.5 of folds predict positive
    votes = np.mean(binary_preds, axis=0)
    return (votes >= 0.5).astype(int)

# Thresholds tuned per fold during CV
thresholds = [0.42, 0.45, 0.38, 0.44, 0.41]
final_preds = per_fold_vote(test_fold_preds, thresholds)
```

## Workflow

1. During CV, optimize a threshold per fold on its validation set (maximize F1, IoU, etc.)
2. Save per-fold thresholds alongside fold models
3. At inference, generate predictions from each fold model
4. Binarize each fold's predictions using its own threshold
5. Majority vote across folds for the final prediction

## Key Decisions

- **Threshold tuning metric**: F1 for balanced, IoU for entity matching, custom for competition metrics
- **Vote threshold**: 0.5 (majority) is standard; lower for higher recall
- **vs probability averaging**: Voting is more robust when folds have different calibration
- **Soft alternative**: Weight votes by fold validation performance

## References

- [Public: 0.861 | PyKakasi & Radian Coordinates](https://www.kaggle.com/code/nlztrk/public-0-861-pykakasi-radian-coordinates)
