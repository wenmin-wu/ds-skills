---
name: tabular-per-feature-bias-correction
description: Post-processing correction for multi-output regression — scale each output by its train-derived mean ratio to fix systematic per-feature bias
domain: tabular
---

# Per-Feature Bias Correction

## Overview

Multi-output models often have systematic per-feature biases (some outputs consistently over/under-predicted). Compute the mean ratio of true labels to OOF predictions per output on training data, then apply as a multiplicative correction to test predictions. Clip extreme corrections to avoid amplifying noise.

## Quick Start

```python
import numpy as np

def compute_bias_scales(y_true, y_pred, clip_range=(0.99, 1.01)):
    """Compute per-feature correction scales from OOF predictions.
    
    Args:
        y_true: (N, K) ground truth
        y_pred: (N, K) OOF predictions
        clip_range: (lo, hi) to prevent extreme corrections
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        scales = np.nanmean(y_true / y_pred, axis=0)
    scales = np.clip(scales, *clip_range)
    scales[np.isnan(scales)] = 1.0
    return scales

def apply_bias_correction(predictions, scales):
    return predictions * scales[np.newaxis, :]

# Usage
scales = compute_bias_scales(train_labels, oof_predictions)
test_corrected = apply_bias_correction(test_predictions, scales)
```

## Key Decisions

- **Multiplicative correction**: preserves relative scale within each feature
- **Clip to ±1%**: prevents large corrections from amplifying noise
- **OOF-based**: compute on out-of-fold predictions to avoid data leakage
- **Per-feature independent**: each output corrected separately

## References

- Source: [neurips-scale-sigmas](https://www.kaggle.com/code/gromml/neurips-scale-sigmas)
- Competition: NeurIPS - Ariel Data Challenge 2024
