---
name: timeseries-unknown-class-residual-probability
description: Estimate out-of-distribution class probability as the product of (1 - p_i) across all known classes, scaled by a calibrated prior
domain: timeseries
---

# Unknown Class Residual Probability

## Overview

When test data may contain an unseen class not present in training (open-set classification), estimate its probability using the "residual" approach: multiply (1 - p_i) across all known classes. If the model is confident in no known class, the product is high — indicating a likely OOD sample. Scale by a prior reflecting the expected OOD fraction, then normalize by the mean to calibrate.

## Quick Start

```python
import numpy as np

def add_unknown_class_prob(preds, prior=0.14):
    """Add an unknown-class probability column to predictions.
    
    Args:
        preds: (n_samples, n_classes) array of known-class probabilities
        prior: expected fraction of OOD samples (tune on validation)
    Returns:
        (n_samples, n_classes+1) array with unknown class appended
    """
    # Product of complements — high when no known class is confident
    ood_score = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        ood_score *= (1 - preds[:, i])
    
    # Scale by prior and normalize
    ood_prob = prior * ood_score / np.mean(ood_score)
    
    # Append and renormalize
    full_preds = np.column_stack([preds, ood_prob])
    full_preds /= full_preds.sum(axis=1, keepdims=True)
    return full_preds
```

## Key Decisions

- **prior=0.14**: fraction of OOD samples expected; tune on leaderboard or validation
- **Product not max**: product captures "confident in nothing"; max(1-p) misses partial confidence
- **Normalize by mean**: prevents the OOD column from dominating or vanishing
- **Renormalize rows**: ensures probabilities sum to 1 after adding the new column

## References

- Source: [simple-neural-net-for-time-series-classification](https://www.kaggle.com/code/meaninglesslives/simple-neural-net-for-time-series-classification)
- Competition: PLAsTiCC Astronomical Classification
