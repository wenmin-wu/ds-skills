---
name: tabular-distribution-matching-postprocess
description: >
  Reshapes model predictions to match the known label distribution from training data using rank-based mapping.
---
# Distribution Matching Post-Processing

## Overview

When the target distribution is known (e.g., from training labels), reshape raw predictions to match it. Convert predictions to ranks, map rank positions to the corresponding quantiles of the target distribution. This corrects systematic distribution shifts without retraining.

## Quick Start

```python
import numpy as np
from scipy.stats import rankdata

def distribution_match(preds, target_distribution):
    """Map predictions to match target label distribution.

    Args:
        preds: array of raw predictions, shape (n_samples,)
        target_distribution: sorted array of target values to match

    Returns:
        array with predictions reshaped to target distribution
    """
    ranks = rankdata(preds, method="ordinal") - 1
    n_preds = len(preds)
    n_target = len(target_distribution)
    indices = (ranks * (n_target - 1)) // (n_preds - 1)
    return target_distribution[indices.astype(int)]
```

## Workflow

1. Collect training label distribution (sorted values or histogram)
2. Generate raw model predictions on test set
3. Rank test predictions
4. Map each rank position to the corresponding quantile in training distribution
5. Output matched predictions as final submission

## Key Decisions

- **Per-target matching**: Apply independently per target column in multi-output tasks
- **When to use**: Best when evaluation metric is rank-based (Spearman) or bounded
- **Granularity**: With few unique target values, use histogram bins instead of raw values
- **Combination**: Stack after rank averaging for additional calibration

## References

- Google QUEST Q&A Labeling competition, 1st place solution (Kaggle)
- Source: [1st-place-solution](https://www.kaggle.com/code/ddanevskyi/1st-place-solution)
