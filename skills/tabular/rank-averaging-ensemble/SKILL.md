---
name: tabular-rank-averaging-ensemble
description: >
  Ensembles multiple model predictions by converting to ranks, averaging, and normalizing back to [0,1].
---
# Rank Averaging Ensemble

## Overview

Combine predictions from multiple models by converting each to ranks, averaging the ranks, then normalizing to [0,1]. This is robust to different prediction scales and distributions across models, making it ideal when blending heterogeneous model families.

## Quick Start

```python
import numpy as np
from scipy.stats import rankdata

def rank_average(predictions_list):
    """Ensemble by rank averaging.

    Args:
        predictions_list: list of arrays, each shape (n_samples,)

    Returns:
        array of shape (n_samples,) with averaged ranks normalized to [0,1]
    """
    ranks = np.column_stack([
        rankdata(preds) for preds in predictions_list
    ])
    avg_ranks = ranks.mean(axis=1)
    return (avg_ranks - avg_ranks.min()) / (avg_ranks.max() - avg_ranks.min())
```

## Workflow

1. Generate predictions from each model on the same test set
2. Convert each prediction vector to ranks via `scipy.stats.rankdata`
3. Average ranks across models (optionally with weights)
4. Normalize averaged ranks to [0, 1] via min-max scaling
5. Use as final submission or feed into calibration

## Key Decisions

- **Equal vs weighted**: Start equal; tune weights on CV if models differ in quality
- **Tie handling**: `rankdata` uses average ranks by default; fine for most cases
- **vs raw averaging**: Rank averaging wins when models have different output scales
- **Per-target**: For multi-output, apply rank averaging independently per target

## References

- Google QUEST Q&A Labeling competition (Kaggle)
- Source: [distilbert-use-features-oof](https://www.kaggle.com/code/abhishek/distilbert-use-features-oof)
