---
name: nlp-identity-weighted-bce-loss
description: Weight BCE loss by identity subgroup membership to debias predictions — upweight samples where identity conflicts with label
domain: nlp
---

# Identity-Weighted BCE Loss

## Overview

Toxicity classifiers often learn spurious correlations between identity terms ("gay", "muslim") and toxicity. Fix this by upweighting four sample categories: (1) any identity-mentioned sample, (2) background-positive/subgroup-negative (toxic text without identity), (3) background-negative/subgroup-positive (non-toxic text with identity). Each category adds 0.25 to the sample weight, so hard fairness cases get up to 2x weight.

## Quick Start

```python
import numpy as np
import torch.nn as nn

def compute_identity_weights(df, identity_cols, target_col, threshold=0.5):
    """Compute per-sample weights for bias-aware training.
    
    Args:
        df: DataFrame with identity columns and target
        identity_cols: list of identity attribute columns
        target_col: toxicity target column
        threshold: binarization threshold
    """
    weights = np.ones(len(df)) / 4
    has_identity = (df[identity_cols].fillna(0).values >= threshold).any(axis=1)
    is_toxic = df[target_col].values >= threshold
    
    weights += has_identity.astype(float) / 4           # subgroup
    weights += (is_toxic & ~has_identity) / 4            # BPSN
    weights += (~is_toxic & has_identity) / 4            # BNSP
    return weights

# Custom loss with sample weights
def weighted_bce(pred, target_with_weights):
    target = target_with_weights[:, 0:1]
    weight = target_with_weights[:, 1:2]
    return nn.BCEWithLogitsLoss(weight=weight)(pred, target)

weights = compute_identity_weights(train, IDENTITY_COLS, 'target')
```

## Key Decisions

- **Equal 0.25 splits**: each bias category contributes equally; tune if one matters more
- **Threshold 0.5**: binarizes soft identity labels; lower for more aggressive debiasing
- **Combine with auxiliary targets**: pair with multi-task heads on identity subtypes for best results
- **Normalize weights**: divide by mean to keep effective batch size stable

## References

- Source: [simple-lstm-with-identity-parameters-fastai](https://www.kaggle.com/code/kunwar31/simple-lstm-with-identity-parameters-fastai)
- Competition: Jigsaw Unintended Bias in Toxicity Classification
