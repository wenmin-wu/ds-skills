---
name: cv-per-modality-ensemble-averaging
description: Train separate models per imaging modality (FLAIR/T1w/T1wCE/T2w) and average their predictions for final ensemble
---

# Per-Modality Ensemble Averaging

## Overview

Multi-modal medical imaging (MRI sequences, CT windows) captures complementary information. Rather than concatenating modalities into a single multi-channel input, train an independent model per modality and average their sigmoid/softmax outputs at inference. This avoids missing-modality issues and lets each model specialize, often outperforming multi-channel approaches when modality quality varies.

## Quick Start

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def train_per_modality(train_df, val_df, modalities, train_fn, predict_fn):
    """Train one model per modality, return ensemble predictions."""
    models = {}
    for mod in modalities:
        models[mod] = train_fn(train_df, val_df, modality=mod)

    val_preds = np.zeros(len(val_df))
    for mod in modalities:
        val_preds += predict_fn(models[mod], val_df, modality=mod)
    val_preds /= len(modalities)

    auc = roc_auc_score(val_df['label'], val_preds)
    return models, auc

modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
models, auc = train_per_modality(df_train, df_val, modalities, train_fn, predict_fn)
```

## Workflow

1. Identify available imaging modalities per patient
2. Train one model per modality using modality-specific data
3. At inference, run each model on its corresponding modality
4. Average predictions across all modalities
5. Handle missing modalities by averaging only available ones

## Key Decisions

- **Equal weighting**: simple average works well; learned weights add complexity for marginal gain
- **Missing modalities**: average over available ones only, don't zero-fill
- **vs multi-channel**: ensemble is more robust when some modalities have quality issues
- **Model architecture**: same architecture per modality, but weights are independent

## References

- [Efficientnet3D with one MRI type](https://www.kaggle.com/code/rluethy/efficientnet3d-with-one-mri-type)
