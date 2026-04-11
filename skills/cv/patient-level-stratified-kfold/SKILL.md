---
name: cv-patient-level-stratified-kfold
description: >
  Stratifies CV folds at the patient level rather than image level, preventing data leakage when multiple images exist per patient.
---
# Patient-Level Stratified KFold

## Overview

In medical imaging, each patient has multiple images (e.g., left/right breast, multiple slices, follow-up scans). Standard image-level KFold leaks information — images from the same patient can appear in both train and validation, inflating metrics by 0.01–0.05. Patient-level stratification ensures all images from one patient are in the same fold, while still balancing the target distribution across folds. Essential for any medical competition (RSNA, SIIM, VinBigData).

## Quick Start

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Aggregate to patient-level label (any positive image = positive patient)
patient_labels = train_df.groupby('patient_id')['target'].max().reset_index()

# Split at patient level, stratified by patient-level label
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
patient_labels['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(
    patient_labels['patient_id'], patient_labels['target']
)):
    patient_labels.loc[val_idx, 'fold'] = fold

# Map fold back to image-level DataFrame
train_df = train_df.merge(
    patient_labels[['patient_id', 'fold']], on='patient_id'
)

# Use in training
for fold in range(5):
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    # No patient overlap between train_idx and val_idx
```

## Workflow

1. Aggregate target to patient level (`groupby('patient_id').target.max()`)
2. Run StratifiedKFold on patient-level DataFrame
3. Assign fold numbers to patients
4. Merge fold assignments back to image-level DataFrame
5. All images from one patient are in the same fold

## Key Decisions

- **Aggregation**: `max()` for binary (any positive = positive patient); `mean()` for regression
- **Stratification**: On patient-level label, not image-level — ensures balanced class distribution
- **GroupKFold alternative**: `GroupKFold` prevents leakage but doesn't stratify; this does both
- **Multi-label**: For multi-condition labels, stratify on the rarest positive condition

## References

- [fast.ai Starter Pack - Train + Inference](https://www.kaggle.com/code/radek1/fast-ai-starter-pack-train-inference)
