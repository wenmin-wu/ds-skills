---
name: tabular-row-wise-target-normalization
description: >
  Normalizes each sample's multi-output target vector to zero mean and unit variance, removing per-sample scale differences before training.
---
# Row-wise Target Normalization

## Overview

In multi-output regression (e.g., predicting 140 protein levels per cell), each sample's target vector can have a different overall magnitude and offset. Row-wise normalization (subtract row mean, divide by row std) removes these per-sample differences, letting the model focus on relative patterns across outputs. This is critical when the evaluation metric is row-wise Pearson correlation, which is inherently scale- and offset-invariant — training on unnormalized targets wastes capacity learning per-sample baselines.

## Quick Start

```python
import numpy as np

# Y shape: (n_samples, n_targets)
row_mean = Y.mean(axis=1, keepdims=True)
row_std = Y.std(axis=1, keepdims=True)
Y_norm = (Y - row_mean) / (row_std + 1e-8)

# Train model on normalized targets
model.fit(X_train, Y_norm)

# At inference — no denormalization needed if metric is correlation
preds = model.predict(X_test)
```

## Workflow

1. Compute per-row mean and std across all target columns
2. Subtract mean and divide by std for each row
3. Train model on normalized targets
4. At inference, predict directly — no inverse transform needed for correlation metrics

## Key Decisions

- **When to apply**: When metric is correlation-based (Pearson, Spearman) or cosine similarity
- **Epsilon**: Add 1e-8 to std to avoid division by zero for constant rows
- **Denormalization**: Only needed if metric is MSE/MAE — store row_mean and row_std from training
- **vs column normalization**: Column-wise (StandardScaler) normalizes features; row-wise normalizes the target profile shape

## References

- [All in one: CITEseq & Multiome with Keras](https://www.kaggle.com/code/pourchot/all-in-one-citeseq-multiome-with-keras)
