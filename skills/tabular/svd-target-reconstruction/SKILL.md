---
name: tabular-svd-target-reconstruction
description: >
  Compresses high-dimensional targets with TruncatedSVD, trains on the reduced space, then reconstructs full predictions via the components matrix.
---
# SVD Target Reconstruction

## Overview

When predicting thousands of outputs (e.g., 23,418 gene expressions), training directly is slow and noisy. TruncatedSVD compresses targets to 64-512 components, capturing the dominant variance. The model trains on these compact representations, then at inference, predictions are projected back to full dimensionality by multiplying with the SVD components matrix. This reduces training time by 10-100x while preserving 95%+ of target variance, and acts as implicit regularization by discarding noise in low-variance components.

## Quick Start

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy.sparse

# Y_train shape: (n_samples, 23418) — can be sparse
svd = TruncatedSVD(n_components=128, random_state=42)
Y_reduced = svd.fit_transform(Y_train)  # (n_samples, 128)

print(f"Variance retained: {svd.explained_variance_ratio_.sum():.3f}")

# Train model on reduced targets
model.fit(X_train, Y_reduced)

# Reconstruct full predictions at inference
preds_reduced = model.predict(X_test)  # (n_test, 128)
preds_full = preds_reduced @ svd.components_  # (n_test, 23418)
```

## Workflow

1. Fit TruncatedSVD on training targets (works with sparse matrices)
2. Transform targets to reduced space (64-512 dims)
3. Train model on reduced targets
4. At inference, multiply predictions by `svd.components_` to reconstruct

## Key Decisions

- **n_components**: 64-512; check `explained_variance_ratio_.sum()` — aim for >0.90
- **TruncatedSVD vs PCA**: TruncatedSVD works on sparse matrices without centering; PCA requires dense
- **Both inputs and targets**: Can reduce both independently for ultra-high-dimensional problems
- **Fold-wise SVD**: Fit SVD per CV fold to avoid target leakage (minor effect in practice)

## References

- [MSCI CITEseq Quickstart](https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart)
- [LB T15 MSCI Multiome CatBoostRegressor](https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor)
