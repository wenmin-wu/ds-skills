---
name: tabular-gmm-feature-augmentation
description: >
  Fits a Gaussian Mixture Model on the joint feature-target space and samples synthetic data pairs to augment small tabular datasets.
---
# GMM Feature Augmentation

## Overview

Small tabular datasets (<1000 rows) often underfit — there isn't enough signal for tree models or neural nets to generalize. GMM augmentation fits a Gaussian Mixture Model on the joint space of features and targets, then samples synthetic (X, y) pairs that follow the same distribution. Unlike SMOTE (which only interpolates between neighbors), GMM captures the full multimodal density, generating diverse samples that respect cluster boundaries. This is especially effective for regression tasks where SMOTE doesn't apply.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def gmm_augment(X, y, n_samples=1000, n_components=5, random_state=42):
    """Generate synthetic samples from GMM fitted on joint (X, y) space."""
    df = pd.DataFrame(X).copy()
    df.columns = df.columns.astype(str)
    df['_target'] = y

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    synthetic, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic, columns=df.columns)

    augmented = pd.concat([df, synthetic_df], ignore_index=True)
    X_aug = augmented.drop(columns='_target').values
    y_aug = augmented['_target'].values
    return X_aug, y_aug

X_aug, y_aug = gmm_augment(X_train, y_train, n_samples=2000, n_components=10)
```

## Workflow

1. Concatenate features and target into a joint matrix
2. Fit GMM with K components on the joint space
3. Sample N synthetic rows from the fitted GMM
4. Split synthetic rows back into features and target
5. Concatenate with original data for training

## Key Decisions

- **n_components**: 5-10 for small datasets; use BIC to select optimal K
- **n_samples**: 1-3x the original dataset size; too many synthetic samples can dilute real signal
- **Feature scaling**: Standardize before fitting GMM so all dimensions contribute equally
- **Multi-target**: Include all targets in the joint space so synthetic samples have consistent labels

## References

- [Extra Data with FS (Starting Point)](https://www.kaggle.com/code/alejandrolopezrincon/extra-data-with-fs-starting-point)
