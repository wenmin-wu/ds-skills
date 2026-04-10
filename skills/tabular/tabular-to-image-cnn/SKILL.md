---
name: tabular-tabular-to-image-cnn
description: >
  Reshapes tabular features into 2D pseudo-images via random feature permutation, enabling CNN-based feature interaction learning.
---
# Tabular-to-Image CNN

## Overview

Convert flat tabular features into a 2D grid by repeating the feature vector with random permutations, then apply 2D convolutions. Each row of the "image" is a different random ordering of features, so the CNN learns local interactions between different feature pairs. This captures non-linear feature interactions that tree models and MLPs may miss.

## Quick Start

```python
import numpy as np
import tensorflow as tf
from random import choice

def build_tabular_cnn(n_feats, n_repeats=50):
    # Create random permutation mask
    mask = np.zeros((n_repeats, n_feats), dtype=np.int32)
    for i in range(n_repeats):
        indices = list(range(n_feats))
        for j in range(n_feats):
            mask[i, j] = indices.pop(choice(range(len(indices))))

    inp = tf.keras.layers.Input(shape=(n_feats,))
    x = tf.keras.layers.Lambda(lambda x: tf.gather(x, mask, axis=1))(inp)
    x = tf.keras.layers.Reshape((n_repeats, n_feats, 1))(x)
    x = tf.keras.layers.Conv2D(32, (n_repeats, n_feats), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)
```

## Workflow

1. Generate a random permutation mask of shape (n_repeats, n_features)
2. Apply mask to each input vector → 2D grid (n_repeats × n_features)
3. Add channel dimension → feed into Conv2D layers
4. Flatten and pass through dense head for prediction
5. Use as one model in an ensemble alongside GBDT

## Key Decisions

- **n_repeats**: 50 gives diverse feature orderings; more adds computation, less limits interactions
- **Kernel size**: Large kernels (covering full width) let the CNN see all features at once
- **When to use**: As a diversity member in ensembles — rarely beats GBDT alone but adds decorrelated predictions
- **Freeze mask**: Generate mask once at init, reuse across epochs for reproducibility

## References

- 2019 Data Science Bowl (Kaggle)
- Source: [convert-to-regression](https://www.kaggle.com/code/braquino/convert-to-regression)
