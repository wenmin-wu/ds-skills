---
name: cv-separable-temporal-spectral-cnn
description: 2D CNN with asymmetric kernels — temporal convolutions (Nx1) then spectral convolutions (1xM) — to decouple time and feature extraction
domain: cv
---

# Separable Temporal-Spectral CNN

## Overview

For 2D inputs with (time, feature) structure (spectrograms, sensor arrays, multi-channel time series), use asymmetric convolution kernels: first apply tall kernels (3x1) along the time axis, then wide kernels (1x3) along the feature axis. This decouples temporal pattern extraction from cross-feature learning, reducing parameters vs square kernels.

## Quick Start

```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization

def build_separable_cnn(time_steps, n_features, n_outputs):
    inp = Input((time_steps, n_features, 1))
    # Temporal convolutions
    x = Conv2D(32, (3, 1), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 1))(x)
    # Spectral convolutions
    x = Conv2D(128, (1, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 2))(x)
    x = Conv2D(64, (1, 3), activation='relu', padding='same')(x)
    # Head
    x = GlobalAveragePooling2D()(x)
    out = Dense(n_outputs)(x)
    return Model(inp, out)
```

## Key Decisions

- **Temporal first**: capture local time patterns before mixing features
- **Asymmetric pooling**: pool along the axis being convolved — (2,1) for time, (1,2) for features
- **Fewer params**: (3,1) + (1,3) has 6 params vs (3,3) with 9 — plus captures axis-specific patterns
- **Generalizable**: works for spectrograms, mel-frequency features, multi-sensor grids

## References

- Source: [host-starter-solution](https://www.kaggle.com/code/gordonyip/host-starter-solution)
- Competition: NeurIPS - Ariel Data Challenge 2024
