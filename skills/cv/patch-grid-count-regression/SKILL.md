---
name: cv-patch-grid-count-regression
description: Tile a large aerial image into fixed-size patches, accumulate per-class point-annotation counts into a grid tensor aligned with the tiles, and train a small CNN to regress per-class object counts per patch under MSE
---

## Overview

When you only have point annotations and the target is a *count* (not bounding boxes or pixel masks), patch-level count regression is the lightest-weight recipe that works. Tile the image into fixed-size patches, accumulate point labels into a `(grid_x, grid_y, n_classes)` count tensor, then train a small CNN with a linear head to predict per-class counts per patch under MSE. At inference you sum the patch predictions over the image. It dodges the complexity of detection, segmentation, and density maps, and it's exactly what won NOAA Steller Sea Lion Population Count top kernels.

## Quick Start

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

PATCH = 300
H, W = img.shape[:2]
grid = np.zeros((W // PATCH + 1, H // PATCH + 1, n_classes), dtype='int16')
for x, y, cls in points:
    grid[x // PATCH, y // PATCH, cls] += 1

X, Y = [], []
for i in range(W // PATCH):
    for j in range(H // PATCH):
        X.append(img[j*PATCH:(j+1)*PATCH, i*PATCH:(i+1)*PATCH])
        Y.append(grid[i, j])
X = np.array(X); Y = np.array(Y, dtype='float32')

model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(PATCH, PATCH, 3)),
    Conv2D(64, 3, activation='relu', padding='same'), MaxPooling2D(),
    Flatten(), Dense(256, activation='relu'),
    Dense(n_classes, activation='linear'),     # linear head — count regression
])
model.compile(loss='mse', optimizer='adam')
model.fit(X, Y, epochs=20, batch_size=32)
```

## Workflow

1. Choose a patch size matched to object scale (patch ≈ 10× object diameter works well)
2. Accumulate point annotations into a `(grid_x, grid_y, n_classes)` integer tensor
3. Crop the image into patches aligned to the grid; flatten to `(N_patches, H, W, 3)` and `(N_patches, n_classes)`
4. Train a small CNN with a **linear** output head under MSE — no softmax, no ReLU on the final layer
5. At inference, predict per-patch counts and sum across patches for the image-level total per class

## Key Decisions

- **Linear output head**: count regression is unbounded; softmax/sigmoid would cap it. ReLU is fine too but linear is cleaner for MSE.
- **Patch size**: too small → most patches contain 0 objects and the model overfits background; too large → counts are high-variance and hard to regress. Tune empirically.
- **Integer count tensor, float labels**: accumulate as int16 during labeling, cast to float32 right before training.
- **Sample pos:neg ~1:3**: purely random tiles are mostly empty; downsample empties so rare classes are seen.
- **vs density maps**: density maps need per-pixel labels and kernel tuning; patch counts need neither and are 10× simpler to ship.

## References

- [Use keras to count sea lions](https://www.kaggle.com/code/outrunner/use-keras-to-count-sea-lions)
