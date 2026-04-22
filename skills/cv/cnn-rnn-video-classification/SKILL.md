---
name: cv-cnn-rnn-video-classification
description: Extract per-frame CNN features then classify the temporal sequence with stacked GRU layers and a boolean mask for variable-length video inputs
---

# CNN-RNN Video Classification

## Overview

For video classification that benefits from temporal context (action recognition, deepfake detection), extract fixed-length feature vectors from each frame using a pretrained CNN (InceptionV3, ResNet), then feed the sequence to a GRU/LSTM. A boolean mask handles variable-length videos by ignoring padded positions. This two-stage approach decouples spatial feature learning from temporal modeling.

## Quick Start

```python
import tensorflow as tf
from tensorflow import keras

feature_extractor = keras.applications.InceptionV3(
    weights="imagenet", include_top=False, pooling="avg")

MAX_SEQ = 30
FEAT_DIM = 2048

frame_input = keras.Input((MAX_SEQ, FEAT_DIM))
mask_input = keras.Input((MAX_SEQ,), dtype="bool")
x = keras.layers.GRU(16, return_sequences=True)(frame_input, mask=mask_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model([frame_input, mask_input], output)
model.compile(loss="binary_crossentropy", optimizer="adam")
```

## Workflow

1. Sample frames from each video (fixed max, e.g., 30)
2. Extract per-frame features: `feature_extractor.predict(frame)` → (2048,) vector
3. Stack into a matrix of shape (max_seq, feat_dim), zero-pad shorter videos
4. Create a boolean mask: `True` for real frames, `False` for padding
5. Train GRU on (features, mask) → binary label
6. At inference, apply the same sampling and feature extraction

## Key Decisions

- **CNN backbone**: InceptionV3 (2048-d) or EfficientNet (1280-d); freeze weights for speed
- **RNN type**: GRU is faster than LSTM with comparable performance for short sequences
- **Stacking**: 2 GRU layers (16→8 units) is sufficient; deeper stacks overfit on small datasets
- **Masking**: essential for variable-length inputs — without it, the GRU learns to predict from padding
- **vs. 3D CNN**: CNN-RNN is more parameter-efficient and easier to pretrain; 3D CNNs capture fine-grained motion better

## References

- [Deep Fake Detection on Images and Videos](https://www.kaggle.com/code/krooz0/deep-fake-detection-on-images-and-videos)
