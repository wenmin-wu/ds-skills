---
name: cv-conv1d-lstm-stroke-classifier
description: Stack 1D convolutions for local feature extraction before bidirectional LSTMs to classify variable-length stroke sequences into hundreds of doodle categories
---

# Conv1D-LSTM Stroke Classifier

## Overview

Sketches can be represented as sequences of (x, y, stroke_id) points rather than rasterized images. A Conv1D-LSTM architecture processes this raw sequence: 1D convolutions extract local patterns (corners, curves), then LSTMs capture long-range stroke order dependencies. This approach works directly on stroke coordinates without rendering, preserving resolution-independent information.

## Quick Start

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, Dense,
    Dropout, BatchNormalization)

n_classes = 340
max_points = 128

model = Sequential([
    BatchNormalization(input_shape=(max_points, 3)),
    Conv1D(48, 5, activation="relu"),
    Dropout(0.3),
    Conv1D(64, 5, activation="relu"),
    Dropout(0.3),
    Conv1D(96, 3, activation="relu"),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(n_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
```

## Workflow

1. Parse strokes into Nx3 matrix: (x, y, stroke_flag) where flag marks stroke boundaries
2. Pad or truncate to fixed length (128-256 points)
3. Normalize x, y coordinates to [0, 1]
4. Feed through Conv1D layers (increasing filters: 48→64→96) for local pattern extraction
5. Feed through stacked LSTMs for sequential modeling
6. Dense layers → softmax over N classes

## Key Decisions

- **Input format**: (x, y, stroke_flag) preserves drawing order; stroke_flag = 1 (continue) or 2 (new stroke)
- **Sequence length**: 128 captures most sketches; longer sequences add diminishing returns
- **Conv before LSTM**: Conv1D reduces sequence length and extracts local features, making LSTM more effective
- **Dropout**: 0.3 between every layer prevents overfitting on the large number of classes
- **vs. image CNN**: stroke-based models are resolution-independent and faster; image CNNs are more accurate with pretrained weights

## References

- [QuickDraw Baseline LSTM Reading and Submission](https://www.kaggle.com/code/kmader/quickdraw-baseline-lstm-reading-and-submission)
