---
name: timeseries-keras-multi-quantile-loss
description: Single neural network outputting all quantiles simultaneously via pinball loss over a quantile vector for joint probabilistic forecasting
---

# Keras Multi-Quantile Loss

## Overview

Instead of training one model per quantile, a single neural network outputs all quantile predictions at once. The final Dense layer has `num_quantiles` outputs, and the loss function computes the pinball (quantile) loss across all quantiles jointly. This is 9x faster than training separate models and naturally produces coherent quantile orderings through shared representations.

## Quick Start

```python
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

def multi_quantile_loss(y_true, y_pred):
    q = tf.constant(np.array([QUANTILES]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)

def build_model(input_dim, num_quantiles=9):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(500, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_quantiles, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=multi_quantile_loss)
    return model

model = build_model(input_dim=50, num_quantiles=len(QUANTILES))
```

## Workflow

1. Define target quantiles as a constant tensor
2. Build a network with `num_quantiles` outputs in the final layer
3. Compute pinball loss: `max(q * e, (q-1) * e)` where `e = y_true - y_pred`
4. Train with standard gradient descent — loss handles all quantiles jointly
5. Predictions are a vector of quantile estimates per sample

## Key Decisions

- **Quantile crossing**: outputs may cross; post-hoc sort or add monotonicity constraint
- **Pinball loss**: asymmetric — penalizes under-prediction more for high quantiles
- **vs separate models**: joint training is faster but may compromise extreme quantiles
- **Target shape**: `y_true` is broadcast across the quantile dimension

## References

- [Quantile-Regression-with-Keras](https://www.kaggle.com/code/ulrich07/quantile-regression-with-keras)
