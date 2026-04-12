---
name: tabular-pearson-correlation-loss
description: >
  Uses negative row-wise Pearson correlation as a differentiable loss function for multi-output regression, directly optimizing the competition metric.
---
# Pearson Correlation Loss

## Overview

When the evaluation metric is row-wise Pearson correlation (average correlation between each sample's predicted and true target vectors), standard MSE is a poor proxy — it penalizes scale and offset errors that don't affect correlation. A differentiable negative Pearson correlation loss directly optimizes the metric. Each row's predictions are mean-centered, then the cosine similarity with the true values gives the correlation. Negating it makes it a minimizable loss. This is especially effective for multi-output regression with 100+ targets.

## Quick Start

```python
import tensorflow as tf
from tensorflow.keras import backend as K

def negative_correlation_loss(y_true, y_pred):
    """Negative row-wise Pearson correlation loss."""
    my = K.mean(y_pred, axis=1, keepdims=True)
    ym = y_pred - my
    r_num = K.sum(y_true * ym, axis=1)
    r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * tf.cast(tf.shape(y_true)[1], tf.float32))
    r = tf.reduce_mean(r_num / r_den)
    return -r

model.compile(optimizer='adam', loss=negative_correlation_loss)
model.fit(X_train, Y_train, epochs=50, batch_size=64)
```

## Workflow

1. Mean-center predictions per row (subtract row mean)
2. Compute dot product of true values and centered predictions per row
3. Divide by the norm of centered predictions times sqrt(n_targets)
4. Average across rows and negate for minimization

## Key Decisions

- **vs MSE**: MSE penalizes scale; correlation loss is scale-invariant
- **Numerical stability**: Add epsilon to denominator to avoid division by zero
- **y_true centering**: Not needed if targets are already normalized per row
- **PyTorch variant**: Replace Keras ops with `torch.mean`, `torch.sum`, `torch.sqrt`

## References

- [All in one: CITEseq & Multiome with Keras](https://www.kaggle.com/code/pourchot/all-in-one-citeseq-multiome-with-keras)
