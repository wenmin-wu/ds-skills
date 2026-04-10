---
name: timeseries-class-weighted-multiclass-logloss
description: Custom multiclass log-loss that weights per-class contributions by class frequency and domain importance, usable as both training loss and eval metric
domain: timeseries
---

# Class-Weighted Multiclass Log-Loss

## Overview

For imbalanced multiclass problems, standard log-loss is dominated by frequent classes. Weight each class's contribution by (1) its inverse frequency and (2) an optional domain-importance multiplier. Normalize the sum by total weight. Implement as both a Keras/TF training loss and a numpy evaluation metric for consistent optimization.

## Quick Start

```python
import numpy as np
import tensorflow as tf

def weighted_logloss_np(y_true_ohe, y_pred, class_weights, class_counts):
    """Numpy version for OOF evaluation.
    
    Args:
        y_true_ohe: one-hot encoded ground truth
        y_pred: predicted probabilities
        class_weights: dict {class_id: importance_weight}
        class_counts: per-class sample counts
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    per_class_loss = np.sum(y_true_ohe * np.log(y_pred), axis=0)
    weights = np.array([class_weights[c] for c in sorted(class_weights)])
    weighted_loss = per_class_loss * weights / class_counts
    return -np.sum(weighted_loss) / np.sum(weights)

def weighted_logloss_tf(class_freq):
    """Keras-compatible training loss."""
    freq = tf.constant(class_freq, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        per_class = tf.reduce_mean(y_true * tf.math.log(y_pred), axis=0)
        return -tf.reduce_mean(per_class / freq)
    return loss
```

## Key Decisions

- **Divide by class count**: normalizes so rare classes contribute equally to the loss
- **Domain weights**: multiply by importance (e.g. 2x for critical rare classes)
- **Clip predictions**: 1e-15 floor prevents log(0) explosion
- **Consistent metric**: use same formula for training loss and validation metric

## References

- Source: [simple-neural-net-for-time-series-classification](https://www.kaggle.com/code/meaninglesslives/simple-neural-net-for-time-series-classification)
- Competition: PLAsTiCC Astronomical Classification
