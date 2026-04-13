---
name: cv-differentiable-soft-f1-loss
description: Use a soft macro-F1 loss `1 − mean(2·tp / (2·tp + fp + fn))` computed from raw sigmoid probabilities (no thresholding) as a direct training objective for multi-label classification, optionally combined with BCE — closes the gap between training surrogate and the F1 metric the leaderboard scores
---

## Overview

When the eval metric is macro-F1, BCE is a poor proxy: it weights every label equally regardless of base rate, and it doesn't push the model to actually pass a threshold. The fix is a *differentiable* macro-F1 surrogate computed directly from sigmoid probabilities — no thresholding, no rounding. Sum probability-weighted TP/FP/FN over the batch, compute per-class F1, average, return `1 − mean_F1`. The loss is well-behaved as long as you mask NaNs from empty classes. In practice it is most effective as a co-loss (`α·BCE + (1−α)·F1_soft`) — pure F1 loss has weak gradients early when no class is being predicted at all; BCE bootstraps confidence and F1 sharpens it.

## Quick Start

```python
import keras.backend as K
import tensorflow as tf

def f1_loss(y_true, y_pred):
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def combo_loss(alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()
    return lambda y, p: alpha * bce(y, p) + (1 - alpha) * f1_loss(y, p)

model.compile(optimizer='adam', loss=combo_loss(0.5))
```

## Workflow

1. Implement the soft-F1 reduction over the batch axis (axis=0), per class
2. Mask NaN / Inf from classes with zero positives in the batch — `tf.where(tf.is_nan(f1), 0, f1)`
3. Train with combo loss for the first ~10 epochs (BCE-weighted), then anneal `alpha` to 0 to fully optimize F1
4. Pair with per-class threshold tuning at inference for the final lift
5. Watch out for very small batches — soft-F1 estimate becomes noisy when batch < 32

## Key Decisions

- **No thresholding inside the loss**: rounding kills gradients; use raw probabilities and let the loss push them apart.
- **Macro vs. micro**: macro-F1 (averaged across classes) is the right pairing for class-imbalanced multi-label; micro-F1 is dominated by frequent classes and barely differs from BCE.
- **Combo with BCE is almost always better than pure F1**: BCE provides stable early gradients that pure F1 lacks when predictions are still around 0.5.
- **Larger batches help**: 64+ samples per batch give enough positives per rare class for the soft-F1 estimate to be reliable.
- **Don't average across batch examples**: average across classes, sum across the batch — that's the macro definition.
- **NaN guard is mandatory**: classes with no positives produce 0/0; without the mask the model collapses.

## References

- [Best loss function for F1-score metric](https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric)
