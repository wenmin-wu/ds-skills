---
name: cv-per-class-soft-f1-threshold-fitting
description: Optimize per-class decision thresholds for macro-F1 by replacing the non-differentiable hard threshold with a sigmoid-sharpened soft-F1 surrogate and fitting the per-class threshold vector via least-squares — averaged over multiple random validation splits to suppress overfitting on rare classes
---

## Overview

Picking one global sigmoid threshold (e.g. 0.5) for multi-label classification leaves macro-F1 points on the table because rare classes need lower thresholds and easy classes need higher ones. The naive per-class grid search overfits when validation has only a handful of positives. The trick: replace the hard `(p > th)` step with a sigmoid `σ(d·(p − th))` surrogate, plug into the F1 formula to get a differentiable per-class soft-F1, and use `scipy.optimize.leastsq` with a small L2 penalty to fit the threshold vector. Then average over 10 random train/val splits to denoise. This was the per-class threshold trick that pushed top HPA notebooks from 0.40 to 0.46 LB.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import optimize as opt

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))

def F1_soft(preds, targs, th=0.0, d=25.0):
    p = sigmoid_np(d * (preds - th))
    return 2.0 * (p * targs).sum(0) / ((p + targs).sum(0) + 1e-6)

def fit_thresholds(preds, targs, n_classes, wd=1e-5):
    params = np.zeros(n_classes)
    err = lambda p: np.concatenate(
        (F1_soft(preds, targs, p) - 1.0, wd * p), axis=None)
    p, _ = opt.leastsq(err, params)
    return p

# Average across 10 random splits
th = np.zeros(n_classes)
for i in range(10):
    xt, xv, yt, yv = train_test_split(val_pred, val_y, test_size=0.5, random_state=i)
    th += fit_thresholds(xt, yt, n_classes)
th /= 10
```

## Workflow

1. Hold out enough validation examples that every class has ≥ a few positives
2. Run model inference and keep raw sigmoid scores (not hard predictions)
3. Implement `F1_soft` with sharpness `d ≈ 20–30` — too low and the surrogate isn't tight, too high and gradients vanish
4. Wrap in a `leastsq` residual that drives soft-F1 to 1.0, with small L2 weight decay
5. Fit on multiple random subsamples of validation and average — gives a stable per-class threshold vector
6. Apply at inference: `pred_binary = (test_score > th).astype(int)`

## Key Decisions

- **Sigmoid surrogate sharpness `d`**: 25 is the sweet spot — high enough to track hard-F1, low enough to keep gradients non-zero.
- **Why `leastsq` not gradient descent**: F1 surface has many flats; least-squares with small wd converges in milliseconds and is reproducible.
- **Average across splits, don't fit once**: rare classes overfit a single split easily; 10 splits cuts variance ~3x.
- **Initialize at 0**: the sigmoid centers naturally; non-zero init biases towards a global threshold.
- **Don't fit on training data**: the model is overconfident on train and the thresholds will be too high.
- **Generalizes to any per-instance multi-label problem**: NLP entity tagging, audio multi-tag, recsys multi-genre.

## References

- [Pretrained ResNet34 with RGBY (0.460 public LB)](https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb)
